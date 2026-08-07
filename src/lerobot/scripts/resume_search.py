#!/usr/bin/env python
"""LoLA resume 搜索: 在 run 集合目录下按训练配置匹配 checkpoint, 选步数最多者。

背景: 原 --resume 只能指向 (a) 含 latest 指针的 run 目录, 或 (b) 具体 tag 目录
(step_XXXXXX / final)。本模块增加第三种形态: (c) run 集合目录 — 扫描其中所有
run 的 training_config.json, 与当前训练配置做【语义子集】匹配, 在匹配的 run 中
选 latest 步数最多者。

"语义子集"而非整份 JSON 完全匹配的原因 (整份永远匹配不上):
  - training_args = vars(args) 含 resume/wandb_id/ckpt_dir 等 run 身份字段
    (resume 字段本身每次 resume 都不同);
  - distributed 含 world_rank/node_rank/device 等 rank0 现场值, 只有 world_size
    有语义 (ZeRO-3 按 rank 分片保存, 不同 world_size 的 checkpoint 不可互载);
  - lola_config.vlm_path 等路径字段在本地化 IO 下会变成节点本地镜像路径。

匹配规则:
  - lola_config: 全量比较, 仅排除 LOLA_CONFIG_EXCLUDE_KEYS (路径/运行时/性能开关);
    新增配置字段自动纳入比较, 防止"忘了加白名单"导致的错误匹配。
  - training_args: 白名单比较 (TRAINING_ARGS_INCLUDE_KEYS), 只含影响训练语义/
    checkpoint 兼容性的键。
  - distributed: 仅 world_size。
  - dataset_metadata: 全量 (total_episodes/total_frames/fps/features shape+type,
    不含路径)。

步数语义: run 的 latest 指针解析 tag; step_N → N; final → 视为无穷大
(同配置下 final 意味着该配置的训练已完成; 并列时按 run 目录名(时间戳)取新者)。

使用方式:
  1) trainer 内 (test_azure_v07.sh / 非本地化): main() 在构建 trainer 前调用
     resolve_resume_auto, 三种形态自动识别。
  2) launcher bash 内 (test_azure_v07c.sh 本地化): 先用 azcopy 把集合目录下的
     latest + training_config.json 元数据拉回本地, 再以 CLI 方式调用本模块:
       python resume_search.py --resolve_parent <本地集合目录> \
           --local_dataset_root <本地数据集根> --world_size <N> -- <trainer 参数原样透传>
     stdout 仅输出选中的 run 目录名 (供 bash 捕获), 候选表打到 stderr;
     无匹配时 stdout 为空, bash 据此降级为从头训练。

无匹配语义: 搜索模式下无候选/无匹配 → 返回 None + ⚠️ 日志, 训练从头开始
(resume 禁用); 但 --resume 路径不存在仍响亮报错 (防打错路径白训一场)。

注意: 严格匹配是有意的 — 改了训练配置想续训时, 请直接把 --resume 指向具体
run 目录 (旧行为, 绕过搜索)。搜索模式只回答"同一配置最近一次跑到哪了"。
"""

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

# lola_config 比较时排除的键:
#   路径 (本地化 IO 会改写), 运行时现场值, 纯性能开关 (不影响 checkpoint 兼容性)
LOLA_CONFIG_EXCLUDE_KEYS = frozenset({
    "vlm_path",
    "device",
    "compile_model",
    "compile_mode",
    "gradient_checkpointing",
    "dit_gradient_checkpointing",
})

# training_args 比较白名单: 影响训练语义或 checkpoint 兼容性的键。
# 语义覆盖说明: 大部分模型/训练超参住在 lola_config 里 (全量比较), 这里只补
# lola_config 之外的训练语义。排除: 路径 (dataset_root/ckpt_dir/resume/vlm_path),
# run 身份 (wandb_*), 操作性参数 (num_workers/log_every/save_every/静态padding),
# 纯性能参数 (deepspeed bucket sizes)。
TRAINING_ARGS_INCLUDE_KEYS = frozenset({
    "strategy",
    "batch_size",
    "max_steps",
    "max_epochs",
    "learning_rate",
    "gradient_clip_val",
    "deepspeed_zero_stage",
    "dataset_repo_id",
    "episodes",
    "stats_mode",
})

_STEP_TAG_RE = re.compile(r"step_(\d+)")
_TAG_DIR_RE = re.compile(r"(step_\d+|final)")


def make_serializable(obj):
    """与 training_config.json 写入时相同的序列化 (train_lola_v07_azure.train 原逻辑)。"""
    import torch

    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return make_serializable(dataclasses.asdict(obj))
    return obj


def canon(obj):
    """归一化为 JSON round-trip 后的纯 python 结构 (tuple→list, enum/dtype→str),
    使当前快照与磁盘读回的 training_config.json 可按值比较。"""
    return json.loads(json.dumps(make_serializable(obj), default=str))


def build_current_snapshot(args, lola_config, dataset_metadata_dict, world_size):
    """构建当前训练配置的语义快照 (与 training_config.json 同构的可比较子集)。

    Args:
        args: trainer argparse namespace (当前运行)
        lola_config: 完整构建的 LoLAV07Config (含 __post_init__ 解析结果)
        dataset_metadata_dict: build_dataset_metadata_snapshot() 的产物
        world_size: 分布式总进程数
    """
    ta = vars(args)
    return {
        "lola_config": {
            k: v for k, v in canon(dataclasses.asdict(lola_config)).items()
            if k not in LOLA_CONFIG_EXCLUDE_KEYS
        },
        "training_args": {
            k: canon(ta[k]) for k in sorted(TRAINING_ARGS_INCLUDE_KEYS) if k in ta
        },
        "distributed": {"world_size": int(world_size)},
        "dataset_metadata": canon(dataset_metadata_dict),
    }


def _candidate_comparable(candidate_json):
    """从磁盘 training_config.json 提取可比较子集 (与 build_current_snapshot 同构)。"""
    return {
        "lola_config": {
            k: v for k, v in (candidate_json.get("lola_config") or {}).items()
            if k not in LOLA_CONFIG_EXCLUDE_KEYS
        },
        "training_args": {
            k: v for k, v in (candidate_json.get("training_args") or {}).items()
            if k in TRAINING_ARGS_INCLUDE_KEYS
        },
        "distributed": {
            "world_size": (candidate_json.get("distributed") or {}).get("world_size")
        },
        "dataset_metadata": candidate_json.get("dataset_metadata") or {},
    }


def _deep_diff(current, candidate, prefix, diffs, max_diffs=20):
    """递归按键路径比较, 收集差异描述 (最多 max_diffs 条)。"""
    if len(diffs) >= max_diffs:
        return
    if isinstance(current, dict) and isinstance(candidate, dict):
        keys = sorted(set(current) | set(candidate))
        for k in keys:
            path = f"{prefix}.{k}" if prefix else str(k)
            if k not in current:
                diffs.append(f"{path}: 候选多出 (当前无此键)")
            elif k not in candidate:
                diffs.append(f"{path}: 候选缺失 (当前={current[k]!r})")
            else:
                _deep_diff(current[k], candidate[k], path, diffs, max_diffs)
            if len(diffs) >= max_diffs:
                return
    elif current != candidate:
        diffs.append(f"{prefix}: 当前={current!r} 候选={candidate!r}")


def diff_snapshot(current_snapshot, candidate_json):
    """返回当前快照与候选 training_config.json 的差异列表 (空 = 匹配)。"""
    diffs = []
    _deep_diff(current_snapshot, _candidate_comparable(candidate_json), "", diffs)
    return diffs


def run_latest_step(run_dir):
    """解析 run 目录的最新步数。

    Returns:
        (step, tag): latest 指向 step_N → (N, tag); 指向 final → (inf, "final");
        latest 缺失/不可读 → (None, None)。
    """
    latest_path = os.path.join(run_dir, "latest")
    if not os.path.isfile(latest_path):
        return None, None
    try:
        tag = open(latest_path).read().strip()
    except OSError:
        return None, None
    m = _STEP_TAG_RE.fullmatch(tag)
    if m:
        return int(m.group(1)), tag
    if tag == "final":
        # 同配置下 final 意味着该配置训练已完成, 优先级最高 (见模块 docstring)
        return float("inf"), tag
    # 未知 tag: 视为 step 0 (不跳过, 但排序最低)
    return 0, tag


def resolve_resume_search(parent_dir, current_snapshot, log=print):
    """在集合目录下按训练配置匹配 run, 返回 latest 步数最多者的路径。

    Returns:
        选中 run 目录的路径; 无候选 / 无匹配时返回 None (调用方应降级为从头训练
        并保留上方 ⚠️ 日志)。

    Raises:
        RuntimeError: 搜索根目录不存在 (大概率路径打错, 响亮报错优于静默从头训练)。
    """
    if not os.path.isdir(parent_dir):
        raise RuntimeError(f"[resume-search] 搜索根目录不存在: {parent_dir}")

    candidates = []  # (run_name, run_dir, step, tag, diffs)
    for name in sorted(os.listdir(parent_dir)):
        run_dir = os.path.join(parent_dir, name)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "training_config.json")
        if not os.path.isfile(cfg_path):
            continue  # 非 run 目录 (或无 config 的旧格式), 跳过
        try:
            with open(cfg_path) as f:
                cand_json = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log(f"[resume-search]   {name}: 跳过 (training_config.json 不可读: {e})")
            continue
        step, tag = run_latest_step(run_dir)
        if step is None:
            log(f"[resume-search]   {name}: 跳过 (无 latest 指针, run 尚未保存过 checkpoint)")
            continue
        diffs = diff_snapshot(current_snapshot, cand_json)
        candidates.append((name, run_dir, step, tag, diffs))

    if not candidates:
        log(f"[resume-search] ⚠️ {parent_dir} 下未找到任何带 training_config.json + latest 的 run 目录"
            f" — 找不到匹配的 checkpoint, 将从头开始训练 (resume 已禁用)")
        return None

    log(f"[resume-search] 在 {parent_dir} 下找到 {len(candidates)} 个候选 run:")
    for name, _, step, tag, diffs in candidates:
        step_str = "final" if step == float("inf") else str(step)
        if not diffs:
            log(f"[resume-search]   ✅ {name}: latest={tag} (step {step_str}) — 配置匹配")
        else:
            log(f"[resume-search]   ❌ {name}: latest={tag} (step {step_str}) — 配置不匹配, 差异 {len(diffs)} 处:")
            for d in diffs[:5]:
                log(f"[resume-search]        {d}")
            if len(diffs) > 5:
                log(f"[resume-search]        ... 其余 {len(diffs) - 5} 处从略")

    matches = [c for c in candidates if not c[4]]
    if not matches:
        # 给出最接近的候选, 便于判断是配置漂移还是路径问题
        closest = min(candidates, key=lambda c: len(c[4]))
        log(f"[resume-search] ⚠️ {len(candidates)} 个候选 run 的配置均不匹配当前训练配置"
            f" — 找不到匹配的 checkpoint, 将从头开始训练 (resume 已禁用)")
        log(f"[resume-search]   最接近的候选 {closest[0]} 差异 {len(closest[4])} 处:")
        for d in closest[4][:10]:
            log(f"[resume-search]     {d}")
        log(f"[resume-search]   若确认要从某 run 续训 (配置漂移是有意的), 请直接把 --resume 指向该 run 目录")
        return None

    # 步数最多者; 并列按目录名 (含时间戳) 取新者
    best = max(matches, key=lambda c: (c[2], c[0]))
    step_str = "final" if best[2] == float("inf") else str(best[2])
    log(f"[resume-search] 命中 {len(matches)} 个匹配 run, 选中步数最多者: "
        f"{best[0]} (latest={best[3]}, step {step_str})")
    return best[1]


def resolve_resume_auto(path, current_snapshot, log=print):
    """--resume 三形态自动识别: tag 目录 / 含 latest 的 run 目录 / run 集合目录。

    tag 目录与 run 目录形态原样返回 (旧行为); 集合目录进入搜索模式, 无匹配时
    返回 None (调用方降级为从头训练); 路径不存在仍报错 (防打错路径白训一场)。
    """
    if not path:
        return path
    base = os.path.basename(path.rstrip("/"))
    if os.path.isfile(os.path.join(path, "latest")):
        return path  # run 目录: DS latest 指针在, 旧行为
    if _TAG_DIR_RE.fullmatch(base) and os.path.isdir(path):
        return path  # 具体 tag 目录 (step_XXXXXX / final), 旧行为
    if os.path.isdir(path):
        log(f"[resume-search] --resume 指向 run 集合目录, 进入搜索模式: {path}")
        return resolve_resume_search(path, current_snapshot, log=log)
    raise RuntimeError(f"[resume-search] --resume 路径不存在: {path}")


def _cli(argv):
    """bash 辅助入口: 重建当前配置快照并在本地元数据目录中解析 resume run。

    stdout: 仅一行 — 选中的 run 目录名 (供 bash $(...) 捕获)。
    stderr: 候选表与日志。
    """
    ap = argparse.ArgumentParser(description="LoLA resume 搜索 (launcher 本地化模式辅助)")
    ap.add_argument("--resolve_parent", required=True, help="run 集合目录 (本地, 已预下载元数据)")
    ap.add_argument("--local_dataset_root", default=None,
                    help="本地数据集根 (覆盖透传参数里的 --dataset_root, 后者可能还是挂载点路径)")
    ap.add_argument("--world_size", type=int, required=True, help="总训练进程数 (nnodes × nproc)")
    ap.add_argument("--dump_snapshot", action="store_true", help="只打印当前配置快照 (调试用)")
    known, remaining = ap.parse_known_args(argv)
    # "--" 分隔符可能被 argparse 留在 remaining 里; 不剥掉的话第二轮 parse 会把
    # 其后的 trainer 参数全部当作位置参数 (静默丢失 → 快照退回默认值)
    if "--" in remaining:
        remaining.remove("--")

    # 与 trainer 共享 parser / config 构建, 保证快照与实际训练进程严格同源
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_lola_v07_azure import (
        build_arg_parser,
        build_dataset_metadata_snapshot,
        build_lola_config,
    )

    args, _ = build_arg_parser().parse_known_args(remaining)
    if known.local_dataset_root:
        args.dataset_root = known.local_dataset_root

    import contextlib

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    # 快照构建期的日志 (_log/警告等) 一律走 stderr, stdout 只留最终结果行,
    # 供 bash RUN_NAME=$(...) 捕获
    with contextlib.redirect_stdout(sys.stderr):
        meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
        config, features, _, _ = build_lola_config(args, meta)
        snapshot = build_current_snapshot(
            args, config, build_dataset_metadata_snapshot(meta, features), known.world_size
        )
    if known.dump_snapshot:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return

    selected = resolve_resume_search(
        known.resolve_parent, snapshot,
        log=lambda m: print(m, file=sys.stderr, flush=True),
    )
    # 无匹配: stdout 留空 (exit 0), bash 据此降级为从头训练
    if selected is not None:
        print(os.path.basename(selected.rstrip("/")))


if __name__ == "__main__":
    _cli(sys.argv[1:])
