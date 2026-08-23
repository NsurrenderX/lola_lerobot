"""ZeRO-3 save / resume / VLM-unfreeze 集成测试 (方案 §25)。

必须 world_size >= 2 —— 单卡 ZeRO-3 的参数分片是平凡的, "grad 分区视图 shape 失配"
这类只在真实分片下才出现的问题复现不出来, 单卡通过不算验收完成。

覆盖的 10 步:
  1-2  启动 summary 模型 (VLM 冻结), 训练 >=2 步
  3-4  保存 checkpoint, 销毁全部进程          -> phase a
  5-7  fresh 进程 resume, 再训练 >=2 步,
       检查参数更新 / buffer / optimizer coverage / 无 shape error   -> phase b
  8-10 人为触发 VLM unfreeze 重建 (第三条参数组构造路径),
       再训练 >=2 步, 复查覆盖与 pool 参数是否仍在更新             -> phase c

用真实的 LoLAV07Pytorch + build_lola_v07_param_groups + get_deepspeed_config,
VLM 用一个小 MLP 替身 (只需要参与计算图并能被冻结/解冻)。

运行 (conda env lerobot-gcr3, 2x GPU):

    cd <repo> && export PYTHONPATH=$PWD/src
    PY=$HOME/anaconda3/envs/lerobot-gcr3/bin/python
    D=/tmp/lola_zero3_test && rm -rf $D && mkdir -p $D
    for p in a b c; do
        $PY -m torch.distributed.run --nproc_per_node=2 \
            tests/test_lola_v07_zero3_resume.py --phase $p --dir $D || break
    done
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lerobot" / "scripts"))

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn

from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config
from lerobot.policies.lola_v07.modeling_lola_v07 import (
    LoLAV07Pytorch,
    assert_param_group_coverage,
    build_lola_v07_param_groups,
)
from train_lola_v07_azure import get_deepspeed_config

PASS, FAIL = "✅", "❌"
failures = []

VLM_HIDDEN, EXTRACT_LAYERS, VLM_LEN = 128, (0, 1, 2), 16
MICRO_BATCH, BUDGET, STATE_DIM = 2, 32, 7
BASE_LR = 1e-3  # 放大 LR, 让"参数确实在更新"这件事在几步内可观测
STATE_MEAN = torch.arange(STATE_DIM, dtype=torch.float32) * 0.1
STATE_STD = torch.full((STATE_DIM,), 0.5)


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def make_config() -> LoLAV07Config:
    return LoLAV07Config(
        action_dim=7, state_dim=STATE_DIM, action_chunk_size=8, pred_chunk_size=16,
        dit_hidden_size=512, dit_num_heads=8, dit_double_layers=2, dit_single_layers=2,
        action_bottleneck_dim=256, grip_bottleneck_dim=128,
        state_bottleneck_dim=256, state_grip_bottleneck_dim=128,
        vlm_hidden_size=VLM_HIDDEN, vlm_extract_layers=EXTRACT_LAYERS, vlm_bridge_mode="legacy",
        history_type="state", state_encoder_mode="unified",
        load_full_history=True, use_special_tokens=True, use_previous_task_end=True,
        hist_action_token_drop_rate=0.0, transition_mask_rate=0.0,
        completed_tasks_history_len=4,
        history_tokenization_mode="segment_summary", history_architecture_version=1,
        history_summary_num_heads=8,
        max_transition_summary_frames=BUDGET, max_task_summary_frames=BUDGET,
        encoder_lr_mult=1.0, gradient_checkpointing=False,
    )


class FakePolicy(nn.Module):
    """结构与 LoLAV07Policy 一致的最小替身: .model / .vlm / .config。

    VLM 用小 MLP 代替真权重, 但真实接进计算图 —— 否则解冻路径测不出东西。
    """

    def __init__(self, cfg: LoLAV07Config):
        super().__init__()
        self.config = cfg
        self.model = LoLAV07Pytorch(cfg)
        self.vlm = nn.Sequential(nn.Linear(VLM_HIDDEN, VLM_HIDDEN), nn.SiLU(),
                                 nn.Linear(VLM_HIDDEN, VLM_HIDDEN))

    def forward(self, batch):
        feat = self.vlm(batch["vlm_in"])
        hidden = {i: feat for i in self.config.vlm_extract_layers}
        out = self.model(
            hidden_states_all_layers=hidden,
            input_ids=batch["input_ids"],
            hist_actions=None,
            target_actions=batch["action"],
            segment_history=batch["segment_history"],
        )
        return out["total_loss"]


def make_batch(device, dtype, step: int, rank: int):
    g = torch.Generator().manual_seed(1000 * step + rank)
    b = MICRO_BATCH

    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(device=device)

    tm = torch.zeros(b, BUDGET, dtype=torch.bool)
    tl = torch.zeros(b, dtype=torch.long)
    for i in range(b):
        n = 0 if i % 2 else BUDGET          # 覆盖 present=True / False 两种几何
        if n:
            tm[i, BUDGET - n:] = True
        tl[i] = n
    return {
        "vlm_in": rnd(b, VLM_LEN, VLM_HIDDEN).to(dtype),
        "input_ids": torch.zeros(b, VLM_LEN, dtype=torch.long, device=device),
        "action": rnd(b, 16, 7),
        "segment_history": {
            "transition_states": rnd(b, BUDGET, STATE_DIM),
            "transition_frame_mask": tm.to(device),
            "transition_total_length": tl.to(device),
            "task_states": rnd(b, BUDGET, STATE_DIM),
            "task_frame_mask": torch.ones(b, BUDGET, dtype=torch.bool, device=device),
            "task_total_length": torch.full((b,), BUDGET, dtype=torch.long, device=device),
            "transition_drop": (torch.arange(b, device=device) % 3 == 0),
            "task_drop": (torch.arange(b, device=device) % 2 == 0),
        },
    }


def shard_fingerprint(policy: nn.Module) -> dict:
    """fp32 master 分片的 L2 范数 —— 用于判断"参数是否在更新"。

    不能直接看 bf16 权重: lr 量级的更新常低于 bf16 的 ULP (~7.8e-3 @ |w|~1),
    会误判成"参数没更新"。同 world_size 的分片布局确定, 可按 name 逐一比对。
    """
    try:
        from deepspeed.utils import safe_get_local_fp32_param
    except ImportError:
        safe_get_local_fp32_param = None

    out = {}
    for name, p in policy.named_parameters():
        shard = None
        if safe_get_local_fp32_param is not None and hasattr(p, "ds_id"):
            try:
                shard = safe_get_local_fp32_param(p)
            except Exception:  # noqa: BLE001 — 未进优化器的参数没有 master 副本
                shard = None
        if shard is None:
            shard = p.ds_tensor if getattr(p, "ds_tensor", None) is not None else p.data
        out[name] = float(shard.detach().float().norm())
    return out


def module_fingerprint(policy: nn.Module) -> dict:
    """模块权重 (bf16) 分片的 L2 范数 —— 用于判断"权重是否被回载"。

    load_optimizer_states=False 时 fp32 master 不会被回载, 因此 roundtrip 校验
    必须看模块权重而不是 master。
    """
    return {
        name: float((p.ds_tensor if getattr(p, "ds_tensor", None) is not None else p.data)
                    .detach().float().norm())
        for name, p in policy.named_parameters()
    }


def close(a: float, b: float, rtol: float = 5e-3) -> bool:
    """相对比较: bf16 分片的范数在大张量上的舍入误差远超固定绝对阈值。"""
    return abs(a - b) <= rtol * max(1.0, abs(b))


def build_engine(policy, cfg, world_size, include_vlm: bool, train_vlm: bool):
    groups = build_lola_v07_param_groups(
        policy, cfg, BASE_LR, vlm_lr=BASE_LR * cfg.vlm_lr_mult, include_vlm=include_vlm)
    assert_param_group_coverage(policy, groups)
    # deepspeed.initialize 会就地改写 group["params"], 构造后再查就查不到了
    vlm_ids = {id(p) for p in policy.vlm.parameters()}
    n_vlm_groups = sum(1 for g in groups if any(id(x) in vlm_ids for x in g["params"]))
    ds_config = get_deepspeed_config(
        learning_rate=BASE_LR, weight_decay=0.0, gradient_clip_val=1.0,
        train_vlm=train_vlm, batch_size=MICRO_BATCH, world_size=world_size,
        reduce_bucket_size=int(5e7), allgather_bucket_size=int(5e7), zero_stage=3,
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=policy, model_parameters=groups, config=ds_config, dist_init_required=False)
    return engine, n_vlm_groups


def train_steps(engine, device, dtype, n: int, start: int, rank: int) -> list:
    losses = []
    for i in range(n):
        loss = engine(make_batch(device, dtype, start + i, rank))
        engine.backward(loss)
        engine.step()
        losses.append(float(loss.detach().float()))
    return losses


POOL_PREFIX = "model.state_encoder.segment_pool."


def pool_names(policy) -> list:
    return [n for n, _ in policy.named_parameters() if n.startswith(POOL_PREFIX)]


def n_changed(before: dict, after: dict, names) -> int:
    return sum(1 for n in names if abs(before[n] - after[n]) > 1e-7)


def unchanged(before: dict, after: dict, names) -> list:
    return [n.replace(POOL_PREFIX, "") for n in names if abs(before[n] - after[n]) <= 1e-7]


def changed_report(before: dict, after: dict, names) -> str:
    stale = unchanged(before, after, names)
    return f"{len(names) - len(stale)}/{len(names)} 已更新" + (f", 未变: {stale}" if stale else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["a", "b", "c"])
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    deepspeed.init_distributed(dist_backend="nccl")

    is_main = rank == 0
    if world_size < 2:
        if is_main:
            print("❌ 本测试要求 world_size >= 2 (单卡 ZeRO-3 分片平凡, 复现不出 grad 视图失配)")
        sys.exit(1)

    ckpt_dir = os.path.join(args.dir, "ckpt")
    fp_path = os.path.join(args.dir, f"fp_rank{rank}.json")
    torch.manual_seed(0)

    cfg = make_config()
    policy = FakePolicy(cfg)
    policy.model.state_encoder.initialize_history_null_state(STATE_MEAN, STATE_STD)
    expected_null = ((0.0 - STATE_MEAN) / (STATE_STD + 1e-8))

    if args.phase == "a":
        if is_main:
            print(f"[phase a] 构建 (VLM 冻结) → 训练 3 步 → 保存 (world_size={world_size})")
        for p in policy.vlm.parameters():
            p.requires_grad = False
        engine, _ = build_engine(policy, cfg, world_size, include_vlm=False, train_vlm=False)
        dtype = torch.bfloat16 if engine.bfloat16_enabled() else torch.float32

        names = pool_names(policy)
        if is_main:
            check("pool 参数存在且被 ZeRO-3 分片",
                  len(names) > 0 and all(hasattr(dict(policy.named_parameters())[n], "ds_tensor")
                                         for n in names),
                  f"{len(names)} 个 pool 参数")
            null = policy.model.state_encoder.history_null_state
            rel = float((null.float().cpu() - expected_null).abs().max()
                        / expected_null.abs().max())
            check("engine 构建后 null buffer 仍与 stats 一致 (相对误差 < 1%)",
                  rel < 1e-2, f"dtype={null.dtype} rel_err={rel:.2e}")

        before = shard_fingerprint(policy)
        losses = train_steps(engine, device, dtype, 3, 0, rank)
        after = shard_fingerprint(policy)
        if is_main:
            check("3 步训练 loss 有限", all(l == l and abs(l) < 1e6 for l in losses), str(losses))
            check("pool 参数在 ZeRO-3 下确实被更新",
                  n_changed(before, after, names) == len(names),
                  changed_report(before, after, names))

        engine.save_checkpoint(ckpt_dir, tag="step_000003", client_state={"step": 3})
        dist.barrier()
        with open(fp_path, "w") as f:
            json.dump({"after_a": after, "losses": losses}, f)
        if is_main:
            check("checkpoint 已落盘", os.path.isdir(os.path.join(ckpt_dir, "step_000003")))

    elif args.phase == "b":
        if is_main:
            print("[phase b] fresh 进程 → resume → 再训练 3 步")
        for p in policy.vlm.parameters():
            p.requires_grad = False
        engine, _ = build_engine(policy, cfg, world_size, include_vlm=False, train_vlm=False)
        dtype = torch.bfloat16 if engine.bfloat16_enabled() else torch.float32
        names = pool_names(policy)

        fresh = shard_fingerprint(policy)
        load_path, client = engine.load_checkpoint(
            ckpt_dir, tag="step_000003", load_optimizer_states=True,
            load_lr_scheduler_states=False, load_module_strict=False)
        for p in engine.module.parameters():   # ZeRO-3 resume 会留下 0 尺寸的过期 grad 视图
            p.grad = None
        resumed = shard_fingerprint(policy)
        with open(fp_path.replace(f"rank{rank}", f"rank{rank}"), "r") as f:
            saved = json.load(f)

        if is_main:
            check("load_checkpoint 成功", load_path is not None, str(load_path))
            check("client_state 恢复", (client or {}).get("step") == 3, str(client))
        check(f"[rank{rank}] resume 后所有参数分片 == 保存时",
              all(close(resumed[n], saved["after_a"][n]) for n in resumed),
              str([n for n in resumed if not close(resumed[n], saved["after_a"][n])][:5]))
        # 用与 phase a 相同的绝对阈值: 3 步小 lr 的范数变化远低于任何相对阈值
        check(f"[rank{rank}] pool 参数确实被 checkpoint 覆盖 (而非停留在随机初始化)",
              n_changed(fresh, resumed, names) == len(names),
              changed_report(fresh, resumed, names))
        if is_main:
            null = policy.model.state_encoder.history_null_state
            rel = float((null.float().cpu() - expected_null).abs().max()
                        / expected_null.abs().max())
            check("persistent buffer 随 checkpoint 正确恢复", rel < 1e-2, f"rel_err={rel:.2e}")

        before = shard_fingerprint(policy)
        losses = train_steps(engine, device, dtype, 3, 100, rank)
        after = shard_fingerprint(policy)
        if is_main:
            check("resume 后继续训练无 shape error 且 loss 有限",
                  all(l == l and abs(l) < 1e6 for l in losses), str(losses))
            check("resume 后 pool / query / gate 参数仍在更新",
                  n_changed(before, after, names) == len(names),
                  changed_report(before, after, names))
            bank = [n for n in names if n.endswith(("summary_queries", "segment_type_embeddings",
                                                    "last_chunk_gates"))]
            check("三个 Parameter bank 均更新 (单次 expand 未破坏梯度链路)",
                  len(bank) == 3 and n_changed(before, after, bank) == 3, str(bank))

    else:  # phase c
        if is_main:
            print("[phase c] 冻结训练 2 步 → unfreeze 重建 engine → 再训练 2 步")
        for p in policy.vlm.parameters():
            p.requires_grad = False
        engine, _ = build_engine(policy, cfg, world_size, include_vlm=False, train_vlm=False)
        dtype = torch.bfloat16 if engine.bfloat16_enabled() else torch.float32
        names = pool_names(policy)
        train_steps(engine, device, dtype, 2, 200, rank)

        # 复刻 _unfreeze_vlm_deepspeed 的 ZeRO-3 路径: roundtrip 保存 → destroy → 重建 → 回载
        rt_dir = os.path.join(args.dir, "roundtrip")
        engine.save_checkpoint(rt_dir, tag="unfreeze", exclude_frozen_parameters=False)
        # load_optimizer_states=False 不会回载 fp32 master, 因此这里比模块权重
        pre = module_fingerprint(policy)
        engine.destroy()
        for p in policy.parameters():
            if hasattr(p, "_z3_optimizer"):
                p._z3_optimizer = None
            p.grad = None
        engine.optimizer = None
        engine = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        for p in policy.vlm.parameters():
            p.requires_grad = True
        engine2, n_vlm_groups = build_engine(policy, cfg, world_size, include_vlm=True, train_vlm=True)
        load_path, _ = engine2.load_checkpoint(
            rt_dir, tag="unfreeze", load_optimizer_states=False,
            load_lr_scheduler_states=False, load_module_strict=False)
        post = module_fingerprint(policy)

        if is_main:
            check("unfreeze 重建后 roundtrip 权重回载成功", load_path is not None, str(load_path))
            check("重建时的参数组含 VLM 组 (第三条构造路径)", n_vlm_groups == 1, str(n_vlm_groups))
        check(f"[rank{rank}] 重建 + 回载后模块权重与重建前一致",
              all(close(post[n], pre[n]) for n in pre),
              str([n for n in pre if not close(post[n], pre[n])][:5]))

        before = shard_fingerprint(policy)
        losses = train_steps(engine2, device, dtype, 2, 300, rank)
        after = shard_fingerprint(policy)
        if is_main:
            check("unfreeze 后继续训练无 shape error 且 loss 有限",
                  all(l == l and abs(l) < 1e6 for l in losses), str(losses))
            check("unfreeze 后 pool 参数仍在更新 (未被静默冻结)",
                  n_changed(before, after, names) == len(names),
                  changed_report(before, after, names))
            vlm_names = [n for n, _ in policy.named_parameters() if n.startswith("vlm.")]
            check("unfreeze 后 VLM 参数开始更新",
                  n_changed(before, after, vlm_names) == len(vlm_names),
                  f"{n_changed(before, after, vlm_names)}/{len(vlm_names)}")

    dist.barrier()
    n_fail = torch.tensor([len(failures)], device=device)
    dist.all_reduce(n_fail)
    if is_main:
        print()
        if int(n_fail) > 0:
            print(f"FAILED (phase {args.phase}): 全 rank 合计 {int(n_fail)} 项")
        else:
            print(f"ALL PASS (phase {args.phase})")
    dist.barrier()
    sys.exit(1 if int(n_fail) > 0 else 0)


if __name__ == "__main__":
    main()
