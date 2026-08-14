#!/usr/bin/env python
"""
LoLA V07 t-网格探针 (t-stratified loss probe)

诊断目的: 定位 flow matching 误差在 t 维度上的分布, 用于验证 warmup t-truncation
截断点、指导 time_sampling_beta 调参、以及推理求解器步长分配。

流程 (对每个验证集样本):
    1. 采一个噪声 ε ~ N(0,1), 在该样本的所有 t 上固定复用
       (common random numbers, 降方差且让逐样本曲线可比)
    2. 对 t 网格上每个 t (低 t 端加密), 按训练插值公式在 latent 空间构造
       z_t = (1-t)·latent(x0) + t·ε   (由模型 forward 内部完成, 与训练逐字节一致)
    3. 单次前向 DiT(z_t, t, 条件) → x0̂(z_t, t)
    4. 逐样本记录: v_loss (latent 空间), arm Huber/MSE (归一化 action 空间),
       gripper BCE + 阈值化判别精度, 按训练权重合成的总损失

实现说明:
    - 复用 LoLAV07Policy.forward 的 time=/noise= 注入接口, 模型内部完成
      latent 编码、插值、v/x0 换算, 探针不复制任何模型逻辑
    - 每个 t 点全 batch 共用同一 t, 各样本固定自己的 ε
    - 条件 (VLM/hist 特征) 与 t 无关但当前每个 t 都重算 (方案 B, 未做缓存优化);
      成本 ≈ len(t_grid) × forward_loss 验证, 大样本量时可再加条件缓存

使用方法:
    python src/lerobot/scripts/probe_lola_v07_tcurve.py \
        --training_config /path/to/run_dir/training_config.json \
        --checkpoint_path /path/to/run_dir \
        --val_dataset_repo_id <val_dataset> \
        --max_samples 200 \
        --output_file probe_tcurve.json

    # 自定义 t 网格
    python src/lerobot/scripts/probe_lola_v07_tcurve.py ... \
        --t_grid "0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.95"
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)
# train_lola_v07_azure 内部 `from resume_search import ...` (同目录模块), 需要 scripts 目录在 sys.path
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "lerobot", "scripts"))

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola_v07 import LoLAV07Config, LoLAV07Policy
from lerobot.policies.factory import make_pre_post_processors

from lerobot.scripts.train_lola_v07_azure import (
    create_lola_dataset,
    make_collate_fn,
    compute_vlm_max_length,
)

# 复用 validate_lola_v07 的纯函数 (checkpoint 加载 / 特殊字段提取 / training_config 匹配)
from lerobot.scripts.validate_lola_v07 import (
    extract_special_fields,
    load_deepspeed_checkpoint,
    _load_training_config,
    _apply_training_config,
)

# 默认 t 网格: 34 点, 低 t 端 (x0 预测难、误差集中) 加密
DEFAULT_T_GRID = [
    0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
    0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45,
    0.475, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.925, 0.95, 0.975, 0.99,
]

# 逐样本指标 (模型 forward return_per_sample=True 返回的 keys)
PER_SAMPLE_METRICS = [
    "v_loss", "arm_huber", "arm_mse", "gripper_bce", "gripper_acc", "total_loss",
]


def probe_t_curve(policy, preprocessor, val_loader, device, t_grid, max_samples,
                  num_workers_log_rank="0"):
    """CRN t-网格探针主循环。

    Returns:
        dict: {
            "per_t": {metric: {"mean": [...], "std": [...]}},  # 跨样本聚合
            "arm_huber_per_dim_mean": [[arm_dim] * len(t_grid)],  # 逐 t 逐 arm 维
            "per_sample": [{"index": i, metric: [...]}, ...],
            "num_samples": int,
        }
    """
    policy.eval()
    policy.model.eval()
    if hasattr(policy, 'vlm'):
        policy.vlm.eval()

    n_t = len(t_grid)
    # 跨样本聚合累加器 (float64 保精度)
    sums = {m: [0.0] * n_t for m in PER_SAMPLE_METRICS}
    sq_sums = {m: [0.0] * n_t for m in PER_SAMPLE_METRICS}
    counts = [0] * n_t
    arm_dim = None
    per_dim_sums = None   # [n_t][arm_dim], 按 batch size 加权
    per_sample_records = []

    sample_offset = 0
    n_batches = len(val_loader)
    t_start = time.time()

    for batch_idx, batch in enumerate(val_loader):
        if sample_offset >= max_samples:
            break

        # 与训练/验证一致: 特殊字段在 preprocessor 前提取, 之后恢复
        special_data = extract_special_fields(batch)
        if "action" not in special_data:
            print(f"[probe] batch {batch_idx}: no 'action' key, skipped")
            continue
        batch = preprocessor(batch)
        for k, v in special_data.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            else:
                batch[k] = v

        b = special_data["action"].shape[0]

        # CRN: 每个样本采一个固定噪声 (latent 空间), 该样本所有 t 复用。
        # 先编码 target 拿 latent 形状 (与模型 forward step 3 相同的 FP32 autocast 编码)
        target_actions = policy.prepare_target_actions(batch)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float32):
            _, arm_latent, grip_latent = policy.model.action_encoder(
                target_actions, return_latent=True
            )
        noise_arm = torch.randn_like(arm_latent)
        noise_grip = torch.randn_like(grip_latent)
        noise = (noise_arm, noise_grip)

        batch_records = [
            {"index": sample_offset + i, **{m: [0.0] * n_t for m in PER_SAMPLE_METRICS}}
            for i in range(b)
        ]

        for ti, t_val in enumerate(t_grid):
            time_tensor = torch.full((b,), float(t_val), device=device)
            with torch.no_grad():
                _, loss_dict = policy(
                    batch,
                    time=time_tensor,
                    noise=noise,
                    compute_per_dim=True,
                    return_per_sample=True,
                )

            counts[ti] += b
            for m in PER_SAMPLE_METRICS:
                ps = loss_dict[f"{m}_per_sample"].double().cpu()  # [B]
                sums[m][ti] += ps.sum().item()
                sq_sums[m][ti] += (ps ** 2).sum().item()
                for i in range(b):
                    batch_records[i][m][ti] = ps[i].item()

            # 逐 t 逐 arm 维 Huber (batch mean, 按 B 加权聚合)
            pd = loss_dict["arm_loss_per_dim"].double().cpu()  # [arm_dim]
            if arm_dim is None:
                arm_dim = pd.shape[0]
                per_dim_sums = [[0.0] * arm_dim for _ in range(n_t)]
            for d in range(arm_dim):
                per_dim_sums[ti][d] += pd[d].item() * b

        per_sample_records.extend(batch_records)
        sample_offset += b

        elapsed = time.time() - t_start
        done_fwd = (batch_idx + 1) * n_t
        total_fwd = min(n_batches, math.ceil(max_samples / b)) * n_t
        eta = elapsed / done_fwd * (total_fwd - done_fwd) if done_fwd > 0 else 0
        print(f"[probe] batch {batch_idx + 1}/{min(n_batches, math.ceil(max_samples / b))} "
              f"({sample_offset}/{max_samples} samples, {done_fwd} forwards, "
              f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

    # 聚合: mean / std
    per_t = {}
    for m in PER_SAMPLE_METRICS:
        means, stds = [], []
        for ti in range(n_t):
            n = max(counts[ti], 1)
            mean = sums[m][ti] / n
            var = max(sq_sums[m][ti] / n - mean ** 2, 0.0)
            means.append(mean)
            stds.append(math.sqrt(var))
        per_t[m] = {"mean": means, "std": stds}

    arm_huber_per_dim_mean = None
    if per_dim_sums is not None:
        arm_huber_per_dim_mean = [
            [per_dim_sums[ti][d] / max(counts[ti], 1) for d in range(arm_dim)]
            for ti in range(n_t)
        ]

    return {
        "per_t": per_t,
        "arm_huber_per_dim_mean": arm_huber_per_dim_mean,
        "per_sample": per_sample_records,
        "num_samples": len(per_sample_records),
    }


def plot_curves(t_grid, per_t, out_png):
    """画聚合曲线 (matplotlib 不可用时静默跳过)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe] matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = [
        ("v_loss", "v-loss (latent space)", True),
        ("arm_huber", "arm Huber (normalized action space)", True),
        ("gripper_bce", "gripper BCE", True),
        ("gripper_acc", "gripper threshold accuracy", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, specs):
        mean = per_t[metric]["mean"]
        std = per_t[metric]["std"]
        ax.plot(t_grid, mean, "o-", markersize=3, label="mean")
        ax.fill_between(t_grid,
                        [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)],
                        alpha=0.2, label="±1 std (across samples)")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("t (flow time)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("LoLA V07 t-stratified probe (CRN fixed noise per sample)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[probe] curves saved to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="LoLA V07 t-stratified probe (CRN)")

    # Training config — 自动匹配训练配置 (CLI 参数优先)
    parser.add_argument("--training_config", type=str, default=None,
                        help="Path to training_config.json saved by v07 training. "
                             "When provided, config args are auto-populated; "
                             "CLI args override training_config values.")

    # 验证数据集参数
    parser.add_argument("--val_dataset_repo_id", type=str, default=None)
    parser.add_argument("--val_dataset_root", type=str, default=None)
    parser.add_argument("--val_episodes", type=int, nargs="*", default=None)

    # 模型参数
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--deepspeed_tag", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true", default=False)
    parser.add_argument("--vlm_path", type=str, default=None)
    parser.add_argument("--vlm_model_name", type=str, default=None)

    # LoLAV07Config 参数 (default None → training_config → 内置默认)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--action_chunk_size", type=int, default=None)
    parser.add_argument("--pred_chunk_size", type=int, default=None)
    parser.add_argument("--n_obs_steps", type=int, default=None)
    parser.add_argument("--train_vlm", action="store_true", default=None)
    parser.add_argument("--load_full_history", action="store_true", default=None)
    parser.add_argument("--max_history_length", type=int, default=None)
    parser.add_argument("--history_padding_side", type=str, default=None, choices=["left", "right"])
    parser.add_argument("--vlm_extract_layers", type=int, nargs="+", default=None)
    parser.add_argument("--max_image_pixels", type=int, default=None)
    parser.add_argument("--min_image_pixels", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--static_vlm_padding", action="store_true", default=None)
    parser.add_argument("--vlm_max_length", type=int, default=None)
    parser.add_argument("--empty_cameras", type=int, default=None)
    parser.add_argument("--vlm_hidden_size", type=int, default=None)
    parser.add_argument("--vlm_backbone", type=str, default=None)
    parser.add_argument("--dit_hidden_size", type=int, default=None)
    parser.add_argument("--empty_token_id", type=int, default=None)

    # v07 架构/行为参数
    parser.add_argument("--history_type", type=str, default=None, choices=["action", "state"])
    parser.add_argument("--state_dim", type=int, default=None)
    parser.add_argument("--state_encoder_mode", type=str, default=None, choices=["unified", "separated"])
    parser.add_argument("--use_state_condition", action="store_true", default=None)
    parser.add_argument("--vlm_bridge_mode", type=str, default=None, choices=["legacy", "transformer"])
    parser.add_argument("--vlm_bridge_width", type=int, default=None)
    parser.add_argument("--vlm_bridge_layers", type=int, default=None)
    parser.add_argument("--vlm_bridge_num_heads", type=int, default=None)
    parser.add_argument("--vlm_bridge_ffn_ratio", type=float, default=None)
    parser.add_argument("--obs_prev_chunk_frame", action="store_true", default=None)
    parser.add_argument("--action_bottleneck_dim", type=int, default=None)
    parser.add_argument("--grip_bottleneck_dim", type=int, default=None)
    parser.add_argument("--state_bottleneck_dim", type=int, default=None)
    parser.add_argument("--state_grip_bottleneck_dim", type=int, default=None)

    # v07 文本模板 / 特殊 token
    parser.add_argument("--task_text_template_version", type=str, default=None,
                        choices=["raw", "v1_with_completed"])
    parser.add_argument("--completed_tasks_use_ann", action="store_true", default=None)
    parser.add_argument("--no_completed_tasks_use_ann", action="store_true")
    parser.add_argument("--completed_tasks_history_len", type=int, default=None)
    parser.add_argument("--max_transition_len", type=int, default=None)
    parser.add_argument("--use_special_tokens", action="store_true", default=None)
    parser.add_argument("--no_use_special_tokens", action="store_true")

    # 归一化参数
    parser.add_argument("--norm_mode", type=str, default=None,
                        choices=["default", "robovlm", "zscore"])
    parser.add_argument("--norm_min", type=float, default=None)
    parser.add_argument("--norm_max", type=float, default=None)
    parser.add_argument("--stats_mode", type=str, default=None,
                        choices=["original", "incremental"])

    # 夹爪参数
    parser.add_argument("--gripper_dims", type=str, default=None)
    parser.add_argument("--gripper_threshold", type=float, default=None)
    parser.add_argument("--gripper_loss_weight", type=float, default=None)
    parser.add_argument("--action_loss_weight", type=float, default=None)

    # 探针参数
    parser.add_argument("--t_grid", type=str, default=None,
                        help=f"Comma-separated t values. Default: {len(DEFAULT_T_GRID)}-point grid, "
                             f"dense at low t: {DEFAULT_T_GRID[:5]}...{DEFAULT_T_GRID[-3:]}")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max validation samples to probe")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_per_sample", action="store_true",
                        help="Only save aggregated per-t curves, skip per-sample records")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (controls CRN noise draws and dataset order)")

    # 输出
    parser.add_argument("--output_file", type=str, default="probe_tcurve_results.json")
    parser.add_argument("--no_plot", action="store_true")

    args = parser.parse_args()

    # ─── Apply training_config.json if provided ─────────────────────────────
    if args.training_config is not None:
        lola_cfg, train_args = _load_training_config(args.training_config)
        _apply_training_config(args, lola_cfg, train_args)

    # ─── 默认值填充 (与 validate_lola_v07 相同) ──────────────────────────────
    if args.history_type is None:
        args.history_type = "action"
    if args.state_encoder_mode is None:
        args.state_encoder_mode = "unified"
    if args.use_state_condition is None:
        args.use_state_condition = False
    if args.vlm_bridge_mode is None:
        args.vlm_bridge_mode = "legacy"
    if args.vlm_bridge_width is None:
        args.vlm_bridge_width = 2048
    if args.vlm_bridge_layers is None:
        args.vlm_bridge_layers = 8
    if args.vlm_bridge_num_heads is None:
        args.vlm_bridge_num_heads = 0
    if args.vlm_bridge_ffn_ratio is None:
        args.vlm_bridge_ffn_ratio = 4.0
    if args.obs_prev_chunk_frame is None:
        args.obs_prev_chunk_frame = False
    if args.norm_mode is None:
        args.norm_mode = "default"
    if args.norm_min is None:
        args.norm_min = -0.65
    if args.norm_max is None:
        args.norm_max = 0.65
    if args.stats_mode is None:
        args.stats_mode = "original"
    if args.gripper_threshold is None:
        args.gripper_threshold = 0.5
    if args.gripper_loss_weight is None:
        args.gripper_loss_weight = 1.0
    if args.action_loss_weight is None:
        args.action_loss_weight = 10.0
    if args.action_chunk_size is None:
        args.action_chunk_size = 10
    if args.pred_chunk_size is None:
        args.pred_chunk_size = 50
    if args.n_obs_steps is None:
        args.n_obs_steps = 1
    if args.max_history_length is None:
        args.max_history_length = 100
    if args.history_padding_side is None:
        args.history_padding_side = "left"
    if args.vlm_extract_layers is None:
        args.vlm_extract_layers = [8, 16, 24]
    if args.max_image_pixels is None:
        args.max_image_pixels = 230400
    if args.min_image_pixels is None:
        args.min_image_pixels = 65536
    if args.num_inference_steps is None:
        args.num_inference_steps = 10
    if args.vlm_backbone is None:
        args.vlm_backbone = "qwen3_5"
    if args.dit_hidden_size is None:
        args.dit_hidden_size = 1024
    if args.empty_cameras is None:
        args.empty_cameras = 0
    if args.action_bottleneck_dim is None:
        args.action_bottleneck_dim = 256
    if args.grip_bottleneck_dim is None:
        args.grip_bottleneck_dim = 128
    if args.state_bottleneck_dim is None:
        args.state_bottleneck_dim = 256
    if args.state_grip_bottleneck_dim is None:
        args.state_grip_bottleneck_dim = 128
    if args.train_vlm is None:
        args.train_vlm = False
    if args.load_full_history is None:
        args.load_full_history = False
    if args.static_vlm_padding is None:
        args.static_vlm_padding = False
    if args.task_text_template_version is None:
        args.task_text_template_version = "raw"
    if args.completed_tasks_use_ann is None:
        args.completed_tasks_use_ann = True
    if args.no_completed_tasks_use_ann:
        args.completed_tasks_use_ann = False
    if args.completed_tasks_history_len is None:
        args.completed_tasks_history_len = 5
    if args.max_transition_len is None:
        args.max_transition_len = 64
    if args.use_special_tokens is None:
        args.use_special_tokens = False
    if args.no_use_special_tokens:
        args.use_special_tokens = False
    if args.vlm_path is None:
        args.vlm_path = "/data_16T/deepseek/qwen3_5/Qwen3.5-4B/"
        print(f"WARNING: --vlm_path not provided and not in training_config, "
              f"falling back to {args.vlm_path}")

    if args.val_dataset_repo_id is None and args.val_dataset_root is None:
        raise ValueError("Either --val_dataset_repo_id or --val_dataset_root must be provided "
                         "(not found in training_config either).")

    # t 网格
    if args.t_grid is not None:
        t_grid = sorted(set(float(x.strip()) for x in args.t_grid.split(",")))
        for t in t_grid:
            if not (0.0 < t < 1.0):
                raise ValueError(f"t value {t} out of range (0, 1); "
                                 f"t=0/1 会退化 (v 换算除以 t, 且 t=0 无噪声)")
    else:
        t_grid = DEFAULT_T_GRID
    print(f"t grid: {len(t_grid)} points, [{t_grid[0]}, ..., {t_grid[-1]}]")

    # seed: 控制 CRN 噪声采样与数据顺序
    torch.manual_seed(args.seed)

    # 设备 (探针为单进程脚本; 多 GPU 可分片 --val_episodes 多次运行)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 加载数据集元数据
    print(f"Loading validation dataset metadata from {args.val_dataset_repo_id or args.val_dataset_root}...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.val_dataset_repo_id,
        root=args.val_dataset_root,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    elif args.action_dim is not None:
        action_dim = args.action_dim
    else:
        raise ValueError("action_dim could not be determined")

    if args.state_dim is not None:
        state_dim = args.state_dim
    elif "observation.state" in features:
        state_dim = features["observation.state"].shape[0]
    else:
        state_dim = action_dim

    # 解析夹爪维度索引
    gripper_dim_indices = []
    if args.gripper_dims is not None:
        raw_indices = [int(x.strip()) for x in args.gripper_dims.split(",")]
        for idx in raw_indices:
            resolved = action_dim + idx if idx < 0 else idx
            if resolved < 0 or resolved >= action_dim:
                raise ValueError(f"Gripper dim index {idx} out of range [0, {action_dim})")
            gripper_dim_indices.append(resolved)
        gripper_dim_indices = sorted(set(gripper_dim_indices))
        print(f"Gripper dim indices: {gripper_dim_indices}")

    print(f"Dataset: {dataset_metadata.total_episodes} episodes, "
          f"{dataset_metadata.total_frames} frames, action_dim={action_dim}, state_dim={state_dim}")

    # static VLM padding 但未给 vlm_max_length: 与训练一致自动计算
    if args.static_vlm_padding and args.vlm_max_length is None:
        frames_per_cam = 2 if args.obs_prev_chunk_frame else max(1, args.n_obs_steps)
        args.vlm_max_length = compute_vlm_max_length(
            dataset_metadata,
            vlm_path=args.vlm_path,
            min_image_pixels=args.min_image_pixels,
            max_image_pixels=args.max_image_pixels,
            frames_per_cam=frames_per_cam,
        )
        print(f"Auto-computed vlm_max_length={args.vlm_max_length}")

    # ─── 创建 LoLAV07 配置 (train-only 字段强制为验证语义, 与 validate_lola_v07 一致) ───
    config = LoLAV07Config(
        vlm_backbone=args.vlm_backbone,
        vlm_path=args.vlm_path,
        vlm_model_name=args.vlm_model_name,
        action_dim=action_dim,
        action_chunk_size=args.action_chunk_size,
        pred_chunk_size=args.pred_chunk_size,
        n_obs_steps=args.n_obs_steps,
        input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
        output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
        train_vlm=args.train_vlm,
        load_full_history=args.load_full_history,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
        history_type=args.history_type,
        state_dim=state_dim,
        state_encoder_mode=args.state_encoder_mode,
        use_state_condition=args.use_state_condition,
        gradient_checkpointing=False,
        dit_gradient_checkpointing=False,
        vlm_extract_layers=tuple(args.vlm_extract_layers),
        vlm_bridge_mode=args.vlm_bridge_mode,
        vlm_bridge_width=args.vlm_bridge_width,
        vlm_bridge_layers=args.vlm_bridge_layers,
        vlm_bridge_num_heads=args.vlm_bridge_num_heads,
        vlm_bridge_ffn_ratio=args.vlm_bridge_ffn_ratio,
        use_special_tokens=args.use_special_tokens,
        max_image_pixels=args.max_image_pixels,
        min_image_pixels=args.min_image_pixels,
        gripper_loss_weight=args.gripper_loss_weight,
        action_loss_weight=args.action_loss_weight,
        gripper_dim_indices=tuple(gripper_dim_indices),
        gripper_threshold=args.gripper_threshold,
        hist_action_token_drop_rate=0.0,   # train-only
        static_vlm_padding=args.static_vlm_padding,
        vlm_max_length=args.vlm_max_length,
        empty_cameras=args.empty_cameras,
        vlm_hidden_size=args.vlm_hidden_size,
        dit_hidden_size=args.dit_hidden_size,
        num_inference_steps=args.num_inference_steps,
        empty_token_id=args.empty_token_id,
        task_text_template_version=args.task_text_template_version,
        completed_tasks_use_ann=args.completed_tasks_use_ann,
        completed_tasks_history_len=args.completed_tasks_history_len,
        transition_mask_rate=0.0,            # train-only
        max_transition_len=args.max_transition_len,
        action_bottleneck_dim=args.action_bottleneck_dim,
        grip_bottleneck_dim=args.grip_bottleneck_dim,
        state_bottleneck_dim=args.state_bottleneck_dim,
        state_grip_bottleneck_dim=args.state_grip_bottleneck_dim,
        obs_prev_chunk_frame=args.obs_prev_chunk_frame,
        visual_token_drop_rate=0.0,          # train-only
    )
    config.device = "cuda:0"

    print(f"Config: history_type={config.history_type}, "
          f"use_special_tokens={config.use_special_tokens}, "
          f"task_template={config.task_text_template_version}, "
          f"vlm_backbone={config.vlm_backbone}, "
          f"obs_prev_chunk_frame={config.obs_prev_chunk_frame}, "
          f"action_bottleneck={config.action_bottleneck_dim}, "
          f"grip_bottleneck={config.grip_bottleneck_dim}")

    # 归一化模式: 优先 training_config 保存的精确映射
    saved_norm_mapping = getattr(args, "normalization_mapping", None)
    if saved_norm_mapping:
        config.normalization_mapping = {
            k: NormalizationMode(v) for k, v in saved_norm_mapping.items()
        }
        print(f"normalization_mapping from training_config: "
              f"{ {k: v.value for k, v in config.normalization_mapping.items()} }")
    elif args.norm_mode == "robovlm":
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    elif args.norm_mode == "zscore":
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.IDENTITY,
        }

    # 加载模型
    print("Loading LoLAV07 model...")
    policy = LoLAV07Policy(config)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=dataset_metadata.stats,
    )

    # 加载 checkpoint (与 validate_lola_v07 相同的两条路径)
    ckpt_metadata = {}
    ckpt_type = "none"
    if args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            ckpt_type = "single_file"
            print(f"Loading single-file checkpoint: {args.checkpoint_path}")
            ckpt = torch.load(args.checkpoint_path, map_location="cpu")

            if "model_state_dict" in ckpt:
                ckpt_state_dict = ckpt["model_state_dict"]
                if "step" in ckpt:
                    ckpt_metadata["step"] = ckpt["step"]
                if "epoch" in ckpt:
                    ckpt_metadata["epoch"] = ckpt["epoch"]
            elif "state_dict" in ckpt:
                ckpt_state_dict = ckpt["state_dict"]
            else:
                ckpt_state_dict = ckpt

            dit_sd = {}
            vlm_sd_raw = {}
            for key, value in ckpt_state_dict.items():
                if key.startswith("policy.model."):
                    dit_sd[key[len("policy.model."):]] = value
                elif key.startswith("policy.vlm."):
                    vlm_sd_raw[key[len("policy.vlm."):]] = value
                elif key.startswith("model."):
                    dit_sd[key[len("model."):]] = value
                elif key.startswith("vlm."):
                    vlm_sd_raw[key[len("vlm."):]] = value
                else:
                    dit_sd[key] = value

            if dit_sd:
                current_dit_sd = policy.model.state_dict()
                dit_loaded = sum(1 for k in current_dit_sd if k in dit_sd)
                for key in current_dit_sd:
                    if key in dit_sd:
                        current_dit_sd[key] = dit_sd[key]
                policy.model.load_state_dict(current_dit_sd)
                dit_missing = len(current_dit_sd) - dit_loaded
                print(f"DiT weights: {dit_loaded} loaded, {dit_missing} missing")
                if dit_missing > 0:
                    print(f"WARNING: {dit_missing} DiT params not found — "
                          f"确认 config 与训练一致 (bottleneck dims / bridge / special tokens / history_type)")

            if vlm_sd_raw:
                current_vlm_sd = policy.vlm.state_dict()
                vlm_loaded = sum(1 for k in current_vlm_sd if k in vlm_sd_raw)
                for key in current_vlm_sd:
                    if key in vlm_sd_raw:
                        current_vlm_sd[key] = vlm_sd_raw[key]
                policy.vlm.load_state_dict(current_vlm_sd)
                print(f"VLM weights: {vlm_loaded} loaded")

            if args.use_ema:
                ema_state = ckpt.get("ema_state") if isinstance(ckpt, dict) else None
                if ema_state:
                    current_full_sd = policy.state_dict()
                    n_ema = 0
                    for key, value in ema_state.items():
                        full_key = key if key.startswith("policy.") else f"policy.{key}"
                        if full_key in current_full_sd:
                            current_full_sd[full_key] = value.to(dtype=current_full_sd[full_key].dtype)
                            n_ema += 1
                    policy.load_state_dict(current_full_sd)
                    print(f"[EMA] 覆盖 {n_ema} 个参数 (单文件 ema_state)")
                    ckpt_metadata["use_ema"] = True
                else:
                    print("WARNING: --use_ema 指定但 checkpoint 无 ema_state, 使用非 EMA 权重")

        elif os.path.isdir(args.checkpoint_path):
            ckpt_type = "deepspeed"
            ckpt_metadata = load_deepspeed_checkpoint(
                checkpoint_dir=args.checkpoint_path,
                policy=policy,
                tag=args.deepspeed_tag,
                local_rank=0,
                use_ema=args.use_ema,
            )
            print(f"DeepSpeed checkpoint metadata: {ckpt_metadata}")
        else:
            raise ValueError(f"Checkpoint path does not exist: {args.checkpoint_path}")
    else:
        print("No checkpoint provided, using randomly initialized model")

    policy._device = device
    policy.model = policy.model.to(device)
    policy.vlm = policy.vlm.to(device)
    policy.eval()
    policy.model.eval()
    policy.vlm.eval()

    # 创建验证数据集
    print("Creating validation dataset...")
    if args.norm_mode == "robovlm":
        norm_action = True
    elif args.norm_mode == "zscore":
        norm_action = "zscore"
    else:
        norm_action = False
    val_dataset = create_lola_dataset(
        repo_id=args.val_dataset_repo_id,
        config=config,
        root=args.val_dataset_root,
        episodes=args.val_episodes,
        use_lola_dataset=args.load_full_history,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
        norm_action=norm_action,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
        gripper_dim_indices_abs=config.gripper_dim_indices_abs,
        dataset_stats=dataset_metadata.stats,
        history_type=args.history_type,
        state_dim=state_dim,
        track_completed_tasks=config.task_text_template_version == "v1_with_completed",
        transition_mask_rate=0.0,
        completed_tasks_use_ann=config.completed_tasks_use_ann,
        completed_tasks_history_len=config.completed_tasks_history_len,
        max_transition_len=config.max_transition_len,
        stats_mode=args.stats_mode,
    )
    print(f"Total validation samples: {len(val_dataset)}")

    # 顺序遍历 (shuffle=False): CRN 曲线跨运行可复现
    use_static_padding = args.load_full_history
    static_max_len = args.max_history_length if use_static_padding else None
    collate = make_collate_fn(static_max_len=static_max_len)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # ─── 运行探针 ────────────────────────────────────────────────────────────
    start_time = time.time()
    probe_results = probe_t_curve(
        policy, preprocessor, val_loader, device, t_grid, args.max_samples,
    )
    elapsed = time.time() - start_time

    # 打印聚合曲线摘要
    print("=" * 78)
    print("t-stratified probe results (mean across samples)")
    print("=" * 78)
    header = f"{'t':>6} | {'v_loss':>9} | {'arm_huber':>9} | {'arm_mse':>9} | {'grip_bce':>9} | {'grip_acc':>9} | {'total':>9}"
    print(header)
    print("-" * len(header))
    for ti, t_val in enumerate(t_grid):
        row = f"{t_val:>6.3f} | "
        for m in PER_SAMPLE_METRICS:
            row += f"{probe_results['per_t'][m]['mean'][ti]:>9.5f} | "
        print(row[:-3])

    # 保存 JSON
    output = {
        "dataset": args.val_dataset_repo_id or args.val_dataset_root,
        "training_config": args.training_config,
        "checkpoint_info": {"type": ckpt_type, "path": args.checkpoint_path or "N/A", **ckpt_metadata},
        "seed": args.seed,
        "t_grid": t_grid,
        "num_samples": probe_results["num_samples"],
        "action_dim": action_dim,
        "gripper_dim_indices": gripper_dim_indices,
        "action_loss_weight": args.action_loss_weight,
        "gripper_loss_weight": args.gripper_loss_weight,
        "per_t": probe_results["per_t"],
        "arm_huber_per_dim_mean": probe_results["arm_huber_per_dim_mean"],
        "elapsed_s": elapsed,
    }
    if not args.no_per_sample:
        output["per_sample"] = probe_results["per_sample"]

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output_file}")

    if not args.no_plot:
        plot_curves(t_grid, probe_results["per_t"],
                    os.path.splitext(args.output_file)[0] + ".png")


if __name__ == "__main__":
    main()
