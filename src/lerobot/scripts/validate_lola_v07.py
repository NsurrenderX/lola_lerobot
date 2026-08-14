#!/usr/bin/env python
"""
LoLA V07 模型验证脚本 - 在验证集上评估模型质量

与 validate_lola.py (v06) 的区别:
    - 使用 LoLAV07Config / LoLAV07Policy (bottleneck 空间 latent flow matching)
    - 数据集/_collate 复用 train_lola_v07_azure (hist_states, completed_tasks,
      n_transition_chunks, stats_mode)
    - 支持 --training_config: 从训练保存的 training_config.json 自动匹配配置
      (参照 eval_calvin.py; CLI 参数优先级高于 training_config)

支持两种验证模式:
    forward_loss: 计算 v-loss 和 action_loss（与训练相同的前向传播）
    inference:    运行实际推理去噪管线，对比预测动作与真实动作（MSE, L1）
    both:         同时运行两种模式

使用方法:
    # 推荐: 用 training_config.json 自动匹配训练配置
    python src/lerobot/scripts/validate_lola_v07.py \
        --training_config /path/to/run_dir/training_config.json \
        --checkpoint_path /path/to/run_dir/step_000100 \
        --val_dataset_repo_id <val_dataset> \
        --mode both

    # 多 GPU
    torchrun --nproc_per_node=4 src/lerobot/scripts/validate_lola_v07.py \
        --training_config /path/to/run_dir/training_config.json \
        --checkpoint_path /path/to/run_dir \
        --val_dataset_repo_id <val_dataset>

    # CLI 覆盖 training_config 中的值 (CLI 优先)
    python src/lerobot/scripts/validate_lola_v07.py \
        --training_config /path/to/training_config.json \
        --checkpoint_path /path/to/ckpt \
        --history_type state --use_special_tokens \
        --val_dataset_repo_id <val_dataset>
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

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

# 从 V07 训练脚本复用 (hist_states / completed_tasks / n_transition_chunks / stats_mode 支持)
from lerobot.scripts.train_lola_v07_azure import (
    create_lola_dataset,
    make_collate_fn,
    compute_vlm_max_length,
)


def unnormalize_lola_actions(actions, dataset_stats, action_dim, norm_mode, action_key="action"):
    """将动作从模型输出空间反归一化回原始空间。

    default (MEAN_STD) 模式: actions * std + mean
    robovlm 模式: 数据集已用 normalize_action 归一化到 [-1,1]（夹爪保持原值），
                  无需反归一化。
    zscore 模式: 与 MEAN_STD 相同公式 actions * std + mean，
                 用于横向对比时将预测和标签都还原到原始空间。

    action_key: stats_mode="incremental" 时为 "action_incremental"，否则 "action"。
    """
    if norm_mode == "robovlm":
        # robovlm 模式下夹爪值本身就是原始的 {-1,1}/{0,1}
        return actions
    else:
        stats_entry = dataset_stats.get(action_key) or dataset_stats["action"]
        action_mean = stats_entry["mean"][:action_dim]
        action_std = stats_entry["std"][:action_dim]
        if not isinstance(action_mean, torch.Tensor):
            action_mean = torch.tensor(action_mean, dtype=torch.float32)
        if not isinstance(action_std, torch.Tensor):
            action_std = torch.tensor(action_std, dtype=torch.float32)
        action_mean = action_mean.to(actions.device)
        action_std = action_std.to(actions.device)
        return actions * action_std + action_mean


def extract_special_fields(batch):
    """提取特殊字段，避免被preprocessor处理（与 V07 训练脚本逻辑一致）。

    V07 相比 v06 新增: hist_states_* (history_type="state") 和
    n_transition / n_transition_chunks (5 个特殊 token 中 previous_task_end 的定位)。
    completed_tasks / completed_tasks_ann 不在此提取 — 它们必须随 batch 一起
    经过 preprocessor (任务文本模板在 processor 内组装)。
    """
    special_data = {}
    keys_to_extract = [
        "hist_actions_full", "hist_actions_mask", "hist_actions_length",
        "hist_states_full", "hist_states_mask", "hist_states_length",
        "n_transition", "n_transition_chunks",
    ]
    for key in keys_to_extract:
        if key in batch:
            special_data[key] = batch.pop(key)
    if "action" in batch:
        special_data["action"] = batch.pop("action")
    return special_data


def load_deepspeed_checkpoint(
    checkpoint_dir: str,
    policy: LoLAV07Policy,
    tag: str | None = None,
    local_rank: int = 0,
    use_ema: bool = False,
) -> dict:
    """从 DeepSpeed ZeRO checkpoint 目录加载权重到 policy。

    支持两种路径格式：
      - 基目录（包含 'latest' 文件）：自动检测 tag
      - tag 子目录（如 'ckpt_dir/step_000100/'）：从路径推导 tag

    由于训练时使用 exclude_frozen_parameters=True，冻结的 VLM 权重不在 checkpoint 中。
    仅加载 DiT 权重 (policy.model)。VLM 已从本地模型路径初始化。

    Args:
        checkpoint_dir: DeepSpeed checkpoint 路径（基目录或 tag 子目录）
        policy: LoLAV07Policy 实例（VLM 已初始化）
        tag: 显式指定 tag。如果 None，从 'latest' 文件或路径自动检测。
        local_rank: GPU rank，用于日志
        use_ema: 是否用 EMA 分片权重覆盖

    Returns:
        dict: 包含 step/epoch 元数据（如果可解析）
    """
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    except ImportError:
        raise ImportError(
            "DeepSpeed is required to load ZeRO checkpoint directories. "
            "Install it with: pip install deepspeed"
        )

    import glob
    import re

    # 检测路径类型并推导 checkpoint_dir 和 tag
    resolved_dir = checkpoint_dir
    resolved_tag = tag

    if os.path.isfile(os.path.join(checkpoint_dir, "latest")):
        # 基目录：DeepSpeed 会从 'latest' 文件读取 tag
        resolved_dir = checkpoint_dir
        if tag is None:
            latest_file = os.path.join(checkpoint_dir, "latest")
            with open(latest_file, "r") as f:
                resolved_tag = f.read().strip()
            print(f"[Rank {local_rank}] Auto-detected DeepSpeed tag from 'latest': {resolved_tag}")
    elif glob.glob(os.path.join(checkpoint_dir, "*_model_states.pt")):
        # tag 子目录：从目录名推导 tag
        tag_name = os.path.basename(checkpoint_dir)
        parent_dir = os.path.dirname(checkpoint_dir)
        resolved_dir = parent_dir
        resolved_tag = tag_name
        print(f"[Rank {local_rank}] Detected DeepSpeed tag subdirectory: tag={resolved_tag}, base_dir={resolved_dir}")
    else:
        contents = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
        raise ValueError(
            f"Directory does not appear to be a valid DeepSpeed checkpoint: {checkpoint_dir}\n"
            f"Expected 'latest' file or '*_model_states.pt' files.\n"
            f"Directory contents: {contents[:20]}"
        )

    # 提取合并的 FP32 state_dict（此操作内存密集，需要足够 CPU RAM）
    print(f"[Rank {local_rank}] Extracting FP32 state_dict from DeepSpeed checkpoint "
          f"(dir={resolved_dir}, tag={resolved_tag})...")
    ckpt_state_dict = get_fp32_state_dict_from_zero_checkpoint(
        checkpoint_dir=resolved_dir,
        tag=resolved_tag,
        exclude_frozen_parameters=True,
    )

    # 分组：按前缀分离 DiT 和 VLM 权重
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
            # DeepSpeed checkpoint 在 exclude_frozen_parameters=True 时
            # 通常使用裸参数名（无 "policy." 前缀），属于 DiT 权重
            dit_sd[key] = value

    # 加载 DiT 权重
    if dit_sd:
        current_dit_sd = policy.model.state_dict()
        dit_loaded = 0
        dit_missing = 0
        for key in current_dit_sd:
            if key in dit_sd:
                current_dit_sd[key] = dit_sd[key]
                dit_loaded += 1
            else:
                dit_missing += 1
        policy.model.load_state_dict(current_dit_sd)
        print(f"[Rank {local_rank}] DiT weights from DeepSpeed: {dit_loaded} loaded, "
              f"{dit_missing} missing (VLM keys absent: exclude_frozen_parameters=True)")
        if dit_missing > 0:
            print(f"[Rank {local_rank}] WARNING: {dit_missing} DiT params not found in checkpoint — "
                  f"确认 config 与训练一致 (bottleneck dims / bridge / special tokens / history_type)")

    # 加载 VLM 权重（仅在 train_vlm=True 时可能存在）
    if vlm_sd_raw:
        current_vlm_sd = policy.vlm.state_dict()
        vlm_loaded = 0
        for key in current_vlm_sd:
            if key in vlm_sd_raw:
                current_vlm_sd[key] = vlm_sd_raw[key]
                vlm_loaded += 1
        policy.vlm.load_state_dict(current_vlm_sd, strict=False)
        print(f"[Rank {local_rank}] VLM weights from DeepSpeed: {vlm_loaded} loaded "
              f"(frozen VLM params are not in checkpoint)")
    else:
        print(f"[Rank {local_rank}] No VLM weights in DeepSpeed checkpoint "
              f"(expected: VLM is frozen, loaded from local path)")

    # 从 tag 名称解析 step/epoch 元数据
    metadata = {}
    if resolved_tag:
        step_match = re.search(r"step_?(\d+)", resolved_tag)
        if step_match:
            metadata["step"] = int(step_match.group(1))
        if resolved_tag == "final":
            metadata["is_final"] = True

    # EMA 覆盖: 用 EMA 权重替换刚加载的训练权重 (ZeRO-3 按 rank 分片保存, 需重建)
    if use_ema:
        tag_dir = os.path.join(resolved_dir, resolved_tag) if resolved_tag else resolved_dir
        n_overlaid = overlay_ema_into_policy(policy, tag_dir, local_rank=local_rank)
        if n_overlaid == 0:
            print(f"[Rank {local_rank}] WARNING: --use_ema 指定但 {tag_dir} 下无 ema_rank_*.pt, "
                  f"使用非 EMA 权重")
        else:
            metadata["use_ema"] = True

    return metadata


def overlay_ema_into_policy(policy, tag_dir: str, local_rank: int = 0) -> int:
    """将 tag_dir 下的 ema_rank_*.pt (ZeRO-3 shard-local EMA) 重建并覆盖到 policy。

    训练侧 EMA 以 policy.named_parameters() 的名字 ("model.*" / "vlm.*") 为键,
    每个 rank 只保存自己拥有的参数分片 (扁平 1-D, 可能带 padding)。
    重建方式: 按 rank 序拼接 → 截断到目标 numel → reshape 成目标形状。

    Returns:
        int: 成功覆盖的参数个数 (0 = 未找到 EMA 文件)
    """
    import glob as _glob
    import re

    def _rank_of(path):
        m = re.search(r"ema_rank_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    ema_files = sorted(_glob.glob(os.path.join(tag_dir, "ema_rank_*.pt")), key=_rank_of)
    if not ema_files:
        return 0

    # name -> [(rank, shard), ...]
    shards = {}
    for f in ema_files:
        sd = torch.load(f, map_location="cpu")
        for name, t in sd.items():
            shards.setdefault(name, []).append((_rank_of(f), t))

    target_sd = policy.state_dict()
    dit_sd = policy.model.state_dict()
    vlm_sd = policy.vlm.state_dict()
    n_overlaid = 0
    for name, parts in shards.items():
        key = name[len("policy."):] if name.startswith("policy.") else name
        if key not in target_sd:
            print(f"[Rank {local_rank}] [EMA] 跳过未知参数: {key}")
            continue
        parts.sort(key=lambda x: x[0])
        flat = torch.cat([t.reshape(-1) for _, t in parts])
        tgt = target_sd[key]
        if flat.numel() < tgt.numel():
            print(f"[Rank {local_rank}] [EMA] 跳过 {key}: 分片总元素 {flat.numel()} < 目标 {tgt.numel()}")
            continue
        full = flat[: tgt.numel()].reshape(tgt.shape).to(dtype=tgt.dtype)
        if key.startswith("model.") and key[len("model."):] in dit_sd:
            dit_sd[key[len("model."):]] = full
            n_overlaid += 1
        elif key.startswith("vlm.") and key[len("vlm."):] in vlm_sd:
            vlm_sd[key[len("vlm."):]] = full
            n_overlaid += 1
    policy.model.load_state_dict(dit_sd)
    policy.vlm.load_state_dict(vlm_sd, strict=False)
    print(f"[Rank {local_rank}] [EMA] 覆盖 {n_overlaid} 个参数 (来自 {len(ema_files)} 个 rank 分片)")
    return n_overlaid


def validate_forward_loss(policy, preprocessor, val_loader, device,
                          action_dim=None, gripper_dim_indices=None,
                          compute_per_dim=False):
    """在验证集上计算 forward loss（v-loss, arm_loss, gripper_loss, total_loss）

    支持每维度的 arm_loss 分解。
    """
    gripper_dim_indices = gripper_dim_indices or []
    continuous_dim_indices = [i for i in range(action_dim) if i not in gripper_dim_indices]
    arm_dim = action_dim - len(gripper_dim_indices)
    need_per_dim = compute_per_dim or len(gripper_dim_indices) > 0

    # forward loss 需要模型在 train 模式（flow matching 需要随机采样噪声和时间步）
    # 注意: 只开 policy.model 的 train 模式; policy 自身保持 eval,
    # 避免 V07 的 visual_token_drop (self.training 门控) 在验证时误触发。
    policy.eval()
    policy.model.train()
    # 但冻结的 VLM 保持 eval
    if not policy.config.train_vlm and hasattr(policy, 'vlm'):
        policy.vlm.eval()

    total_loss = 0.0
    total_v_loss = 0.0
    total_arm_loss = 0.0
    total_gripper_loss = 0.0
    num_batches = 0

    per_dim_arm_loss_sum = torch.zeros(arm_dim, dtype=torch.float64) if need_per_dim else None

    print("Running forward loss validation...")
    for batch_idx, batch in enumerate(val_loader):
        # 提取特殊字段（在 preprocessor 之前，与训练一致）
        special_data = extract_special_fields(batch)

        # 应用预处理器（内含 DeviceProcessorStep 会将数据移到 config.device）
        batch = preprocessor(batch)

        # 恢复特殊字段并移动到设备
        for k, v in special_data.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            else:
                batch[k] = v

        with torch.no_grad():
            loss, loss_dict = policy(batch, compute_per_dim=need_per_dim)

        total_loss += loss_dict["loss"]
        total_v_loss += loss_dict["v_loss"]
        total_arm_loss += loss_dict["arm_loss"]
        total_gripper_loss += loss_dict["gripper_loss"]
        num_batches += 1

        if need_per_dim and "arm_loss_per_dim" in loss_dict:
            per_dim_arm_loss_sum += loss_dict["arm_loss_per_dim"].double().cpu()

        if (batch_idx + 1) % 10 == 0:
            print(f"[Rank {os.environ.get('LOCAL_RANK', '0')}] Forward loss: {batch_idx + 1}/{len(val_loader)} batches done")

    policy.model.eval()

    # 多卡同步：构建单一 flat tensor
    is_distributed = dist.is_initialized()
    if is_distributed:
        header_size = 5
        per_dim_size = arm_dim if need_per_dim else 0
        total_size = header_size + per_dim_size

        stats_tensor = torch.zeros(total_size, dtype=torch.float64, device=device)
        stats_tensor[0] = total_loss
        stats_tensor[1] = total_v_loss
        stats_tensor[2] = total_arm_loss
        stats_tensor[3] = total_gripper_loss
        stats_tensor[4] = num_batches

        if need_per_dim:
            stats_tensor[header_size:header_size + arm_dim] = per_dim_arm_loss_sum.to(device)

        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        total_loss = stats_tensor[0].item()
        total_v_loss = stats_tensor[1].item()
        total_arm_loss = stats_tensor[2].item()
        total_gripper_loss = stats_tensor[3].item()
        num_batches = stats_tensor[4].item()

        if need_per_dim:
            per_dim_arm_loss_sum = stats_tensor[header_size:header_size + arm_dim].cpu()

    if num_batches == 0:
        return {}

    results = {
        "val_total_loss": total_loss / num_batches,
        "val_v_loss": total_v_loss / num_batches,
        "val_arm_loss": total_arm_loss / num_batches,
        "val_gripper_loss": total_gripper_loss / num_batches,
    }

    # 每维度 arm_loss（arm_loss_per_dim 只含 arm 维度，不含夹爪）
    if need_per_dim:
        per_dim_arm_loss_avg = per_dim_arm_loss_sum / num_batches
        for i, dim_idx in enumerate(continuous_dim_indices):
            results[f"val_arm_loss_dim_{dim_idx}"] = per_dim_arm_loss_avg[i].item()

        results["val_continuous_arm_loss"] = per_dim_arm_loss_avg.mean().item()

    return results


def validate_inference(policy, preprocessor, val_loader, device, max_samples=100,
                       action_dim=None, gripper_dim_indices=None, gripper_threshold=0.0,
                       compute_per_dim=False, norm_mode="default", dataset_stats=None,
                       num_act_exec=None, action_key="action"):
    """运行推理去噪管线，对比预测动作与真实动作。

    支持每维度 MSE/L1 指标和夹爪分类指标（accuracy, precision, recall, F1）。
    action_key: stats_mode="incremental" 时为 "action_incremental"。
    """
    policy.eval()
    policy.model.eval()

    gripper_dim_indices = gripper_dim_indices or []
    num_gripper_dims = len(gripper_dim_indices)
    continuous_dim_indices = [i for i in range(action_dim) if i not in gripper_dim_indices]
    need_per_dim = compute_per_dim or len(gripper_dim_indices) > 0

    # 累积器
    mse_sum = 0.0
    l1_sum = 0.0
    n_batches = 0
    sample_count = 0

    per_dim_mse_sum = torch.zeros(action_dim, dtype=torch.float64) if need_per_dim else None
    per_dim_l1_sum = torch.zeros(action_dim, dtype=torch.float64) if need_per_dim else None

    gripper_tp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_fp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_fn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_tn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_total = 0

    print("Running inference validation...")
    for batch_idx, batch in enumerate(val_loader):
        if sample_count >= max_samples:
            break

        # 保存 ground truth action
        special_data = extract_special_fields(batch)
        ground_truth_actions = special_data["action"]  # [B, T, action_dim]

        # 应用预处理器（内含 DeviceProcessorStep 会将数据移到 config.device）
        batch = preprocessor(batch)

        # 恢复历史/transition 字段（推理需要），并移动到设备
        # ("action" 不恢复 — 它是 GT, 不是模型输入)
        for key, value in special_data.items():
            if key == "action":
                continue
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            else:
                batch[key] = value

        # 推理
        with torch.no_grad():
            predicted_actions = policy.predict_action_chunk(batch)  # [B, pred_len, action_dim]

        # 对齐长度
        if ground_truth_actions.ndim == 2:
            ground_truth_actions = ground_truth_actions.unsqueeze(1)

        pred_len = predicted_actions.shape[1]
        gt_len = ground_truth_actions.shape[1]
        min_len = min(pred_len, gt_len)
        if num_act_exec is not None:
            min_len = min(min_len, num_act_exec)

        pred_matched = predicted_actions[:, :min_len, :]
        gt_matched = ground_truth_actions[:, :min_len, :].to(device)

        # Unnormalize for MSE/L1 comparison in original space (for cross-model comparison).
        # - robovlm (IDENTITY): predictions already in original space, no unnormalization needed.
        # - default (MEAN_STD): arm dims need unnormalization; gripper dims are already
        #   discretized to {-1, 1} (original space) via sigmoid thresholding in sample_actions.
        # - zscore: both pred and GT need unnormalization for cross-model comparison;
        #   gripper dims stay in original space.
        if norm_mode in ("default", "zscore") and dataset_stats is not None:
            pred_for_metric = unnormalize_lola_actions(
                pred_matched, dataset_stats, action_dim, norm_mode, action_key)
            # Overwrite gripper dims with original predictions (already in original space)
            for g_dim in gripper_dim_indices:
                pred_for_metric[:, :, g_dim] = pred_matched[:, :, g_dim]

            if norm_mode == "zscore":
                # zscore: GT is also normalized, must unnormalize for original-space comparison
                gt_for_metric = unnormalize_lola_actions(
                    gt_matched, dataset_stats, action_dim, norm_mode, action_key)
                for g_dim in gripper_dim_indices:
                    gt_for_metric[:, :, g_dim] = gt_matched[:, :, g_dim]
            else:
                gt_for_metric = gt_matched
        else:
            pred_for_metric = pred_matched
            gt_for_metric = gt_matched

        # 总体 MSE/L1（向后兼容）
        mse = F.mse_loss(pred_for_metric, gt_for_metric, reduction="mean")
        l1 = F.l1_loss(pred_for_metric, gt_for_metric, reduction="mean")
        mse_sum += mse.item()
        l1_sum += l1.item()
        n_batches += 1
        sample_count += predicted_actions.shape[0]

        # 每维度 MSE/L1
        if need_per_dim:
            per_dim_mse = F.mse_loss(pred_for_metric, gt_for_metric, reduction="none").mean(dim=(0, 1))
            per_dim_l1 = F.l1_loss(pred_for_metric, gt_for_metric, reduction="none").mean(dim=(0, 1))
            per_dim_mse_sum += per_dim_mse.double().cpu()
            per_dim_l1_sum += per_dim_l1.double().cpu()

        # 夹爪分类指标
        if num_gripper_dims > 0:
            # Gripper dims are already discretized to {-1, 1} via sigmoid thresholding
            # in sample_actions, so they are already in original space regardless of
            # norm_mode. No unnormalization needed for gripper dims.
            pred_for_gripper = pred_matched

            for g_idx, g_dim in enumerate(gripper_dim_indices):
                pred_gripper = pred_for_gripper[:, :, g_dim]  # [B, min_len]
                gt_gripper = gt_matched[:, :, g_dim]          # [B, min_len]

                pred_binary = (pred_gripper > 0.0).reshape(-1).float()
                gt_binary = (gt_gripper > 0.0).reshape(-1).float()

                tp = (pred_binary * gt_binary).sum().double().cpu()
                fp = ((1 - gt_binary) * pred_binary).sum().double().cpu()
                fn = (gt_binary * (1 - pred_binary)).sum().double().cpu()
                tn = ((1 - gt_binary) * (1 - pred_binary)).sum().double().cpu()

                gripper_tp[g_idx] += tp
                gripper_fp[g_idx] += fp
                gripper_fn[g_idx] += fn
                gripper_tn[g_idx] += tn

            gripper_total += pred_matched.shape[0] * pred_matched.shape[1]

        if (batch_idx + 1) % 10 == 0:
            print(f"[Rank {os.environ.get('LOCAL_RANK', '0')}] Inference: {sample_count}/{max_samples} samples done")

    # 多卡同步：构建单一 flat tensor，一次 all_reduce
    is_distributed = dist.is_initialized()
    if is_distributed:
        header_size = 4  # mse_sum, l1_sum, n_batches, sample_count
        per_dim_size = action_dim * 2 if need_per_dim else 0
        gripper_size = num_gripper_dims * 4 + 1 if num_gripper_dims > 0 else 0
        total_size = header_size + per_dim_size + gripper_size

        stats_tensor = torch.zeros(total_size, dtype=torch.float64, device=device)
        stats_tensor[0] = mse_sum
        stats_tensor[1] = l1_sum
        stats_tensor[2] = n_batches
        stats_tensor[3] = sample_count

        offset = header_size
        if need_per_dim:
            stats_tensor[offset:offset + action_dim] = per_dim_mse_sum.to(device)
            offset += action_dim
            stats_tensor[offset:offset + action_dim] = per_dim_l1_sum.to(device)
            offset += action_dim

        if num_gripper_dims > 0:
            stats_tensor[offset:offset + num_gripper_dims] = gripper_tp.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = gripper_fp.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = gripper_fn.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = gripper_tn.to(device)
            offset += num_gripper_dims
            stats_tensor[offset] = gripper_total

        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        mse_sum = stats_tensor[0].item()
        l1_sum = stats_tensor[1].item()
        n_batches = stats_tensor[2].item()
        sample_count = stats_tensor[3].item()

        offset = header_size
        if need_per_dim:
            per_dim_mse_sum = stats_tensor[offset:offset + action_dim].cpu()
            offset += action_dim
            per_dim_l1_sum = stats_tensor[offset:offset + action_dim].cpu()
            offset += action_dim

        if num_gripper_dims > 0:
            gripper_tp = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            gripper_fp = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            gripper_fn = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            gripper_tn = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            gripper_total = stats_tensor[offset].item()

    if n_batches == 0:
        return {}

    results = {
        "val_action_mse": mse_sum / n_batches,
        "val_action_l1": l1_sum / n_batches,
    }

    # 每维度指标
    if need_per_dim:
        per_dim_mse_avg = per_dim_mse_sum / n_batches
        per_dim_l1_avg = per_dim_l1_sum / n_batches

        for i in range(action_dim):
            results[f"val_mse_dim_{i}"] = per_dim_mse_avg[i].item()
            results[f"val_l1_dim_{i}"] = per_dim_l1_avg[i].item()

        # 仅连续维度的聚合 MSE/L1（排除夹爪）
        if continuous_dim_indices:
            continuous_mse = per_dim_mse_avg[continuous_dim_indices].mean().item()
            continuous_l1 = per_dim_l1_avg[continuous_dim_indices].mean().item()
            results["val_continuous_mse"] = continuous_mse
            results["val_continuous_l1"] = continuous_l1

    # 夹爪分类指标
    if num_gripper_dims > 0 and gripper_total > 0:
        for g_idx, g_dim in enumerate(gripper_dim_indices):
            tp = gripper_tp[g_idx].item()
            fp = gripper_fp[g_idx].item()
            fn = gripper_fn[g_idx].item()
            tn = gripper_tn[g_idx].item()
            total = tp + fp + fn + tn

            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[f"val_gripper_dim_{g_dim}_accuracy"] = accuracy
            results[f"val_gripper_dim_{g_dim}_precision"] = precision
            results[f"val_gripper_dim_{g_dim}_recall"] = recall
            results[f"val_gripper_dim_{g_dim}_f1"] = f1

    return results


# ─── training_config.json 自动匹配 (参照 eval_calvin.py) ─────────────────────

def _load_training_config(config_path, rank=0):
    """Load training_config.json and return (lola_config, training_args) dicts."""
    print(f"[Rank {rank}] Loading training config from {config_path}...")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    lola_cfg = cfg.get("lola_config", {})
    train_args = cfg.get("training_args", {})
    print(f"[Rank {rank}] Training config loaded: history_type={lola_cfg.get('history_type')}, "
          f"use_special_tokens={lola_cfg.get('use_special_tokens')}, "
          f"state_dim={lola_cfg.get('state_dim')}, action_dim={lola_cfg.get('action_dim')}")
    return lola_cfg, train_args


def _apply_training_config(args, lola_cfg, train_args):
    """Apply training_config values to args (CLI overrides take precedence)."""
    # Mapping: (args attr, lola_config key, training_args key)
    mapping = [
        ("action_dim", "action_dim", "action_dim"),
        ("action_chunk_size", "action_chunk_size", "action_chunk_size"),
        ("pred_chunk_size", "pred_chunk_size", "pred_chunk_size"),
        ("n_obs_steps", "n_obs_steps", "n_obs_steps"),
        ("max_history_length", "max_history_length", "max_history_length"),
        ("history_padding_side", "history_padding_side", "history_padding_side"),
        ("vlm_extract_layers", "vlm_extract_layers", "vlm_extract_layers"),
        ("max_image_pixels", "max_image_pixels", "max_image_pixels"),
        ("min_image_pixels", "min_image_pixels", "min_image_pixels"),
        ("num_inference_steps", "num_inference_steps", "num_inference_steps"),
        ("vlm_backbone", "vlm_backbone", "vlm_backbone"),
        ("vlm_hidden_size", "vlm_hidden_size", "vlm_hidden_size"),
        ("dit_hidden_size", "dit_hidden_size", "dit_hidden_size"),
        ("empty_cameras", "empty_cameras", "empty_cameras"),
        ("empty_token_id", "empty_token_id", "empty_token_id"),
        ("gripper_threshold", "gripper_threshold", "gripper_threshold"),
        ("gripper_loss_weight", "gripper_loss_weight", "gripper_loss_weight"),
        ("action_loss_weight", "action_loss_weight", "action_loss_weight"),
        ("vlm_model_name", "vlm_model_name", "vlm_model_name"),
        # v07 架构/行为字段
        ("history_type", "history_type", "history_type"),
        ("state_dim", "state_dim", "state_dim"),
        ("state_encoder_mode", "state_encoder_mode", "state_encoder_mode"),
        ("use_state_condition", "use_state_condition", "use_state_condition"),
        ("vlm_bridge_mode", "vlm_bridge_mode", "vlm_bridge_mode"),
        ("vlm_bridge_width", "vlm_bridge_width", "vlm_bridge_width"),
        ("vlm_bridge_layers", "vlm_bridge_layers", "vlm_bridge_layers"),
        ("vlm_bridge_num_heads", "vlm_bridge_num_heads", "vlm_bridge_num_heads"),
        ("vlm_bridge_ffn_ratio", "vlm_bridge_ffn_ratio", "vlm_bridge_ffn_ratio"),
        ("obs_prev_chunk_frame", "obs_prev_chunk_frame", "obs_prev_chunk_frame"),
        ("action_bottleneck_dim", "action_bottleneck_dim", "action_bottleneck_dim"),
        ("grip_bottleneck_dim", "grip_bottleneck_dim", "grip_bottleneck_dim"),
        ("state_bottleneck_dim", "state_bottleneck_dim", "state_bottleneck_dim"),
        ("state_grip_bottleneck_dim", "state_grip_bottleneck_dim", "state_grip_bottleneck_dim"),
        # Text template & completed tasks
        ("task_text_template_version", "task_text_template_version", "task_text_template_version"),
        ("completed_tasks_use_ann", "completed_tasks_use_ann", "completed_tasks_use_ann"),
        ("completed_tasks_history_len", "completed_tasks_history_len", "completed_tasks_history_len"),
        ("max_transition_len", "max_transition_len", "max_transition_len"),
        # Special tokens
        ("use_special_tokens", "use_special_tokens", "use_special_tokens"),
        # Normalization
        ("norm_mode", None, "norm_mode"),
        ("norm_min", None, "norm_min"),
        ("norm_max", None, "norm_max"),
        ("stats_mode", None, "stats_mode"),
    ]
    for attr, lola_key, train_key in mapping:
        if getattr(args, attr) is not None:
            continue  # CLI override takes precedence
        if lola_key and lola_key in lola_cfg:
            setattr(args, attr, lola_cfg[lola_key])
        elif train_key and train_key in train_args:
            setattr(args, attr, train_args[train_key])

    # Special handling for boolean flags / paths
    if args.vlm_path is None:
        args.vlm_path = lola_cfg.get("vlm_path") or train_args.get("vlm_path")
    if args.train_vlm is None:
        args.train_vlm = lola_cfg.get("train_vlm", False)
    if args.load_full_history is None:
        args.load_full_history = lola_cfg.get("load_full_history", False)
    if args.gripper_dims is None:
        gd = lola_cfg.get("gripper_dim_indices", train_args.get("gripper_dims"))
        if isinstance(gd, (list, tuple)):
            args.gripper_dims = ",".join(str(x) for x in gd)
        elif isinstance(gd, str):
            args.gripper_dims = gd
    if args.static_vlm_padding is None:
        args.static_vlm_padding = lola_cfg.get("static_vlm_padding", False)
    if args.vlm_max_length is None:
        args.vlm_max_length = lola_cfg.get("vlm_max_length")
    # normalization_mapping: 训练时保存的精确映射 (json 中是 "MEAN_STD" 等字符串),
    # 在 main 中转换回 NormalizationMode
    if getattr(args, "normalization_mapping", None) is None:
        args.normalization_mapping = lola_cfg.get("normalization_mapping")

    # 验证数据集: 默认用训练时保存的 val 数据集 (没有则回退训练数据集)
    if args.val_dataset_repo_id is None:
        args.val_dataset_repo_id = train_args.get("val_dataset_repo_id") or train_args.get("dataset_repo_id")
        if args.val_dataset_repo_id:
            print(f"Auto-filled val_dataset_repo_id from training_config: {args.val_dataset_repo_id}")
    if args.val_dataset_root is None:
        args.val_dataset_root = train_args.get("val_dataset_root") or train_args.get("dataset_root")
        if args.val_dataset_root:
            print(f"Auto-filled val_dataset_root from training_config: {args.val_dataset_root}")


def main():
    parser = argparse.ArgumentParser(description="LoLA V07 Model Validation")

    # Training config — 自动匹配训练配置 (CLI 参数优先)
    parser.add_argument("--training_config", type=str, default=None,
                        help="Path to training_config.json saved by v07 training. "
                             "When provided, config args are auto-populated; "
                             "CLI args override training_config values.")

    # 验证数据集参数
    parser.add_argument("--val_dataset_repo_id", type=str, default=None,
                        help="Validation dataset repo ID (default: from training_config)")
    parser.add_argument("--val_dataset_root", type=str, default=None,
                        help="Local root for validation dataset (default: from training_config)")
    parser.add_argument("--val_episodes", type=int, nargs="*", default=None,
                        help="Specific validation episodes to load (optional)")

    # 模型参数
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint. Supports: single file (.ckpt/.pt), "
                             "DeepSpeed ZeRO directory (base dir or tag subdirectory)")
    parser.add_argument("--deepspeed_tag", type=str, default=None,
                        help="Explicit tag for DeepSpeed checkpoint (e.g., 'step_000100' or 'final'). "
                             "Auto-detected from 'latest' file or directory name if not provided.")
    parser.add_argument("--use_ema", action="store_true", default=False,
                        help="用 checkpoint 内的 EMA 权重 (ema_rank_*.pt / ema_state) 覆盖训练权重后再验证")
    parser.add_argument("--vlm_path", type=str, default=None,
                        help="Path to local VLM model (default: from training_config)")
    parser.add_argument("--vlm_model_name", type=str, default=None,
                        help="HF model name for the VLM backbone. None → backbone default "
                             "(qwen3_5: Qwen/Qwen3.5-4B, cosmos3_nano: nvidia/Cosmos3-Nano)")

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
    parser.add_argument("--static_vlm_padding", action="store_true", default=None,
                        help="Pad VLM tokens to fixed max_length for consistent tensor shapes")
    parser.add_argument("--vlm_max_length", type=int, default=None,
                        help="Override tokenizer max_length when static_vlm_padding=True")
    parser.add_argument("--empty_cameras", type=int, default=None)
    parser.add_argument("--vlm_hidden_size", type=int, default=None,
                        help="VLM hidden dim (None → backbone default: qwen3_5=2560, cosmos3_nano=4096)")
    parser.add_argument("--vlm_backbone", type=str, default=None,
                        help="VLM backbone registry key (e.g. qwen3_5, cosmos3_nano); must match training")
    parser.add_argument("--dit_hidden_size", type=int, default=None)
    parser.add_argument("--empty_token_id", type=int, default=None,
                        help="Empty token ID (None → backbone default)")

    # v07 架构/行为参数
    parser.add_argument("--history_type", type=str, default=None, choices=["action", "state"])
    parser.add_argument("--state_dim", type=int, default=None)
    parser.add_argument("--state_encoder_mode", type=str, default=None, choices=["unified", "separated"])
    parser.add_argument("--use_state_condition", action="store_true", default=None)
    parser.add_argument("--vlm_bridge_mode", type=str, default=None, choices=["legacy", "transformer"])
    parser.add_argument("--vlm_bridge_width", type=int, default=None)
    parser.add_argument("--vlm_bridge_layers", type=int, default=None)
    parser.add_argument("--vlm_bridge_num_heads", type=int, default=None,
                        help="Bridge attention heads (0 → auto width // 128)")
    parser.add_argument("--vlm_bridge_ffn_ratio", type=float, default=None,
                        help="Bridge SwiGLU FFN expansion ratio")
    parser.add_argument("--obs_prev_chunk_frame", action="store_true", default=None,
                        help="观测 = [上一 chunk 起始帧, 当前帧]; 必须与训练一致 (改变数据集取帧)")
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
    parser.add_argument("--use_special_tokens", action="store_true", default=None,
                        help="Insert 5 special token embeddings in DiT sequence; must match training")
    parser.add_argument("--no_use_special_tokens", action="store_true")

    # 验证模式
    parser.add_argument("--mode", type=str, default="both",
                        choices=["forward_loss", "inference", "both"],
                        help="Validation mode")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                        help="Max number of samples for inference validation")
    parser.add_argument("--num_act_exec", type=int, default=None,
                        help="Number of action steps to compare in inference validation. "
                             "Only the first num_act_exec actions are used for loss and gripper metrics")

    # DataLoader 参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # 归一化参数
    parser.add_argument("--norm_mode", type=str, default=None,
                        choices=["default", "robovlm", "zscore"])
    parser.add_argument("--norm_min", type=float, default=None)
    parser.add_argument("--norm_max", type=float, default=None)
    parser.add_argument("--stats_mode", type=str, default=None,
                        choices=["original", "incremental"],
                        help="Stats mode for z-score normalization; must match training "
                             "(incremental → action_incremental stats)")

    # 每维度指标与夹爪分类参数
    parser.add_argument("--gripper_dims", type=str, default=None,
                        help="Comma-separated indices of gripper dims. Supports negative indices "
                             "(e.g., '-1' for last dim, '-1,-11' for dual-arm). "
                             "Default: from training_config")
    parser.add_argument("--gripper_threshold", type=float, default=None)
    parser.add_argument("--gripper_loss_weight", type=float, default=None)
    parser.add_argument("--action_loss_weight", type=float, default=None,
                        help="Huber loss weight for continuous arm dimensions (v07 default 10.0)")
    parser.add_argument("--per_dim_metrics", action="store_true", default=False,
                        help="Compute per-dimension MSE/L1 for all action dims")

    # 输出
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save validation results to JSON file")

    args = parser.parse_args()

    # ─── Apply training_config.json if provided ─────────────────────────────
    if args.training_config is not None:
        lola_cfg, train_args = _load_training_config(args.training_config)
        _apply_training_config(args, lola_cfg, train_args)

    # ─── 默认值填充 (training_config 与 CLI 都未提供时) ──────────────────────
    if args.state_dim is None:
        args.state_dim = None  # 加载数据集 features 后解析, 见下文
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
        args.vlm_bridge_num_heads = 0  # 0 → auto width // 128
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
        args.action_loss_weight = 10.0  # v07 默认 (v06 为 1.0)
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
        args.vlm_backbone = "qwen3_5"  # 兼容无该字段的旧 checkpoint
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

    # set seed
    seed = 42
    torch.manual_seed(seed)

    # 确定设备
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 初始化分布式（如果使用 torchrun）
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    # 加载数据集元数据
    print(f"Loading validation dataset metadata from {args.val_dataset_repo_id or args.val_dataset_root}...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.val_dataset_repo_id,
        root=args.val_dataset_root,
    )

    # 获取 features
    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    elif args.action_dim is not None:
        action_dim = args.action_dim
    else:
        raise ValueError("action_dim could not be determined from dataset features, "
                         "training_config, or --action_dim")

    # state_dim: CLI/training_config > dataset features > action_dim
    if args.state_dim is not None:
        state_dim = args.state_dim
    elif "observation.state" in features:
        state_dim = features["observation.state"].shape[0]
    else:
        state_dim = action_dim  # fallback

    # 解析夹爪维度索引
    gripper_dim_indices = []
    if args.gripper_dims is not None:
        raw_indices = [int(x.strip()) for x in args.gripper_dims.split(",")]
        for idx in raw_indices:
            if idx < 0:
                resolved = action_dim + idx
            else:
                resolved = idx
            if resolved < 0 or resolved >= action_dim:
                raise ValueError(f"Gripper dim index {idx} resolves to {resolved}, "
                                 f"out of range [0, {action_dim})")
            gripper_dim_indices.append(resolved)
        gripper_dim_indices = sorted(set(gripper_dim_indices))
        print(f"Gripper dim indices: {gripper_dim_indices}")

    compute_per_dim = args.per_dim_metrics or len(gripper_dim_indices) > 0

    # stats_mode → action stats key
    action_key = "action_incremental" if args.stats_mode == "incremental" else "action"
    if args.stats_mode == "incremental" and action_key not in (dataset_metadata.stats or {}):
        print(f"WARNING: stats_mode=incremental 但数据集 stats 中无 '{action_key}', "
              f"unnormalize 将回退到 'action'")

    print(f"Validation Dataset Info:")
    print(f"  - Total episodes: {dataset_metadata.total_episodes}")
    print(f"  - Total frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")
    print(f"  - Action dim: {action_dim}, State dim: {state_dim}")

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

    # ─── 创建 LoLAV07 配置 ──────────────────────────────────────────────────
    # 注意: train-only 字段强制为验证语义 (与推理分布对齐):
    #   hist_action_token_drop_rate=0 / transition_mask_rate=0 / visual_token_drop_rate=0
    #   gradient_checkpointing=False (eval 无需重算)
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
        transition_mask_rate=0.0,            # train-only: 验证不对 transition token 做随机 mask
        max_transition_len=args.max_transition_len,
        action_bottleneck_dim=args.action_bottleneck_dim,
        grip_bottleneck_dim=args.grip_bottleneck_dim,
        state_bottleneck_dim=args.state_bottleneck_dim,
        state_grip_bottleneck_dim=args.state_grip_bottleneck_dim,
        obs_prev_chunk_frame=args.obs_prev_chunk_frame,
        visual_token_drop_rate=0.0,          # train-only
    )
    # 设置 config.device 为当前 rank 对应的 GPU
    # 这会影响 preprocessor 中的 DeviceProcessorStep 以及模型加载
    config.device = f"cuda:{local_rank}"

    print(f"[Rank {local_rank}] Config: history_type={config.history_type}, "
          f"use_special_tokens={config.use_special_tokens}, "
          f"task_template={config.task_text_template_version}, "
          f"vlm_backbone={config.vlm_backbone}, "
          f"obs_prev_chunk_frame={config.obs_prev_chunk_frame}, "
          f"action_bottleneck={config.action_bottleneck_dim}, "
          f"grip_bottleneck={config.grip_bottleneck_dim}, "
          f"stats_mode={args.stats_mode}")

    # 归一化模式: 优先用 training_config 保存的精确 normalization_mapping
    # (反映训练时 preprocessor 的实际行为, 如 zscore 训练是 STATE=MEAN_STD/ACTION=IDENTITY)
    saved_norm_mapping = getattr(args, "normalization_mapping", None)
    if saved_norm_mapping:
        config.normalization_mapping = {
            k: NormalizationMode(v) for k, v in saved_norm_mapping.items()
        }
        print(f"[Rank {local_rank}] normalization_mapping from training_config: "
              f"{ {k: v.value for k, v in config.normalization_mapping.items()} }")
    elif args.norm_mode == "robovlm":
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    elif args.norm_mode == "zscore":
        # 与 train_lola_v07_azure 一致: zscore 下 STATE 走 MEAN_STD
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.IDENTITY,
        }

    # 加载模型
    print(f"[Rank {local_rank}] Loading LoLAV07 model...")

    # 先创建 policy 和 preprocessor（模型在 CPU 上）
    policy = LoLAV07Policy(config)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=dataset_metadata.stats,
    )

    # 加载 checkpoint 权重到 CPU，避免 torch.load 默认把 tensor 加载到 cuda:0
    ckpt_metadata = {}
    ckpt_type = "none"

    if args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            # --- 单文件 checkpoint (.pt / .ckpt) ---
            ckpt_type = "single_file"
            print(f"[Rank {local_rank}] Loading single-file checkpoint: {args.checkpoint_path}")
            ckpt = torch.load(args.checkpoint_path, map_location="cpu")

            # 提取 state_dict
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

            # 分组：按前缀分离 DiT 和 VLM 权重
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

            # 加载 DiT 权重
            if dit_sd:
                current_dit_sd = policy.model.state_dict()
                dit_loaded = 0
                dit_missing = 0
                for key in current_dit_sd:
                    if key in dit_sd:
                        current_dit_sd[key] = dit_sd[key]
                        dit_loaded += 1
                    else:
                        dit_missing += 1
                policy.model.load_state_dict(current_dit_sd)
                print(f"[Rank {local_rank}] DiT weights: {dit_loaded} loaded, {dit_missing} missing")
                if dit_missing > 0:
                    print(f"[Rank {local_rank}] WARNING: {dit_missing} DiT params not found — "
                          f"确认 config 与训练一致 (bottleneck dims / bridge / special tokens / history_type)")

            # 加载 VLM 权重
            if vlm_sd_raw:
                current_vlm_sd = policy.vlm.state_dict()
                vlm_loaded = 0
                vlm_missing = 0
                for key in current_vlm_sd:
                    if key in vlm_sd_raw:
                        current_vlm_sd[key] = vlm_sd_raw[key]
                        vlm_loaded += 1
                    else:
                        vlm_missing += 1
                policy.vlm.load_state_dict(current_vlm_sd)
                print(f"[Rank {local_rank}] VLM weights: {vlm_loaded} loaded, {vlm_missing} missing")

            # EMA 覆盖 (DDP 单文件: ema_state 为完整参数 dict, 非分片)
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
                    print(f"[Rank {local_rank}] [EMA] 覆盖 {n_ema} 个参数 (单文件 ema_state)")
                else:
                    print(f"[Rank {local_rank}] WARNING: --use_ema 指定但 checkpoint 无 ema_state, "
                          f"使用非 EMA 权重")

        elif os.path.isdir(args.checkpoint_path):
            # --- DeepSpeed ZeRO checkpoint 目录 ---
            ckpt_type = "deepspeed"
            ckpt_metadata = load_deepspeed_checkpoint(
                checkpoint_dir=args.checkpoint_path,
                policy=policy,
                tag=args.deepspeed_tag,
                local_rank=local_rank,
                use_ema=args.use_ema,
            )
            if ckpt_metadata:
                step = ckpt_metadata.get("step", "N/A")
                is_final = ckpt_metadata.get("is_final", False)
                print(f"[Rank {local_rank}] DeepSpeed checkpoint metadata: "
                      f"step={step}, is_final={is_final}")

        else:
            raise ValueError(f"Checkpoint path does not exist: {args.checkpoint_path}")
    else:
        print(f"[Rank {local_rank}] No checkpoint provided, using randomly initialized model")

    # 将模型移动到当前 rank 对应的 GPU（关键：从 CPU 移到 cuda:{local_rank}）
    policy._device = device
    policy.model = policy.model.to(device)
    policy.vlm = policy.vlm.to(device)
    policy.eval()
    policy.model.eval()
    if not policy.config.train_vlm:
        policy.vlm.eval()

    # 验证设备
    dit_device = next(policy.model.parameters()).device
    vlm_device = next(policy.vlm.parameters()).device
    print(f"[Rank {local_rank}] DiT device: {dit_device}, VLM device: {vlm_device}")

    # 创建验证数据集 (V07 参数: history_type / state_dim / completed_tasks / stats_mode)
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
        # V2: completed tasks + transition masking
        track_completed_tasks=config.task_text_template_version == "v1_with_completed",
        transition_mask_rate=0.0,  # train-only: 验证不做随机 transition mask
        completed_tasks_use_ann=config.completed_tasks_use_ann,
        completed_tasks_history_len=config.completed_tasks_history_len,
        max_transition_len=config.max_transition_len,
        stats_mode=args.stats_mode,
    )
    print(f"Total validation samples: {len(val_dataset)}")

    # 创建 DataLoader（分布式时使用 DistributedSampler 分片数据）
    sampler = None
    is_distributed = dist.is_initialized()
    if is_distributed:
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

    # Collate function with optional static padding (matches training config)
    use_static_padding = args.load_full_history
    static_max_len = args.max_history_length if use_static_padding else None
    collate = make_collate_fn(static_max_len=static_max_len)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler else True,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # 运行验证
    all_metrics = {}
    start_time = time.time()

    if args.mode in ("forward_loss", "both"):
        forward_metrics = validate_forward_loss(
            policy, preprocessor, val_loader, device,
            action_dim=action_dim,
            gripper_dim_indices=gripper_dim_indices,
            compute_per_dim=compute_per_dim,
        )
        all_metrics.update(forward_metrics)

    if args.mode in ("inference", "both"):
        inference_metrics = validate_inference(
            policy, preprocessor, val_loader, device,
            max_samples=args.num_inference_samples,
            action_dim=action_dim,
            gripper_dim_indices=gripper_dim_indices,
            gripper_threshold=args.gripper_threshold,
            compute_per_dim=compute_per_dim,
            norm_mode=args.norm_mode,
            dataset_stats=dataset_metadata.stats,
            num_act_exec=args.num_act_exec,
            action_key=action_key,
        )
        all_metrics.update(inference_metrics)

    elapsed = time.time() - start_time

    # 输出结果
    def _print_results():
        print("=" * 60)
        print("LoLA V07 Validation Results")
        print("=" * 60)
        print(f"Dataset: {args.val_dataset_repo_id or args.val_dataset_root}")
        print(f"Checkpoint: {args.checkpoint_path or 'N/A'}")
        print(f"Mode: {args.mode}")
        print(f"Validation samples: {len(val_dataset)}")
        print("-" * 60)
        for name, value in all_metrics.items():
            print(f"  {name}: {value:.6f}")
        print(f"  Elapsed time: {elapsed:.1f}s")
        print("=" * 60)

    if dist.is_initialized():
        for rank in range(dist.get_world_size()):
            if dist.get_rank() == rank:
                _print_results()
            dist.barrier()
    else:
        _print_results()

    # 保存结果（仅主进程保存）
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    if args.output_file and is_main:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 构建 checkpoint 信息
        ckpt_info = {"type": ckpt_type, "path": args.checkpoint_path or "N/A"}
        if ckpt_metadata:
            ckpt_info.update(ckpt_metadata)

        results = {
            "dataset": args.val_dataset_repo_id or args.val_dataset_root,
            "training_config": args.training_config,
            "checkpoint_info": ckpt_info,
            "mode": args.mode,
            "num_samples": len(val_dataset),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "history_type": args.history_type,
            "use_special_tokens": args.use_special_tokens,
            "gripper_dim_indices": gripper_dim_indices,
            "gripper_threshold": args.gripper_threshold,
            "norm_mode": args.norm_mode,
            "stats_mode": args.stats_mode,
            "metrics": {k: float(v) for k, v in all_metrics.items()},
            "elapsed_s": elapsed,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
