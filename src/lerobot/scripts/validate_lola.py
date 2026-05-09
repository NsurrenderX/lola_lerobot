#!/usr/bin/env python
"""
LoLA 模型验证脚本 - 在验证集上评估模型质量

支持两种验证模式:
    forward_loss: 计算 v-loss 和 action_loss（与训练相同的前向传播）
    inference:    运行实际推理去噪管线，对比预测动作与真实动作（MSE, L1）
    both:         同时运行两种模式

使用方法:
    # 验证 Lightning checkpoint（两种模式）
    python src/lerobot/scripts/validate_lola.py \
        --checkpoint_path /path/to/lola.ckpt \
        --val_dataset_repo_id <val_dataset> \
        --mode both

    # 仅前向 loss 验证
    python src/lerobot/scripts/validate_lola.py \
        --checkpoint_path /path/to/lola.ckpt \
        --val_dataset_repo_id <val_dataset> \
        --mode forward_loss

    # 多 GPU
    torchrun --nproc_per_node=4 src/lerobot/scripts/validate_lola.py \
        --checkpoint_path /path/to/lola.ckpt \
        --val_dataset_repo_id <val_dataset> \
        --strategy fsdp
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# 从训练脚本复用
from lerobot.scripts.train_lola_multigpu import (
    create_lola_dataset,
    collate_fn,
)


def unnormalize_lola_actions(actions, dataset_stats, action_dim, norm_mode):
    """将动作从模型输出空间反归一化回原始空间。

    default (MEAN_STD) 模式: actions * std + mean
    robovlm 模式: 数据集已用 normalize_action 归一化到 [-1,1]（夹爪保持原值），
                  需用 unnoramalize_action 反归一化，但夹爪仍保持原值。
    """
    if norm_mode == "robovlm":
        # 使用 robovlm_dataset 的反归一化（线性映射 [-1,1] → [norm_min, norm_max]，夹爪保持原值）
        from lerobot.datasets.robovlm_dataset import unnoramalize_action
        # 需要获取 norm_min/norm_max，从 dataset_stats 中没有直接存储，
        # 所以从命令行参数传入不现实；robovlm 模式下夹爪值本身就是原始的 {-1,1}/{0,1}
        # 不需要反归一化就能做分类，所以这里返回原值
        return actions
    else:
        # MEAN_STD 模式: 反归一化 = action * std + mean
        action_mean = dataset_stats["action"]["mean"][:action_dim].to(actions.device)
        action_std = dataset_stats["action"]["std"][:action_dim].to(actions.device)
        return actions * action_std + action_mean


def extract_special_fields(batch):
    """提取特殊字段，避免被preprocessor处理（与训练脚本逻辑一致）"""
    special_data = {}
    keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]
    for key in keys_to_extract:
        if key in batch:
            special_data[key] = batch.pop(key)
    if "action" in batch:
        special_data["action"] = batch.pop("action")
    return special_data


def validate_forward_loss(policy, preprocessor, val_loader, device,
                          action_dim=None, gripper_dim_indices=None,
                          compute_per_dim=False):
    """在验证集上计算 forward loss（v-loss, arm_loss, gripper_loss, total_loss）

    支持每维度的 arm_loss 分解。
    """
    gripper_dim_indices = gripper_dim_indices or []
    continuous_dim_indices = [i for i in range(action_dim) if i not in gripper_dim_indices]
    need_per_dim = compute_per_dim or len(gripper_dim_indices) > 0

    # forward loss 需要模型在 train 模式（flow matching 需要随机采样噪声和时间步）
    policy.model.train()
    # 但冻结的 VLM 保持 eval
    if not policy.config.train_vlm and hasattr(policy, 'vlm'):
        policy.vlm.eval()

    total_loss = 0.0
    total_v_loss = 0.0
    total_arm_loss = 0.0
    total_gripper_loss = 0.0
    num_batches = 0

    per_dim_arm_loss_sum = torch.zeros(action_dim, dtype=torch.float64) if need_per_dim else None

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
        per_dim_size = action_dim if need_per_dim else 0
        total_size = header_size + per_dim_size

        stats_tensor = torch.zeros(total_size, dtype=torch.float64, device=device)
        stats_tensor[0] = total_loss
        stats_tensor[1] = total_v_loss
        stats_tensor[2] = total_arm_loss
        stats_tensor[3] = total_gripper_loss
        stats_tensor[4] = num_batches

        if need_per_dim:
            stats_tensor[header_size:header_size + action_dim] = per_dim_arm_loss_sum.to(device)

        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        total_loss = stats_tensor[0].item()
        total_v_loss = stats_tensor[1].item()
        total_arm_loss = stats_tensor[2].item()
        total_gripper_loss = stats_tensor[3].item()
        num_batches = stats_tensor[4].item()

        if need_per_dim:
            per_dim_arm_loss_sum = stats_tensor[header_size:header_size + action_dim].cpu()

    if num_batches == 0:
        return {}

    results = {
        "val_total_loss": total_loss / num_batches,
        "val_v_loss": total_v_loss / num_batches,
        "val_arm_loss": total_arm_loss / num_batches,
        "val_gripper_loss": total_gripper_loss / num_batches,
    }

    # 每维度 arm_loss
    if need_per_dim:
        per_dim_arm_loss_avg = per_dim_arm_loss_sum / num_batches
        for i in range(action_dim):
            results[f"val_arm_loss_dim_{i}"] = per_dim_arm_loss_avg[i].item()

        if continuous_dim_indices:
            results["val_continuous_arm_loss"] = per_dim_arm_loss_avg[continuous_dim_indices].mean().item()

    return results


def validate_inference(policy, preprocessor, val_loader, device, max_samples=100,
                       action_dim=None, gripper_dim_indices=None, gripper_threshold=0.0,
                       compute_per_dim=False, norm_mode="default", dataset_stats=None):
    """运行推理去噪管线，对比预测动作与真实动作。

    支持每维度 MSE/L1 指标和夹爪分类指标（accuracy, precision, recall, F1）。
    """
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

        # 恢复历史 action 字段（推理需要），并移动到设备
        for key in ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]:
            if key in special_data:
                batch[key] = special_data[key].to(device)

        # 推理
        with torch.no_grad():
            predicted_actions = policy.predict_action_chunk(batch)  # [B, pred_len, action_dim]

        # 对齐长度
        if ground_truth_actions.ndim == 2:
            ground_truth_actions = ground_truth_actions.unsqueeze(1)

        pred_len = predicted_actions.shape[1]
        gt_len = ground_truth_actions.shape[1]
        min_len = min(pred_len, gt_len)

        pred_matched = predicted_actions[:, :min_len, :]
        gt_matched = ground_truth_actions[:, :min_len, :].to(device)

        # Unnormalize predictions for MSE/L1 comparison against original-space ground truth.
        # - robovlm (IDENTITY): predictions already in original space, no unnormalization needed.
        # - default (MEAN_STD): arm dims need unnormalization; gripper dims are already
        #   discretized to {-1, 1} (original space) via sigmoid thresholding in sample_actions.
        if norm_mode == "default" and dataset_stats is not None:
            pred_for_metric = unnormalize_lola_actions(
                pred_matched, dataset_stats, action_dim, norm_mode)
            # Overwrite gripper dims with original predictions (already in original space)
            # since unnormalize_lola_actions applies MEAN_STD to ALL dims including gripper.
            for g_dim in gripper_dim_indices:
                pred_for_metric[:, :, g_dim] = pred_matched[:, :, g_dim]
        else:
            pred_for_metric = pred_matched

        # 总体 MSE/L1（向后兼容）
        mse = F.mse_loss(pred_for_metric, gt_matched, reduction="mean")
        l1 = F.l1_loss(pred_for_metric, gt_matched, reduction="mean")
        mse_sum += mse.item()
        l1_sum += l1.item()
        n_batches += 1
        sample_count += predicted_actions.shape[0]

        # 每维度 MSE/L1
        if need_per_dim:
            per_dim_mse = F.mse_loss(pred_for_metric, gt_matched, reduction="none").mean(dim=(0, 1))
            per_dim_l1 = F.l1_loss(pred_for_metric, gt_matched, reduction="none").mean(dim=(0, 1))
            per_dim_mse_sum += per_dim_mse.double().cpu()
            per_dim_l1_sum += per_dim_l1.double().cpu()

        # 夹爪分类指标
        if num_gripper_dims > 0:
            # Dual-Token/Dual-Expert: gripper dims are already discretized to {-1, 1}
            # via sigmoid thresholding in sample_actions, so they are already in original
            # space regardless of norm_mode. No unnormalization needed for gripper dims.
            pred_for_gripper = pred_matched

            for g_idx, g_dim in enumerate(gripper_dim_indices):
                pred_gripper = pred_for_gripper[:, :, g_dim]  # [B, min_len]
                gt_gripper = gt_matched[:, :, g_dim]          # [B, min_len]

                pred_binary = (pred_gripper > gripper_threshold).reshape(-1).float()
                gt_binary = (gt_gripper > gripper_threshold).reshape(-1).float()

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


def main():
    parser = argparse.ArgumentParser(description="LoLA Model Validation")

    # 验证数据集参数
    parser.add_argument("--val_dataset_repo_id", type=str, default=None,
                        help="Validation dataset repo ID")
    parser.add_argument("--val_dataset_root", type=str, default=None,
                        help="Local root for validation dataset")
    parser.add_argument("--val_episodes", type=int, nargs="*", default=None,
                        help="Specific validation episodes to load (optional)")

    # 模型参数
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--vlm_path", type=str, default="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/",
                        help="Path to local Qwen3.5-4B model")
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--train_vlm", action="store_true", default=False)
    parser.add_argument("--load_full_history", action="store_true")
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--vlm_extract_layers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--max_image_pixels", type=int, default=230400)
    parser.add_argument("--min_image_pixels", type=int, default=65536)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--gradient_checkpointting", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointting", action="store_true")

    # 验证模式
    parser.add_argument("--mode", type=str, default="both",
                        choices=["forward_loss", "inference", "both"],
                        help="Validation mode")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                        help="Max number of samples for inference validation")

    # DataLoader 参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tolerance_frames", type=int, default=2,
                        help="Video frame decode tolerance (frames)")

    # 分布式参数
    parser.add_argument("--strategy", type=str, default="auto",
                        choices=["fsdp", "deepspeed", "ddp", "auto"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])

    # 归一化参数
    parser.add_argument("--norm_mode", type=str, default="default",
                        choices=["default", "robovlm"])
    parser.add_argument("--norm_min", type=float, default=-0.65)
    parser.add_argument("--norm_max", type=float, default=0.65)

    # 每维度指标与夹爪分类参数
    parser.add_argument("--gripper_dims", type=str, default=None,
                        help="Comma-separated indices of gripper dims. Supports negative indices "
                             "(e.g., '-1' for last dim, '-1,-11' for dual-arm)")
    parser.add_argument("--gripper_threshold", type=float, default=0.0,
                        help="Threshold for discretizing gripper predictions. "
                             "0.0 for {-1,1} range, 0.5 for {0,1} range")
    parser.add_argument("--gripper_loss_weight", type=float, default=1.0,
                        help="BCE loss weight for gripper dimension")
    parser.add_argument("--per_dim_metrics", action="store_true", default=False,
                        help="Compute per-dimension MSE/L1 for all action dims")

    # 输出
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save validation results to JSON file")

    args = parser.parse_args()

    if args.val_dataset_repo_id is None and args.val_dataset_root is None:
        raise ValueError("Either --val_dataset_repo_id or --val_dataset_root must be provided.")

    # 确定设备
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 初始化分布式（如果使用 torchrun）
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        import torch.distributed as dist
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
    action_dim = features["action"].shape[0] if "action" in features else args.action_dim

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

    print(f"Validation Dataset Info:")
    print(f"  - Total episodes: {dataset_metadata.total_episodes}")
    print(f"  - Total frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")
    print(f"  - Action dim: {action_dim}")

    # 创建 LoLA 配置
    gradient_checkpointting = not args.no_gradient_checkpointting
    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        vlm_path=args.vlm_path,
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
        vlm_extract_layers=tuple(args.vlm_extract_layers),
        max_image_pixels=args.max_image_pixels,
        min_image_pixels=args.min_image_pixels,
        gripper_loss_weight=args.gripper_loss_weight,
        gripper_dim_indices=tuple(int(x.strip()) for x in args.gripper_dims.split(",")) if args.gripper_dims else (),
    )
    # draccus.ChoiceRegistry 不接受 gradient_checkpointting 作为构造参数
    config.gradient_checkpointting = gradient_checkpointting

    # 设置 config.device 为当前 rank 对应的 GPU
    # 这会影响 preprocessor 中的 DeviceProcessorStep 以及模型加载
    config.device = f"cuda:{local_rank}"

    # 归一化模式
    if args.norm_mode == "robovlm":
        from lerobot.configs.types import NormalizationMode
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }

    # 加载模型
    print(f"[Rank {local_rank}] Loading LoLA model...")

    # 先创建 policy 和 preprocessor（模型在 CPU 上）
    policy = LoLAPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=dataset_metadata.stats,
    )

    # 加载 checkpoint 权重到 CPU，避免 torch.load 默认把 tensor 加载到 cuda:0
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        print(f"[Rank {local_rank}] Loading checkpoint: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")

        # 提取 state_dict
        if "model_state_dict" in ckpt:
            ckpt_state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            ckpt_state_dict = ckpt["state_dict"]
        else:
            ckpt_state_dict = ckpt

        # DDP 训练保存的 checkpoint 键名格式：
        #   DiT:  model.vlm_bridge.xxx  → 需要去掉 "model." 前缀匹配 policy.model 的 state_dict
        #   VLM:  vlm.language_model.xxx → 需要去掉 "vlm." 前缀匹配 policy.vlm 的 state_dict
        #   也可能直接是 LoLAPolicy 的 state_dict（带 policy.model./policy.vlm. 前缀）

        # 分组：按前缀分离 DiT 和 VLM 权重
        dit_sd = {}  # 给 policy.model 用
        vlm_sd_raw = {}  # 给 policy.vlm 用

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
                # 无前缀的键，可能是 DiT 的
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
    else:
        print(f"[Rank {local_rank}] No checkpoint provided, using randomly initialized model")

    # 将模型移动到当前 rank 对应的 GPU（关键：从 CPU 移到 cuda:{local_rank}）
    policy._device = device
    policy.model = policy.model.to(device)
    policy.vlm = policy.vlm.to(device)
    policy.model.eval()
    if not policy.config.train_vlm:
        policy.vlm.eval()

    # 验证设备
    dit_device = next(policy.model.parameters()).device
    vlm_device = next(policy.vlm.parameters()).device
    print(f"[Rank {local_rank}] DiT device: {dit_device}, VLM device: {vlm_device}")

    # 创建验证数据集
    print("Creating validation dataset...")
    norm_action = (args.norm_mode == "robovlm")
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
        tolerance_frames=args.tolerance_frames,
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
            shuffle=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler else True,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
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
        )
        all_metrics.update(inference_metrics)

    elapsed = time.time() - start_time

    # 输出结果
    print("=" * 60)
    print("LoLA Validation Results")
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

    # 保存结果（仅主进程保存）
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    if args.output_file and is_main:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        results = {
            "dataset": args.val_dataset_repo_id or args.val_dataset_root,
            "checkpoint": args.checkpoint_path,
            "mode": args.mode,
            "num_samples": len(val_dataset),
            "action_dim": action_dim,
            "gripper_dim_indices": gripper_dim_indices,
            "gripper_threshold": args.gripper_threshold,
            "norm_mode": args.norm_mode,
            "metrics": {k: float(v) for k, v in all_metrics.items()},
            "elapsed_s": elapsed,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
