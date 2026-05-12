#!/usr/bin/env python
"""
RoboVLM 模型验证脚本 - 在验证集上评估模型质量

支持两种验证模式:
    forward_loss: 计算 arm loss, gripper loss, gripper accuracy（与训练相同的前向传播）
                  可选: 每维度 arm loss 分解
    inference:    运行实际推理管线，对比预测动作与真实动作
                  支持: 每维度 MSE/L1, 连续维度聚合, 夹爪分类指标(accuracy/precision/recall/F1)
    both:         同时运行两种模式

支持 DDP 多卡并行加速（使用 torchrun 启动）。

使用方法:
    # 单卡验证 checkpoint（两种模式）
    python src/lerobot/scripts/validate_robovlm.py \
        --checkpoint_path /path/to/robovlm.pt \
        --val_dataset_repo_id <val_dataset> \
        --mode both

    # 多卡 DDP 加速验证（2 卡）
    torchrun --nproc_per_node=2 src/lerobot/scripts/validate_robovlm.py \
        --checkpoint_path /path/to/robovlm.pt \
        --val_dataset_repo_id <val_dataset> \
        --mode both

    # 仅前向 loss 验证
    python src/lerobot/scripts/validate_robovlm.py \
        --checkpoint_path /path/to/robovlm.pt \
        --val_dataset_repo_id <val_dataset> \
        --mode forward_loss

    # 每维度指标 + 夹爪分类（默认夹爪为最后一个维度）
    python src/lerobot/scripts/validate_robovlm.py \
        --checkpoint_path /path/to/robovlm.pt \
        --val_dataset_repo_id <val_dataset> \
        --mode inference \
        --gripper_dims -1 --per_dim_metrics
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.robovlm_dataset import RoboVLMDataset, unnoramalize_action
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.robovlm import RoboVLMConfig, RoboVLMPolicy


def validate_forward_loss(policy, val_loader, device, action_dim=None,
                          gripper_dim_indices=None, compute_per_dim=False,
                          original_fwd=10, tolerance_frames=0):
    """在验证集上计算 forward loss（arm loss, gripper loss, gripper accuracy）

    可选: 每维度 arm loss 分解（需模型支持 compute_per_dim 参数）。
    支持多卡 DDP: 使用 all_reduce 聚合所有 rank 的结果。
    """
    policy.model.eval()
    tol = tolerance_frames
    fwd = original_fwd

    gripper_dim_indices = gripper_dim_indices or []
    continuous_dim_indices = [i for i in range(action_dim) if i not in gripper_dim_indices]
    need_per_dim = compute_per_dim or len(gripper_dim_indices) > 0

    total_loss = 0.0
    total_loss_arm = 0.0
    total_loss_gripper = 0.0
    total_acc_gripper = 0.0
    num_batches = 0

    per_dim_arm_loss_sum = torch.zeros(len(continuous_dim_indices), dtype=torch.float64) if need_per_dim else None

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"[Rank {local_rank}] Running forward loss validation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 数据集 action_chunck 形状为 [B, ws, fwd+2*tol, 7],
            # 每个 chunk 包含 [tol past, fwd center, tol future].
            # policy forward 只需要中心部分 [:, :, tol:fwd+tol, :]
            if tol > 0 and batch["action_chunck"].shape[2] > fwd:
                batch = dict(batch)  # shallow copy
                batch["action_chunck"] = batch["action_chunck"][:, :, tol:tol + fwd, :]
                batch["chunck_mask"] = batch["chunck_mask"][:, :, tol:tol + fwd]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = policy(batch, compute_per_dim=need_per_dim)

            total_loss += loss.item()
            total_loss_arm += loss_dict["loss_arm"]
            total_loss_gripper += loss_dict["loss_gripper"]
            total_acc_gripper += loss_dict["acc_gripper"]
            num_batches += 1

            if need_per_dim and "arm_loss_per_dim" in loss_dict:
                per_dim_arm_loss_sum += loss_dict["arm_loss_per_dim"].double().cpu()

            if (batch_idx + 1) % 10 == 0:
                print(f"[Rank {local_rank}] Forward loss: {batch_idx + 1}/{len(val_loader)} batches done")

    if num_batches == 0:
        return {}

    # 多卡同步：构建单一 flat tensor，一次 all_reduce
    is_distributed = dist.is_initialized()
    if is_distributed:
        header_size = 5  # total_loss, total_loss_arm, total_loss_gripper, total_acc_gripper, num_batches
        per_dim_size = len(continuous_dim_indices) if need_per_dim else 0
        total_size = header_size + per_dim_size

        stats_tensor = torch.zeros(total_size, dtype=torch.float64, device=device)
        stats_tensor[0] = total_loss
        stats_tensor[1] = total_loss_arm
        stats_tensor[2] = total_loss_gripper
        stats_tensor[3] = total_acc_gripper
        stats_tensor[4] = num_batches

        if need_per_dim:
            stats_tensor[header_size:header_size + per_dim_size] = per_dim_arm_loss_sum.to(device)

        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        total_loss = stats_tensor[0].item()
        total_loss_arm = stats_tensor[1].item()
        total_loss_gripper = stats_tensor[2].item()
        total_acc_gripper = stats_tensor[3].item()
        num_batches = stats_tensor[4].item()

        if need_per_dim:
            per_dim_arm_loss_sum = stats_tensor[header_size:header_size + per_dim_size].cpu()

    results = {
        "val_loss": total_loss / num_batches,
        "val_loss_arm": total_loss_arm / num_batches,
        "val_loss_gripper": total_loss_gripper / num_batches,
        "val_acc_gripper": total_acc_gripper / num_batches,
    }

    if need_per_dim and per_dim_arm_loss_sum is not None:
        per_dim_arm_loss_avg = per_dim_arm_loss_sum / num_batches
        for idx, dim_i in enumerate(continuous_dim_indices):
            results[f"val_arm_loss_dim_{dim_i}"] = per_dim_arm_loss_avg[idx].item()

    return results


def _compute_gripper_confusion_strict(pred_gripper, gt_gripper, gripper_threshold):
    """严格模式：逐帧对比，每个时刻的 pred 和 gt 必须完全匹配。"""
    pred_binary = (pred_gripper > gripper_threshold).reshape(-1).float()
    gt_binary = (gt_gripper > gripper_threshold).reshape(-1).float()
    tp = (pred_binary * gt_binary).sum().double().cpu()
    fp = ((1 - gt_binary) * pred_binary).sum().double().cpu()
    fn = (gt_binary * (1 - pred_binary)).sum().double().cpu()
    tn = ((1 - gt_binary) * (1 - pred_binary)).sum().double().cpu()
    return tp, fp, fn, tn


def _compute_gripper_confusion_lenient(pred_gripper, gt_extended_gripper, gripper_threshold,
                                         tolerance_frames, gt_mask=None):
    """宽松模式：对于 pred 与 gt 不匹配的帧，在扩展窗口 ±tolerance_frames 范围内搜索有效 gt 标签。

    pred_gripper:   [B, fwd_pred_next_n]              — 预测夹爪值（中心窗口）
    gt_extended_gripper: [B, fwd + 2*tol]             — 扩展 gt 窗口
                     中心部分 (indices [tol:tol+fwd]) 与 pred 对齐
    tolerance_frames: 搜索半径（帧数）
    gt_mask:         [B, fwd + 2*tol]                  — 有效性掩码 (True=有效帧, None 表示全部有效)

    在稳定区域（远离翻转点），±tol 内只有一种 gt 状态 → 不宽恕。
    在翻转边界附近，±tol 内两种状态共存 → 0.2s 级时序偏差被宽恕。
    """
    B, fwd = pred_gripper.shape
    offset = tolerance_frames  # pred 在 gt_extended 中的起始偏移

    pred_binary = (pred_gripper > gripper_threshold)             # [B, fwd] bool
    gt_binary_ext = (gt_extended_gripper > gripper_threshold)    # [B, ext_len] bool

    # 中心窗口的 gt（与 pred 对齐）
    gt_center = gt_binary_ext[:, offset:offset + fwd]            # [B, fwd] bool

    # 严格模式初步判定
    match = pred_binary == gt_center                              # [B, fwd] bool

    # 对不匹配的帧进行容忍搜索
    if tolerance_frames > 0 and not match.all():
        # 构建有效帧的标签掩码：只在有效帧中搜索
        if gt_mask is not None:
            valid_mask = gt_mask.bool()                           # [B, ext_len]
            valid_positive = (gt_binary_ext & valid_mask).float().unsqueeze(1)
            valid_negative = (~gt_binary_ext & valid_mask).float().unsqueeze(1)
        else:
            valid_positive = gt_binary_ext.float().unsqueeze(1)
            valid_negative = (~gt_binary_ext).float().unsqueeze(1)

        kernel_size = 2 * tolerance_frames + 1
        # Replication padding + max_pool1d 做膨胀
        # 填充后 ext_len+2*tol，输出 (ext_len+2*tol)-(2*tol+1)+1 = ext_len
        padded_pos = F.pad(valid_positive, (tolerance_frames, tolerance_frames), mode='replicate')
        padded_neg = F.pad(valid_negative, (tolerance_frames, tolerance_frames), mode='replicate')
        dilated_pos = F.max_pool1d(padded_pos, kernel_size, stride=1).squeeze(1)  # [B, ext_len]
        dilated_neg = F.max_pool1d(padded_neg, kernel_size, stride=1).squeeze(1)  # [B, ext_len]

        # 提取中心窗口的膨胀结果（对应 pred 各帧）
        dilated_pos_center = dilated_pos[:, offset:offset + fwd]   # [B, fwd]
        dilated_neg_center = dilated_neg[:, offset:offset + fwd]   # [B, fwd]

        # pred 预测的标签在 ±tol 内有有效 gt 验证 → 不算 fp/fn
        lenient_match_pos = pred_binary & (dilated_pos_center > 0)
        lenient_match_neg = (~pred_binary) & (dilated_neg_center > 0)
        lenient_match = lenient_match_pos | lenient_match_neg

        match = match | lenient_match

    # 根据最终匹配结果计算混淆矩阵（基于中心窗口）
    pred_flat = pred_binary.reshape(-1).float()
    gt_flat = gt_center.reshape(-1).float()
    match_flat = match.reshape(-1)

    tp = (pred_flat * gt_flat * match_flat.float()).sum().double().cpu()
    tn = ((1 - pred_flat) * (1 - gt_flat) * match_flat.float()).sum().double().cpu()
    fp = ((pred_flat * (1 - gt_flat)) * (~match_flat).float()).sum().double().cpu()
    fn = (((1 - pred_flat) * gt_flat) * (~match_flat).float()).sum().double().cpu()

    return tp, fp, fn, tn


def validate_inference(policy, val_loader, device, max_samples=100,
                       action_dim=None, gripper_dim_indices=None,
                       gripper_threshold=0.0, compute_per_dim=False,
                       tolerance_frames=0, original_fwd=10, window_size=8):
    """运行推理管线，对比预测动作与真实动作。

    支持每维度 MSE/L1 和夹爪分类指标（accuracy, precision, recall, F1）。
    夹爪分类同时返回严格模式和宽松模式（±tolerance_frames 容忍窗口）的结果。

    数据集 action_chunck 形状为 [B, ws, fwd+2*tol, 7], 每个 chunk 包含:
      - [0, tol):        过去容差帧
      - [tol, tol+fwd):  中心帧（与 pred 对齐）
      - [tol+fwd, fwd+2*tol): 未来容差帧

    支持多卡 DDP: 使用 all_reduce 聚合所有 rank 的结果。
    """
    policy.model.eval()
    fwd = original_fwd
    tol = tolerance_frames

    gripper_dim_indices = gripper_dim_indices or []
    num_gripper_dims = len(gripper_dim_indices)
    continuous_dim_indices = [i for i in range(action_dim) if i not in gripper_dim_indices]
    need_per_dim = compute_per_dim or num_gripper_dims > 0

    mse_sum = 0.0
    l1_sum = 0.0
    n_batches = 0
    sample_count = 0

    per_dim_mse_sum = torch.zeros(action_dim, dtype=torch.float64) if need_per_dim else None
    per_dim_l1_sum = torch.zeros(action_dim, dtype=torch.float64) if need_per_dim else None

    # 严格模式混淆矩阵
    strict_tp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    strict_fp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    strict_fn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    strict_tn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    # 宽松模式混淆矩阵
    lenient_tp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    lenient_fp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    lenient_fn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    lenient_tn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_total = 0

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"[Rank {local_rank}] Running inference validation (tolerance_frames={tol}, "
          f"original_fwd={fwd})...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if sample_count >= max_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_actions = policy.predict_action_chunk(batch)  # [B, fwd, 7]

            # ---- 从最后 chunk 直接提取扩展 gt 窗口 ----
            # action_chunck[:, -1, :, :] 已包含 [tol past, fwd center, tol future]
            gt_extended = batch["action_chunck"][:, -1, :, :].clone()   # [B, fwd+2*tol, 7]
            gt_ext_mask = batch["chunck_mask"][:, -1, :]                 # [B, fwd+2*tol]

            # Unnormalize + binarize gripper on extended window
            gt_extended = unnoramalize_action(gt_extended, policy.config.norm_min, policy.config.norm_max)
            gt_extended[..., -1] = (gt_extended[..., -1] > 0).float() * 2 - 1

            # MSE/L1 比较仅用中心部分 [tol : tol+fwd]
            gt_center = gt_extended[:, tol:tol + fwd, :]  # [B, fwd, 7]
            min_len = min(pred_actions.shape[1], fwd)
            pred_matched = pred_actions[:, :min_len, :]
            gt_matched = gt_center[:, :min_len, :].to(pred_matched.device)

            mse = F.mse_loss(pred_matched, gt_matched, reduction="mean")
            l1 = F.l1_loss(pred_matched, gt_matched, reduction="mean")
            mse_sum += mse.item()
            l1_sum += l1.item()
            n_batches += 1
            sample_count += pred_actions.shape[0]

            # 每维度 MSE/L1
            if need_per_dim:
                per_dim_mse = F.mse_loss(pred_matched, gt_matched, reduction="none").mean(dim=(0, 1))
                per_dim_l1 = F.l1_loss(pred_matched, gt_matched, reduction="none").mean(dim=(0, 1))
                per_dim_mse_sum += per_dim_mse.double().cpu()
                per_dim_l1_sum += per_dim_l1.double().cpu()

            # 夹爪分类指标
            if num_gripper_dims > 0:
                for g_idx, g_dim in enumerate(gripper_dim_indices):
                    pred_gripper = pred_matched[:, :, g_dim]          # [B, min_len]
                    gt_center_gripper = gt_matched[:, :, g_dim]       # [B, min_len]
                    gt_ext_gripper = gt_extended[:, :, g_dim]         # [B, fwd+2*tol]

                    # 严格模式：中心窗口逐帧对比
                    tp, fp, fn, tn = _compute_gripper_confusion_strict(
                        pred_gripper, gt_center_gripper, gripper_threshold)
                    strict_tp[g_idx] += tp
                    strict_fp[g_idx] += fp
                    strict_fn[g_idx] += fn
                    strict_tn[g_idx] += tn

                    # 宽松模式：扩展窗口 ±tol 搜索
                    tp, fp, fn, tn = _compute_gripper_confusion_lenient(
                        pred_gripper, gt_ext_gripper, gripper_threshold,
                        tolerance_frames=tol, gt_mask=gt_ext_mask)
                    lenient_tp[g_idx] += tp
                    lenient_fp[g_idx] += fp
                    lenient_fn[g_idx] += fn
                    lenient_tn[g_idx] += tn

                gripper_total += pred_matched.shape[0] * pred_matched.shape[1]

            if (batch_idx + 1) % 10 == 0:
                print(f"[Rank {local_rank}] Inference: {sample_count}/{max_samples} samples done")

    # 多卡同步：构建单一 flat tensor，一次 all_reduce
    is_distributed = dist.is_initialized()
    if is_distributed:
        header_size = 4  # mse_sum, l1_sum, n_batches, sample_count
        per_dim_size = action_dim * 2 if need_per_dim else 0
        # strict + lenient 各 4 组 (tp/fp/fn/tn) + gripper_total
        gripper_size = num_gripper_dims * 8 + 1 if num_gripper_dims > 0 else 0
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
            # strict
            stats_tensor[offset:offset + num_gripper_dims] = strict_tp.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = strict_fp.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = strict_fn.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = strict_tn.to(device)
            offset += num_gripper_dims
            # lenient
            stats_tensor[offset:offset + num_gripper_dims] = lenient_tp.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = lenient_fp.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = lenient_fn.to(device)
            offset += num_gripper_dims
            stats_tensor[offset:offset + num_gripper_dims] = lenient_tn.to(device)
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
            strict_tp = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            strict_fp = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            strict_fn = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            strict_tn = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            lenient_tp = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            lenient_fp = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            lenient_fn = stats_tensor[offset:offset + num_gripper_dims].cpu()
            offset += num_gripper_dims
            lenient_tn = stats_tensor[offset:offset + num_gripper_dims].cpu()
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

        if continuous_dim_indices:
            continuous_mse = per_dim_mse_avg[continuous_dim_indices].mean().item()
            continuous_l1 = per_dim_l1_avg[continuous_dim_indices].mean().item()
            results["val_continuous_mse"] = continuous_mse
            results["val_continuous_l1"] = continuous_l1

    # 夹爪分类指标
    if num_gripper_dims > 0 and gripper_total > 0:
        def _gripper_metrics(tp_v, fp_v, fn_v, tn_v):
            total = tp_v + fp_v + fn_v + tn_v
            accuracy = (tp_v + tn_v) / total if total > 0 else 0.0
            precision = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
            recall = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return accuracy, precision, recall, f1

        for g_idx, g_dim in enumerate(gripper_dim_indices):
            # 严格模式
            acc, prec, rec, f1 = _gripper_metrics(
                strict_tp[g_idx].item(), strict_fp[g_idx].item(),
                strict_fn[g_idx].item(), strict_tn[g_idx].item())
            results[f"val_gripper_dim_{g_dim}_strict_accuracy"] = acc
            results[f"val_gripper_dim_{g_dim}_strict_precision"] = prec
            results[f"val_gripper_dim_{g_dim}_strict_recall"] = rec
            results[f"val_gripper_dim_{g_dim}_strict_f1"] = f1

            # 宽松模式
            acc, prec, rec, f1 = _gripper_metrics(
                lenient_tp[g_idx].item(), lenient_fp[g_idx].item(),
                lenient_fn[g_idx].item(), lenient_tn[g_idx].item())
            results[f"val_gripper_dim_{g_dim}_lenient_accuracy"] = acc
            results[f"val_gripper_dim_{g_dim}_lenient_precision"] = prec
            results[f"val_gripper_dim_{g_dim}_lenient_recall"] = rec
            results[f"val_gripper_dim_{g_dim}_lenient_f1"] = f1

    return results


def main():
    parser = argparse.ArgumentParser(description="RoboVLM Model Validation")

    # 验证数据集参数
    parser.add_argument("--val_dataset_repo_id", type=str, default=None,
                        help="Validation dataset repo ID")
    parser.add_argument("--val_dataset_root", type=str, default=None,
                        help="Local root for validation dataset")

    # 模型参数
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--vlm_pretrained_path", type=str, default="/data_16T/deepseek/kosmos-2-patch14-224/",
                        help="Path to local Kosmos-2 model")
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--no_train_vision", action="store_true", default=False)
    parser.add_argument("--no_train_text_embedding", action="store_true", default=False)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--fwd_pred_next_n", type=int, default=10)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--use_hand_rgb", action="store_true", default=True)
    parser.add_argument("--no_use_hand_rgb", action="store_true", default=False)
    parser.add_argument("--skip_action_normalize", action="store_true", default=True)
    parser.add_argument("--no_skip_action_normalize", action="store_true", default=False)
    parser.add_argument("--arm_gripper_loss_ratio", type=float, default=1.0)

    # 验证模式
    parser.add_argument("--mode", type=str, default="both",
                        choices=["forward_loss", "inference", "both"],
                        help="Validation mode")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                        help="Max number of samples for inference validation")

    # DataLoader 参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # 每维度指标与夹爪分类参数
    parser.add_argument("--gripper_dims", type=str, default=None,
                        help="Comma-separated indices of gripper dims. Supports negative indices "
                             "(e.g., '-1' for last dim). Default for RoboVLM: last dim is gripper")
    parser.add_argument("--gripper_threshold", type=float, default=0.0,
                        help="Threshold for discretizing gripper predictions. "
                             "0.0 for {-1,1} range, 0.5 for {0,1} range")
    parser.add_argument("--gripper_tolerance_s", type=float, default=0.1,
                        help="Tolerance window in seconds for lenient gripper matching. "
                             "If pred and gt disagree at time t, but gt has the matching "
                             "label within ±tolerance_s, it is not counted as an error. "
                             "Set to 0 to disable lenient mode.")
    parser.add_argument("--per_dim_metrics", action="store_true", default=False,
                        help="Compute per-dimension MSE/L1 for all action dims")

    # 输出
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save validation results to JSON file")

    args = parser.parse_args()

    if args.val_dataset_repo_id is None and args.val_dataset_root is None:
        raise ValueError("Either --val_dataset_repo_id or --val_dataset_root must be provided.")

    # set random seed
    seed = 42
    torch.manual_seed(seed)
    
    # 解析布尔参数
    use_state = args.use_state
    use_hand_rgb = args.use_hand_rgb and not args.no_use_hand_rgb
    train_vision = not args.no_train_vision
    train_text_embedding = not args.no_train_text_embedding
    skip_action_normalize = not args.no_skip_action_normalize if args.no_skip_action_normalize else args.skip_action_normalize

    # 确定设备和 local rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 初始化分布式（如果使用 torchrun）
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    # 创建 RoboVLM 配置
    config = RoboVLMConfig(
        vlm_pretrained_path=args.vlm_pretrained_path,
        freeze_backbone=args.freeze_backbone,
        train_vision=train_vision,
        train_text_embedding=train_text_embedding,
        use_state=use_state,
        use_hand_rgb=use_hand_rgb,
        skip_action_normalize=skip_action_normalize,
        window_size=args.window_size,
        fwd_pred_next_n=args.fwd_pred_next_n,
        arm_gripper_loss_ratio=args.arm_gripper_loss_ratio,
    )

    # 加载模型（先在 CPU 上创建，再移到对应 GPU）
    print(f"[Rank {local_rank}] Loading RoboVLM model...")
    policy = RoboVLMPolicy(config)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        print(f"[Rank {local_rank}] Loading checkpoint: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")

        # 提取 state_dict
        if "model_state_dict" in ckpt:
            model_sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            model_sd = ckpt["state_dict"]
        else:
            model_sd = ckpt

        # 处理 DDP 的 module. 前缀
        current_sd = policy.model.state_dict()
        loaded_keys = []
        for key in current_sd:
            if key in model_sd:
                current_sd[key] = model_sd[key]
                loaded_keys.append(key)
            elif key.startswith("module.") and key[len("module."):] in model_sd:
                current_sd[key] = model_sd[key[len("module."):]]
                loaded_keys.append(key)
        policy.model.load_state_dict(current_sd)
        print(f"[Rank {local_rank}] Loaded {len(loaded_keys)} keys from checkpoint")

    # 将模型移动到当前 rank 对应的 GPU
    policy.model = policy.model.to(device)
    policy.model.eval()
    print(f"[Rank {local_rank}] Model loaded on {device}")

    # 获取 tokenizer
    tokenizer = policy.model.tokenizer

    # 先加载 metadata 获取 fps, 用于计算 tolerance_frames
    pre_meta = LeRobotDatasetMetadata(args.val_dataset_repo_id, root=args.val_dataset_root)
    fps = pre_meta.fps
    tolerance_frames = max(0, round(args.gripper_tolerance_s * fps))
    print(f"[Rank {local_rank}] fps={fps}, gripper_tolerance_s={args.gripper_tolerance_s}, "
          f"tolerance_frames={tolerance_frames}")

    # 设置 dataset_tolerance_frames 以加载额外 past/future 容差帧
    # 模型的 fwd_pred_next_n 不变，仅数据集加载范围扩展
    config.dataset_tolerance_frames = tolerance_frames

    # 创建验证数据集
    print(f"[Rank {local_rank}] Creating validation dataset from {args.val_dataset_repo_id or args.val_dataset_root}...")
    val_dataset = RoboVLMDataset(
        repo_id=args.val_dataset_repo_id,
        config=config,
        root=args.val_dataset_root,
        tokenizer=tokenizer,
    )
    print(f"[Rank {local_rank}] Total validation samples: {len(val_dataset)}")

    # 更新 config features
    dataset_metadata = val_dataset.meta
    features = dataset_to_policy_features(dataset_metadata.features)
    if not config.output_features:
        config.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    if not config.input_features:
        config.input_features = {k: ft for k, ft in features.items() if k not in config.output_features}

    # 解析夹爪维度索引
    action_dim = config.action_dim
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
        print(f"[Rank {local_rank}] Gripper dim indices: {gripper_dim_indices}")

    compute_per_dim = args.per_dim_metrics or len(gripper_dim_indices) > 0

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collater,
    )

    # 运行验证
    all_metrics = {}
    start_time = time.time()

    if args.mode in ("forward_loss", "both"):
        forward_metrics = validate_forward_loss(
            policy, val_loader, device,
            action_dim=action_dim,
            gripper_dim_indices=gripper_dim_indices,
            compute_per_dim=compute_per_dim,
            original_fwd=config.fwd_pred_next_n,
            tolerance_frames=tolerance_frames,
        )
        all_metrics.update(forward_metrics)

    if args.mode in ("inference", "both"):
        inference_metrics = validate_inference(
            policy, val_loader, device,
            max_samples=args.num_inference_samples,
            action_dim=action_dim,
            gripper_dim_indices=gripper_dim_indices,
            gripper_threshold=args.gripper_threshold,
            compute_per_dim=compute_per_dim,
            tolerance_frames=tolerance_frames,
            original_fwd=config.fwd_pred_next_n,
            window_size=config.window_size,
        )
        all_metrics.update(inference_metrics)

    elapsed = time.time() - start_time

    # 输出结果（仅 rank 0 打印，避免多卡信息混杂）
    is_main = not is_distributed or dist.get_rank() == 0
    if is_main:
        lines = []
        lines.append("=" * 60)
        lines.append("RoboVLM Validation Results")
        lines.append("=" * 60)
        lines.append(f"Dataset: {args.val_dataset_repo_id or args.val_dataset_root}")
        lines.append(f"Checkpoint: {args.checkpoint_path or 'N/A'}")
        lines.append(f"Mode: {args.mode}")
        lines.append(f"Validation samples: {len(val_dataset)}")
        lines.append("-" * 60)
        for name, value in all_metrics.items():
            lines.append(f"  {name}: {value:.6f}")
        lines.append(f"  Elapsed time: {elapsed:.1f}s")
        lines.append("=" * 60)
        print("\n".join(lines))

    # 保存结果（仅主进程保存）
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
            "gripper_tolerance_s": args.gripper_tolerance_s,
            "metrics": {k: float(v) for k, v in all_metrics.items()},
            "elapsed_s": elapsed,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()