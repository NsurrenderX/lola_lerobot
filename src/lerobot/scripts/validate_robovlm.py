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
from lerobot.datasets.robovlm_dataset import RoboVLMDataset, unnoramalize_action
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.robovlm import RoboVLMConfig, RoboVLMPolicy


def validate_forward_loss(policy, val_loader, device, action_dim=None,
                          gripper_dim_indices=None, compute_per_dim=False):
    """在验证集上计算 forward loss（arm loss, gripper loss, gripper accuracy）

    可选: 每维度 arm loss 分解（需模型支持 compute_per_dim 参数）。
    支持多卡 DDP: 使用 all_reduce 聚合所有 rank 的结果。
    """
    policy.model.eval()

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


def validate_inference(policy, val_loader, device, max_samples=100,
                       action_dim=None, gripper_dim_indices=None,
                       gripper_threshold=0.0, compute_per_dim=False):
    """运行推理管线，对比预测动作与真实动作。

    支持每维度 MSE/L1 和夹爪分类指标（accuracy, precision, recall, F1）。
    支持多卡 DDP: 使用 all_reduce 聚合所有 rank 的结果。
    """
    policy.model.eval()

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

    gripper_tp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_fp = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_fn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_tn = torch.zeros(num_gripper_dims, dtype=torch.float64) if num_gripper_dims > 0 else None
    gripper_total = 0

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"[Rank {local_rank}] Running inference validation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if sample_count >= max_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Ground truth: last timestep's action chunk
            gt_actions = batch["action_chunck"][:, -1, :, :].clone()  # [B, fwd, 7]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_actions = policy.predict_action_chunk(batch)  # [B, fwd, 7]

            # Unnormalize ground truth to match prediction space
            gt_actions = unnoramalize_action(gt_actions, policy.config.norm_min, policy.config.norm_max)
            # Binarize gripper dim in gt to match prediction
            gt_actions[..., -1] = (gt_actions[..., -1] > 0).float() * 2 - 1

            # Match lengths
            min_len = min(pred_actions.shape[1], gt_actions.shape[1])
            pred_matched = pred_actions[:, :min_len, :]
            gt_matched = gt_actions[:, :min_len, :].to(pred_actions.device)

            # 总体 MSE/L1（向后兼容）
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
                    pred_gripper = pred_matched[:, :, g_dim]  # [B, min_len]
                    gt_gripper = gt_matched[:, :, g_dim]       # [B, min_len]

                    # pred 和 gt 的夹爪值都已经在 {-1, 1} 空间，threshold=0 将其分为两类
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
                print(f"[Rank {local_rank}] Inference: {sample_count}/{max_samples} samples done")

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
    parser.add_argument("--per_dim_metrics", action="store_true", default=False,
                        help="Compute per-dimension MSE/L1 for all action dims")

    # 输出
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save validation results to JSON file")

    args = parser.parse_args()

    if args.val_dataset_repo_id is None and args.val_dataset_root is None:
        raise ValueError("Either --val_dataset_repo_id or --val_dataset_root must be provided.")

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
            shuffle=False,
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
        )
        all_metrics.update(inference_metrics)

    elapsed = time.time() - start_time

    # 输出结果（所有 rank 都打印，方便观察）
    print("=" * 60)
    print(f"[Rank {local_rank}] RoboVLM Validation Results")
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
    is_main = not is_distributed or dist.get_rank() == 0
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
            "metrics": {k: float(v) for k, v in all_metrics.items()},
            "elapsed_s": elapsed,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()