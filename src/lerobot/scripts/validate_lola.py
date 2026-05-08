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
from torch.utils.data import DataLoader

import pytorch_lightning as pl

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# 从训练脚本复用
from lerobot.scripts.train_lola_multigpu import (
    LoLALightningModule,
    create_lola_dataset,
    collate_fn,
    get_fsdp_strategy,
    get_deepspeed_strategy,
)


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


def validate_forward_loss(policy, preprocessor, val_loader, device):
    """在验证集上计算 forward loss（v-loss, action_loss, total_loss）"""
    # forward loss 需要模型在 train 模式（flow matching 需要随机采样噪声和时间步）
    policy.model.train()
    # 但冻结的 VLM 保持 eval
    if not policy.config.train_vlm and hasattr(policy, 'vlm'):
        policy.vlm.eval()

    total_loss = 0.0
    total_v_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0

    print("Running forward loss validation...")
    for batch_idx, batch in enumerate(val_loader):
        # 移动到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 提取特殊字段
        special_data = extract_special_fields(batch)

        # 应用预处理器
        batch = preprocessor(batch)

        # 恢复特殊字段
        batch.update(special_data)

        with torch.no_grad():
            loss, loss_dict = policy(batch)

        total_loss += loss_dict["loss"]
        total_v_loss += loss_dict["v_loss"]
        total_action_loss += loss_dict["action_loss"]
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Forward loss: {batch_idx + 1}/{len(val_loader)} batches done")

    policy.model.eval()

    return {
        "val_total_loss": total_loss / num_batches,
        "val_v_loss": total_v_loss / num_batches,
        "val_action_loss": total_action_loss / num_batches,
    }


def validate_inference(policy, preprocessor, val_loader, device, max_samples=100):
    """运行推理去噪管线，对比预测动作与真实动作"""
    policy.model.eval()

    all_mse = []
    all_l1 = []
    sample_count = 0

    print("Running inference validation...")
    for batch_idx, batch in enumerate(val_loader):
        if sample_count >= max_samples:
            break

        # 保存 ground truth action
        special_data = extract_special_fields(batch)
        ground_truth_actions = special_data["action"]  # [B, T, action_dim]

        # 移动到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 应用预处理器
        batch = preprocessor(batch)

        # 恢复历史 action 字段（推理需要）
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

        mse = F.mse_loss(pred_matched, gt_matched, reduction="mean")
        l1 = F.l1_loss(pred_matched, gt_matched, reduction="mean")

        all_mse.append(mse.item())
        all_l1.append(l1.item())
        sample_count += predicted_actions.shape[0]

        if (batch_idx + 1) % 10 == 0:
            print(f"  Inference: {sample_count}/{max_samples} samples done")

    if not all_mse:
        return {}

    return {
        "val_action_mse": sum(all_mse) / len(all_mse),
        "val_action_l1": sum(all_l1) / len(all_l1),
    }


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

    # 加载数据集元数据
    print(f"Loading validation dataset metadata from {args.val_dataset_repo_id or args.val_dataset_root}...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.val_dataset_repo_id,
        root=args.val_dataset_root,
    )

    # 获取 features
    features = dataset_to_policy_features(dataset_metadata.features)
    action_dim = features["action"].shape[0] if "action" in features else args.action_dim

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
        gradient_checkpointting=gradient_checkpointting,
        vlm_extract_layers=tuple(args.vlm_extract_layers),
        max_image_pixels=args.max_image_pixels,
        min_image_pixels=args.min_image_pixels,
    )

    # 归一化模式
    if args.norm_mode == "robovlm":
        from lerobot.configs.types import NormalizationMode
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }

    # 加载模型
    print("Loading LoLA model...")
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        # Lightning checkpoint
        print(f"Loading from Lightning checkpoint: {args.checkpoint_path}")
        lightning_model = LoLALightningModule.load_from_checkpoint(
            args.checkpoint_path,
            config=config,
            dataset_stats=dataset_metadata.stats,
        )
        policy = lightning_model.policy
        preprocessor = lightning_model.preprocessor
        postprocessor = lightning_model.postprocessor
    else:
        # 从头创建策略（用于手动加载权重）
        print("Creating LoLA Policy from scratch (no checkpoint loaded)...")
        policy = LoLAPolicy(config)
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            dataset_stats=dataset_metadata.stats,
        )

    # 将模型移动到设备
    policy._device = device
    policy.model = policy.model.to(device)
    policy.vlm = policy.vlm.to(device)
    policy.model.eval()

    print(f"Model loaded on {device}")

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
    )
    print(f"Total validation samples: {len(val_dataset)}")

    # 创建 DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 运行验证
    all_metrics = {}
    start_time = time.time()

    if args.mode in ("forward_loss", "both"):
        forward_metrics = validate_forward_loss(policy, preprocessor, val_loader, device)
        all_metrics.update(forward_metrics)

    if args.mode in ("inference", "both"):
        inference_metrics = validate_inference(
            policy, preprocessor, val_loader, device,
            max_samples=args.num_inference_samples,
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

    # 保存结果
    if args.output_file:
        results = {
            "dataset": args.val_dataset_repo_id or args.val_dataset_root,
            "checkpoint": args.checkpoint_path,
            "mode": args.mode,
            "num_samples": len(val_dataset),
            "metrics": {k: float(v) for k, v in all_metrics.items()},
            "elapsed_s": elapsed,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
