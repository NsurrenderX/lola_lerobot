#!/usr/bin/env python
"""
RoboVLM 模型验证脚本 - 在验证集上评估模型质量

支持两种验证模式:
    forward_loss: 计算 arm loss, gripper loss, gripper accuracy（与训练相同的前向传播）
    inference:    运行实际推理管线，对比预测动作与真实动作（MSE, L1）
    both:         同时运行两种模式

使用方法:
    # 验证 checkpoint（两种模式）
    python src/lerobot/scripts/validate_robovlm.py \
        --checkpoint_path /path/to/robovlm.pt \
        --val_dataset_repo_id <val_dataset> \
        --mode both

    # 仅前向 loss 验证
    python src/lerobot/scripts/validate_robovlm.py \
        --checkpoint_path /path/to/robovlm.pt \
        --val_dataset_repo_id <val_dataset> \
        --mode forward_loss
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.robovlm_dataset import RoboVLMDataset, unnoramalize_action
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.robovlm import RoboVLMConfig, RoboVLMPolicy


def validate_forward_loss(policy, val_loader, device):
    """在验证集上计算 forward loss（arm loss, gripper loss, gripper accuracy）"""
    policy.model.eval()

    total_loss = 0.0
    total_loss_arm = 0.0
    total_loss_gripper = 0.0
    total_acc_gripper = 0.0
    num_batches = 0

    print("Running forward loss validation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = policy(batch)

            total_loss += loss.item()
            total_loss_arm += loss_dict["loss_arm"]
            total_loss_gripper += loss_dict["loss_gripper"]
            total_acc_gripper += loss_dict["acc_gripper"]
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Forward loss: {batch_idx + 1}/{len(val_loader)} batches done")

    if num_batches == 0:
        return {}

    return {
        "val_loss": total_loss / num_batches,
        "val_loss_arm": total_loss_arm / num_batches,
        "val_loss_gripper": total_loss_gripper / num_batches,
        "val_acc_gripper": total_acc_gripper / num_batches,
    }


def validate_inference(policy, val_loader, device, max_samples=100):
    """运行推理管线，对比预测动作与真实动作"""
    policy.model.eval()

    all_mse = []
    all_l1 = []
    sample_count = 0

    print("Running inference validation...")
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
            mse = F.mse_loss(pred_actions[:, :min_len], gt_actions[:, :min_len].to(pred_actions.device))
            l1 = F.l1_loss(pred_actions[:, :min_len], gt_actions[:, :min_len].to(pred_actions.device))

            all_mse.append(mse.item())
            all_l1.append(l1.item())
            sample_count += pred_actions.shape[0]

            if (batch_idx + 1) % 10 == 0:
                print(f"  Inference: {sample_count}/{max_samples} samples done")

    if not all_mse:
        return {}

    return {
        "val_action_mse": sum(all_mse) / len(all_mse),
        "val_action_l1": sum(all_l1) / len(all_l1),
    }


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

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 加载模型
    print("Loading RoboVLM model...")
    policy = RoboVLMPolicy(config)
    policy.model = policy.model.to(device)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        print(f"Loading checkpoint: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model_sd = ckpt.get("model_state_dict", ckpt)

        # 处理 DDP/FSDP 的 module. 前缀
        current_sd = policy.model.state_dict()
        loaded_keys = []
        for key in current_sd:
            if key in model_sd:
                current_sd[key] = model_sd[key]
                loaded_keys.append(key)
        policy.model.load_state_dict(current_sd)
        print(f"Loaded {len(loaded_keys)} keys from checkpoint")

    policy.model.eval()
    print(f"Model loaded on {device}")

    # 获取 tokenizer
    tokenizer = policy.model.tokenizer

    # 创建验证数据集
    print(f"Creating validation dataset from {args.val_dataset_repo_id or args.val_dataset_root}...")
    val_dataset = RoboVLMDataset(
        repo_id=args.val_dataset_repo_id,
        config=config,
        root=args.val_dataset_root,
        tokenizer=tokenizer,
    )
    print(f"Total validation samples: {len(val_dataset)}")

    # 更新 config features
    dataset_metadata = val_dataset.meta
    features = dataset_to_policy_features(dataset_metadata.features)
    if not config.output_features:
        config.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    if not config.input_features:
        config.input_features = {k: ft for k, ft in features.items() if k not in config.output_features}

    # 创建 DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collater,
    )

    # 运行验证
    all_metrics = {}
    start_time = time.time()

    if args.mode in ("forward_loss", "both"):
        forward_metrics = validate_forward_loss(policy, val_loader, device)
        all_metrics.update(forward_metrics)

    if args.mode in ("inference", "both"):
        inference_metrics = validate_inference(
            policy, val_loader, device,
            max_samples=args.num_inference_samples,
        )
        all_metrics.update(inference_metrics)

    elapsed = time.time() - start_time

    # 输出结果
    print("=" * 60)
    print("RoboVLM Validation Results")
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
