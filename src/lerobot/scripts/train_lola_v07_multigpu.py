#!/usr/bin/env python
"""
LoLA 多卡分布式训练脚本 - 使用 LeRobotDataset

本脚本加载实际的 LoLA 模型（包含 Qwen3.5-4B VLM），使用 LeRobotDataset 进行训练。

使用方法:
    # 使用 LeRobotDataset 训练 (需要指定数据集)
    torchrun --nproc_per_node=4 src/lerobot/scripts/train_lola_multigpu.py \
        --dataset_repo_id lerobot/pusht \
        --strategy fsdp

    # DeepSpeed 训练 (4卡)
    torchrun --nproc_per_node=4 src/lerobot/scripts/train_lola_multigpu.py \
        --dataset_repo_id lerobot/pusht \
        --strategy deepspeed

    # 单卡测试
    python src/lerobot/scripts/train_lola_multigpu.py --devices 1 --dataset_repo_id lerobot/pusht
    
    # 启用完整历史action加载
    python src/lerobot/scripts/train_lola_multigpu.py \
        --load_full_history \
        --max_history_length 100 \
        --history_padding_side left
"""

import argparse
import datetime
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 关键：让 PyTorch 延迟初始化 CUDA，避免在 import 时占用 GPU
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.lola_dataset import LoLADataset  # 新增：支持完整历史action的数据集
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola_v07 import LoLAV07Config, LoLAV07Policy
from lerobot.policies.factory import make_pre_post_processors


# ----------------------------------------------------------------------
# LoLA Lightning Module - 包装实际模型
# ----------------------------------------------------------------------
class LoLAV07LightningModule(pl.LightningModule):
    """LoLA 的 PyTorch Lightning 模块 - 包装实际的 LoLAV07Policy"""
    
    def __init__(
        self,
        config: LoLAV07Config,
        dataset_stats: dict | None = None,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        warmup_ratio: float = 0.03,
        train_vlm: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "dataset_stats"])
        
        self.config = config
        self.dataset_stats = dataset_stats
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.train_vlm = train_vlm
        
        # 延迟加载模型到 setup() 阶段
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None

        # 计时
        self._step_start_time = None
        
    def setup(self, stage=None):
        """在 distributed 环境初始化后加载模型"""
        if self.policy is not None:
            return  # 已经加载
            
        # 获取当前进程的 local_rank
        if hasattr(self.trainer.strategy, 'local_rank') and self.trainer.strategy.local_rank is not None:
            local_rank = self.trainer.strategy.local_rank
        else:
            # 从环境变量获取
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
        # 设置当前设备 - 关键：确保每个进程使用正确的 GPU
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        print(f"[Rank {self.global_rank}] Loading LoLA Policy on {device}...")
        print(f"[Rank {self.global_rank}] VLM Path: {self.config.vlm_path}")
        
        # 加载 LoLA Policy
        self.policy = LoLAV07Policy(self.config)
        
        # 将模型移动到正确的设备
        self.policy._device = device
        self.policy.model = self.policy.model.to(device)
        self.policy.vlm = self.policy.vlm.to(device)
        
        # 创建预处理器和后处理器
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            dataset_stats=self.dataset_stats,
        )
        
        print(f"[Rank {self.global_rank}] LoLA Policy loaded on {device}!")
        print(f"[Rank {self.global_rank}] VLM device: {next(self.policy.vlm.parameters()).device}")
        print(f"[Rank {self.global_rank}] DiT device: {next(self.policy.model.parameters()).device}")
        
        # 如果不训练 VLM，冻结 VLM 参数
        if not self.train_vlm and hasattr(self.policy, 'vlm'):
            print(f"[Rank {self.global_rank}] Freezing VLM parameters...")
            for param in self.policy.vlm.parameters():
                param.requires_grad = False
            self.policy.vlm.eval()
            
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"[Rank {self.global_rank}] Trainable params: {trainable_params:,} / {total_params:,}")

        # DeepSpeed 模式下：切换梯度检查点为 DeepSpeed 实现
        is_deepspeed = hasattr(self.trainer, 'strategy') and isinstance(self.trainer.strategy, DeepSpeedStrategy)
        if is_deepspeed and self.config.gradient_checkpointing:
            print(f"[Rank {self.global_rank}] Configuring DeepSpeed activation checkpointing...")
            self.policy.model.set_deepspeed_checkpointing()
            if self.train_vlm:
                import deepspeed
                ds_fn = deepspeed.checkpointing.non_reentrant_checkpoint
                vlm = self.policy.vlm
                if hasattr(vlm, '_gradient_checkpointing_func'):
                    vlm._gradient_checkpointing_func = ds_fn
                for module in vlm.modules():
                    if hasattr(module, '_gradient_checkpointing_func'):
                        module._gradient_checkpointing_func = ds_fn
                if hasattr(self.policy, '_vlm_forward_mode'):
                    self.policy._vlm_forward_mode = "output_hidden_states"
        
    def forward(self, batch, time=None):
        """前向传播"""
        return self.policy(batch, time=time)
    
    def _extract_special_fields(self, batch):
        """
        提取特殊字段，避免被preprocessor处理。

        包括：
        - hist_actions_full, hist_actions_mask, hist_actions_length: LoLADataset的历史action字段
        - hist_states_full, hist_states_mask, hist_states_length: LoLADataset的历史state字段
        - action: 目标action字段（shape可能不匹配stats中的定义）

        这些字段不在数据集stats中定义，或者shape可能与stats不匹配。
        """
        special_data = {}
        # LoLADataset 的历史action和state字段
        keys_to_extract = [
            "hist_actions_full", "hist_actions_mask", "hist_actions_length",
            "hist_states_full", "hist_states_mask", "hist_states_length",
        ]
        for key in keys_to_extract:
            if key in batch:
                special_data[key] = batch.pop(key)

        # 提取action字段（因为其shape可能与stats不匹配）
        # 当使用delta_timestamps加载多步action时，shape是(B, T, action_dim)
        # 但stats期望的是(B, action_dim)
        if "action" in batch:
            special_data["action"] = batch.pop("action")

        return special_data
    
    def _restore_special_fields(self, batch, special_data):
        """恢复特殊字段到batch"""
        batch.update(special_data)
        return batch
    
    def _restore_history_actions(self, batch, history_data):
        """恢复历史action相关字段到batch"""
        batch.update(history_data)
        return batch
    
    def training_step(self, batch, batch_idx):
        """训练步骤 - v07: warmup t-truncation and v-loss alarm"""
        self._step_start_time = time.monotonic()

        # 提取特殊字段（避免被preprocessor处理）
        special_data = self._extract_special_fields(batch)

        # 应用预处理器
        batch = self.preprocessor(batch)

        # 恢复特殊字段
        batch = self._restore_special_fields(batch, special_data)

        # v07: Warmup t-truncation
        warmup_steps = int(self.trainer.estimated_stepping_batches * self.config.warmup_pct)
        if self.global_step < warmup_steps:
            b = batch["action"].shape[0]
            t_raw = torch.distributions.Beta(
                self.config.time_sampling_beta_alpha,
                self.config.time_sampling_beta_beta,
            ).sample((b,)).to(batch["action"].device)
            time_param = t_raw * (self.config.warmup_t_trunc_high - self.config.warmup_t_trunc_low) + self.config.warmup_t_trunc_low
            loss, loss_dict = self(batch, time=time_param)
        else:
            loss, loss_dict = self(batch)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if k != "loss":
                self.log(f"train_{k}", v, prog_bar=False, sync_dist=True)

        # v07: v-loss alarm
        if loss_dict.get("v_loss", 0) > 1.0:
            print(f"[WARNING] v_loss = {loss_dict['v_loss']:.4f} > 1.0 at step {self.global_step}")

        # 记录步耗时
        if self._step_start_time is not None:
            update_s = round(time.monotonic() - self._step_start_time, 2)
            self.log("train_update_s", update_s, prog_bar=False, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤：在验证数据上计算 v-loss 和 arm_loss"""
        special_data = self._extract_special_fields(batch)
        batch = self.preprocessor(batch)
        batch = self._restore_special_fields(batch, special_data)

        loss, loss_dict = self(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if k != "loss":
                self.log(f"val_{k}", v, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """配置优化器 - v07: Separate parameter groups with encoder LR multiplier"""
        # v07: Separate parameter groups
        encoder_lr_mult = self.config.encoder_lr_mult
        base_lr = self.learning_rate

        param_groups = [
            {"params": list(self.policy.model.dit.parameters()), "lr": base_lr},
            {"params": list(self.policy.model.vlm_bridge.parameters()), "lr": base_lr},
            {"params": list(self.policy.model.action_encoder.parameters()), "lr": base_lr * encoder_lr_mult},
            {"params": list(self.policy.model.arm_dit_to_latent.parameters()), "lr": base_lr * encoder_lr_mult},
            {"params": list(self.policy.model.grip_dit_to_latent.parameters()), "lr": base_lr * encoder_lr_mult},
        ]
        if self.policy.model.state_encoder is not None:
            param_groups.append({"params": list(self.policy.model.state_encoder.parameters()), "lr": base_lr * encoder_lr_mult})

        # Filter out params that don't require grad
        for group in param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Cosine decay scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        warmup_ratio = min(self.hparams.warmup_ratio, 0.1)
        total_steps = self.max_steps if self.max_steps is not None else int(self.trainer.estimated_stepping_batches)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy='cos',
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# ----------------------------------------------------------------------
# 数据集工具函数
# ----------------------------------------------------------------------
def create_lola_dataset(
    repo_id: str,
    config: LoLAV07Config,
    root: str | None = None,
    episodes: list | None = None,
    image_transforms=None,
    video_backend: str | None = None,
    use_lola_dataset: bool = False,
    max_history_length: int = 100,
    history_padding_side: str = "left",
    norm_action: bool | str = False,
    norm_min: float = -0.65,
    norm_max: float = 0.65,
    tolerance_frames: int | None = None,
    gripper_dim_indices_abs: tuple[int, ...] | None = None,
    dataset_stats: dict | None = None,
    history_type: str = "action",
    state_dim: int | None = None,
    # V2: completed tasks + transition masking
    track_completed_tasks: bool = True,
    transition_mask_rate: float = 0.0,
    completed_tasks_use_ann: bool = True,
    completed_tasks_history_len: int = 5,
    max_transition_len: int = 64,
) -> LeRobotDataset | LoLADataset:
    """
    创建 LoLA 训练用的数据集。

    根据 LoLA 的配置设置 delta_timestamps：
    - observation.state: 用于历史动作输入
    - action: 用于预测目标动作
    - observation.images.*: 用于视觉输入（如果存在）

    Args:
        repo_id: 数据集仓库ID (如 "lerobot/pusht")
        config: LoLA 配置
        root: 本地数据集根目录
        episodes: 指定加载的 episode 列表
        image_transforms: 图像变换
        video_backend: 视频后端
        use_lola_dataset: 是否使用 LoLADataset（加载完整历史action）
        max_history_length: 历史action最大长度
        history_padding_side: padding方向
        norm_action: 归一化模式 (False/True/"minmax"/"zscore")
        norm_min: RoboVLM min-max 归一化下界
        norm_max: RoboVLM min-max 归一化上界
        gripper_dim_indices_abs: gripper维度的绝对索引（z-score模式需要）
        dataset_stats: 数据集统计信息（z-score模式需要）

    Returns:
        配置好的 LeRobotDataset 或 LoLADataset
    """
    # 获取数据集元数据以确定 fps
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = dataset_metadata.fps

    # 构建 delta_timestamps
    # 将帧索引转换为时间戳（秒）
    delta_timestamps = {}

    # 观测状态：使用 n_obs_steps 个历史帧
    delta_timestamps["observation.state"] = [
        i / fps for i in config.observation_delta_indices
    ]

    # 动作：预测 pred_chunk_size 步
    delta_timestamps["action"] = [
        i / fps for i in config.action_delta_indices
    ]

    # 图像/视频观测：与状态同步
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [
            i / fps for i in config.observation_delta_indices
    ]

    print(f"[Dataset] delta_timestamps: {delta_timestamps}")

    # 创建数据集
    if use_lola_dataset:
        # 使用 LoLADataset：支持加载完整历史action
        print(f"[Dataset] Using LoLADataset with max_history_length={max_history_length}, action_chunk_size={config.action_chunk_size}, padding_side={history_padding_side}")
        dataset = LoLADataset(
            repo_id=repo_id,
            max_history_length=max_history_length,
            action_chunk_size=config.action_chunk_size,
            history_padding_side=history_padding_side,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            tolerance_frame=tolerance_frames if tolerance_frames is not None else 2,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            norm_action=norm_action,
            norm_min=norm_min,
            norm_max=norm_max,
            gripper_dim_indices_abs=gripper_dim_indices_abs,
            history_type=history_type,
            state_dim=state_dim,
            # V2: completed tasks + transition masking
            track_completed_tasks=track_completed_tasks,
            transition_mask_rate=transition_mask_rate,
            completed_tasks_use_ann=completed_tasks_use_ann,
            completed_tasks_history_len=completed_tasks_history_len,
            hist_action_token_drop_rate=config.hist_action_token_drop_rate,
            max_transition_len=max_transition_len,
        )
    else:
        # 使用标准 LeRobotDataset
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_frames=tolerance_frames,
            video_backend=video_backend,
        )
        # Z-score mode with LeRobotDataset: wrap with per-item normalization
        if norm_action == "zscore":
            from lerobot.datasets.robovlm_dataset import normalize_action_zscore
            import numpy as np
            _mean = dataset_stats["action"]["mean"]
            _std = dataset_stats["action"]["std"]
            action_mean = torch.tensor(_mean, dtype=torch.float32) if isinstance(_mean, np.ndarray) else _mean.float()
            action_std = torch.tensor(_std, dtype=torch.float32) if isinstance(_std, np.ndarray) else _std.float()
            dataset = _ZScoreActionDataset(dataset, action_mean, action_std, gripper_dim_indices_abs)
            print(f"[Dataset] Wrapped LeRobotDataset with ZScoreActionDataset (gripper dims: {gripper_dim_indices_abs})")

    return dataset


class _ZScoreActionDataset:
    """Wrapper that applies z-score arm normalization + gripper binarization to LeRobotDataset items."""

    def __init__(self, dataset, action_mean: torch.Tensor, action_std: torch.Tensor,
                 gripper_dim_indices_abs: tuple[int, ...] | None = None):
        self.dataset = dataset
        self.action_mean = action_mean
        self.action_std = action_std
        self.gripper_dim_indices_abs = gripper_dim_indices_abs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from lerobot.datasets.robovlm_dataset import normalize_action_zscore
        item = self.dataset[idx]
        if "action" in item and isinstance(item["action"], torch.Tensor):
            item["action"] = normalize_action_zscore(
                item["action"], self.action_mean, self.action_std,
                self.gripper_dim_indices_abs,
            )
        return item


def make_collate_fn(static_max_len: int | None = None):
    """Create a collate function with optional static padding length.

    If static_max_len is provided, hist_actions_full and hist_actions_mask
    are always padded to this fixed length, producing constant-size tensors
    every step. This eliminates CUDA memory fragmentation and stabilizes
    DeepSpeed ZeRO-2 reduce-scatter timing.

    If static_max_len is None, falls back to dynamic per-batch padding.
    """
    variable_length_keys = {"hist_actions_full", "hist_actions_mask", "hist_states_full", "hist_states_mask"}

    def collate_fn(batch):
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]

            if key == "task":
                result[key] = values
            elif key in ("completed_tasks", "completed_tasks_ann"):
                result[key] = values  # list[list[str]], pass through as-is
            elif key.startswith("observation.images."):
                result[key] = values
            elif key in variable_length_keys and isinstance(values[0], torch.Tensor):
                max_len = static_max_len if static_max_len is not None else max(v.shape[0] for v in values)
                padded_values = []
                for v in values:
                    if v.shape[0] < max_len:
                        pad_len = max_len - v.shape[0]
                        if key in {"hist_actions_full", "hist_states_full"}:
                            padding = torch.zeros(pad_len, v.shape[1], dtype=v.dtype)
                        else:
                            padding = torch.zeros(pad_len, dtype=v.dtype)
                        v = torch.cat([padding, v], dim=0)  # left padding
                    elif v.shape[0] > max_len:
                        v = v[-max_len:]  # truncate from left (keep most recent)
                    padded_values.append(v)
                result[key] = torch.stack(padded_values)
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values

        return result

    return collate_fn


# ----------------------------------------------------------------------
# DeepSpeed 配置
# ----------------------------------------------------------------------
def get_deepspeed_config(
    learning_rate: float = 2.5e-5,
    weight_decay: float = 0.01,
    gradient_clip_val: float = 1.0,
    reduce_bucket_size: float = 5e7,
    allgather_bucket_size: float = 5e7,
):
    """Generate default DeepSpeed ZeRO-2 config for B200 GPUs (~183GB each)."""
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_bucket_size": allgather_bucket_size,
            "reduce_bucket_size": reduce_bucket_size,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": gradient_clip_val,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": weight_decay,
            },
        },
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
        },
    }


def compute_vlm_max_length(
    dataset_metadata,
    vlm_path: str,
    min_image_pixels: int = 65536,
    max_image_pixels: int = 230400,
) -> int:
    """Auto-compute vlm_max_length from dataset info for static VLM padding.

    Computes: visual_tokens + structural_tokens + max_task_text_tokens + 1 (empty)
    """
    import math
    from transformers import AutoProcessor

    merge_size = 2
    patch_size = 16
    factor = merge_size * patch_size

    visual_tokens_total = 0
    num_images = 0
    for cam_key in dataset_metadata.camera_keys:
        feat = dataset_metadata.features.get(cam_key, {})
        info = feat.get("info", {})
        h = info.get("video.height", feat.get("shape", (256, 256, 3))[0])
        w = info.get("video.width", feat.get("shape", (256, 256, 3))[1])

        h_bar = max(1, math.ceil(h / factor))
        w_bar = max(1, math.ceil(w / factor))

        if h_bar * w_bar * factor * factor > max_image_pixels:
            ratio = (h_bar * w_bar * factor * factor) / max_image_pixels
            h_bar = max(1, math.floor(h_bar / math.sqrt(ratio)))
            w_bar = max(1, math.floor(w_bar / math.sqrt(ratio)))
        elif h_bar * w_bar * factor * factor < min_image_pixels:
            ratio = min_image_pixels / (h_bar * w_bar * factor * factor)
            h_bar = max(1, math.ceil(h_bar * math.sqrt(ratio)))
            w_bar = max(1, math.ceil(w_bar * math.sqrt(ratio)))
            while h_bar * w_bar * factor * factor < min_image_pixels:
                if h_bar <= w_bar:
                    h_bar += 1
                else:
                    w_bar += 1

        tokens = h_bar * w_bar // (merge_size ** 2)
        visual_tokens_total += tokens
        num_images += 1

    structural_tokens = 10 + 2 * num_images

    max_task_tokens = 0
    if dataset_metadata.total_tasks > 0:
        import pandas as pd
        tasks_path = dataset_metadata.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            df = pd.read_parquet(tasks_path)
            processor = AutoProcessor.from_pretrained(vlm_path, local_files_only=True)
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            for task in df.index:
                token_ids = tokenizer.encode(str(task))
                max_task_tokens = max(max_task_tokens, len(token_ids))

    vlm_max_length = visual_tokens_total + structural_tokens + max_task_tokens + 1
    print(f"Auto-computed vlm_max_length={vlm_max_length} "
          f"(visual={visual_tokens_total}, structural={structural_tokens}, "
          f"max_text={max_task_tokens}, empty=1)")
    return vlm_max_length


# ----------------------------------------------------------------------
# FSDP 配置
# ----------------------------------------------------------------------
def get_fsdp_strategy(gradient_checkpointing=True):
    """获取 FSDP 策略配置"""
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5VisionBlock
    from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock, Flux2SingleTransformerBlock
    from lerobot.policies.lola.modeling_lola import LolaVLMFeatureExtractor, LoLADualExpertDoubleBlock, LoLADualExpertSingleBlock
    from lerobot.policies.lola_v07.modeling_lola_v07 import LolaV07ActionEncoder, LolaV07StateEncoder

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # 使用 transformer_auto_wrap_policy 正确分片 transformer 层
    auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
        module, recurse, nonwrapped_numel,
        transformer_layer_cls={
            Qwen3_5DecoderLayer,
            Qwen3_5VisionBlock,
            Flux2TransformerBlock,
            Flux2SingleTransformerBlock,
            LolaVLMFeatureExtractor,
            LoLADualExpertDoubleBlock,
            LoLADualExpertSingleBlock,
            LolaV07ActionEncoder,
            LolaV07StateEncoder,
        }
    )

    # v07: Activation checkpointing for DiT transformer blocks
    activation_checkpointing = None
    if gradient_checkpointing:
        activation_checkpointing = [LoLADualExpertDoubleBlock, LoLADualExpertSingleBlock]
        print("FSDP activation checkpointing enabled for DiT blocks")

    strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        cpu_offload=False,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=activation_checkpointing,
        use_orig_params=True,  # 兼容优化器
        state_dict_type=StateDictType.FULL_STATE_DICT,
    )
    return strategy


def get_deepspeed_strategy(learning_rate=2.5e-5, gradient_clip_val=1.0, deepspeed_config_path=None,
                           reduce_bucket_size=5e7, allgather_bucket_size=5e7):
    """获取 DeepSpeed ZeRO-2 策略配置，针对 B200 GPU 调优"""
    ds_config = get_deepspeed_config(
        learning_rate=learning_rate,
        gradient_clip_val=gradient_clip_val,
        reduce_bucket_size=reduce_bucket_size,
        allgather_bucket_size=allgather_bucket_size,
    )
    if deepspeed_config_path is not None:
        import json
        with open(deepspeed_config_path) as f:
            custom_config = json.load(f)
        ds_config.update(custom_config)

    # Lightning manages these keys — remove from DeepSpeed config to avoid MisconfigurationException
    for key in ["gradient_accumulation_steps", "train_batch_size", "train_micro_batch_size_per_gpu"]:
        ds_config.pop(key, None)

    strategy = DeepSpeedStrategy(config=ds_config)
    return strategy


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    os.environ['WANDB_API_KEY'] = "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"
    parser = argparse.ArgumentParser(description="LoLA V07 Multi-GPU Training with LeRobotDataset")
    
    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None,
                        help="HuggingFace dataset repo ID (e.g., lerobot/pusht)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Local dataset root directory (optional)")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Specific episodes to load (optional)")

    # 验证集参数
    parser.add_argument("--val_dataset_repo_id", type=str, default=None,
                        help="Validation dataset repo ID (separate from training)")
    parser.add_argument("--val_dataset_root", type=str, default=None,
                        help="Local root for validation dataset (optional)")
    parser.add_argument("--val_batch_size", type=int, default=None,
                        help="Validation batch size (defaults to batch_size)")
    parser.add_argument("--val_frequency", type=int, default=None,
                        help="Validate every N training steps")
    
    # 训练策略参数
    parser.add_argument("--strategy", type=str, default="fsdp", choices=["fsdp", "deepspeed", "ddp"])
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to custom DeepSpeed config JSON. Default: ZeRO-2 config tuned for B200.")
    parser.add_argument("--devices", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (mutually exclusive with --max_epochs)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max training epochs (mutually exclusive with --max_steps)")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=None,
                        help="Save checkpoint every N training steps (mutually exclusive with --save_every_n_epochs)")
    parser.add_argument("--save_every_n_epochs", type=int, default=None,
                        help="Save checkpoint every N epochs (mutually exclusive with --save_every_n_steps)")

    # 模型参数
    parser.add_argument("--vlm_path", type=str, default="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/",
                        help="Path to local Qwen3.5-4B model")
    parser.add_argument("--train_vlm", action="store_true", help="Whether to train VLM (default: False)")
    parser.add_argument("--ckpt_dir", type=str, default="/data_16T/deepseek/checkpoints/lola",
                        help="Path to save LoLA checkpoints.")
    
    # LoLA 特定参数
    parser.add_argument("--action_dim", type=int, default=14, help="Action dimension")
    parser.add_argument("--action_chunk_size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--pred_chunk_size", type=int, default=50, help="Prediction chunk size")
    parser.add_argument("--n_obs_steps", type=int, default=1, help="Number of observation steps")
    
    # 历史action加载参数
    parser.add_argument("--load_full_history", action="store_true",
                        help="Load full episode history actions (use LoLADataset)")
    parser.add_argument("--max_history_length", type=int, default=100,
                        help="Maximum history length for padding/truncation")
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"],
                        help="Padding side for history actions")
    parser.add_argument("--history_type", type=str, default="action", choices=["action", "state"],
                        help="History type: 'action' uses historical actions, 'state' uses historical observation states")
    parser.add_argument("--state_dim", type=int, default=None,
                        help="State dimension (auto-detected from dataset if not provided)")
    parser.add_argument("--state_encoder_mode", type=str, default="unified", choices=["unified", "separated"],
                        help="State encoder mode: 'unified' (single MLP → 2*hidden, split) or 'separated' (arm/grip separate MLPs)")

    # V2: Text template + completed tasks + transition masking
    parser.add_argument("--task_text_template_version", type=str, default="raw", choices=["raw", "v1_with_completed"],
                        help="Text template version: 'raw' = old behavior, 'v1_with_completed' = new template with completed tasks")
    parser.add_argument("--completed_tasks_use_ann", action="store_true", default=True,
                        help="Use descriptive 'ann' text for completed tasks (default: True)")
    parser.add_argument("--no_completed_tasks_use_ann", action="store_true",
                        help="Use concise 'task' label instead of 'ann' for completed tasks")
    parser.add_argument("--completed_tasks_history_len", type=int, default=5,
                        help="Only keep the most recent N completed tasks (default: 5)")
    parser.add_argument("--transition_mask_rate", type=float, default=0.0,
                        help="Mask rate for transition-dominant hist tokens (0=no mask, 1=full mask)")
    parser.add_argument("--max_transition_len", type=int, default=64,
                        help="Max history frames before annotation (must match conversion)")

    # LoLA 模型配置参数
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="启用梯度检查点（默认开启）")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="关闭梯度检查点")
    parser.add_argument("--compile_model", action="store_true",
                        help="启用 torch.compile 优化")
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                        help="torch.compile 模式")
    parser.add_argument("--vlm_lr", type=float, default=1e-6,
                        help="VLM 学习率（仅 train_vlm=True 时生效）")
    parser.add_argument("--vlm_extract_layers", type=int, nargs="+", default=[8, 16, 24],
                        help="VLM 提取层索引")
    parser.add_argument("--max_image_pixels", type=int, default=230400,
                        help="每张图片最大像素数（控制 visual token 数）")
    parser.add_argument("--min_image_pixels", type=int, default=65536,
                        help="每张图片最小像素数")
    parser.add_argument("--num_inference_steps", type=int, default=10,
                        help="Flow matching 推理去噪步数")
    parser.add_argument("--gripper_loss_weight", type=float, default=1.0,
                        help="BCE loss weight for gripper dimension")
    parser.add_argument("--action_loss_weight", type=float, default=1.0,
                        help="Huber loss weight for continuous arm dimensions")
    parser.add_argument("--gripper_dims", type=str, default="-1",
                        help="Comma-separated gripper dim indices (supports negative)")
    parser.add_argument("--hist_action_token_drop_rate", type=float, default=0.0,
                        help="Probability of dropping each valid history action token during training (0.0 = no dropout)")

    # V07: Bottleneck dimensions
    parser.add_argument("--action_bottleneck_dim", type=int, default=256,
                        help="Arm latent dimension for flow matching (default: 256)")
    parser.add_argument("--grip_bottleneck_dim", type=int, default=128,
                        help="Grip latent dimension for flow matching (default: 128)")
    parser.add_argument("--state_bottleneck_dim", type=int, default=256,
                        help="StateEncoder unified mode arm bottleneck dimension (default: 256)")
    parser.add_argument("--state_grip_bottleneck_dim", type=int, default=128,
                        help="StateEncoder unified mode grip bottleneck dimension (default: 128)")
    parser.add_argument("--encoder_lr_mult", type=float, default=1.5,
                        help="Encoder LR multiplier relative to base LR (default: 1.5)")
    parser.add_argument("--warmup_pct", type=float, default=0.1,
                        help="Warm-up fraction of total steps (default: 0.1)")

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 数量")
    parser.add_argument("--tolerance_frames", type=int, default=None,
                        help="Video frame decode tolerance (frames), overrides default tolerance_s")

    # 归一化参数
    parser.add_argument("--norm_mode", type=str, default="default",
                        choices=["default", "robovlm", "zscore"],
                        help="归一化模式: default(LoLA默认MEAN_STD), robovlm(min-max→[-1,1],全IDENTITY), zscore(arm=z-score,gripper=二值化{0,1})")
    parser.add_argument("--norm_min", type=float, default=-0.65,
                        help="RoboVLM 归一化下界")
    parser.add_argument("--norm_max", type=float, default=0.65,
                        help="RoboVLM 归一化上界")

    # DeepSpeed 参数
    parser.add_argument("--deepspeed_reduce_bucket_size", type=float, default=5e7,
                        help="DeepSpeed ZeRO-2 reduce bucket size (default: 5e7 for B200 NVLink)")
    parser.add_argument("--deepspeed_allgather_bucket_size", type=float, default=5e7,
                        help="DeepSpeed ZeRO-2 allgather bucket size (default: 5e7 for B200 NVLink)")

    # Static padding parameters
    parser.add_argument("--static_collate_padding", action="store_true", default=True,
                        help="Use static max_history_length padding in collate (default: enabled)")
    parser.add_argument("--no_static_collate_padding", action="store_true",
                        help="Disable static padding, use dynamic per-batch padding")
    parser.add_argument("--static_vlm_padding", action="store_true",
                        help="Pad VLM tokens to fixed max_length for consistent tensor shapes")
    parser.add_argument("--vlm_max_length", type=int, default=None,
                        help="Override tokenizer max_length for static VLM padding; auto-compute if None")

    args = parser.parse_args()

    # 检查数据集是否有参数
    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    # 检查训练终止条件参数
    if args.max_steps is None and args.max_epochs is None:
        raise ValueError("Either --max_steps or --max_epochs must be provided.")
    if args.max_steps is not None and args.max_epochs is not None:
        raise ValueError("--max_steps and --max_epochs are mutually exclusive. Please specify only one.")

    # 检查保存间隔参数
    if args.save_every_n_steps is not None and args.save_every_n_epochs is not None:
        raise ValueError("--save_every_n_steps and --save_every_n_epochs are mutually exclusive. Please specify only one.")

    # 检查 DeepSpeed 可用性
    if args.strategy == "deepspeed" and not HAS_DEEPSPEED:
        raise ImportError("DeepSpeed required for strategy='deepspeed'. Install: pip install deepspeed")

    # 设置策略
    if args.strategy == "fsdp":
        strategy = get_fsdp_strategy(gradient_checkpointing=not args.no_gradient_checkpointing)
    elif args.strategy == "deepspeed":
        strategy = get_deepspeed_strategy(
            learning_rate=args.learning_rate,
            gradient_clip_val=1.0,
            deepspeed_config_path=args.deepspeed_config,
            reduce_bucket_size=args.deepspeed_reduce_bucket_size,
            allgather_bucket_size=args.deepspeed_allgather_bucket_size,
        )
    else:
        strategy = "auto"
    
    # 获取数据集元数据
    print(f"Loading dataset metadata from {args.dataset_repo_id}...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )
    
    # 获取数据集的 features 并转换为 policy features
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 获取 action_dim 从数据集
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    # 获取 state_dim 从数据集
    if "observation.state" in features:
        state_dim = features["observation.state"].shape[0]
    elif args.state_dim is not None:
        state_dim = args.state_dim
    else:
        state_dim = action_dim  # fallback
    
    print(f"Dataset info:")
    print(f"  - Total episodes: {dataset_metadata.total_episodes}")
    print(f"  - Total frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Features: {list(features.keys())}")

    # Auto-compute vlm_max_length if static VLM padding is enabled but no override given
    if args.static_vlm_padding and args.vlm_max_length is None:
        args.vlm_max_length = compute_vlm_max_length(
            dataset_metadata,
            vlm_path=args.vlm_path,
            min_image_pixels=args.min_image_pixels,
            max_image_pixels=args.max_image_pixels,
        )

    # 创建 LoLA 配置
    gradient_checkpointing = not args.no_gradient_checkpointing
    config = LoLAV07Config(
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
        history_type=args.history_type,
        state_dim=state_dim,
        state_encoder_mode=args.state_encoder_mode,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        vlm_lr=args.vlm_lr,
        vlm_extract_layers=tuple(args.vlm_extract_layers),
        max_image_pixels=args.max_image_pixels,
        min_image_pixels=args.min_image_pixels,
        gripper_loss_weight=args.gripper_loss_weight,
        action_loss_weight=args.action_loss_weight,
        gripper_dim_indices=tuple(int(x.strip()) for x in args.gripper_dims.split(",")),
        hist_action_token_drop_rate=args.hist_action_token_drop_rate,
        static_vlm_padding=args.static_vlm_padding,
        vlm_max_length=args.vlm_max_length,
        # V2: text template + completed tasks + transition masking
        task_text_template_version=args.task_text_template_version,
        completed_tasks_use_ann=not args.no_completed_tasks_use_ann,
        completed_tasks_history_len=args.completed_tasks_history_len,
        transition_mask_rate=args.transition_mask_rate,
        max_transition_len=args.max_transition_len,
        # V07: Bottleneck dimensions
        action_bottleneck_dim=args.action_bottleneck_dim,
        grip_bottleneck_dim=args.grip_bottleneck_dim,
        state_bottleneck_dim=args.state_bottleneck_dim,
        state_grip_bottleneck_dim=args.state_grip_bottleneck_dim,
        encoder_lr_mult=args.encoder_lr_mult,
        warmup_pct=args.warmup_pct,
        gradient_checkpointing=gradient_checkpointing,
    )

    # 归一化模式
    if args.norm_mode == "robovlm":
        from lerobot.configs.types import NormalizationMode
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    elif args.norm_mode == "zscore":
        from lerobot.configs.types import NormalizationMode
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.IDENTITY,
        }

    # 创建训练和验证数据集
    print("Creating datasets...")
    if args.norm_mode == "robovlm":
        norm_action = True
    elif args.norm_mode == "zscore":
        norm_action = "zscore"
    else:
        norm_action = False
    train_dataset = create_lola_dataset(
        repo_id=args.dataset_repo_id,
        config=config,
        root=args.dataset_root,
        episodes=args.episodes,
        use_lola_dataset=args.load_full_history,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
        norm_action=norm_action,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
        tolerance_frames=args.tolerance_frames,
        gripper_dim_indices_abs=config.gripper_dim_indices_abs,
        dataset_stats=dataset_metadata.stats,
        history_type=args.history_type,
        state_dim=state_dim,
        # V2: completed tasks + transition masking
        track_completed_tasks=config.task_text_template_version == "v1_with_completed",
        transition_mask_rate=config.transition_mask_rate,
        completed_tasks_use_ann=config.completed_tasks_use_ann,
        completed_tasks_history_len=config.completed_tasks_history_len,
        max_transition_len=config.max_transition_len,
    )
    print("Done.\n Train Data Example:")
    for key, value in train_dataset[0].items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, {value.dtype}")
        else:
            print(f"{key}: {value}")

    print(f"\nTotal training samples: {len(train_dataset)}")

    # 创建 DataLoader
    use_static_padding = not args.no_static_collate_padding and args.load_full_history
    static_max_len = args.max_history_length if use_static_padding else None
    if static_max_len is not None:
        print(f"Using static collate padding to max_history_length={static_max_len}")
    collate = make_collate_fn(static_max_len=static_max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # 创建验证集 DataLoader（如果提供了验证数据集）
    val_loader = None
    if args.val_dataset_repo_id or args.val_dataset_root:
        print("Creating validation dataset...")
        val_dataset = create_lola_dataset(
            repo_id=args.val_dataset_repo_id,
            config=config,
            root=args.val_dataset_root,
            use_lola_dataset=args.load_full_history,
            max_history_length=args.max_history_length,
            history_padding_side=args.history_padding_side,
            norm_action=norm_action,
            norm_min=args.norm_min,
            norm_max=args.norm_max,
            tolerance_frames=args.tolerance_frames,
            gripper_dim_indices_abs=config.gripper_dim_indices_abs,
            dataset_stats=dataset_metadata.stats,
            history_type=args.history_type,
            state_dim=state_dim,
            # V2: completed tasks + transition masking
            track_completed_tasks=config.task_text_template_version == "v1_with_completed",
            transition_mask_rate=config.transition_mask_rate,
            completed_tasks_use_ann=config.completed_tasks_use_ann,
            max_transition_len=config.max_transition_len,
        )
        val_batch_size = args.val_batch_size or args.batch_size
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )
        print(f"Total validation samples: {len(val_dataset)}")

    # 创建模型
    model = LoLAV07LightningModule(
        config=config,
        dataset_stats=dataset_metadata.stats,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        train_vlm=args.train_vlm,
    )
    
    # 回调函数
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"lola-v07-{time_str}")

    # Save all training configurations as JSON at training start (rank 0 only)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        import json as _json
        import dataclasses
        from pathlib import Path

        os.makedirs(ckpt_dir, exist_ok=True)
        config_path = os.path.join(ckpt_dir, "training_config.json")

        def _make_serializable(obj):
            if isinstance(obj, (torch.dtype, torch.device)):
                return str(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return _make_serializable(dataclasses.asdict(obj))
            return obj

        full_config = _make_serializable({
            "lola_config": config,
            "training_args": vars(args),
            "dataset_metadata": {
                "total_episodes": dataset_metadata.total_episodes,
                "total_frames": dataset_metadata.total_frames,
                "fps": dataset_metadata.fps,
                "features": {k: {"shape": list(v.shape), "type": str(v.type)} for k, v in features.items()},
            },
        })

        with open(config_path, "w") as f:
            _json.dump(full_config, f, indent=2, default=str)
        print(f"Training config saved to {config_path}")

    checkpoint_kwargs = {
        "dirpath": ckpt_dir,
        "filename": "lola-{step:06d}",
        "save_top_k": -1,
        "save_last": True,
    }
    if args.save_every_n_steps is not None:
        checkpoint_kwargs["every_n_train_steps"] = args.save_every_n_steps
    if args.save_every_n_epochs is not None:
        checkpoint_kwargs["every_n_epochs"] = args.save_every_n_epochs

    callbacks = [
        ModelCheckpoint(**checkpoint_kwargs),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Logger
    logger_name = args.dataset_repo_id.replace('/', '-') if args.dataset_repo_id else os.path.basename(args.dataset_root)
    logger = WandbLogger(
        project="lola-multigpu",
        name=f"lola-v07-{args.strategy}-{logger_name}",
        save_dir="logs",
    )
    
    # 创建 Trainer
    gradient_clip_val = 1.0 if args.strategy != "fsdp" else None

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=1,
        benchmark=True,
        enable_progress_bar=True,
        sync_batchnorm=True,
    )
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps
    if args.max_epochs is not None:
        trainer_kwargs["max_epochs"] = args.max_epochs
    if args.val_frequency is not None:
        trainer_kwargs["val_check_interval"] = args.val_frequency

    trainer = pl.Trainer(**trainer_kwargs)
    
    # 打印配置信息
    print("=" * 60)
    print("LoLA V07 Multi-GPU Training with LeRobotDataset")
    print("=" * 60)
    print(f"Dataset: {args.dataset_repo_id}")
    print(f"Strategy: {args.strategy}")
    print(f"Devices: {args.devices}")
    print(f"Num Nodes: {args.num_nodes}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Steps: {args.max_steps or 'N/A (epoch-based)'}")
    print(f"Max Epochs: {args.max_epochs or 'N/A (step-based)'}")
    print(f"Save Every N Steps: {args.save_every_n_steps or 'N/A'}")
    print(f"Save Every N Epochs: {args.save_every_n_epochs or 'N/A'}")
    print(f"Precision: {args.precision}")
    print(f"VLM Path: {args.vlm_path}")
    print(f"Train VLM: {args.train_vlm}")
    print(f"Action Dim: {action_dim}")
    print(f"Action Chunk Size: {args.action_chunk_size}")
    print(f"Pred Chunk Size: {args.pred_chunk_size}")
    print(f"Load Full History: {args.load_full_history}")
    print(f"Max History Length: {args.max_history_length}")
    print(f"History Padding Side: {args.history_padding_side}")
    print(f"Validation Dataset: {args.val_dataset_repo_id or args.val_dataset_root or 'N/A'}")
    print(f"Val Batch Size: {args.val_batch_size or args.batch_size}")
    print(f"Val Frequency: {args.val_frequency or 'N/A'}")
    print("=" * 60)
    
    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("Training completed!")
    print(f"Last checkpoint: {callbacks[0].last_model_path}")


if __name__ == "__main__":
    main()