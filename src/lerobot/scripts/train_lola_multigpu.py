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
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy

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
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors


# ----------------------------------------------------------------------
# LoLA Lightning Module - 包装实际模型
# ----------------------------------------------------------------------
class LoLALightningModule(pl.LightningModule):
    """LoLA 的 PyTorch Lightning 模块 - 包装实际的 LoLAPolicy"""
    
    def __init__(
        self,
        config: LoLAConfig,
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
        self.policy = LoLAPolicy(self.config)
        
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
        
    def forward(self, batch):
        """前向传播"""
        return self.policy(batch)
    
    def _extract_special_fields(self, batch):
        """
        提取特殊字段，避免被preprocessor处理。
        
        包括：
        - hist_actions_full, hist_actions_mask, hist_actions_length: LoLADataset的历史action字段
        - action: 目标action字段（shape可能不匹配stats中的定义）
        
        这些字段不在数据集stats中定义，或者shape可能与stats不匹配。
        """
        special_data = {}
        # LoLADataset 的历史action字段
        keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]
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
        """训练步骤"""
        self._step_start_time = time.monotonic()

        # 提取特殊字段（避免被preprocessor处理）
        special_data = self._extract_special_fields(batch)
        
        # 应用预处理器
        batch = self.preprocessor(batch)
        
        # 恢复特殊字段
        batch = self._restore_special_fields(batch, special_data)
        
        loss, loss_dict = self(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if k != "loss":
                self.log(f"train_{k}", v, prog_bar=False, sync_dist=True)

        # 记录步耗时
        if self._step_start_time is not None:
            update_s = round(time.monotonic() - self._step_start_time, 2)
            self.log("train_update_s", update_s, prog_bar=False, sync_dist=False)

        return loss

    def configure_optimizers(self):
        """配置优化器"""
        # 只优化需要梯度的参数
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
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
            max_lr=self.learning_rate,
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
    config: LoLAConfig,
    root: str | None = None,
    episodes: list | None = None,
    image_transforms=None,
    video_backend: str | None = None,
    use_lola_dataset: bool = False,
    max_history_length: int = 100,
    history_padding_side: str = "left",
    norm_action: bool = False,
    norm_min: float = -0.65,
    norm_max: float = 0.65,
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
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            norm_action=norm_action,
            norm_min=norm_min,
            norm_max=norm_max,
        )
    else:
        # 使用标准 LeRobotDataset
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )
    
    return dataset


def collate_fn(batch):
    """
    自定义 collate 函数，处理 LeRobotDataset 返回的数据。

    LeRobotDataset 返回的键包括：
    - observation.state: 状态数据
    - observation.images.*: 图像数据
    - action: 动作数据
    - task: 任务描述字符串
    - episode_index, frame_index, timestamp 等元数据
    - hist_actions_full: 历史动作（LoLADataset，变长）
    - hist_actions_mask: 历史动作mask（LoLADataset，变长）
    - hist_actions_length: 历史动作实际长度（LoLADataset）

    对于变长序列（hist_actions_full, hist_actions_mask），会 pad 到 batch 内最大长度，
    类似 flash attention 的处理方式。
    """
    result = {}

    # 变长序列的 key，需要特殊处理
    variable_length_keys = {"hist_actions_full", "hist_actions_mask"}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if key == "task":
            # 任务描述保持为字符串列表
            result[key] = values
        elif key.startswith("observation.images."):
            # 相机图像保持为 list，不做 torch.stack
            # LolaImageProcessor batch mode 需要 list 格式来检测 batch 维度
            result[key] = values
        elif key in variable_length_keys and isinstance(values[0], torch.Tensor):
            # 变长序列：pad 到 batch 内最大长度
            # 找到最大长度
            max_len = max(v.shape[0] for v in values)

            # Pad 每个序列到最大长度（左侧 padding）
            padded_values = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_len = max_len - v.shape[0]
                    if key == "hist_actions_full":
                        # action: [seq_len, action_dim] -> pad 到 [max_len, action_dim]
                        padding = torch.zeros(pad_len, v.shape[1], dtype=v.dtype)
                    else:  # hist_actions_mask
                        # mask: [seq_len] -> pad 到 [max_len]
                        padding = torch.zeros(pad_len, dtype=v.dtype)
                    # 左侧 padding
                    v = torch.cat([padding, v], dim=0)
                padded_values.append(v)

            result[key] = torch.stack(padded_values)
        elif isinstance(values[0], torch.Tensor):
            # 固定长度张量数据堆叠
            result[key] = torch.stack(values)
        else:
            # 其他类型保持列表
            result[key] = values

    return result


# ----------------------------------------------------------------------
# FSDP 配置
# ----------------------------------------------------------------------
def get_fsdp_strategy():
    """获取 FSDP 策略配置"""
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5VisionBlock
    from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock, Flux2SingleTransformerBlock
    from lerobot.policies.lola.modeling_lola import LolaVLMFeatureExtractor

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
        }
    )
    
    strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        cpu_offload=False,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,  # 兼容优化器
        state_dict_type=StateDictType.FULL_STATE_DICT,
    )
    return strategy


def get_deepspeed_strategy():
    """获取 DeepSpeed 策略配置"""
    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=False,
        offload_parameters=False,
    )
    return strategy


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    os.environ['WANDB_API_KEY'] = "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"
    parser = argparse.ArgumentParser(description="LoLA Multi-GPU Training with LeRobotDataset")
    
    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None,
                        help="HuggingFace dataset repo ID (e.g., lerobot/pusht)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Local dataset root directory (optional)")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Specific episodes to load (optional)")
    
    # 训练策略参数
    parser.add_argument("--strategy", type=str, default="fsdp", choices=["fsdp", "deepspeed", "ddp"])
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

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 数量")

    # 归一化参数
    parser.add_argument("--norm_mode", type=str, default="default",
                        choices=["default", "robovlm"],
                        help="归一化模式: default(LoLA默认MEAN_STD), robovlm(min-max→[-1,1],全IDENTITY)")
    parser.add_argument("--norm_min", type=float, default=-0.65,
                        help="RoboVLM 归一化下界")
    parser.add_argument("--norm_max", type=float, default=0.65,
                        help="RoboVLM 归一化上界")

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
    
    # 设置策略
    if args.strategy == "fsdp":
        strategy = get_fsdp_strategy()
    elif args.strategy == "deepspeed":
        strategy = get_deepspeed_strategy()
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
    
    print(f"Dataset info:")
    print(f"  - Total episodes: {dataset_metadata.total_episodes}")
    print(f"  - Total frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Features: {list(features.keys())}")
    
    # 创建 LoLA 配置
    gradient_checkpointing = not args.no_gradient_checkpointing
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
        gradient_checkpointing=gradient_checkpointing,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        vlm_lr=args.vlm_lr,
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

    # 创建训练和验证数据集
    print("Creating datasets...")
    norm_action = (args.norm_mode == "robovlm")
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
    )
    print("Done.\n Train Data Example:")
    for key, value in train_dataset[0].items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, {value.dtype}")
        else:
            print(f"{key}: {value}")

    print(f"\nTotal training samples: {len(train_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 创建模型
    model = LoLALightningModule(
        config=config,
        dataset_stats=dataset_metadata.stats,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        train_vlm=args.train_vlm,
    )
    
    # 回调函数
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"lola-{time_str}")
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
        name=f"lola-{args.strategy}-{logger_name}",
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

    trainer = pl.Trainer(**trainer_kwargs)
    
    # 打印配置信息
    print("=" * 60)
    print("LoLA Multi-GPU Training with LeRobotDataset")
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
    print("=" * 60)
    
    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    print("Training completed!")
    print(f"Last checkpoint: {callbacks[0].last_model_path}")


if __name__ == "__main__":
    main()