#!/usr/bin/env python
"""
LoLA Azure 分布式训练脚本 - 使用原生 PyTorch DDP

本脚本适用于 Azure/AWS 等云平台的多节点训练，使用环境变量初始化分布式。

与 train_lola_multigpu.py 的区别：
- 使用原生 PyTorch 分布式初始化（从环境变量获取 WORLD_SIZE, RANK 等）
- 不依赖 torchrun，适合在 Azure ML、AWS SageMaker 等平台运行
- 支持 DDP 和 FSDP 两种分布式策略
- 支持 Wandb 日志记录

环境变量（由平台自动设置）：
- WORLD_SIZE: 总进程数
- RANK: 全局 rank
- LOCAL_RANK: 节点内 rank
- NODE_RANK: 节点 rank
- MASTER_ADDR: 主节点 IP
- MASTER_PORT: 主节点端口

使用方法:
    # 单节点多卡
    python -m torch.distributed.launch --nproc_per_node=4 src/lerobot/scripts/train_lola_azure.py \
        --dataset_root /path/to/dataset

    # Azure ML 多节点训练（环境变量自动设置）
    python src/lerobot/scripts/train_lola_azure.py \
        --dataset_root /path/to/dataset \
        --strategy fsdp

    # 使用 Wandb 日志
    python src/lerobot/scripts/train_lola_azure.py \
        --dataset_root /path/to/dataset \
        --wandb_project my-project \
        --wandb_name experiment-1

    # 禁用 Wandb
    python src/lerobot/scripts/train_lola_azure.py \
        --dataset_root /path/to/dataset \
        --disable_wandb
"""

import argparse
import datetime
import logging
import os
import sys
import time
from datetime import timedelta
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.lola_dataset import LoLADataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [Rank {os.environ.get('RANK', '0')}] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """
    从环境变量初始化分布式训练。

    环境变量由 Azure/AWS 等平台自动设置：
    - WORLD_SIZE: 总进程数
    - RANK: 全局 rank
    - LOCAL_RANK: 节点内 rank
    - NODE_RANK: 节点 rank
    - MASTER_ADDR: 主节点 IP
    - MASTER_PORT: 主节点端口
    """
    # 获取环境变量
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    master_uri = "tcp://%s:%s" % (master_addr, master_port)

    # 设置当前设备

    if world_size > 1:
        
        # 初始化进程组
        dist.init_process_group(
            backend="nccl",
            init_method=master_uri,
            world_size=world_size,
            timeout=timedelta(minutes=60),
            rank=world_rank,
        )

        logger.info(f"Distributed initialized: rank={world_rank}, local_rank={local_rank}, "
                    f"world_size={world_size}, master={master_uri}")
    else:
        logger.info(f"Single GPU mode: local_rank={local_rank}")
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return {
        "world_size": world_size,
        "local_rank": local_rank,
        "world_rank": world_rank,
        "node_rank": node_rank,
        "device": device,
        "is_distributed": world_size > 1,
    }


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


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
) -> LeRobotDataset | LoLADataset:
    """创建 LoLA 训练用的数据集。"""
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = dataset_metadata.fps

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    logger.info(f"delta_timestamps: {delta_timestamps}")

    if use_lola_dataset:
        logger.info(f"Using LoLADataset with max_history_length={max_history_length}")
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
        )
    else:
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
    """自定义 collate 函数。"""
    result = {}
    variable_length_keys = {"hist_actions_full", "hist_actions_mask"}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if key == "task":
            result[key] = values
        elif key in variable_length_keys and isinstance(values[0], torch.Tensor):
            max_len = max(v.shape[0] for v in values)
            padded_values = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_len = max_len - v.shape[0]
                    if key == "hist_actions_full":
                        padding = torch.zeros(pad_len, v.shape[1], dtype=v.dtype)
                    else:
                        padding = torch.zeros(pad_len, dtype=v.dtype)
                    v = torch.cat([padding, v], dim=0)
                padded_values.append(v)
            result[key] = torch.stack(padded_values)
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values

    return result


# ----------------------------------------------------------------------
# 训练器
# ----------------------------------------------------------------------
class LoLATrainer:
    """原生 PyTorch 训练器，支持 DDP 和 FSDP"""

    def __init__(
        self,
        config: LoLAConfig,
        dataset_stats: dict | None,
        dist_info: dict,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        max_steps: int = 30000,
        train_vlm: bool = False,
        strategy: str = "ddp",
        gradient_clip_val: float = 1.0,
        ckpt_dir: str = "/data_16T/deepseek/checkpoints/lola",
        save_every_n_steps: int = 500,
        log_every_n_steps: int = 10,
        # Wandb 参数
        wandb_project: str = "lola-azure",
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_id: str | None = None,
    ):
        self.config = config
        self.dataset_stats = dataset_stats
        self.dist_info = dist_info
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.train_vlm = train_vlm
        self.strategy = strategy
        self.gradient_clip_val = gradient_clip_val
        self.ckpt_dir = ckpt_dir
        self.save_every_n_steps = save_every_n_steps
        self.log_every_n_steps = log_every_n_steps

        # Wandb 配置
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_entity = wandb_entity
        self.wandb_id = wandb_id
        self.use_wandb = HAS_WANDB and dist_info["world_rank"] == 0

        self.device = dist_info["device"]
        self.local_rank = dist_info["local_rank"]
        self.world_rank = dist_info["world_rank"]
        self.world_size = dist_info["world_size"]
        self.is_distributed = dist_info["is_distributed"]
        self.is_main_process = self.world_rank == 0

        # 模型和优化器
        self.policy = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.preprocessor = None
        self.postprocessor = None

        # 混合精度：BF16 不需要 GradScaler，FP16 才需要
        self.use_bf16 = True  # 使用 BF16 精度
        self.scaler = None if self.use_bf16 else torch.amp.GradScaler("cuda")

        # 训练状态
        self.global_step = 0
        self.best_loss = float("inf")

    def setup_model(self):
        """设置模型"""
        logger.info(f"Loading LoLA Policy on {self.device}...")

        # 加载 LoLA Policy
        self.policy = LoLAPolicy(self.config)
        self.policy._device = self.device
        self.policy.model = self.policy.model.to(self.device)
        self.policy.vlm = self.policy.vlm.to(self.device)

        # 创建预处理器和后处理器
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            dataset_stats=self.dataset_stats,
        )

        # 冻结 VLM 参数
        if not self.train_vlm and hasattr(self.policy, "vlm"):
            logger.info("Freezing VLM parameters...")
            for param in self.policy.vlm.parameters():
                param.requires_grad = False
            self.policy.vlm.eval()

        # 打印参数统计
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")

        # 设置分布式
        if self.is_distributed:
            if self.strategy == "fsdp":
                self._setup_fsdp()
            else:
                self._setup_ddp()
        else:
            self.model = self.policy

    def _setup_ddp(self):
        """设置 DDP"""
        logger.info("Setting up DDP...")
        self.model = DDP(
            self.policy,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

    def _setup_fsdp(self):
        """设置 FSDP"""
        logger.info("Setting up FSDP...")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5VisionBlock
        from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock, Flux2SingleTransformerBlock
        from lerobot.policies.lola.modeling_lola import LolaVLMFeatureExtractor

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

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

        self.model = FSDP(
            self.policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
        )

    def setup_optimizer(self):
        """设置优化器"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        from torch.optim.lr_scheduler import OneCycleLR
        warmup_ratio = min(self.warmup_ratio, 0.1)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.max_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
        )

    def _extract_special_fields(self, batch):
        """提取特殊字段"""
        special_data = {}
        keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]
        for key in keys_to_extract:
            if key in batch:
                special_data[key] = batch.pop(key)
        if "action" in batch:
            special_data["action"] = batch.pop("action")
        return special_data

    def _restore_special_fields(self, batch, special_data):
        """恢复特殊字段"""
        batch.update(special_data)
        return batch

    def training_step(self, batch):
        """单步训练"""
        # 移动数据到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 提取特殊字段
        special_data = self._extract_special_fields(batch)

        # 预处理
        batch = self.preprocessor(batch)
        batch = self._restore_special_fields(batch, special_data)

        # 前向传播（混合精度）
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = self.model(batch)

        return loss, loss_dict

    def train(self, train_loader, start_step: int = 0):
        """训练循环"""
        self.global_step = start_step
        self.model.train()

        # 创建 checkpoint 目录
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, f"lola-azure-{time_str}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Checkpoint directory: {ckpt_dir}")

        # 初始化 Wandb
        if self.use_wandb:
            wandb_run_name = self.wandb_name or f"lola-{self.strategy}-{time_str}"
            wandb.init(
                project=self.wandb_project,
                name=wandb_run_name,
                entity=self.wandb_entity,
                id=self.wandb_id,
                resume="allow" if self.wandb_id else None,
                config={
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "max_steps": self.max_steps,
                    "batch_size": train_loader.batch_size,
                    "strategy": self.strategy,
                    "world_size": self.world_size,
                    "train_vlm": self.train_vlm,
                    "gradient_clip_val": self.gradient_clip_val,
                },
            )
            logger.info(f"Wandb initialized: {wandb_run_name}")

        logger.info(f"Starting training from step {start_step} to {self.max_steps}")

        # 计算 resume 时需要跳过的 batch 数
        # Map-style 数据集：可以用 skip_epochs + break 快速跳过完整 epoch
        # IterableDataset（streaming）：没有 __len__，只能逐 batch 迭代丢弃
        try:
            batches_per_epoch = len(train_loader)
            logger.info(f"Total batches per epoch: {batches_per_epoch}")
        except TypeError:
            batches_per_epoch = None
            logger.info("IterableDataset detected: cannot determine batches per epoch")

        if start_step > 0 and batches_per_epoch is not None:
            skip_epochs = start_step // batches_per_epoch
            skip_batches = start_step % batches_per_epoch
            logger.info(f"Resuming: skipping {skip_epochs} epochs + {skip_batches} batches")
        elif start_step > 0 and batches_per_epoch is None:
            # IterableDataset: 无法按 epoch 跳过，for 循环会创建新迭代器从头开始
            # 所以 streaming resume 时数据会从头开始，但模型/优化器/scheduler 状态已恢复
            skip_epochs = 0
            skip_batches = 0
            logger.warning(
                f"Resuming from step {start_step} with IterableDataset: "
                "data will restart from the beginning (model/optimizer/scheduler states are restored). "
                "For precise data resume, use map-style dataset or add start_index to IterableDataset."
            )
        else:
            skip_epochs = 0
            skip_batches = 0

        epoch = 0
        while self.global_step < self.max_steps:
            epoch += 1
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader):
                if self.global_step >= self.max_steps:
                    break

                # Map-style 数据集：跳过已训练的 batch
                if skip_epochs > 0 or skip_batches > 0:
                    if skip_epochs > 0:
                        skip_epochs -= 1
                        break  # 跳过整个 epoch
                    skip_batches -= 1
                    continue

                step_start = time.monotonic()

                self.optimizer.zero_grad()

                loss, loss_dict = self.training_step(batch)

                # 反向传播
                if self.use_bf16:
                    # BF16: 直接 backward，不需要 scaler
                    loss.backward()
                else:
                    # FP16: 使用 scaler
                    self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.gradient_clip_val > 0:
                    if not self.use_bf16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val,
                    )

                # 优化器步进
                if self.use_bf16:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # 学习率调度
                self.scheduler.step()

                self.global_step += 1

                # 计算步耗时
                update_s = round(time.monotonic() - step_start, 2)

                # 日志
                if self.global_step % self.log_every_n_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    if self.is_main_process:
                        logger.info(
                            f"Step {self.global_step}/{self.max_steps} | "
                            f"Loss: {loss.item():.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Update: {update_s}s"
                        )
                        # Wandb 日志
                        if self.use_wandb:
                            log_dict = {
                                "train/loss": loss.item(),
                                "train/learning_rate": lr,
                                "train/step": self.global_step,
                                "train/epoch": epoch,
                                "train/update_s": update_s,
                            }
                            # 添加 loss_dict 中的其他 loss
                            for k, v in loss_dict.items():
                                if k != "loss" and isinstance(v, (int, float)):
                                    log_dict[f"train/{k}"] = v
                            wandb.log(log_dict)

                # 保存 checkpoint
                if self.global_step % self.save_every_n_steps == 0 and self.is_main_process:
                    self.save_checkpoint(ckpt_dir, self.global_step)

        # 保存最终 checkpoint
        if self.is_main_process:
            self.save_checkpoint(ckpt_dir, self.global_step, is_final=True)
            logger.info(f"Training completed! Final checkpoint saved at step {self.global_step}")

        # 关闭 Wandb
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, ckpt_dir: str, step: int, is_final: bool = False):
        """保存 checkpoint"""
        if self.strategy == "fsdp":
            from torch.distributed.checkpoint import save as save_fsdp_checkpoint
            from torch.distributed.checkpoint.state_dict import get_state_dict

            # FSDP checkpoint 保存：用 get_state_dict 获取模型和优化器的分片 state_dict
            model_sd, optimizer_sd = get_state_dict(self.model, self.optimizer)
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}" if not is_final else "final")
            save_fsdp_checkpoint(
                {
                    "model": model_sd,
                    "optimizer": optimizer_sd,
                    "step": [step],
                },
                checkpoint_id=ckpt_path,
            )
            # scheduler 不支持 torch.distributed.checkpoint，单独用 torch.save 保存
            if self.is_main_process:
                torch.save(
                    {"scheduler_state_dict": self.scheduler.state_dict()},
                    os.path.join(ckpt_path, "scheduler.pt"),
                )
        else:
            # DDP checkpoint 保存
            state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
            ckpt_name = f"lola-step-{step:06d}.pt" if not is_final else "lola-final.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save({
                "step": step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }, ckpt_path)

        logger.info(f"Checkpoint saved: {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str):
        """加载 checkpoint"""
        if self.strategy == "fsdp":
            from torch.distributed.checkpoint import load as load_fsdp_checkpoint
            from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

            # FSDP checkpoint 加载：先获取空 state_dict 容器，再 load 填充，最后 set 回模型/优化器
            model_sd, optimizer_sd = get_state_dict(self.model, self.optimizer)
            # 用 list 包装 step，因为 int 是不可变对象，load 无法原地修改
            step_container = [0]
            load_fsdp_checkpoint(
                {"model": model_sd, "optimizer": optimizer_sd, "step": step_container},
                checkpoint_id=ckpt_path,
            )
            set_state_dict(self.model, self.optimizer, model_state_dict=model_sd, optim_state_dict=optimizer_sd)
            self.global_step = step_container[0]
            # 恢复 scheduler 状态
            scheduler_path = os.path.join(ckpt_path, "scheduler.pt")
            if os.path.exists(scheduler_path):
                scheduler_ckpt = torch.load(scheduler_path, map_location=self.device)
                self.scheduler.load_state_dict(scheduler_ckpt["scheduler_state_dict"])
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if self.is_distributed:
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.global_step = checkpoint.get("step", 0)

        logger.info(f"Checkpoint loaded from: {ckpt_path}, starting from step {self.global_step}")


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    # 初始化分布式
    dist_info = setup_distributed()

    # 参数解析
    parser = argparse.ArgumentParser(description="LoLA Azure Distributed Training")

    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)

    # 训练参数
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # 模型参数
    parser.add_argument("--vlm_path", type=str, default="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/")
    parser.add_argument("--train_vlm", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="/data_16T/deepseek/checkpoints/lola")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # LoLA 参数
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)

    # 历史action参数
    parser.add_argument("--load_full_history", action="store_true")
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"])

    # Wandb 参数
    parser.add_argument("--wandb_project", type=str, default="lola-azure", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run id (for resume)")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # 检查数据集参数
    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    # 打印配置
    if dist_info["world_rank"] == 0:
        logger.info("=" * 60)
        logger.info("LoLA Azure Distributed Training")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset_repo_id or args.dataset_root}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"World Size: {dist_info['world_size']}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info(f"Max Steps: {args.max_steps}")
        logger.info(f"VLM Path: {args.vlm_path}")
        logger.info(f"Train VLM: {args.train_vlm}")
        logger.info("=" * 60)

    # 获取数据集元数据
    logger.info(f"Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    logger.info(f"Dataset: {dataset_metadata.total_episodes} episodes, {dataset_metadata.total_frames} frames")
    logger.info(f"Action dim: {action_dim}")

    # 创建 LoLA 配置
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
    )

    # 创建数据集
    logger.info("Creating dataset...")
    train_dataset = create_lola_dataset(
        repo_id=args.dataset_repo_id,
        config=config,
        root=args.dataset_root,
        episodes=args.episodes,
        use_lola_dataset=args.load_full_history,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
    )
    logger.info(f"Dataset size: {len(train_dataset)}")

    # 创建 DataLoader（使用 DistributedSampler）
    sampler = None
    shuffle = True
    if dist_info["is_distributed"]:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_info["world_size"],
            rank=dist_info["world_rank"],
            shuffle=True,
        )
        shuffle = False  # sampler 已处理 shuffle

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # 分布式训练建议 drop_last
    )

    # 创建训练器
    trainer = LoLATrainer(
        config=config,
        dataset_stats=dataset_metadata.stats,
        dist_info=dist_info,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        train_vlm=args.train_vlm,
        strategy=args.strategy,
        gradient_clip_val=args.gradient_clip_val,
        ckpt_dir=args.ckpt_dir,
        save_every_n_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        # Wandb 参数
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        wandb_id=args.wandb_id,
    )

    # 禁用 wandb
    if args.disable_wandb:
        trainer.use_wandb = False

    # 设置模型和优化器
    trainer.setup_model()
    trainer.setup_optimizer()

    # 加载 checkpoint
    start_step = 0
    if args.resume:
        trainer.load_checkpoint(args.resume)
        start_step = trainer.global_step

    # 开始训练
    trainer.train(train_loader, start_step=start_step)

    # 清理
    cleanup_distributed()
    logger.info("Training completed!")


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = "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"
    
    main()
