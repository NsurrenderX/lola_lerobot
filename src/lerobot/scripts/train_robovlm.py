#!/usr/bin/env python
"""
RoboVLM 分布式微调脚本

使用 LeRobotDataset 加载 lerobot 3.0 格式数据集，适配 RoboVLM (Kosmos-2 + LSTMDecoder) 模型。
适用于小规模微调场景，无需流式数据集。

使用方法:
    # 单 GPU
    python src/lerobot/scripts/train_robovlm.py \
        --dataset_repo_id <repo_id> --dataset_root <path>

    # 多 GPU (torchrun)
    torchrun --nproc_per_node=4 src/lerobot/scripts/train_robovlm.py \
        --dataset_repo_id <repo_id> --strategy ddp

    # 多节点 (torchrun)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_endpoint=MASTER_ADDR:MASTER_PORT \
        src/lerobot/scripts/train_robovlm.py \
        --dataset_repo_id <repo_id> --strategy ddp
"""

import argparse
import datetime
import logging
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.robovlm import RoboVLMConfig, RoboVLMPolicy
from lerobot.policies.factory import make_pre_post_processors

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [Rank {os.environ.get('RANK', '0')}] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rank = os.environ.get("RANK", "0")
    print(f"[{ts}] [Rank {rank}] {msg}", flush=True)


def setup_distributed():
    """Initialize distributed training from environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        master_uri = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=master_uri,
            world_size=world_size,
            timeout=timedelta(minutes=60),
            rank=world_rank,
        )
        _log(f"Distributed initialized: rank={world_rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        _log(f"Single GPU mode: local_rank={local_rank}")

    return {
        "world_size": world_size,
        "local_rank": local_rank,
        "world_rank": world_rank,
        "node_rank": node_rank,
        "device": device,
        "is_distributed": world_size > 1,
    }


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_robovlm_dataset(
    repo_id: str,
    config: RoboVLMConfig,
    root: str | None = None,
):
    """Create a LeRobotDataset with delta_timestamps matching RoboVLM config."""
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = dataset_metadata.fps

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    _log(f"delta_timestamps: {delta_timestamps}")
    _log(f"Camera keys: {dataset_metadata.camera_keys}")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )

    return dataset, dataset_metadata


class RoboVLMTrainer:
    def __init__(
        self,
        config: RoboVLMConfig,
        dataset_stats: dict,
        dist_info: dict,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        max_steps: int = 10000,
        strategy: str = "ddp",
        fsdp_sharding: str = "full_shard",
        gradient_clip_val: float = 1.0,
        ckpt_dir: str = "runs/checkpoints",
        save_every_n_steps: int = 500,
        log_every_n_steps: int = 10,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
    ):
        self.config = config
        self.dataset_stats = dataset_stats
        self.dist_info = dist_info
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.strategy = strategy
        self.fsdp_sharding = fsdp_sharding
        self.gradient_clip_val = gradient_clip_val
        self.ckpt_dir = ckpt_dir
        self.save_every_n_steps = save_every_n_steps
        self.log_every_n_steps = log_every_n_steps

        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_entity = wandb_entity
        self.use_wandb = HAS_WANDB and dist_info["world_rank"] == 0

        self.device = dist_info["device"]
        self.local_rank = dist_info["local_rank"]
        self.world_rank = dist_info["world_rank"]
        self.world_size = dist_info["world_size"]
        self.is_distributed = dist_info["is_distributed"]
        self.is_main_process = self.world_rank == 0

        self.policy = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.preprocessor = None
        self.postprocessor = None

        self.global_step = 0
        self.best_loss = float("inf")

    def setup_model(self):
        """Instantiate RoboVLM policy and apply DDP/FSDP wrapping."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        _log(f"Loading RoboVLM Policy on {self.device}...")
        self.policy = RoboVLMPolicy(self.config)
        self.policy.model = self.policy.model.to(self.device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            dataset_stats=self.dataset_stats,
        )

        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy.parameters())
        _log(f"Trainable params: {trainable_params:,} / {total_params:,}")

        if self.is_distributed:
            if self.strategy == "fsdp":
                self._setup_fsdp()
            else:
                self._setup_ddp()
        else:
            self.model = self.policy

    def _setup_ddp(self):
        _log("Setting up DDP...")
        self.model = DDP(
            self.policy,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

    def _setup_fsdp(self):
        _log("Setting up FSDP...")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Kosmos-2 specific layer classes for FSDP wrapping
        try:
            from transformers.models.kosmos2.modeling_kosmos2 import (
                Kosmos2DecoderLayer,
                Kosmos2VisionEncoder,
            )
            wrap_classes = {Kosmos2DecoderLayer, Kosmos2VisionEncoder, LSTMDecoder}
        except ImportError:
            _log("Warning: Kosmos-2 layer classes not found for FSDP wrapping")
            from lerobot.policies.robovlm.modeling_robovlm import LSTMDecoder
            wrap_classes = {LSTMDecoder}

        from lerobot.policies.robovlm.modeling_robovlm import LSTMDecoder
        wrap_classes.add(LSTMDecoder)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
            module, recurse, nonwrapped_numel,
            transformer_layer_cls=wrap_classes,
        )

        sharding_strategy = (
            ShardingStrategy.FULL_SHARD if self.fsdp_sharding == "full_shard"
            else ShardingStrategy.SHARD_GRAD_OP
        )

        self.model = FSDP(
            self.policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            device_id=self.local_rank,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )

    def setup_optimizer(self):
        """Create optimizer and scheduler."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        from torch.optim.lr_scheduler import OneCycleLR
        warmup_pct = self.config.scheduler_warmup_steps / self.max_steps
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.max_steps,
            pct_start=min(warmup_pct, 0.1),
            anneal_strategy="cos",
        )

    def training_step(self, batch):
        """Single training step."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Extract action and task before preprocessing (preprocessor doesn't handle them)
        action = batch.pop("action", None)
        task = batch.pop("task", None)
        action_is_pad = batch.pop("action_is_pad", None)

        # Preprocess remaining observation data
        batch = self.preprocessor(batch)

        # Restore action and task
        if action is not None:
            batch["action"] = action
        if task is not None:
            batch["task"] = task
        if action_is_pad is not None:
            batch["action_is_pad"] = action_is_pad

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = self.model(batch)

        return loss, loss_dict

    def train(self, train_loader, start_step: int = 0):
        """Training loop."""
        self.global_step = start_step
        self.model.train()

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, f"robovlm-{time_str}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            _log(f"Checkpoint directory: {ckpt_dir}")

        if self.use_wandb:
            wandb_run_name = self.wandb_name or f"robovlm-{self.strategy}-{time_str}"
            wandb.init(
                project=self.wandb_project,
                name=wandb_run_name,
                entity=self.wandb_entity,
                config={
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "max_steps": self.max_steps,
                    "batch_size": train_loader.batch_size,
                    "strategy": self.strategy,
                    "world_size": self.world_size,
                },
            )

        _log(f"Starting training from step {start_step} to {self.max_steps}")

        data_yield_start = time.monotonic()
        data_iter = iter(train_loader)

        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                if self.is_distributed and hasattr(train_loader, "sampler"):
                    train_loader.sampler.set_epoch(self.global_step // len(train_loader))
                data_iter = iter(train_loader)
                _log("DataLoader exhausted, restarting (new epoch)")
                try:
                    batch = next(data_iter)
                except StopIteration:
                    _log("DataLoader produced no data after restart, stopping training")
                    break

            data_yield_s = time.monotonic() - data_yield_start
            step_start = time.monotonic()

            self.optimizer.zero_grad()

            fwd_start = time.monotonic()
            loss, loss_dict = self.training_step(batch)
            fwd_s = time.monotonic() - fwd_start

            bwd_start = time.monotonic()
            loss.backward()
            bwd_s = time.monotonic() - bwd_start

            if self.gradient_clip_val > 0:
                if self.strategy == "fsdp":
                    grad_norm = self.model.clip_grad_norm_(self.gradient_clip_val)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
            else:
                grad_norm = None

            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            update_s = time.monotonic() - step_start

            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                gpu_mem_alloc = torch.cuda.memory_allocated(self.device) / 1e9
                gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9

                if self.is_main_process:
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                    _log(
                        f"[Step {self.global_step}/{self.max_steps}] "
                        f"Loss={loss.item():.4f} LR={lr:.2e} "
                        f"Update={update_s:.2f}s DataWait={data_yield_s:.2f}s"
                    )
                    if grad_norm_val is not None:
                        _log(f"  grad_norm={grad_norm_val:.4f}")
                    _log(
                        f"  Timing: fwd={fwd_s:.3f}s bwd={bwd_s:.3f}s"
                    )
                    _log(f"  GPU: alloc={gpu_mem_alloc:.1f}GB reserved={gpu_mem_reserved:.1f}GB")

                    for k, v in loss_dict.items():
                        if isinstance(v, (int, float)):
                            _log(f"  {k}={v:.4f}")

                    if self.use_wandb:
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/learning_rate": lr,
                            "train/step": self.global_step,
                            "timing/fwd_s": fwd_s,
                            "timing/bwd_s": bwd_s,
                            "memory/gpu_alloc_gb": gpu_mem_alloc,
                            "memory/gpu_reserved_gb": gpu_mem_reserved,
                        }
                        if grad_norm_val is not None:
                            log_dict["train/grad_norm"] = grad_norm_val
                        for k, v in loss_dict.items():
                            if isinstance(v, (int, float)):
                                log_dict[f"train/{k}"] = v
                        wandb.log(log_dict)

            # Checkpointing
            if self.global_step % self.save_every_n_steps == 0:
                self.save_checkpoint(ckpt_dir, self.global_step)

            data_yield_start = time.monotonic()

        self.save_checkpoint(ckpt_dir, self.global_step, is_final=True)
        _log(f"Training completed! Final checkpoint saved at step {self.global_step}")

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, ckpt_dir, step, is_final=False):
        """Save checkpoint."""
        if not self.is_main_process:
            return

        if self.strategy == "fsdp":
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.model.state_dict()
            path = os.path.join(ckpt_dir, f"step-{step}.pt")
            torch.save({"model_state_dict": state_dict, "global_step": step}, path)
        else:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            path = os.path.join(ckpt_dir, f"step-{step}.pt")
            torch.save({
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": step,
            }, path)

        tag = "final" if is_final else f"step-{step}"
        _log(f"Checkpoint saved: {path} ({tag})")

    def load_checkpoint(self, path):
        """Load checkpoint for resuming training."""
        ckpt = torch.load(path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.global_step = ckpt.get("global_step", 0)
        _log(f"Checkpoint loaded from {path}, resuming from step {self.global_step}")


def parse_args():
    parser = argparse.ArgumentParser(description="RoboVLM Fine-tuning")

    # Dataset
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="LeRobot dataset repo ID")
    parser.add_argument("--dataset_root", type=str, default=None, help="Local dataset root path")

    # Training
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=8)

    # Model overrides
    parser.add_argument("--vlm_pretrained_path", type=str, default=".vlms/kosmos-2-patch14-224")
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--train_vision", action="store_true", default=True)
    parser.add_argument("--train_text_embedding", action="store_true", default=True)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--fwd_pred_next_n", type=int, default=10)

    # Distributed
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--fsdp_sharding", type=str, default="full_shard", choices=["full_shard", "shard_grad_op"])
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # Checkpoint / logging
    parser.add_argument("--ckpt_dir", type=str, default="runs/checkpoints")
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="robovlm")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    dist_info = setup_distributed()

    # Build config
    config = RoboVLMConfig(
        vlm_pretrained_path=args.vlm_pretrained_path,
        freeze_backbone=args.freeze_backbone,
        train_vision=args.train_vision,
        train_text_embedding=args.train_text_embedding,
        window_size=args.window_size,
        fwd_pred_next_n=args.fwd_pred_next_n,
        optimizer_lr=args.learning_rate,
    )

    # Create dataset
    train_dataset, dataset_metadata = create_robovlm_dataset(
        repo_id=args.dataset_repo_id,
        config=config,
        root=args.dataset_root,
    )

    # Setup dataset features in config
    features = dataset_to_policy_features(dataset_metadata.features)
    if not config.output_features:
        config.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    if not config.input_features:
        config.input_features = {k: ft for k, ft in features.items() if k not in config.output_features}
    config.validate_features()

    dataset_stats = dataset_metadata.stats

    # Create DataLoader
    sampler = None
    if dist_info["is_distributed"]:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_info["world_size"],
            rank=dist_info["world_rank"],
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    _log(f"Dataset size: {len(train_dataset)}, batch_size: {args.batch_size}, "
         f"batches_per_epoch: {len(train_loader)}")

    # Create trainer
    trainer = RoboVLMTrainer(
        config=config,
        dataset_stats=dataset_stats,
        dist_info=dist_info,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        strategy=args.strategy,
        fsdp_sharding=args.fsdp_sharding,
        gradient_clip_val=args.gradient_clip_val,
        ckpt_dir=args.ckpt_dir,
        save_every_n_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
    )

    if args.disable_wandb:
        trainer.use_wandb = False

    trainer.setup_model()
    trainer.setup_optimizer()

    start_step = 0
    if args.resume:
        trainer.load_checkpoint(args.resume)
        start_step = trainer.global_step

    trainer.train(train_loader, start_step=start_step)

    cleanup_distributed()
    _log("Training completed!")


if __name__ == "__main__":
    main()