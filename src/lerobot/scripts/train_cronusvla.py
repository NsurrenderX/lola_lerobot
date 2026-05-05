#!/usr/bin/env python
"""
CronusVLA 分布式微调脚本

使用 CronusVLADataset 加载 lerobot 3.0 格式数据集，适配 CronusVLA (PrismaticVLM + DiT Diffusion) 模型。
适用于小规模微调场景，无需流式数据集。

使用方法:
    # 单 GPU
    python src/lerobot/scripts/train_cronusvla.py \
        --dataset_repo_id <repo_id> --dataset_root <path>

    # 多 GPU (torchrun)
    torchrun --nproc_per_node=4 src/lerobot/scripts/train_cronusvla.py \
        --dataset_repo_id <repo_id> --strategy ddp

    # 多节点 (torchrun)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_endpoint=MASTER_ADDR:MASTER_PORT \
        src/lerobot/scripts/train_cronusvla.py \
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
from lerobot.datasets.cronusvla_dataset import CronusVLADataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.cronusvla import CronusVLAConfig, CronusVLAPolicy

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


class CronusVLATrainer:
    def __init__(
        self,
        config: CronusVLAConfig,
        dist_info: dict,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        strategy: str = "ddp",
        fsdp_sharding: str = "full_shard",
        gradient_clip_val: float = 1.0,
        ckpt_dir: str = "runs/checkpoints",
        save_every_n_steps: int = 500,
        save_every_n_epochs: int | None = None,
        log_every_n_steps: int = 10,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
    ):
        self.config = config
        self.dist_info = dist_info
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.strategy = strategy
        self.fsdp_sharding = fsdp_sharding
        self.gradient_clip_val = gradient_clip_val
        self.ckpt_dir = ckpt_dir
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
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

        self.global_step = 0
        self.current_epoch = 0
        self.max_steps = max_steps  # may be overridden by epoch mode
        self.best_loss = float("inf")

    def setup_model(self, pretrained_checkpoint=None):
        """Instantiate CronusVLA policy and apply DDP/FSDP wrapping."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        _log(f"Loading CronusVLA Policy on {self.device}...")
        self.policy = CronusVLAPolicy(self.config)
        self.policy.model = self.policy.model.to(self.device)

        # Load pretrained weights (partial load -- missing keys stay randomly initialized)
        if pretrained_checkpoint is not None:
            _log(f"Loading pretrained checkpoint: {pretrained_checkpoint}")
            ckpt = torch.load(pretrained_checkpoint, map_location=self.device)
            model_sd = ckpt["model_state_dict"]
            current_sd = self.policy.model.state_dict()

            # Match keys that exist in both
            loaded_keys = []
            missing_in_pretrained = []
            for key in current_sd:
                if key in model_sd:
                    current_sd[key] = model_sd[key]
                    loaded_keys.append(key)
                else:
                    missing_in_pretrained.append(key)

            self.policy.model.load_state_dict(current_sd)
            _log(f"Pretrained weights loaded: {len(loaded_keys)} keys matched")
            _log(f"Keys missing in pretrained (randomly initialized): {missing_in_pretrained}")

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

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Use CronusVLA's own FSDP wrapping policy (covers PrismaticVLM + DiT)
        auto_wrap_policy = self.policy.model.get_fsdp_wrapping_policy()

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
        """Create optimizer and LR scheduler."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        warmup_steps = self.config.scheduler_warmup_steps
        if self.config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            # Default: "constant" -- warmup then flat LR
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )

    def training_step(self, batch):
        """Single training step -- data already preprocessed by CronusVLADataset.

        CronusVLA computes diffusion loss from VLM cognition features + action targets.
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = self.model(batch)

        return loss, loss_dict

    def train(self, train_loader, start_step: int = 0):
        """Training loop supporting both step-based and epoch-based modes."""
        self.global_step = start_step
        self.model.train()

        steps_per_epoch = len(train_loader)

        # Compute max_steps from epochs if using epoch mode
        if self.max_epochs is not None:
            self.max_steps = self.max_epochs * steps_per_epoch
            _log(f"Epoch mode: {self.max_epochs} epochs x {steps_per_epoch} steps/epoch = {self.max_steps} total steps")
        elif self.max_steps is None:
            raise ValueError("Must specify either --max_steps or --max_epochs")

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, f"cronusvla-{time_str}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            _log(f"Checkpoint directory: {ckpt_dir}")

        if self.use_wandb:
            wandb_run_name = self.wandb_name or f"cronusvla-{self.strategy}-{time_str}"
            wandb.init(
                project=self.wandb_project,
                name=wandb_run_name,
                entity=self.wandb_entity,
                config={
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "max_steps": self.max_steps,
                    "max_epochs": self.max_epochs,
                    "batch_size": train_loader.batch_size,
                    "strategy": self.strategy,
                    "world_size": self.world_size,
                    "vlm_base": self.config.vlm_base,
                    "action_model_type": self.config.action_model_type,
                    "repeated_diffusion_steps": self.config.repeated_diffusion_steps,
                },
            )

        _log(f"Starting training from step {start_step} to {self.max_steps}")

        data_yield_start = time.monotonic()
        data_iter = iter(train_loader)
        self.current_epoch = start_step // steps_per_epoch

        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Epoch boundary
                self.current_epoch += 1
                if self.is_distributed and hasattr(train_loader, "sampler"):
                    train_loader.sampler.set_epoch(self.current_epoch)
                data_iter = iter(train_loader)
                _log(f"Epoch {self.current_epoch} completed, starting epoch {self.current_epoch + 1}")

                # Epoch-based checkpointing
                if self.save_every_n_epochs and self.current_epoch % self.save_every_n_epochs == 0:
                    self.save_checkpoint(ckpt_dir, self.global_step, epoch=self.current_epoch)

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
                    epoch_str = f" Epoch={self.current_epoch + 1}/{self.max_epochs}" if self.max_epochs else ""

                    _log(
                        f"[Step {self.global_step}/{self.max_steps}]{epoch_str} "
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
                            "train/epoch": self.current_epoch + 1,
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

            # Step-based checkpointing
            if self.global_step % self.save_every_n_steps == 0:
                self.save_checkpoint(ckpt_dir, self.global_step)

            data_yield_start = time.monotonic()

        self.save_checkpoint(ckpt_dir, self.global_step, is_final=True,
                             epoch=self.current_epoch + 1 if self.max_epochs else None)
        _log(f"Training completed! Final checkpoint saved at step {self.global_step}")

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, ckpt_dir, step, is_final=False, epoch=None):
        """Save checkpoint."""
        if not self.is_main_process:
            return

        # Build filename
        if is_final:
            fname = "final.pt"
        elif epoch is not None:
            fname = f"epoch-{epoch}.pt"
        else:
            fname = f"step-{step}.pt"

        if self.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.model.state_dict()
            path = os.path.join(ckpt_dir, fname)
            torch.save({"model_state_dict": state_dict, "global_step": step, "epoch": epoch}, path)
        else:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            path = os.path.join(ckpt_dir, fname)
            torch.save({
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": step,
                "epoch": epoch,
            }, path)

        tag = "final" if is_final else fname.replace(".pt", "")
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
        self.current_epoch = ckpt.get("epoch", 0) or 0
        _log(f"Checkpoint loaded from {path}, resuming from step {self.global_step}, epoch {self.current_epoch}")


def parse_args():
    parser = argparse.ArgumentParser(description="CronusVLA Fine-tuning")

    # Dataset
    parser.add_argument("--dataset_repo_id", type=str, default=None, help="LeRobot dataset repo ID")
    parser.add_argument("--dataset_root", type=str, default=None, help="Local dataset root path")

    # Training
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    train_mode = parser.add_mutually_exclusive_group()
    train_mode.add_argument("--max_steps", type=int, default=None, help="Max training steps (step-based mode)")
    train_mode.add_argument("--max_epochs", type=int, default=None, help="Max training epochs (epoch-based mode)")
    parser.add_argument("--num_workers", type=int, default=8)

    # CronusVLA-specific model args
    parser.add_argument("--vlm_base", type=str, default="prism-dinosiglip-224px+7b",
                        help="PrismaticVLM base model identifier (resolved via prismatic model registry)")
    parser.add_argument("--action_model_type", type=str, default="DiT-B",
                        choices=["DiT-S", "DiT-B", "DiT-L"],
                        help="DiT diffusion action decoder model size")
    parser.add_argument("--repeated_diffusion_steps", type=int, default=4,
                        help="Number of diffusion step repeats per training batch for variance reduction")
    parser.add_argument("--future_action_window_size", type=int, default=15,
                        help="Number of future action steps to predict")
    parser.add_argument("--past_action_window_size", type=int, default=0,
                        help="Number of past action steps for temporal context")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale for inference")
    parser.add_argument("--use_ddim", action="store_true", default=False,
                        help="Use DDIM sampling for inference instead of DDPM")
    parser.add_argument("--view_sequence_len", type=int, default=1,
                        help="Number of views per timestep (1=primary only, 2=primary+wrist)")
    parser.add_argument("--use_wrist_image", action="store_true", default=False,
                        help="Use wrist camera images alongside primary camera")
    parser.add_argument("--freeze_vision_backbone", action="store_true", default=True,
                        help="Freeze VLM vision backbone during training")
    parser.add_argument("--no_freeze_vision_backbone", action="store_true", default=False,
                        help="Unfreeze VLM vision backbone (full finetune)")
    parser.add_argument("--freeze_llm_backbone", action="store_true", default=True,
                        help="Freeze VLM LLM backbone during training")
    parser.add_argument("--no_freeze_llm_backbone", action="store_true", default=False,
                        help="Unfreeze VLM LLM backbone (full finetune)")
    parser.add_argument("--unfreeze_last_llm_layer", action="store_true", default=False,
                        help="Unfreeze the last LLM decoder layer for finetune")
    parser.add_argument("--action_dim", type=int, default=7,
                        help="Action dimensionality")
    parser.add_argument("--hidden_size", type=int, default=4096,
                        help="VLM hidden size (must match LLM lm_head.in_features)")
    parser.add_argument("--diffusion_steps", type=int, default=100,
                        help="Number of diffusion steps for noise schedule")
    parser.add_argument("--scheduler", type=str, default="constant", choices=["constant", "cosine"],
                        help="LR scheduler type: 'constant' (warmup then flat) or 'cosine' (warmup then cosine decay)")
    parser.add_argument("--scheduler_warmup_steps", type=int, default=250)

    # Distributed
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--fsdp_sharding", type=str, default="full_shard", choices=["full_shard", "shard_grad_op"])
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # Checkpoint / logging
    parser.add_argument("--ckpt_dir", type=str, default="runs/checkpoints")
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every N epochs (epoch-based mode)")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to pretrained CronusVLA checkpoint (.pt). "
                             "Missing keys stay randomly initialized.")
    parser.add_argument("--local_vlm_path", type=str, default=None,
                        help="Path to local .pt checkpoint for VLM weights. When set, builds VLM in "
                             "inference_mode (no HF Hub LLM download) and loads weights from file. "
                             "Format: {'model': {'vision_backbone': ..., 'llm_backbone': ..., "
                             "'projector': ..., 'action_model': ...}}")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token for gated models (e.g., Llama-2-7b-hf)")
    parser.add_argument("--resume", type=str, default=None)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="cronusvla")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    dist_info = setup_distributed()

    # Resolve freeze flags (handle --no_freeze_* overrides)
    freeze_vision = not args.no_freeze_vision_backbone if args.no_freeze_vision_backbone else args.freeze_vision_backbone
    freeze_llm = not args.no_freeze_llm_backbone if args.no_freeze_llm_backbone else args.freeze_llm_backbone

    # Build config
    config = CronusVLAConfig(
        vlm_base=args.vlm_base,
        action_model_type=args.action_model_type,
        repeated_diffusion_steps=args.repeated_diffusion_steps,
        future_action_window_size=args.future_action_window_size,
        past_action_window_size=args.past_action_window_size,
        cfg_scale=args.cfg_scale,
        use_ddim=args.use_ddim,
        view_sequence_len=args.view_sequence_len,
        use_wrist_image=args.use_wrist_image,
        freeze_vision_backbone=freeze_vision,
        freeze_llm_backbone=freeze_llm,
        unfreeze_last_llm_layer=args.unfreeze_last_llm_layer,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        diffusion_steps=args.diffusion_steps,
        optimizer_lr=args.learning_rate,
        optimizer_weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        local_vlm_path=args.local_vlm_path,
        hf_token=args.hf_token,
    )

    # Setup model first (need tokenizer + image_transform from VLM for dataset)
    trainer = CronusVLATrainer(
        config=config,
        dist_info=dist_info,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        strategy=args.strategy,
        fsdp_sharding=args.fsdp_sharding,
        gradient_clip_val=args.gradient_clip_val,
        ckpt_dir=args.ckpt_dir,
        save_every_n_steps=args.save_every_n_steps,
        save_every_n_epochs=args.save_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
    )

    if args.disable_wandb:
        trainer.use_wandb = False

    trainer.setup_model(pretrained_checkpoint=args.pretrained_checkpoint)

    # Extract tokenizer and image_transform from the PrismaticVLM inside CronusVLAModel
    tokenizer = trainer.policy.model.vlm.llm_backbone.tokenizer
    image_transform = trainer.policy.model.vlm.vision_backbone.image_transform

    # Create CronusVLADataset with tokenizer and image_transform from the model
    train_dataset = CronusVLADataset(
        repo_id=args.dataset_repo_id,
        config=config,
        root=args.dataset_root,
        tokenizer=tokenizer,
        image_transform=image_transform,
    )

    # Setup dataset features in config (needed by PreTrainedPolicy)
    dataset_metadata = train_dataset.meta
    features = dataset_to_policy_features(dataset_metadata.features)
    if not config.output_features:
        config.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    if not config.input_features:
        config.input_features = {k: ft for k, ft in features.items() if k not in config.output_features}
    config.validate_features()

    # Set norm_stats on the CronusVLA model from dataset metadata
    # CronusVLA uses quantile-based (BOUNDS_Q99) normalization for actions
    # Build norm_stats dict from dataset metadata stats
    norm_stats = {}
    repo_id_key = args.dataset_repo_id
    if dataset_metadata.stats is not None and "action" in dataset_metadata.stats:
        action_stats = dataset_metadata.stats["action"]
        norm_stats[repo_id_key] = {
            "action": {
                "q01": action_stats.get("q01", [0] * config.action_dim),
                "q99": action_stats.get("q99", [1] * config.action_dim),
                "mask": [True] * config.action_dim,
            }
        }
    trainer.policy.model.norm_stats = norm_stats
    _log(f"Set norm_stats on model from dataset metadata (key: {repo_id_key})")

    # Create DataLoader with custom collater
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
        collate_fn=train_dataset.collater,
    )

    _log(f"Dataset size: {len(train_dataset)}, batch_size: {args.batch_size}, "
         f"batches_per_epoch: {len(train_loader)}")

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