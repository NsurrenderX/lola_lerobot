#!/usr/bin/env python
"""
LoLA 分布式流式训练脚本

使用 LoLAStreamingDataset / LoLAPretrainStreamingDataset 从远程存储流式加载数据。
适用于数据通过挂载方式访问的场景，无需将整个数据集下载到本地。

与 train_lola_azure.py 的区别：
- 使用 LoLAStreamingDataset（IterableDataset）替代 LoLADataset（map-style）
- 不需要 DistributedSampler（IterableDataset 自带分片）
- 适用于远程挂载存储（Azure Blob 等）
- LoLATrainer 内联定义，无需跨脚本导入

使用方法:
    # 单 GPU
    python src/lerobot/scripts/train_lola_azure_stream.py \
        --dataset_root /mnt/data/lerobot-dataset --pretrain \
        --dataset_to_episodes_path /mnt/data/dataset_to_episodes.json

    # 多 GPU (torchrun)
    torchrun --nproc_per_node=4 src/lerobot/scripts/train_lola_azure_stream.py \
        --dataset_root /mnt/data/lerobot-dataset --strategy ddp
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
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset, AsyncDecodeDataLoader as StreamingAsyncDecodeDataLoader
from lerobot.datasets.lola_pretrain_streaming_dataset import LoLAPretrainStreamingDataset, AsyncDecodeDataLoader as PretrainAsyncDecodeDataLoader
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [Rank {os.environ.get('RANK', '0')}] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """从环境变量初始化分布式训练"""
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
        logger.info(f"Distributed initialized: rank={world_rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        logger.info(f"Single GPU mode: local_rank={local_rank}")

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


class InterconnectMonitor:
    """Monitor NVLink, PCIe, and InfiniBand throughput via NVML and sysfs."""

    def __init__(self, device: torch.device):
        self.available = HAS_NVML
        if not self.available:
            logger.info("InterconnectMonitor: pynvml not available, skipping interconnect metrics")
            return

        self.gpu_index = device.index or 0
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as e:
            logger.warning(f"InterconnectMonitor: NVML init failed ({e}), skipping")
            self.available = False
            return

        # Detect NVLink capability
        self._nvlink_supported = True
        self._active_nvlink_links = []
        for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
            try:
                state = pynvml.nvmlDeviceGetNvLinkState(self.handle, link)
                if state == pynvml.NVML_NVLINK_STATE_ACTIVE:
                    self._active_nvlink_links.append(link)
            except pynvml.NVMLError:
                pass

        # Pre-check NVLink byte counter fields
        if self._active_nvlink_links:
            try:
                vals = pynvml.nvmlDeviceGetFieldValues(self.handle, [
                    pynvml.NVML_FI_DEV_NVLINK_COUNT_RCV_BYTES,
                    pynvml.NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES,
                ])
                if any(v.nvmlReturn != 0 for v in vals):
                    logger.info("InterconnectMonitor: NVLink byte counters not supported, skipping NVLink metrics")
                    self._nvlink_supported = False
            except Exception:
                self._nvlink_supported = False
        else:
            logger.info(f"InterconnectMonitor: No active NVLink links (GPU {self.gpu_index}), skipping NVLink metrics")
            self._nvlink_supported = False

        # Pre-check PCIe byte counter fields
        self._pcie_supported = True
        try:
            vals = pynvml.nvmlDeviceGetFieldValues(self.handle, [
                pynvml.NVML_FI_DEV_PCIE_COUNT_RX_BYTES,
                pynvml.NVML_FI_DEV_PCIE_COUNT_TX_BYTES,
            ])
            if any(v.nvmlReturn != 0 for v in vals):
                logger.info("InterconnectMonitor: PCIe byte counters not supported, will use nvmlDeviceGetPcieThroughput")
                self._pcie_supported = False
        except Exception:
            self._pcie_supported = False

        # Discover IB devices from sysfs
        self._ib_supported = True
        self._ib_counter_paths = []
        ib_base = "/sys/class/infiniband"
        try:
            ib_devs = os.listdir(ib_base)
        except OSError:
            ib_devs = []

        for dev_name in ib_devs:
            dev_path = os.path.join(ib_base, dev_name)
            try:
                ports = os.listdir(os.path.join(dev_path, "ports"))
            except OSError:
                continue
            for port_name in ports:
                counters_dir = os.path.join(dev_path, "ports", port_name, "counters")
                rcv_path = os.path.join(counters_dir, "port_rcv_data")
                xmit_path = os.path.join(counters_dir, "port_xmit_data")
                if os.path.isfile(rcv_path) and os.path.isfile(xmit_path):
                    self._ib_counter_paths.append((rcv_path, xmit_path))

        if not self._ib_counter_paths:
            logger.info("InterconnectMonitor: No IB devices found, skipping IB metrics")
            self._ib_supported = False

        # State for delta computation
        self._prev_pcie_rx = None
        self._prev_pcie_tx = None
        self._prev_nvlink_rcv = None
        self._prev_nvlink_xmit = None
        self._prev_ib_rcv = None
        self._prev_ib_xmit = None
        self._prev_timestamp = None

        logger.info(
            f"InterconnectMonitor initialized for GPU {self.gpu_index}: "
            f"PCIe={self._pcie_supported} NVLink={self._nvlink_supported} IB={self._ib_supported}"
        )

    def snapshot(self) -> dict:
        """Take a snapshot and compute throughput from delta with previous snapshot."""
        if not self.available:
            return {}

        now = time.monotonic()
        metrics = {}

        # PCIe throughput
        if self._pcie_supported:
            try:
                vals = pynvml.nvmlDeviceGetFieldValues(self.handle, [
                    pynvml.NVML_FI_DEV_PCIE_COUNT_RX_BYTES,
                    pynvml.NVML_FI_DEV_PCIE_COUNT_TX_BYTES,
                ])
                rx = vals[0].value.ullVal
                tx = vals[1].value.ullVal
                if self._prev_pcie_rx is not None and self._prev_timestamp is not None:
                    dt = now - self._prev_timestamp
                    if dt > 0:
                        metrics["pcie_rx_gb_s"] = (rx - self._prev_pcie_rx) / dt / 1e9
                        metrics["pcie_tx_gb_s"] = (tx - self._prev_pcie_tx) / dt / 1e9
                self._prev_pcie_rx = rx
                self._prev_pcie_tx = tx
            except Exception:
                pass

        if not self._pcie_supported and not metrics:
            # Fallback: use instantaneous PCIe throughput (KB/s -> GB/s)
            try:
                rx_kbs = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                tx_kbs = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                metrics["pcie_rx_gb_s"] = rx_kbs / 1e6
                metrics["pcie_tx_gb_s"] = tx_kbs / 1e6
            except Exception:
                pass

        # NVLink throughput
        if self._nvlink_supported:
            try:
                vals = pynvml.nvmlDeviceGetFieldValues(self.handle, [
                    pynvml.NVML_FI_DEV_NVLINK_COUNT_RCV_BYTES,
                    pynvml.NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES,
                ])
                rcv = vals[0].value.ullVal
                xmit = vals[1].value.ullVal
                if self._prev_nvlink_rcv is not None and self._prev_timestamp is not None:
                    dt = now - self._prev_timestamp
                    if dt > 0:
                        metrics["nvlink_rx_gb_s"] = (rcv - self._prev_nvlink_rcv) / dt / 1e9
                        metrics["nvlink_tx_gb_s"] = (xmit - self._prev_nvlink_xmit) / dt / 1e9
                self._prev_nvlink_rcv = rcv
                self._prev_nvlink_xmit = xmit
            except Exception:
                pass

        # IB throughput
        if self._ib_supported:
            try:
                total_rcv = 0
                total_xmit = 0
                for rcv_path, xmit_path in self._ib_counter_paths:
                    with open(rcv_path) as f:
                        total_rcv += int(f.read().strip())
                    with open(xmit_path) as f:
                        total_xmit += int(f.read().strip())
                if self._prev_ib_rcv is not None and self._prev_timestamp is not None:
                    dt = now - self._prev_timestamp
                    if dt > 0:
                        metrics["ib_rx_gb_s"] = (total_rcv - self._prev_ib_rcv) / dt / 1e9
                        metrics["ib_tx_gb_s"] = (total_xmit - self._prev_ib_xmit) / dt / 1e9
                self._prev_ib_rcv = total_rcv
                self._prev_ib_xmit = total_xmit
            except Exception:
                pass

        self._prev_timestamp = now
        return metrics

    def close(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.available = False


def create_lola_streaming_dataset(
    repo_id: str,
    config: LoLAConfig,
    root: str | None = None,
    episodes: list | None = None,
    image_transforms=None,
    max_history_length: int = 100,
    history_padding_side: str = "left",
    streaming: bool = True,
    buffer_size: int = 1000,
    seed: int = 42,
    shuffle: bool = True,
    deferred_video_decode: bool = True,
    async_decode: bool = False,
    decode_device: str = "cpu",
    decode_num_threads: int = 1,
    num_dataloader_workers: int = 0,
    start_index: int = 0,
):
    """创建 LoLA 流式数据集"""
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = dataset_metadata.fps

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    logger.info(f"delta_timestamps: {delta_timestamps}")

    dataset = LoLAStreamingDataset(
        repo_id=repo_id,
        max_history_length=max_history_length,
        action_chunk_size=config.action_chunk_size,
        history_padding_side=history_padding_side,
        root=root,
        episodes=episodes,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
        streaming=streaming,
        buffer_size=buffer_size,
        seed=seed,
        shuffle=shuffle,
        deferred_video_decode=deferred_video_decode,
        async_decode=async_decode,
        decode_device=decode_device,
        decode_num_threads=decode_num_threads,
        num_dataloader_workers=num_dataloader_workers,
        start_index=start_index,
    )

    return dataset


def create_lola_pretrain_streaming_dataset(
    repo_id: str,
    config: LoLAConfig,
    root: str | None = None,
    sub_root: str | None = None,
    episodes: list | None = None,
    image_transforms=None,
    max_history_length: int = 100,
    history_padding_side: str = "left",
    streaming: bool = True,
    buffer_size: int = 1000,
    seed: int = 42,
    shuffle: bool = True,
    deferred_video_decode: bool = True,
    async_decode: bool = False,
    decode_device: str = "cpu",
    decode_num_threads: int = 1,
    num_dataloader_workers: int = 0,
    dataset_to_episodes_path: str | None = None,
    temp_process: bool = False,
    episode_chunk_size: int = 8,
    start_index: int = 0,
):
    """创建 LoLA 预训练流式数据集（支持多子数据集 per-dataset 归一化）"""
    dataset_metadata = LoLAPretrainStreamingDataset._build_metadata_polars(repo_id, root=root, revision=None)
    fps = dataset_metadata.fps

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    logger.info(f"delta_timestamps: {delta_timestamps}")

    dataset = LoLAPretrainStreamingDataset(
        repo_id=repo_id,
        max_history_length=max_history_length,
        action_chunk_size=config.action_chunk_size,
        history_padding_side=history_padding_side,
        root=root,
        sub_root=sub_root,
        episodes=episodes,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
        streaming=streaming,
        buffer_size=buffer_size,
        seed=seed,
        shuffle=shuffle,
        deferred_video_decode=deferred_video_decode,
        async_decode=async_decode,
        decode_device=decode_device,
        decode_num_threads=decode_num_threads,
        num_dataloader_workers=num_dataloader_workers,
        dataset_to_episodes_path=dataset_to_episodes_path,
        temp_process=temp_process,
        episode_chunk_size=episode_chunk_size,
        start_index=start_index,
    )

    return dataset


# ----------------------------------------------------------------------
# 训练器（内联定义，自包含，不依赖 train_lola_azure.py）
# ----------------------------------------------------------------------
class LoLATrainer:
    """原生 PyTorch 训练器，支持 DDP 和 FSDP，增强 wandb 日志"""

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
        wandb_project: str = "lola-azure-stream",
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

        self.policy = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.preprocessor = None
        self.postprocessor = None

        self.use_bf16 = True
        self.scaler = None if self.use_bf16 else torch.amp.GradScaler("cuda")

        self.global_step = 0
        self.best_loss = float("inf")
        self.interconnect_monitor = None

    def setup_model(self):
        """设置模型"""
        logger.info(f"Loading LoLA Policy on {self.device}...")

        self.policy = LoLAPolicy(self.config)
        self.policy._device = self.device
        self.policy.model = self.policy.model.to(self.device)
        self.policy.vlm = self.policy.vlm.to(self.device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            dataset_stats=self.dataset_stats,
        )

        if not self.train_vlm and hasattr(self.policy, "vlm"):
            logger.info("Freezing VLM parameters...")
            for param in self.policy.vlm.parameters():
                param.requires_grad = False
            self.policy.vlm.eval()

        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")

        if self.is_distributed:
            cap = torch.cuda.get_device_capability(self.device)
            torch_cuda_ver = torch.version.cuda
            logger.info(f"GPU compute capability: sm_{cap[0]}{cap[1]}, torch CUDA: {torch_cuda_ver}")
            if self.strategy == "fsdp":
                self._setup_fsdp()
            else:
                self._setup_ddp()
        else:
            self.model = self.policy

        self.interconnect_monitor = InterconnectMonitor(self.device)

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

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
            module, recurse, nonwrapped_numel,
            transformer_layer_cls={Qwen3_5DecoderLayer, Qwen3_5VisionBlock}
        )

        self.model = FSDP(
            self.policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
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
        """提取特殊字段（action / hist_actions_*），绕过 preprocessor"""
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
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        special_data = self._extract_special_fields(batch)

        batch = self.preprocessor(batch)
        batch = self._restore_special_fields(batch, special_data)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = self.model(batch)

        return loss, loss_dict

    def train(self, train_loader, start_step: int = 0):
        """训练循环，增强 wandb 日志（throughput / timing / GPU metrics）"""
        self.global_step = start_step
        self.model.train()

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, f"lola-azure-stream-{time_str}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Checkpoint directory: {ckpt_dir}")

        if self.use_wandb:
            wandb_run_name = self.wandb_name or f"lola-stream-{self.strategy}-{time_str}"
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

        # 数据跳过由 dataset.start_index 在 __iter__ 内部处理，
        # 这里无需 skip_epochs/skip_batches 逻辑

        for batch_idx, batch in enumerate(train_loader):
            if self.global_step >= self.max_steps:
                break

            step_start = time.monotonic()

            self.optimizer.zero_grad()

            # ── Forward pass (with timing) ───────────────────────
            fwd_start = time.monotonic()
            loss, loss_dict = self.training_step(batch)
            fwd_s = time.monotonic() - fwd_start

            # ── Backward pass (with timing) ──────────────────────
            bwd_start = time.monotonic()
            if self.use_bf16:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()
            bwd_s = time.monotonic() - bwd_start

            # ── Gradient clipping ─────────────────────────────────
            clip_start = time.monotonic()
            if self.gradient_clip_val > 0:
                if not self.use_bf16:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val,
                )
            else:
                grad_norm = None
            clip_s = time.monotonic() - clip_start

            # ── Optimizer step ────────────────────────────────────
            opt_start = time.monotonic()
            if self.use_bf16:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            opt_s = time.monotonic() - opt_start

            self.scheduler.step()

            self.global_step += 1

            update_s = time.monotonic() - step_start
            batch_per_s = 1.0 / update_s if update_s > 0 else 0

            # ── Logging (enhanced wandb metrics) ──────────────────
            if self.global_step % self.log_every_n_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                gpu_mem_alloc = torch.cuda.memory_allocated(self.device) / 1e9
                gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9
                interconnect_metrics = self.interconnect_monitor.snapshot() if self.interconnect_monitor else {}

                if self.is_main_process:
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm if grad_norm is not None else None

                    # ── Console logging (mirrors all wandb metrics) ──
                    logger.info(
                        f"[Step {self.global_step}/{self.max_steps}] "
                        f"Loss={loss.item():.4f} LR={lr:.2e} "
                        f"Update={update_s:.2f}s Throughput={batch_per_s:.2f}batch/s"
                    )
                    print(
                        f"[Step {self.global_step}/{self.max_steps}] "
                        f"Loss={loss.item():.4f} LR={lr:.2e} "
                        f"Update={update_s:.2f}s Throughput={batch_per_s:.2f}batch/s"
                    )
                    if grad_norm_val is not None:
                        logger.info(f"  grad_norm={grad_norm_val:.4f}")
                        print(f"  grad_norm={grad_norm_val:.4f}")
                    logger.info(
                        f"  Timing: fwd={fwd_s:.3f}s bwd={bwd_s:.3f}s "
                        f"clip={clip_s:.3f}s opt={opt_s:.3f}s"
                    )
                    print(
                        f"  Timing: fwd={fwd_s:.3f}s bwd={bwd_s:.3f}s "
                        f"clip={clip_s:.3f}s opt={opt_s:.3f}s"
                    )
                    logger.info(
                        f"  GPU: alloc={gpu_mem_alloc:.1f}GB "
                        f"reserved={gpu_mem_reserved:.1f}GB"
                    )
                    print(
                        f"  GPU: alloc={gpu_mem_alloc:.1f}GB "
                        f"reserved={gpu_mem_reserved:.1f}GB"
                        )
                    if interconnect_metrics:
                        parts = []
                        if "pcie_rx_gb_s" in interconnect_metrics:
                            parts.append(
                                f"PCIe rx={interconnect_metrics['pcie_rx_gb_s']:.2f} "
                                f"tx={interconnect_metrics['pcie_tx_gb_s']:.2f} GB/s"
                            )
                        if "nvlink_rx_gb_s" in interconnect_metrics:
                            parts.append(
                                f"NVLink rx={interconnect_metrics['nvlink_rx_gb_s']:.2f} "
                                f"tx={interconnect_metrics['nvlink_tx_gb_s']:.2f} GB/s"
                            )
                        if "ib_rx_gb_s" in interconnect_metrics:
                            parts.append(
                                f"IB rx={interconnect_metrics['ib_rx_gb_s']:.2f} "
                                f"tx={interconnect_metrics['ib_tx_gb_s']:.2f} GB/s"
                            )
                        logger.info(f"  Interconnect: {' | '.join(parts)}")
                        print(f"  Interconnect: {' | '.join(parts)}")
                    for k, v in loss_dict.items():
                        if k != "loss" and isinstance(v, (int, float)):
                            logger.info(f"  {k}={v:.4f}")
                            print(f"  {k}={v:.4f}")

                    # ── Wandb logging ──────────────────────────────
                    if self.use_wandb:
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/learning_rate": lr,
                            "train/step": self.global_step,
                            "train/update_s": update_s,
                            "train/batch_per_s": batch_per_s,
                            "train/fwd_s": fwd_s,
                            "train/bwd_s": bwd_s,
                            "train/clip_s": clip_s,
                            "train/opt_s": opt_s,
                            "train/gpu_mem_alloc_gb": gpu_mem_alloc,
                            "train/gpu_mem_reserved_gb": gpu_mem_reserved,
                        }
                        if grad_norm_val is not None:
                            log_dict["train/grad_norm"] = grad_norm_val
                        for k, v in loss_dict.items():
                            if k != "loss" and isinstance(v, (int, float)):
                                log_dict[f"train/{k}"] = v
                        for k, v in interconnect_metrics.items():
                            log_dict[f"interconnect/{k}"] = v
                        wandb.log(log_dict)

            if self.global_step % self.save_every_n_steps == 0:
                self.save_checkpoint(ckpt_dir, self.global_step)

        self.save_checkpoint(ckpt_dir, self.global_step, is_final=True)
        logger.info(f"Training completed! Final checkpoint saved at step {self.global_step}")

        if self.interconnect_monitor:
            self.interconnect_monitor.close()

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, ckpt_dir: str, step: int, is_final: bool = False):
        """保存 checkpoint"""
        if self.strategy == "fsdp":
            from torch.distributed.checkpoint import save as save_fsdp_checkpoint
            from torch.distributed.checkpoint.state_dict import get_state_dict

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
            if self.is_main_process:
                torch.save(
                    {"scheduler_state_dict": self.scheduler.state_dict()},
                    os.path.join(ckpt_path, "scheduler.pt"),
                )
        else:
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

            model_sd, optimizer_sd = get_state_dict(self.model, self.optimizer)
            step_container = [0]
            load_fsdp_checkpoint(
                {"model": model_sd, "optimizer": optimizer_sd, "step": step_container},
                checkpoint_id=ckpt_path,
            )
            set_state_dict(self.model, self.optimizer, model_state_dict=model_sd, optim_state_dict=optimizer_sd)
            self.global_step = step_container[0]
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


def main():
    dist_info = setup_distributed()

    parser = argparse.ArgumentParser(description="LoLA Azure Streaming Training")

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
    parser.add_argument("--resume", type=str, default=None)

    # LoLA 参数
    parser.add_argument("--action_dim", type=int, default=20)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)

    # 历史 action 参数
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"])

    # 流式数据集参数
    parser.add_argument("--buffer_size", type=int, default=1000, help="Streaming shuffle buffer size")
    parser.add_argument("--streaming_seed", type=int, default=42, help="Streaming dataset seed")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable streaming shuffle")

    # Wandb 参数
    parser.add_argument("--wandb_project", type=str, default="lola-azure-stream")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    # 视频解码参数
    parser.add_argument("--no_deferred", action="store_true",
                        help="关闭延迟视频解码 (worker 内解码，速度快但 flush 阶段内存峰值)")
    parser.add_argument("--async_decode", action="store_true",
                        help="启用异步解码管线 (独立子进程解码 + 持久化大缓存)")
    parser.add_argument("--decode_device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="视频解码设备 (cpu 或 cuda)")
    parser.add_argument("--decode_num_threads", type=int, default=1,
                        help="异步管线解码线程数")

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4)

    # 预训练参数
    parser.add_argument("--pretrain", action="store_true",
                        help="启用预训练模式（使用 LoLAPretrainStreamingDataset，per-sub-dataset 归一化）")
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None,
                        help="dataset_to_episodes.json 路径（预训练模式必须）")
    parser.add_argument("--sub_root", type=str, default=None,
                        help="子数据集 stats.json 的根目录（预训练模式，默认与 root 相同）")
    parser.add_argument("--temp_process", action="store_true",
                        help="预训练模式：允许维度不匹配时用 0/1 填充归一化 stats")
    parser.add_argument("--episode_chunk_size", type=int, default=8,
                        help="预训练模式：每次加载的连续 episode 数（影响 I/O 效率）")

    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    if dist_info["world_rank"] == 0:
        logger.info("=" * 60)
        logger.info("LoLA Azure Streaming Training")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset_repo_id or args.dataset_root}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"World Size: {dist_info['world_size']}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Streaming: True")
        logger.info(f"Buffer Size: {args.buffer_size}")
        logger.info(f"VLM Path: {args.vlm_path}")
        if args.pretrain:
            logger.info(f"Pretrain Mode: True")
            logger.info(f"Dataset to Episodes: {args.dataset_to_episodes_path}")
            logger.info(f"Sub Root: {args.sub_root}")
            logger.info(f"Temp Process: {args.temp_process}")
            logger.info(f"Episode Chunk Size: {args.episode_chunk_size}")
        logger.info("=" * 60)

    # 获取数据集元数据
    logger.info("Loading dataset metadata...")
    dataset_metadata = LoLAPretrainStreamingDataset._build_metadata_polars(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=None,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    logger.info(f"Dataset: {dataset_metadata.total_episodes} episodes, {dataset_metadata.total_frames} frames")

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
        load_full_history=True,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
    )

    # 预训练模式：归一化由 dataset 内部完成，processor 使用 IDENTITY
    if args.pretrain:
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }

    # 创建流式数据集
    AsyncLoaderClass = PretrainAsyncDecodeDataLoader if args.pretrain else StreamingAsyncDecodeDataLoader

    if args.pretrain:
        # dataset_to_episodes_path is optional for local testing without per-sub-dataset normalization
        logger.info("Creating pretrain streaming dataset...")
        train_dataset = create_lola_pretrain_streaming_dataset(
            repo_id=args.dataset_repo_id,
            config=config,
            root=args.dataset_root,
            sub_root=args.sub_root,
            episodes=args.episodes,
            max_history_length=args.max_history_length,
            history_padding_side=args.history_padding_side,
            buffer_size=args.buffer_size,
            seed=args.streaming_seed,
            shuffle=not args.no_shuffle,
            deferred_video_decode=not args.no_deferred,
            async_decode=args.async_decode,
            decode_device=args.decode_device,
            decode_num_threads=args.decode_num_threads,
            num_dataloader_workers=args.num_workers,
            dataset_to_episodes_path=args.dataset_to_episodes_path,
            temp_process=args.temp_process,
            episode_chunk_size=args.episode_chunk_size,
        )
    else:
        logger.info("Creating streaming dataset...")
        train_dataset = create_lola_streaming_dataset(
            repo_id=args.dataset_repo_id,
            config=config,
            root=args.dataset_root,
            episodes=args.episodes,
            max_history_length=args.max_history_length,
            history_padding_side=args.history_padding_side,
            buffer_size=args.buffer_size,
            seed=args.streaming_seed,
            shuffle=not args.no_shuffle,
            deferred_video_decode=not args.no_deferred,
            async_decode=args.async_decode,
            decode_device=args.decode_device,
            decode_num_threads=args.decode_num_threads,
            num_dataloader_workers=args.num_workers,
        )

    # IterableDataset 数据跳过由 dataset.start_index 内部处理，
    # 需要在 load_checkpoint 之后才能计算 start_index，所以 DataLoader 创建延后

    # 创建训练器
    dataset_stats = {} if args.pretrain else dataset_metadata.stats
    trainer = LoLATrainer(
        config=config,
        dataset_stats=dataset_stats,
        dist_info=dist_info,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        train_vlm=args.train_vlm,
        strategy=args.strategy,
        gradient_clip_val=args.gradient_clip_val,
        ckpt_dir=args.ckpt_dir,
        save_every_n_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        wandb_id=args.wandb_id,
    )

    if args.disable_wandb:
        trainer.use_wandb = False

    trainer.setup_model()
    trainer.setup_optimizer()

    start_step = 0
    if args.resume:
        trainer.load_checkpoint(args.resume)
        start_step = trainer.global_step

    # Resume: 设置数据跳过的 start_index（总样本数 = 步数 × batch_size × world_size）
    if start_step > 0:
        train_dataset.start_index = start_step * args.batch_size * dist_info.world_size
        logger.info(f"Resuming from step {start_step}, dataset.start_index = {train_dataset.start_index}")

    # 创建 DataLoader（必须在 start_index 设置之后，worker fork 时会读取 dataset 属性）
    raw_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )
    train_loader = AsyncLoaderClass(
        dataloader=raw_loader,
        dataset=train_dataset,
        collate_fn=AsyncLoaderClass.make_collate_fn(),
    )

    trainer.train(train_loader, start_step=start_step)

    cleanup_distributed()
    logger.info("Training completed!")


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = "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"
    
    main()