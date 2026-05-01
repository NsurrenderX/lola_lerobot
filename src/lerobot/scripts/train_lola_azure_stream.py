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
from contextlib import nullcontext
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


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rank = os.environ.get("RANK", "0")
    print(f"[{ts}] [Rank {rank}] {msg}", flush=True)


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


class InterconnectMonitor:
    """Monitor NVLink, PCIe, and InfiniBand throughput via NVML and sysfs."""

    def __init__(self, device: torch.device):
        self.available = HAS_NVML
        if not self.available:
            _log("InterconnectMonitor: pynvml not available, skipping interconnect metrics")
            return

        self.gpu_index = device.index or 0
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as e:
            _log(f"InterconnectMonitor: NVML init failed ({e}), skipping")
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
                    _log("InterconnectMonitor: NVLink byte counters not supported, skipping NVLink metrics")
                    self._nvlink_supported = False
            except Exception:
                self._nvlink_supported = False
        else:
            _log(f"InterconnectMonitor: No active NVLink links (GPU {self.gpu_index}), skipping NVLink metrics")
            self._nvlink_supported = False

        # Pre-check PCIe byte counter fields
        self._pcie_supported = True
        try:
            vals = pynvml.nvmlDeviceGetFieldValues(self.handle, [
                pynvml.NVML_FI_DEV_PCIE_COUNT_RX_BYTES,
                pynvml.NVML_FI_DEV_PCIE_COUNT_TX_BYTES,
            ])
            if any(v.nvmlReturn != 0 for v in vals):
                _log("InterconnectMonitor: PCIe byte counters not supported, will use nvmlDeviceGetPcieThroughput")
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
            _log("InterconnectMonitor: No IB devices found, skipping IB metrics")
            self._ib_supported = False

        # State for delta computation
        self._prev_pcie_rx = None
        self._prev_pcie_tx = None
        self._prev_nvlink_rcv = None
        self._prev_nvlink_xmit = None
        self._prev_ib_rcv = None
        self._prev_ib_xmit = None
        self._prev_timestamp = None

        _log(
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

    _log(f"delta_timestamps: {delta_timestamps}")

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
    tier_config_path: str | None = None,
    yield_tier: int | None = None,
):
    """创建 LoLA 预训练流式数据集（支持多子数据集 per-dataset 归一化）"""
    dataset_metadata = LoLAPretrainStreamingDataset._build_metadata_polars(repo_id, root=root, revision=None)
    fps = dataset_metadata.fps

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    _log(f"delta_timestamps: {delta_timestamps}")

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
        tolerance_frames=2,
        tier_config_path=tier_config_path,
        yield_tier=yield_tier,
    )

    return dataset


# ----------------------------------------------------------------------
# Tier 调度自动计算
# ----------------------------------------------------------------------
def compute_balanced_schedule(
    tier_stats: dict,
    tier_micro_batches: list[int],
    target_effective_batch: int,
    balance_mode: str = "frame_weighted",
    seed: int = 42,
):
    """自动计算 per-tier 累积步数，使各 tier 按指定权重平衡参与每个 optimizer step。

    Args:
        tier_stats: tier_config["tier_stats"] dict, keyed by tier index string
        tier_micro_batches: per-tier micro-batch size list
        target_effective_batch: 目标有效 batch size (总样本数)
        balance_mode: "frame_weighted" | "equal" | "episode_weighted"
        seed: 用于 shuffle accum_order 的随机种子

    Returns:
        dict with tier_accum_steps, accum_order, actual_effective_batch,
        balance_weights, balance_mode
    """
    import numpy as np

    num_tiers = len(tier_stats)

    # 1. Compute balance weights
    if balance_mode == "frame_weighted":
        total_frames = sum(tier_stats[str(t)]["frame_count"] for t in range(num_tiers))
        weights = [tier_stats[str(t)]["frame_count"] / total_frames for t in range(num_tiers)]
    elif balance_mode == "episode_weighted":
        total_eps = sum(tier_stats[str(t)]["episode_count"] for t in range(num_tiers))
        weights = [tier_stats[str(t)]["episode_count"] / total_eps for t in range(num_tiers)]
    else:  # equal
        weights = [1.0 / num_tiers] * num_tiers

    # 2. Compute ideal (float) accum steps per tier
    ideal_accum = [target_effective_batch * weights[t] / tier_micro_batches[t] for t in range(num_tiers)]

    # 3. Adaptive rounding: floor first, then distribute remainder
    #    respecting ratio bounds — skip tiers already over-represented
    accum = [max(1, int(a)) for a in ideal_accum]

    def _actual_ratio(accum_list):
        total_s = sum(m * a for m, a in zip(tier_micro_batches, accum_list))
        if total_s == 0:
            return [0.0] * num_tiers
        return [tier_micro_batches[t] * accum_list[t] / total_s for t in range(num_tiers)]

    current_total = sum(m * a for m, a in zip(tier_micro_batches, accum))

    # Greedily add or skip based on ratio deficit
    max_iterations = target_effective_batch  # safety bound
    for _ in range(max_iterations):
        if current_total >= target_effective_batch:
            break

        ratios = _actual_ratio(accum)
        # Find tier with largest deficit (actual < target)
        best_tier = -1
        best_deficit = -float("inf")
        for t in range(num_tiers):
            deficit = weights[t] - ratios[t]
            if deficit > best_deficit:
                best_deficit = deficit
                best_tier = t

        # Only add if adding doesn't push this tier over its target ratio
        # (i.e., the tier is still under-represented)
        if best_tier >= 0 and best_deficit > -1e-6:
            accum[best_tier] += 1
            current_total += tier_micro_batches[best_tier]
        else:
            # All tiers are at or above target ratio; add to the one closest to target
            # This handles the case where total is slightly under target but all ratios match
            smallest_excess = float("inf")
            for t in range(num_tiers):
                excess = ratios[t] - weights[t]
                if excess < smallest_excess:
                    smallest_excess = excess
                    best_tier = t
            if best_tier >= 0:
                accum[best_tier] += 1
                current_total += tier_micro_batches[best_tier]

    # 4. Build interleaved accum_order (round-robin proportional to accum steps)
    accum_order = []
    max_accum = max(accum)
    for step in range(max_accum):
        for t in range(num_tiers):
            if step < accum[t]:
                accum_order.append((t, tier_micro_batches[t]))

    # Shuffle for gradient mixing (deterministic with seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(accum_order)

    return {
        "tier_accum_steps": {str(t): accum[t] for t in range(num_tiers)},
        "accum_order": accum_order,
        "actual_effective_batch": current_total,
        "balance_weights": weights,
        "balance_mode": balance_mode,
    }


def compute_tier_schedule(
    tier_stats: dict,
    effective_batch_size: int,
    balance_mode: str = "frame_weighted",
    gpu_memory_budget_gb: float | None = None,
    tier_micro_batches_override: str | None = None,
    gpu_utilization_target: float = 0.92,
    calibration_path: str | None = None,
    seed: int = 42,
):
    """自动计算 per-tier micro-batch size 和累积调度。

    Args:
        tier_stats: tier_config["tier_stats"] dict
        effective_batch_size: 目标有效 batch size
        balance_mode: "frame_weighted" | "equal" | "episode_weighted"
        gpu_memory_budget_gb: 手动指定 GPU 显存预算 (GB)；None 则自动探测
        tier_micro_batches_override: 手动覆盖 micro-batch sizes, 逗号分隔 (e.g. "8,4,2")
        gpu_utilization_target: GPU 显存利用率目标 (default 0.92 = 92%)
        calibration_path: calibration_coefficients.json 路径
        seed: 用于 shuffle accum_order 的随机种子

    Returns:
        dict with tier_micro_batches, tier_accum_steps, accum_order,
        actual_effective_batch, balance_weights, balance_mode
    """
    num_tiers = len(tier_stats)

    # Step 1: Compute per-tier micro-batch sizes
    if tier_micro_batches_override:
        tier_micro_batches = [int(x) for x in tier_micro_batches_override.split(",")]
        if len(tier_micro_batches) != num_tiers:
            raise ValueError(
                f"tier_micro_batches_override has {len(tier_micro_batches)} values "
                f"but there are {num_tiers} tiers"
            )
    else:
        # Auto-detect GPU memory budget
        if gpu_memory_budget_gb is None:
            if torch.cuda.is_available():
                gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_budget_gb = gpu_total_bytes / 1e9 * gpu_utilization_target
            else:
                gpu_memory_budget_gb = 60.0 * gpu_utilization_target  # conservative default
                _log(f"CUDA not available, using default {gpu_memory_budget_gb:.1f}GB budget")

        # Load calibration to get bytes per equivalent token
        bytes_per_token = 4.0e6  # conservative default
        if calibration_path and os.path.isfile(calibration_path):
            try:
                import json as _json
                with open(calibration_path) as _f:
                    calib = _json.load(_f)
                vlm_calib_bs = calib.get("vlm_calib_batch_size", 4)
                text_slope = calib["measurements"]["text_token_slope_bytes"]
                bytes_per_token = text_slope / vlm_calib_bs
                _log(f"Calibration: bytes_per_equivalent_token={bytes_per_token:.0f} "
                     f"(text_slope={text_slope:.0f}, calib_bs={vlm_calib_bs})")
            except Exception as e:
                _log(f"Failed to load calibration from {calibration_path}: {e}, using default")

        budget_bytes = gpu_memory_budget_gb * 1e9
        tier_micro_batches = []
        for t in range(num_tiers):
            cost = tier_stats[str(t)]["avg_cost"]
            mb = max(1, int(budget_bytes / (cost * bytes_per_token)))
            tier_micro_batches.append(mb)
            _log(f"Tier {t}: avg_cost={cost:.0f}, micro_batch={mb} "
                 f"(budget={gpu_memory_budget_gb:.1f}GB, bytes/token={bytes_per_token:.0f})")

    # Step 2: Compute accumulation schedule
    schedule = compute_balanced_schedule(
        tier_stats=tier_stats,
        tier_micro_batches=tier_micro_batches,
        target_effective_batch=effective_batch_size,
        balance_mode=balance_mode,
        seed=seed,
    )
    schedule["tier_micro_batches"] = tier_micro_batches
    schedule["gpu_utilization_target"] = gpu_utilization_target
    schedule["bytes_per_token"] = bytes_per_token

    return schedule


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
        fsdp_sharding: str = "full_shard",
        gradient_clip_val: float = 1.0,
        ckpt_dir: str = "/data_16T/deepseek/checkpoints/lola",
        save_every_n_steps: int = 500,
        log_every_n_steps: int = 10,
        disable_gradient_checkpointing: bool = False,
        wandb_project: str = "lola-azure-stream",
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_id: str | None = None,
        preprocess_in_loader: bool = False,
        dataloader_timeout: int = 600,
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
        self.fsdp_sharding = fsdp_sharding
        self.gradient_clip_val = gradient_clip_val
        self.ckpt_dir = ckpt_dir
        self.save_every_n_steps = save_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.disable_gradient_checkpointing = disable_gradient_checkpointing
        self.dataloader_timeout = dataloader_timeout

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
        self._preprocess_in_loader = preprocess_in_loader

    def setup_model(self):
        """设置模型"""
        # Enable cuDNN SDPA backend for Blackwell GPUs (cuDNN 9.10+ has dedicated kernels)
        torch.backends.cuda.enable_cudnn_sdp(True)
        cudnn_sdp_available = torch.backends.cuda.cudnn_sdp_enabled()
        _log(f"cuDNN SDPA backend: enabled={cudnn_sdp_available}")

        _log(f"Loading LoLA Policy on {self.device}...")

        self.policy = LoLAPolicy(self.config)
        self.policy._device = self.device
        self.policy.model = self.policy.model.to(self.device)
        self.policy.vlm = self.policy.vlm.to(self.device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            dataset_stats=self.dataset_stats,
        )

        if not self.train_vlm and hasattr(self.policy, "vlm"):
            _log("Freezing VLM parameters...")
            for param in self.policy.vlm.parameters():
                param.requires_grad = False
            self.policy.vlm.eval()

        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy.parameters())
        _log(f"Trainable params: {trainable_params:,} / {total_params:,}")

        # Enable TF32 for matmul and cuDNN on Ampere+ GPUs (B200/H100/A100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _log("TF32 matmul/cudnn: enabled")

        # Silence CUDA Graphs dynamic shape warning for tier-based batching
        # (9 distinct sizes from 3 tiers is expected and harmless)
        # torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

        # torch.compile for kernel fusion (must be done before FSDP wrapping)
        if self.config.compile_model:
            compile_mode = getattr(self.config, 'compile_mode', 'reduce-overhead')
            _log(f"Compiling DiT with torch.compile(mode={compile_mode})...")
            self.policy.model.dit = torch.compile(self.policy.model.dit, mode=compile_mode)
            _log("DiT compilation scheduled (first forward will trigger JIT compilation)")

            # Compile VLM language model layers (split mode avoids hooks, making
            # the full VLM graph compile-friendly)
            _log(f"Compiling VLM decoder layers with torch.compile(mode={compile_mode})...")
            lang_layers = self.policy.vlm.model.language_model.layers
            for i in range(len(lang_layers)):
                lang_layers[i] = torch.compile(lang_layers[i], mode=compile_mode)
            _log("VLM decoder layers compilation scheduled")

            # Compile VLM vision encoder blocks (~287M params, 27 layers)
            _log(f"Compiling VLM vision blocks with torch.compile(mode={compile_mode})...")
            vision_blocks = self.policy.vlm.model.vision_model.blocks
            for i in range(len(vision_blocks)):
                vision_blocks[i] = torch.compile(vision_blocks[i], mode=compile_mode)
            _log("VLM vision blocks compilation scheduled")

        if self.is_distributed:
            cap = torch.cuda.get_device_capability(self.device)
            torch_cuda_ver = torch.version.cuda
            _log(f"GPU compute capability: sm_{cap[0]}{cap[1]}, torch CUDA: {torch_cuda_ver}")
            if self.strategy == "fsdp":
                self._setup_fsdp()
            else:
                self._setup_ddp()
        else:
            self.model = self.policy

        self.interconnect_monitor = InterconnectMonitor(self.device)

        # Disable gradient checkpointing if requested (saves bwd recomputation overhead)
        if self.disable_gradient_checkpointing:
            if hasattr(self.policy, 'vlm') and hasattr(self.policy.vlm, 'gradient_checkpointing_disable'):
                self.policy.vlm.gradient_checkpointing_disable()
                _log("VLM gradient checkpointing DISABLED")
            if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'gradient_checkpointing_disable'):
                self.policy.model.gradient_checkpointing_disable()
                _log("DiT gradient checkpointing DISABLED")

    def _setup_ddp(self):
        """设置 DDP"""
        _log("Setting up DDP...")
        self.model = DDP(
            self.policy,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

    def _setup_fsdp(self):
        """设置 FSDP"""
        _log("Setting up FSDP...")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
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

        sharding_strategy = (
            ShardingStrategy.FULL_SHARD if self.fsdp_sharding == "full_shard"
            else ShardingStrategy.SHARD_GRAD_OP
        )
        _log(f"FSDP sharding strategy: {self.fsdp_sharding} ({sharding_strategy})")

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

    def training_step(self, batch, timing_dict: dict | None = None):
        """单步训练"""
        t0 = time.monotonic()

        if self._preprocess_in_loader:
            # Batch already preprocessed by prefetch thread (CPU steps done, no PIL Images).
            # Only need to move all tensors to GPU.
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            t_preprocess = time.monotonic() - t0  # GPU transfer time only
        else:
            # Old behavior: GPU transfer + extract special + preprocess + restore
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            t_device = time.monotonic() - t0

            t1 = time.monotonic()
            special_data = self._extract_special_fields(batch)
            batch = self.preprocessor(batch)
            batch = self._restore_special_fields(batch, special_data)
            t_preprocess = time.monotonic() - t1

        t2 = time.monotonic()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = self.model(batch)
        t_model_fwd = time.monotonic() - t2

        if timing_dict is not None:
            if self._preprocess_in_loader:
                timing_dict["device_s"] = 0.0  # No separate device transfer before preprocess
                timing_dict["preprocess_s"] = t_preprocess  # GPU transfer time only
            else:
                timing_dict["device_s"] = t_device
                timing_dict["preprocess_s"] = t_preprocess
            timing_dict["model_fwd_s"] = t_model_fwd

        return loss, loss_dict

    def train(self, train_loader, start_step: int = 0):
        """训练循环，增强 wandb 日志（throughput / timing / GPU metrics）"""
        self.global_step = start_step
        self.model.train()

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, f"lola-azure-stream-{time_str}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            _log(f"Checkpoint directory: {ckpt_dir}")

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
            _log(f"Wandb initialized: {wandb_run_name}")

        _log(f"Starting training from step {start_step} to {self.max_steps}")

        # 数据跳过由 dataset.start_index 在 __iter__ 内部处理，
        # 这里无需 skip_epochs/skip_batches 逻辑

        data_yield_start = time.monotonic()
        data_iter = iter(train_loader)
        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset start_index so next epoch starts from beginning
                if hasattr(train_loader, '_dataset') and hasattr(train_loader._dataset, 'start_index'):
                    train_loader._dataset.start_index = 0
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

            # ── Forward pass (with split timing) ────────────────
            fwd_timing = {}
            fwd_start = time.monotonic()
            loss, loss_dict = self.training_step(batch, timing_dict=fwd_timing)
            fwd_s = time.monotonic() - fwd_start
            device_s = fwd_timing.get("device_s", 0)
            preprocess_s = fwd_timing.get("preprocess_s", 0)
            model_fwd_s = fwd_timing.get("model_fwd_s", 0)

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
                if self.strategy == "fsdp":
                    grad_norm = self.model.clip_grad_norm_(self.gradient_clip_val)
                else:
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

                # ── Per-rank memory distribution ──────────────────
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
                local_reserved_pct = torch.cuda.memory_reserved(self.device) / total_mem * 100
                local_alloc_pct = torch.cuda.memory_allocated(self.device) / total_mem * 100
                if self.is_distributed:
                    reserved_tensor = torch.tensor([local_reserved_pct], device=self.device)
                    alloc_tensor = torch.tensor([local_alloc_pct], device=self.device)
                    reserved_gathered = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
                    alloc_gathered = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
                    dist.all_gather(reserved_gathered, reserved_tensor)
                    dist.all_gather(alloc_gathered, alloc_tensor)
                else:
                    reserved_gathered = [torch.tensor([local_reserved_pct])]
                    alloc_gathered = [torch.tensor([local_alloc_pct])]

                if self.is_main_process:
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm if grad_norm is not None else None

                    # ── Console logging (mirrors all wandb metrics) ──
                    _log(
                        f"[Step {self.global_step}/{self.max_steps}] "
                        f"Loss={loss.item():.4f} LR={lr:.2e} "
                        f"Update={update_s:.2f}s Throughput={batch_per_s:.2f}batch/s "
                        f"DataWait={data_yield_s:.2f}s"
                    )
                    if grad_norm_val is not None:
                        _log(f"  grad_norm={grad_norm_val:.4f}")
                    _log(
                        f"  Timing: fwd={fwd_s:.3f}s (device={device_s:.3f}s "
                        f"preprocess={preprocess_s:.3f}s model_fwd={model_fwd_s:.3f}s) "
                        f"bwd={bwd_s:.3f}s clip={clip_s:.3f}s opt={opt_s:.3f}s"
                    )
                    _log(
                        f"  GPU: alloc={gpu_mem_alloc:.1f}GB "
                        f"reserved={gpu_mem_reserved:.1f}GB"
                    )
                    # ── Per-rank memory distribution ──
                    mem_str = " | ".join(
                        f"GPU{i}: r={r.item():.1f}% a={a.item():.1f}%"
                        for i, (r, a) in enumerate(zip(reserved_gathered, alloc_gathered))
                    )
                    _log(f"  Memory Distribution: {mem_str}")
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
                        _log(f"  Interconnect: {' | '.join(parts)}")
                    for k, v in loss_dict.items():
                        if k != "loss" and isinstance(v, (int, float)):
                            _log(f"  {k}={v:.4f}")

                    # ── Wandb logging ──────────────────────────────
                    if self.use_wandb:
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/learning_rate": lr,
                            "train/step": self.global_step,
                            "train/batch_per_s": batch_per_s,
                            "timing/step_s": update_s,
                            "timing/data_yield_s": data_yield_s,
                            "timing/fwd_s": fwd_s,
                            "timing/fwd_device_s": device_s,
                            "timing/fwd_preprocess_s": preprocess_s,
                            "timing/fwd_model_fwd_s": model_fwd_s,
                            "timing/bwd_s": bwd_s,
                            "timing/clip_s": clip_s,
                            "timing/opt_s": opt_s,
                            "memory/gpu_alloc_gb": gpu_mem_alloc,
                            "memory/gpu_reserved_gb": gpu_mem_reserved,
                        }
                        if grad_norm_val is not None:
                            log_dict["train/grad_norm"] = grad_norm_val
                        for k, v in loss_dict.items():
                            if k != "loss" and isinstance(v, (int, float)):
                                log_dict[f"train/{k}"] = v
                        for k, v in interconnect_metrics.items():
                            log_dict[f"interconnect/{k}"] = v
                        for i, (r, a) in enumerate(zip(reserved_gathered, alloc_gathered)):
                            log_dict[f"memory/gpu{i}_reserved_pct"] = r.item()
                            log_dict[f"memory/gpu{i}_alloc_pct"] = a.item()
                        wandb.log(log_dict)

            if self.global_step % self.save_every_n_steps == 0:
                self.save_checkpoint(ckpt_dir, self.global_step)

            # Reset data yield timer for next iteration
            data_yield_start = time.monotonic()

        self.save_checkpoint(ckpt_dir, self.global_step, is_final=True)
        _log(f"Training completed! Final checkpoint saved at step {self.global_step}")

        if self.interconnect_monitor:
            self.interconnect_monitor.close()

        if self.use_wandb:
            wandb.finish()

    def train_with_tiers(self, tier_loaders, tier_batch_sizes, tier_accum_steps,
                         accum_order, balance_weights, target_effective_batch,
                         start_step: int = 0,
                         gpu_utilization_target: float = 0.92,
                         tier_stats: dict | None = None,
                         tier_datasets: list | None = None,
                         async_loader_class=None,
                         dataloader_kwargs: dict | None = None,
                         balance_mode: str = "frame_weighted"):
        """Balanced tier-cycling training loop with gradient accumulation.

        Each optimizer step = len(accum_order) micro-steps.
        The accumulation order is auto-computed to balance tier representation
        proportional to balance_weights (default: frame_weighted).

        Args:
            tier_loaders: list of AsyncDecodeDataLoader per tier
            tier_batch_sizes: list of micro_batch_size per tier
            tier_accum_steps: dict {tier_idx_str: accum_steps}
            accum_order: list of (tier_idx, micro_batch) interleaved + shuffled
            balance_weights: target ratio per tier (e.g. frame-weighted)
            target_effective_batch: target total samples per optimizer step
            start_step: resume offset
            gpu_utilization_target: target GPU memory utilization (default 0.92)
            tier_stats: tier_config["tier_stats"] for recalibration
            tier_datasets: list of tier datasets (for DataLoader recreation)
            async_loader_class: AsyncDecodeDataLoader class to use
            dataloader_kwargs: kwargs for DataLoader creation (num_workers, etc.)
            balance_mode: balance mode used for recalibration (default: frame_weighted)
        """
        self.global_step = start_step
        self.model.train()
        self._gpu_utilization_target = gpu_utilization_target
        self._tier_stats = tier_stats
        self._tier_datasets = tier_datasets
        self._async_loader_class = async_loader_class
        self._dataloader_kwargs = dataloader_kwargs or {}
        self._balance_mode = balance_mode

        num_tiers = len(tier_loaders)
        total_accum_steps = len(accum_order)
        self._tier_loaders = tier_loaders
        self._tier_iters = [iter(loader) for loader in tier_loaders]
        self._tier_epochs = [0] * num_tiers

        _log(f"Accumulation order ({total_accum_steps} micro-steps per optimizer step):")
        for t in range(num_tiers):
            _log(f"  Tier {t}: micro_batch={tier_batch_sizes[t]}, "
                 f"accum_steps={tier_accum_steps[str(t)]}, "
                 f"weight={balance_weights[t]:.3f}, "
                 f"samples/step={tier_batch_sizes[t] * tier_accum_steps[str(t)]}")
        _log(f"  Total effective batch: {target_effective_batch}")

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, f"lola-azure-stream-{time_str}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            _log(f"Checkpoint directory: {ckpt_dir}")
        
        # Compute gradient sync interval for distributed training.
        # Without no_sync(), every backward() triggers all_reduce/reduce_scatter,
        # causing inter-GPU stalls. We sync every N micro-steps to balance
        # throughput (fewer syncs) vs memory (sync releases gradient buffers).
        # Default: sync every 5 steps, dynamically adjusted after step 1.
        sync_interval = total_accum_steps if not self.is_distributed else min(5, total_accum_steps)
        self._sync_interval = sync_interval
        
        if self.use_wandb:
            wandb_run_name = self.wandb_name or f"lola-tier-stream-{self.strategy}-{time_str}"
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
                    "strategy": self.strategy,
                    "world_size": self.world_size,
                    "train_vlm": self.train_vlm,
                    "gradient_clip_val": self.gradient_clip_val,
                    "gradient_accumulation_steps": total_accum_steps,
                    "gradient_sync_interval": sync_interval,
                    "target_effective_batch": target_effective_batch,
                    "tier_batch_sizes": tier_batch_sizes,
                    "tier_accum_steps": tier_accum_steps,
                    "balance_weights": balance_weights,
                },
            )
            _log(f"Wandb initialized: {wandb_run_name}")

        _log(f"Starting tier-cycling training from step {start_step} to {self.max_steps} "
             f"(accum={total_accum_steps} micro-steps, "
             f"effective_batch={target_effective_batch})")

        _log(f"Gradient sync interval: every {sync_interval} micro-steps "
             f"(total_accum={total_accum_steps}, distributed={self.is_distributed})")

        data_yield_start = time.monotonic()
        step_start = time.monotonic()
        while self.global_step < self.max_steps:
            # ── Gradient accumulation cycle ────────────────────────
            self.optimizer.zero_grad()

            accum_loss = 0.0
            accum_samples = 0
            tier_loss_sums = {t: 0.0 for t in range(num_tiers)}
            tier_loss_counts = {t: 0 for t in range(num_tiers)}
            tier_sample_counts = {t: 0 for t in range(num_tiers)}
            tier_memory_costs = {}
            accum_data_yield_s = 0.0
            accum_fwd_s = 0.0
            accum_bwd_s = 0.0
            # Per-tier timing accumulators
            tier_data_yield_s = {t: 0.0 for t in range(num_tiers)}
            tier_fwd_s = {t: 0.0 for t in range(num_tiers)}
            tier_bwd_s = {t: 0.0 for t in range(num_tiers)}

            for micro_idx, (tier_idx, micro_batch) in enumerate(accum_order):
                # Get next batch from tier DataLoader (auto-restart on StopIteration)
                batch = self._get_next_tier_batch(tier_idx)

                data_yield_s = time.monotonic() - data_yield_start
                accum_data_yield_s += data_yield_s
                tier_data_yield_s[tier_idx] += data_yield_s

                # Forward + backward (scaled by 1/total_accum for proper gradient averaging)
                # Periodic gradient sync: sync every sync_interval steps to release
                # gradient buffers and prevent memory buildup, while reducing sync stalls.
                should_sync = (micro_idx + 1) % self._sync_interval == 0 or micro_idx == total_accum_steps - 1
                sync_ctx = nullcontext() if should_sync or not self.is_distributed else self.model.no_sync()

                with sync_ctx:
                    fwd_timing = {}
                    fwd_start = time.monotonic()
                    loss, loss_dict = self.training_step(batch, timing_dict=fwd_timing)
                    fwd_s = time.monotonic() - fwd_start
                    accum_fwd_s += fwd_s
                    tier_fwd_s[tier_idx] += fwd_s

                    bwd_start = time.monotonic()
                    scaled_loss = loss / total_accum_steps
                    scaled_loss.backward()
                    bwd_s = time.monotonic() - bwd_start
                    accum_bwd_s += bwd_s
                    tier_bwd_s[tier_idx] += bwd_s

                accum_loss += loss.item()
                accum_samples += micro_batch
                tier_loss_sums[tier_idx] += loss.item()
                tier_loss_counts[tier_idx] += 1
                tier_sample_counts[tier_idx] += micro_batch

                if "_memory_cost" in batch and isinstance(batch["_memory_cost"], torch.Tensor):
                    tier_memory_costs[tier_idx] = batch["_memory_cost"].mean().item()

                data_yield_start = time.monotonic()

            # ── Gradient clipping + optimizer step ──────────────────
            clip_start = time.monotonic()
            if self.gradient_clip_val > 0:
                if self.strategy == "fsdp":
                    grad_norm = self.model.clip_grad_norm_(self.gradient_clip_val)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val,
                    )
            else:
                grad_norm = None
            clip_s = time.monotonic() - clip_start

            opt_start = time.monotonic()
            self.optimizer.step()
            opt_s = time.monotonic() - opt_start

            self.scheduler.step()
            self.global_step += 1

            # ── Dynamic micro_batch recalibration after first step ──
            if self.global_step == 1 and torch.cuda.is_available() and self._tier_stats is not None:
                actual_peak = torch.cuda.max_memory_allocated(self.device)
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
                actual_ratio = actual_peak / total_mem
                _log(f"Step 1 GPU utilization: {actual_ratio:.2%} (target: {self._gpu_utilization_target:.2%})")

                if actual_ratio < self._gpu_utilization_target * 0.8:
                    scale = min(self._gpu_utilization_target / actual_ratio, 3.0)
                    new_batch_sizes = [max(1, int(bs * scale * 0.95)) for bs in tier_batch_sizes]

                    _log(f"Recalibrating micro_batches: {tier_batch_sizes} -> {new_batch_sizes} "
                         f"(scale={scale:.2f}, safety_margin=0.95)")

                    new_schedule = compute_balanced_schedule(
                        tier_stats=self._tier_stats,
                        tier_micro_batches=new_batch_sizes,
                        target_effective_batch=target_effective_batch,
                        balance_mode=self._balance_mode,
                    )

                    tier_batch_sizes = new_batch_sizes
                    tier_accum_steps = new_schedule["tier_accum_steps"]
                    accum_order = new_schedule["accum_order"]
                    total_accum_steps = len(accum_order)

                    # Recreate DataLoaders with new batch sizes
                    if self._tier_datasets is not None and self._async_loader_class is not None:
                        for t in range(num_tiers):
                            old_loader = self._tier_loaders[t]
                            old_loader.close()
                            self._tier_loaders[t] = self._create_tier_dataloader(
                                tier_ds=self._tier_datasets[t],
                                micro_batch=tier_batch_sizes[t],
                                async_loader_class=self._async_loader_class,
                                **self._dataloader_kwargs,
                            )
                            self._tier_iters[t] = iter(self._tier_loaders[t])

                    _log(f"Updated schedule: {total_accum_steps} micro-steps per optimizer step")
                    for t in range(num_tiers):
                        _log(f"  Tier {t}: micro_batch={tier_batch_sizes[t]}, "
                             f"accum_steps={tier_accum_steps[str(t)]}, "
                             f"samples/step={tier_batch_sizes[t] * tier_accum_steps[str(t)]}")
                    _log(f"  Total effective batch: {sum(tier_batch_sizes[t] * tier_accum_steps[str(t)] for t in range(num_tiers))}")

                torch.cuda.reset_peak_memory_stats(self.device)

                # Dynamically adjust sync_interval based on available memory
                if self.is_distributed and total_accum_steps > 1:
                    # Estimate per-micro-step gradient memory overhead
                    # = peak_mem - post_backward_mem (the gradient buffer size for one step)
                    # Under FSDP, no_sync() accumulates full unsharded gradients instead of
                    # keeping only the local shard. Extra cost per no_sync step ≈
                    # (world_size - 1) / world_size * grad_shard_size.
                    # We measure it empirically by comparing current allocated vs peak.
                    current_mem = torch.cuda.memory_allocated(self.device)
                    total_mem = torch.cuda.get_device_properties(self.device).total_memory
                    budget_mem = total_mem * self._gpu_utilization_target

                    # Estimate gradient buffer overhead per no_sync step
                    # After one step with sync, current_mem = model + optimizer + activations
                    # After one step without sync, peak would be current_mem + grad_buffer
                    # We use the difference between peak and current as grad_buffer estimate
                    # (this is conservative since peak also includes activation peaks)
                    # A better estimate: grad_buffer ≈ param_memory * (world_size-1)/world_size for FSDP
                    param_mem = sum(p.numel() * p.element_size() for p in self.model.parameters())
                    if self.strategy == "fsdp":
                        # FSDP: no_sync accumulates (world_size-1)/world_size of full gradient
                        grad_buffer_per_step = param_mem * (self.world_size - 1) / self.world_size
                    else:
                        # DDP: no_sync keeps full gradient buffer per step
                        grad_buffer_per_step = param_mem

                    # Available room for no_sync gradient accumulation
                    available_for_grads = budget_mem - current_mem
                    max_no_sync_steps = max(1, int(available_for_grads / grad_buffer_per_step)) if grad_buffer_per_step > 0 else 1
                    # Use at most max_no_sync_steps between syncs, but cap at 20 for stability
                    new_sync_interval = min(max_no_sync_steps, 20, total_accum_steps)
                    new_sync_interval = max(1, new_sync_interval)

                    if new_sync_interval != self._sync_interval:
                        _log(f"Sync interval adjusted: {self._sync_interval} -> {new_sync_interval} "
                             f"(grad_buffer/step={grad_buffer_per_step / 1e9:.2f}GB, "
                             f"available={available_for_grads / 1e9:.1f}GB, "
                             f"max_no_sync={max_no_sync_steps})")
                        self._sync_interval = new_sync_interval

            # ── Logging ─────────────────────────────────────────────
            total_step_s = time.monotonic() - step_start
            if self.global_step % self.log_every_n_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                gpu_mem_alloc = torch.cuda.memory_allocated(self.device) / 1e9
                gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9
                interconnect_metrics = self.interconnect_monitor.snapshot() if self.interconnect_monitor else {}

                # Per-rank memory distribution
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
                local_reserved_pct = torch.cuda.memory_reserved(self.device) / total_mem * 100
                local_alloc_pct = torch.cuda.memory_allocated(self.device) / total_mem * 100
                if self.is_distributed:
                    reserved_tensor = torch.tensor([local_reserved_pct], device=self.device)
                    alloc_tensor = torch.tensor([local_alloc_pct], device=self.device)
                    reserved_gathered = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
                    alloc_gathered = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
                    dist.all_gather(reserved_gathered, reserved_tensor)
                    dist.all_gather(alloc_gathered, alloc_tensor)
                else:
                    reserved_gathered = [torch.tensor([local_reserved_pct])]
                    alloc_gathered = [torch.tensor([local_alloc_pct])]

                if self.is_main_process:
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm if grad_norm is not None else None
                    avg_loss = accum_loss / total_accum_steps
                    samples_per_s = accum_samples / total_step_s if total_step_s > 0 else 0

                    # ── Console output ──────────────────────────────
                    _log(
                        f"[Step {self.global_step}/{self.max_steps}] "
                        f"Loss={avg_loss:.4f} LR={lr:.2e} "
                        f"grad_norm={f'{grad_norm_val:.4f}' if grad_norm_val is not None else 'N/A'} "
                        f"samples/s={samples_per_s:.1f}"
                    )
                    _log(
                        f"  Timing: total={total_step_s:.1f}s "
                        f"data={accum_data_yield_s:.1f}s fwd={accum_fwd_s:.1f}s "
                        f"bwd={accum_bwd_s:.1f}s clip={clip_s:.2f}s opt={opt_s:.2f}s"
                    )
                    max_deviation = 0.0
                    for t in range(num_tiers):
                        actual_ratio = tier_sample_counts[t] / accum_samples if accum_samples > 0 else 0
                        deviation = actual_ratio - balance_weights[t]
                        max_deviation = max(max_deviation, abs(deviation))
                        avg_t_loss = tier_loss_sums[t] / tier_loss_counts[t] if tier_loss_counts[t] > 0 else 0
                        t_mem = tier_memory_costs.get(t, 0.0)
                        t_count = tier_loss_counts[t]
                        _log(
                            f"  Tier {t}: loss={avg_t_loss:.4f} samples={tier_sample_counts[t]} "
                            f"ratio={actual_ratio:.3f} (target={balance_weights[t]:.3f}) "
                            f"data={tier_data_yield_s[t]:.1f}s fwd={tier_fwd_s[t]:.1f}s "
                            f"bwd={tier_bwd_s[t]:.1f}s ({t_count} micro-steps) "
                            f"epoch={self._tier_epochs[t]}"
                        )
                    _log(f"  Balance max deviation: {max_deviation:.4f}")

                    # Per-rank memory distribution
                    mem_str = " | ".join(
                        f"GPU{i}: r={r.item():.1f}% a={a.item():.1f}%"
                        for i, (r, a) in enumerate(zip(reserved_gathered, alloc_gathered))
                    )
                    _log(f"  Memory: alloc={gpu_mem_alloc:.1f}GB reserved={gpu_mem_reserved:.1f}GB | {mem_str}")

                    if self.use_wandb:
                        log_dict = {
                            # Core training metrics
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/step": self.global_step,
                            "train/grad_norm": grad_norm_val if grad_norm_val is not None else 0,
                            "train/effective_batch_size": accum_samples,
                            "train/samples_per_s": samples_per_s,

                            # Step-level timing (total optimizer step)
                            "timing/step_s": total_step_s,
                            "timing/data_yield_s": accum_data_yield_s,
                            "timing/fwd_s": accum_fwd_s,
                            "timing/bwd_s": accum_bwd_s,
                            "timing/clip_s": clip_s,
                            "timing/opt_s": opt_s,

                            # Schedule info
                            "schedule/target_effective_batch": target_effective_batch,
                            "schedule/total_accum_steps": total_accum_steps,
                            "schedule/sync_interval": self._sync_interval,
                            "schedule/balance_max_deviation": max_deviation,

                            # GPU memory
                            "memory/gpu_alloc_gb": gpu_mem_alloc,
                            "memory/gpu_reserved_gb": gpu_mem_reserved,
                        }
                        for i, (r, a) in enumerate(zip(reserved_gathered, alloc_gathered)):
                            log_dict[f"memory/gpu{i}_reserved_pct"] = r.item()
                            log_dict[f"memory/gpu{i}_alloc_pct"] = a.item()
                        for k, v in interconnect_metrics.items():
                            log_dict[f"interconnect/{k}"] = v
                        for k, v in loss_dict.items():
                            if k != "loss" and isinstance(v, (int, float)):
                                log_dict[f"train/{k}"] = v
                        for t in range(num_tiers):
                            actual_ratio = tier_sample_counts[t] / accum_samples if accum_samples > 0 else 0
                            avg_t_loss = tier_loss_sums[t] / tier_loss_counts[t] if tier_loss_counts[t] > 0 else 0
                            log_dict[f"tier/{t}/loss"] = avg_t_loss
                            log_dict[f"tier/{t}/samples"] = tier_sample_counts[t]
                            log_dict[f"tier/{t}/actual_ratio"] = actual_ratio
                            log_dict[f"tier/{t}/target_ratio"] = balance_weights[t]
                            log_dict[f"tier/{t}/ratio_deviation"] = actual_ratio - balance_weights[t]
                            log_dict[f"tier/{t}/epoch"] = self._tier_epochs[t]
                            log_dict[f"tier/{t}/data_yield_s"] = tier_data_yield_s[t]
                            log_dict[f"tier/{t}/fwd_s"] = tier_fwd_s[t]
                            log_dict[f"tier/{t}/bwd_s"] = tier_bwd_s[t]
                            if t in tier_memory_costs:
                                log_dict[f"tier/{t}/avg_memory_cost"] = tier_memory_costs[t]
                        wandb.log(log_dict)

            step_start = time.monotonic()

            if self.global_step % self.save_every_n_steps == 0:
                self.save_checkpoint(ckpt_dir, self.global_step)

        self.save_checkpoint(ckpt_dir, self.global_step, is_final=True)
        _log(f"Training completed! Final checkpoint saved at step {self.global_step}")

        if self.interconnect_monitor:
            self.interconnect_monitor.close()

        if self.use_wandb:
            wandb.finish()

    def _create_tier_dataloader(self, tier_ds, micro_batch, num_workers=8,
                                prefetch_factor=4, prefetch_queue_size=0,
                                async_loader_class=None, timeout=600):
        """Create a single tier DataLoader with async decode wrapping."""
        from torch.utils.data import DataLoader

        if async_loader_class is None:
            async_loader_class = self._async_loader_class

        raw_loader = DataLoader(
            tier_ds,
            batch_size=micro_batch,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            timeout=timeout,
        )

        loader_kwargs = {
            "dataloader": raw_loader,
            "dataset": tier_ds,
            "collate_fn": async_loader_class.make_collate_fn(),
        }
        if prefetch_queue_size > 0:
            def make_preprocess_fn(trainer_ref):
                def preprocess_fn(batch):
                    special_data = trainer_ref._extract_special_fields(batch)
                    batch = trainer_ref.preprocessor(batch, skip_device_step=True)
                    batch = trainer_ref._restore_special_fields(batch, special_data)
                    return batch
                return preprocess_fn
            loader_kwargs["prefetch_queue_size"] = prefetch_queue_size
            loader_kwargs["preprocess_fn"] = make_preprocess_fn(self)

        return async_loader_class(**loader_kwargs)

    def _get_next_tier_batch(self, tier_idx: int, max_retries: int = 3):
        """Get next batch from tier DataLoader, with auto-restart and timeout retry."""
        for attempt in range(max_retries + 1):
            try:
                return next(self._tier_iters[tier_idx])
            except StopIteration:
                self._tier_epochs[tier_idx] += 1
                # Wait for any background producer thread to exit cleanly
                # before creating a new iterator on the same DataLoader
                # with persistent_workers.
                import time as _time
                _time.sleep(0.5)
                self._tier_iters[tier_idx] = iter(self._tier_loaders[tier_idx])
                _log(f"Tier {tier_idx} DataLoader restarted (epoch {self._tier_epochs[tier_idx]})")
                continue
            except TimeoutError:
                if attempt < max_retries:
                    _log(f"Tier {tier_idx} DataLoader timed out, retrying "
                         f"(attempt {attempt + 1}/{max_retries})...")
                    import time as _time
                    _time.sleep(1.0)
                    self._tier_iters[tier_idx] = iter(self._tier_loaders[tier_idx])
                    continue
                raise

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

        _log(f"Checkpoint saved: {ckpt_path}")

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

        _log(f"Checkpoint loaded from: {ckpt_path}, starting from step {self.global_step}")


def main():
    dist_info = setup_distributed()

    parser = argparse.ArgumentParser(description="LoLA Azure Streaming Training")

    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)

    # 训练参数
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--fsdp_sharding", type=str, default="full_shard",
                        choices=["full_shard", "shard_grad_op"],
                        help="FSDP sharding strategy: full_shard (ZeRO-3) or shard_grad_op (ZeRO-2)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing on VLM and DiT (saves recomputation overhead, increases memory)")
    parser.add_argument("--compile_model", action="store_true",
                        help="Enable torch.compile for DiT module (kernel fusion, reduces kernel launch overhead)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        help="torch.compile mode: reduce-overhead (CUDA graphs), default, max-autotune")

    # 模型参数
    parser.add_argument("--vlm_path", type=str, default="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/")
    parser.add_argument("--train_vlm", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="/data_16T/deepseek/checkpoints/lola")
    parser.add_argument("--resume", type=str, default=None)

    # VLM 图像分辨率参数
    parser.add_argument("--max_image_pixels", type=int, default=230400,
                        help="Max pixels per image for Qwen3.5 smart_resize (230400 → max_h≈360p)")
    parser.add_argument("--min_image_pixels", type=int, default=65536,
                        help="Min pixels per image for Qwen3.5 smart_resize (65536 → min 64 visual tokens)")

    # LoLA 参数
    parser.add_argument("--action_dim", type=int, default=20)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)

    # 历史 action 参数
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"])

    # 流式数据集参数
    parser.add_argument("--buffer_size", type=int, default=5000, help="Streaming shuffle buffer size per worker")
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
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch_factor per worker (default 2 in PyTorch, we use 4)")
    parser.add_argument("--prefetch_queue_size", type=int, default=4,
                        help="AsyncDecodeDataLoader batch-level prefetch queue size (0=disabled)")
    parser.add_argument("--dataloader_timeout", type=int, default=600,
                        help="DataLoader worker timeout in seconds (default 600)")

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

    # Tier-based batching parameters
    parser.add_argument("--tier_config_path", type=str, default=None,
                        help="Path to tier config JSON from Phase 1 scan (enables per-tier DataLoader + gradient accumulation)")
    parser.add_argument("--effective_batch_size", type=int, default=2048,
                        help="Target effective batch size per optimizer step (sum of tier samples)")
    parser.add_argument("--balance_mode", type=str, default="frame_weighted",
                        choices=["frame_weighted", "equal", "episode_weighted"],
                        help="How to weight tier representation in each optimizer step")
    parser.add_argument("--gpu_utilization_target", type=float, default=0.92,
                        help="Target GPU memory utilization per micro-batch (e.g., 0.92 = 92%%)")
    parser.add_argument("--gpu_memory_budget_gb", type=float, default=None,
                        help="Override GPU memory budget per micro-batch in GB (auto-detect if None)")
    parser.add_argument("--tier_micro_batches_override", type=str, default=None,
                        help="Override auto-computed per-tier micro-batches, comma-separated (e.g., '8,4,2')")

    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    if dist_info["world_rank"] == 0:
        _log("=" * 60)
        _log("LoLA Azure Streaming Training")
        _log("=" * 60)
        _log(f"Dataset: {args.dataset_repo_id or args.dataset_root}")
        _log(f"Strategy: {args.strategy}")
        _log(f"World Size: {dist_info['world_size']}")
        _log(f"Batch Size: {args.batch_size}")
        _log(f"Streaming: True")
        _log(f"Buffer Size: {args.buffer_size}")
        _log(f"VLM Path: {args.vlm_path}")
        if args.pretrain:
            _log(f"Pretrain Mode: True")
            _log(f"Dataset to Episodes: {args.dataset_to_episodes_path}")
            _log(f"Sub Root: {args.sub_root}")
            _log(f"Temp Process: {args.temp_process}")
            _log(f"Episode Chunk Size: {args.episode_chunk_size}")
        if args.tier_config_path:
            _log(f"Tier Config: {args.tier_config_path}")
            _log(f"Effective Batch Size: {args.effective_batch_size}")
            _log(f"Balance Mode: {args.balance_mode}")
            _log(f"GPU Utilization Target: {args.gpu_utilization_target}")
        _log("=" * 60)

    # 获取数据集元数据
    _log("Loading dataset metadata...")
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

    _log(f"Dataset: {dataset_metadata.total_episodes} episodes, {dataset_metadata.total_frames} frames")

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
        max_image_pixels=args.max_image_pixels,
        min_image_pixels=args.min_image_pixels,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
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

    # ── Tier-based batching path ──────────────────────────────────
    if args.pretrain and args.tier_config_path:
        import json
        _log(f"Loading tier config from {args.tier_config_path}...")
        with open(args.tier_config_path) as f:
            tier_config = json.load(f)

        tier_stats = tier_config["tier_stats"]
        num_tiers = len(tier_stats)

        # Auto-compute tier schedule from tier_stats
        calibration_path = tier_config.get("params", {}).get("calibration_path")
        schedule = compute_tier_schedule(
            tier_stats=tier_stats,
            effective_batch_size=args.effective_batch_size,
            balance_mode=args.balance_mode,
            gpu_memory_budget_gb=args.gpu_memory_budget_gb,
            tier_micro_batches_override=args.tier_micro_batches_override,
            gpu_utilization_target=args.gpu_utilization_target,
            calibration_path=calibration_path,
        )

        tier_micro_batches = schedule["tier_micro_batches"]
        tier_accum_steps = schedule["tier_accum_steps"]
        accum_order = schedule["accum_order"]
        balance_weights = schedule["balance_weights"]
        actual_effective_batch = schedule["actual_effective_batch"]

        _log(f"Auto-computed tier schedule:")
        for t in range(num_tiers):
            _log(f"  Tier {t}: micro_batch={tier_micro_batches[t]}, "
                 f"accum_steps={tier_accum_steps[str(t)]}, "
                 f"weight={balance_weights[t]:.3f}, "
                 f"samples/step={tier_micro_batches[t] * tier_accum_steps[str(t)]}")
        _log(f"  Total effective batch: {actual_effective_batch}")
        _log(f"  Balance mode: {args.balance_mode}")

        # Create per-tier datasets + DataLoaders
        tier_datasets = []
        tier_loaders = []

        for tier_idx in range(num_tiers):
            _log(f"Creating dataset + loader for tier {tier_idx} "
                 f"(micro_batch={tier_micro_batches[tier_idx]})...")
            tier_ds = create_lola_pretrain_streaming_dataset(
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
                tier_config_path=args.tier_config_path,
                yield_tier=tier_idx,
            )
            tier_datasets.append(tier_ds)

        # Create trainer first (before DataLoaders, so we can use preprocessor for prefetch)
        dataset_stats = {}
        trainer = LoLATrainer(
            config=config,
            dataset_stats=dataset_stats,
            dist_info=dist_info,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            train_vlm=args.train_vlm,
            strategy=args.strategy,
            fsdp_sharding=args.fsdp_sharding,
            gradient_clip_val=args.gradient_clip_val,
            ckpt_dir=args.ckpt_dir,
            save_every_n_steps=args.save_every_n_steps,
            log_every_n_steps=args.log_every_n_steps,
            disable_gradient_checkpointing=args.disable_gradient_checkpointing,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_entity=args.wandb_entity,
            wandb_id=args.wandb_id,
            preprocess_in_loader=args.prefetch_queue_size > 0,
            dataloader_timeout=args.dataloader_timeout,
        )

        if args.disable_wandb:
            trainer.use_wandb = False

        trainer.setup_model()
        trainer.setup_optimizer()

        start_step = 0
        if args.resume:
            trainer.load_checkpoint(args.resume)
            start_step = trainer.global_step

        # Resume: set start_index for each tier dataset
        if start_step > 0:
            total_accum = len(accum_order)
            for tier_idx, tier_ds in enumerate(tier_datasets):
                tier_ds.start_index = start_step * tier_micro_batches[tier_idx] * dist_info.world_size * total_accum
                _log(f"Tier {tier_idx} start_index = {tier_ds.start_index}")

        # Create per-tier DataLoaders (after start_index is set)
        dataloader_kwargs = {
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "prefetch_queue_size": args.prefetch_queue_size,
            "timeout": args.dataloader_timeout,
        }
        for tier_idx in range(num_tiers):
            tier_loader = trainer._create_tier_dataloader(
                tier_ds=tier_datasets[tier_idx],
                micro_batch=tier_micro_batches[tier_idx],
                async_loader_class=AsyncLoaderClass,
                **dataloader_kwargs,
            )
            tier_loaders.append(tier_loader)

        trainer.train_with_tiers(
            tier_loaders, tier_micro_batches, tier_accum_steps,
            accum_order, balance_weights, actual_effective_batch,
            start_step=start_step,
            gpu_utilization_target=args.gpu_utilization_target,
            tier_stats=tier_stats,
            tier_datasets=tier_datasets,
            async_loader_class=AsyncLoaderClass,
            dataloader_kwargs=dataloader_kwargs,
            balance_mode=args.balance_mode,
        )

    # ── Original single-DataLoader path ───────────────────────────
    elif args.pretrain:
        # dataset_to_episodes_path is optional for local testing without per-sub-dataset normalization
        _log("Creating pretrain streaming dataset...")
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
        _log("Creating streaming dataset...")
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
        fsdp_sharding=args.fsdp_sharding,
        gradient_clip_val=args.gradient_clip_val,
        ckpt_dir=args.ckpt_dir,
        save_every_n_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        disable_gradient_checkpointing=args.disable_gradient_checkpointing,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        wandb_id=args.wandb_id,
        preprocess_in_loader=args.prefetch_queue_size > 0,
        dataloader_timeout=args.dataloader_timeout,
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
        _log(f"Resuming from step {start_step}, dataset.start_index = {train_dataset.start_index}")

    # 创建 DataLoader（必须在 start_index 设置之后，worker fork 时会读取 dataset 属性）
    raw_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        timeout=args.dataloader_timeout,
    )

    # When prefetch is enabled, create a preprocess_fn that runs CPU-only preprocessing
    # in the background thread, so the main training thread only needs GPU transfer.
    # Only PretrainAsyncDecodeDataLoader supports prefetch_queue_size and preprocess_fn.
    loader_kwargs = {
        "dataloader": raw_loader,
        "dataset": train_dataset,
        "collate_fn": AsyncLoaderClass.make_collate_fn(),
    }
    if args.pretrain:
        preprocess_fn_for_loader = None
        if args.prefetch_queue_size > 0:
            def preprocess_fn_for_loader(batch):
                """Preprocess batch in prefetch thread (CPU steps, skip GPU transfer)."""
                special_data = trainer._extract_special_fields(batch)
                batch = trainer.preprocessor(batch, skip_device_step=True)
                batch = trainer._restore_special_fields(batch, special_data)
                return batch
        loader_kwargs["prefetch_queue_size"] = args.prefetch_queue_size
        loader_kwargs["preprocess_fn"] = preprocess_fn_for_loader

    train_loader = AsyncLoaderClass(**loader_kwargs)

    trainer.train(train_loader, start_step=start_step)

    cleanup_distributed()
    _log("Training completed!")


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = "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"
    
    main()