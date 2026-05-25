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

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.lola_dataset import LoLADataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola_v07 import LoLAV07Config, LoLAV07Policy
from lerobot.policies.factory import make_pre_post_processors

# 设置日志
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

        _log(f"Distributed initialized: rank={world_rank}, local_rank={local_rank}, "
                    f"world_size={world_size}, master={master_uri}")
    else:
        _log(f"Single GPU mode: local_rank={local_rank}")
    
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


class InterconnectMonitor:
    """Monitor NVLink, PCIe, and InfiniBand throughput via NVML and sysfs."""

    def __init__(self, device: torch.device):
        self.available = HAS_NVML
        if not self.available:
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
                    self._nvlink_supported = False
            except Exception:
                self._nvlink_supported = False
        else:
            self._nvlink_supported = False

        # Pre-check PCIe byte counter fields
        self._pcie_supported = True
        try:
            vals = pynvml.nvmlDeviceGetFieldValues(self.handle, [
                pynvml.NVML_FI_DEV_PCIE_COUNT_RX_BYTES,
                pynvml.NVML_FI_DEV_PCIE_COUNT_TX_BYTES,
            ])
            if any(v.nvmlReturn != 0 for v in vals):
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
            self._ib_supported = False

        # State for delta computation
        self._prev_pcie_rx = None
        self._prev_pcie_tx = None
        self._prev_nvlink_rcv = None
        self._prev_nvlink_xmit = None
        self._prev_ib_rcv = None
        self._prev_ib_xmit = None
        self._prev_timestamp = None

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


def compute_vlm_max_length(
    dataset_metadata,
    vlm_path: str,
    min_image_pixels: int = 65536,
    max_image_pixels: int = 230400,
) -> int:
    """Auto-compute vlm_max_length from dataset info for static VLM padding.

    Computes: visual_tokens + structural_tokens + max_task_text_tokens + 1 (empty)

    Args:
        dataset_metadata: LeRobotDatasetMetadata with camera_keys, features, tasks info.
        vlm_path: Path to Qwen3.5 model for tokenization.
        min_image_pixels: min_pixels for Qwen smart_resize.
        max_image_pixels: max_pixels for Qwen smart_resize.

    Returns:
        vlm_max_length: fixed tokenizer max_length for static padding.
    """
    import math
    from transformers import AutoProcessor

    # 1. Compute visual tokens per camera using Qwen smart_resize
    merge_size = 2
    patch_size = 16
    factor = merge_size * patch_size  # 32

    visual_tokens_total = 0
    num_images = 0
    for cam_key in dataset_metadata.camera_keys:
        feat = dataset_metadata.features.get(cam_key, {})
        info = feat.get("info", {})
        h = info.get("video.height", feat.get("shape", (256, 256, 3))[0])
        w = info.get("video.width", feat.get("shape", (256, 256, 3))[1])

        # Qwen2.5-VL smart_resize
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

    # 2. Compute structural tokens from chat template
    # For Qwen3.5 with N images:
    # <|im_start|>user\n (3) + N*(<|vision_start|><|vision_end|>) (2*N) +
    # <|im_end|>\n (2) + <|im_start|>assistant\n (3) + ৬\n (2) = 10 + 2*N
    structural_tokens = 10 + 2 * num_images

    # 3. Compute max task text tokens
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

    # 4. Total: visual + structural + text + 1 (empty token from LolaEmptyTokenProcessor)
    vlm_max_length = visual_tokens_total + structural_tokens + max_task_tokens + 1
    _log(f"Auto-computed vlm_max_length={vlm_max_length} "
         f"(visual={visual_tokens_total}, structural={structural_tokens}, "
         f"max_text={max_task_tokens}, empty=1)")
    return vlm_max_length


def get_deepspeed_config(
    learning_rate: float = 2.5e-5,
    weight_decay: float = 0.01,
    gradient_clip_val: float = 1.0,
    train_vlm: bool = False,
    batch_size: int = 4,
    world_size: int = 1,
    reduce_bucket_size: float = 5e7,
    allgather_bucket_size: float = 5e7,
    zero_stage: int = 2,
):
    """Generate default DeepSpeed ZeRO config for B200 GPUs (~183GB each).

    Key design decisions:
    - No CPU offload: 183GB per B200 sufficient for 2B-10B models
    - overlap_comm + reduce_scatter: efficient on NVLink-connected systems (ZeRO-2)
    - contiguous_gradients + round_robin_gradients: memory efficiency (ZeRO-2)
    - Bucket sizes 5e7: finer granularity improves compute/comm overlap on NVLink
    - partition_activations=False: not needed at 4-5B scale on 183GB GPUs; enable for 10B+
    - Optimizer in config: DeepSpeed creates AdamW, ensuring proper ZeRO state partitioning
    - ZeRO-1: optimizer state partitioning only — faster comm, less memory saving than ZeRO-2
    """
    zero_optimization = {
        "stage": zero_stage,
        "allgather_bucket_size": allgather_bucket_size,
        "reduce_bucket_size": reduce_bucket_size,
    }
    if zero_stage >= 2:
        zero_optimization.update({
            "overlap_comm": False,
            "reduce_scatter": True,
            "contiguous_gradients": False,
            "round_robin_gradients": True,
        })

    return {
        "bf16": {"enabled": True},
        "zero_optimization": zero_optimization,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": gradient_clip_val,
        "train_batch_size": batch_size * world_size,
        "train_micro_batch_size_per_gpu": batch_size,
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
    """创建 LoLA 训练用的数据集。"""
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = dataset_metadata.fps

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    _log(f"delta_timestamps: {delta_timestamps}")

    if use_lola_dataset:
        _log(f"Using LoLADataset with max_history_length={max_history_length}")
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
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
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
            _log(f"Wrapped LeRobotDataset with ZScoreActionDataset (gripper dims: {gripper_dim_indices_abs})")

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
# 训练器
# ----------------------------------------------------------------------
class LoLAV07Trainer:
    """原生 PyTorch 训练器，支持 DDP 和 FSDP"""

    def __init__(
        self,
        config: LoLAV07Config,
        dataset_stats: dict | None,
        dist_info: dict,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        train_vlm: bool = False,
        vlm_lr: float = 1e-6,
        strategy: str = "ddp",
        gradient_clip_val: float = 1.0,
        batch_size: int = 4,
        ckpt_dir: str = "/data_16T/deepseek/checkpoints/lola",
        save_every_n_steps: int | None = 500,
        save_every_n_epochs: int | None = None,
        log_every_n_steps: int = 10,
        # Wandb 参数
        wandb_project: str = "lola-azure",
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_id: str | None = None,
        deepspeed_config_path: str | None = None,
        deepspeed_reduce_bucket_size: float = 5e7,
        deepspeed_allgather_bucket_size: float = 5e7,
        deepspeed_zero_stage: int = 2,
        training_args: dict | None = None,
        dataset_metadata: dict | None = None,
    ):
        self.config = config
        self.dataset_stats = dataset_stats
        self.dist_info = dist_info
        self.training_args = training_args
        self.dataset_metadata = dataset_metadata
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.train_vlm = train_vlm
        self.vlm_lr = vlm_lr
        self.strategy = strategy
        self.gradient_clip_val = gradient_clip_val
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.current_epoch = 0
        self.deepspeed_config_path = deepspeed_config_path
        self.deepspeed_reduce_bucket_size = deepspeed_reduce_bucket_size
        self.deepspeed_allgather_bucket_size = deepspeed_allgather_bucket_size
        self.deepspeed_zero_stage = deepspeed_zero_stage

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
        self.model_engine = None  # DeepSpeed engine (set by _setup_deepspeed)
        self.preprocessor = None
        self.postprocessor = None

        # 混合精度：BF16 不需要 GradScaler，FP16 才需要
        self.use_bf16 = True  # 使用 BF16 精度
        self.scaler = None if self.use_bf16 else torch.amp.GradScaler("cuda")

        # 训练状态
        self.global_step = 0
        self.best_loss = float("inf")
        self.interconnect_monitor = None

    def setup_model(self):
        # Enable cuDNN SDPA backend for Blackwell GPUs (cuDNN 9.10+ has dedicated kernels)
        torch.backends.cuda.enable_cudnn_sdp(True)
        cudnn_sdp_available = torch.backends.cuda.cudnn_sdp_enabled()
        _log(f"cuDNN SDPA backend: enabled={cudnn_sdp_available}")
        
        """设置模型"""
        _log(f"Loading LoLA Policy on {self.device}...")

        # 加载 LoLA Policy
        self.policy = LoLAV07Policy(self.config)
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
            _log("Freezing VLM parameters...")
            for param in self.policy.vlm.parameters():
                param.requires_grad = False
            self.policy.vlm.eval()

        # 打印参数统计
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy.parameters())
        _log(f"Trainable params: {trainable_params:,} / {total_params:,}")

        # 设置分布式
        if self.is_distributed:
            if self.strategy == "fsdp":
                self._setup_fsdp()
            elif self.strategy == "deepspeed":
                self.model = self.policy  # DeepSpeed wrapping deferred to _setup_deepspeed()
            else:
                self._setup_ddp()
        else:
            self.model = self.policy

        self.interconnect_monitor = InterconnectMonitor(self.device)

    def _setup_ddp(self):
        """设置 DDP (通信/计算重叠优化)"""
        _log("Setting up DDP...")
        self.model = DDP(
            self.policy,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    def _setup_fsdp(self):
        """设置 FSDP - v07: include v07 encoder classes, activation checkpointing for DiT blocks"""
        _log("Setting up FSDP...")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
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

        self.model = FSDP(
            self.policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
        )

        # v07: Apply FSDP activation checkpointing to DiT transformer blocks
        if self.config.gradient_checkpointing:
            from torch.distributed.fsdp import apply_activation_checkpointing, CheckpointWrapper
            _log("Enabling FSDP activation checkpointing for DiT blocks...")
            # Checkpoint double-stream and single-stream DiT blocks
            # These are the most memory-intensive layers (attention + FFN)
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=CheckpointWrapper,
                check_fn=lambda submodule: isinstance(
                    submodule,
                    (LoLADualExpertDoubleBlock, LoLADualExpertSingleBlock),
                ),
            )
            _log("FSDP activation checkpointing enabled for DiT blocks")


    def _setup_deepspeed(self):
        """Set up DeepSpeed ZeRO-2 engine. Called after setup_model() and setup_optimizer()."""
        if not HAS_DEEPSPEED:
            raise ImportError("DeepSpeed required for strategy='deepspeed'. pip install deepspeed")

        import deepspeed
        _log(f"Setting up DeepSpeed ZeRO-{self.deepspeed_zero_stage}...")

        ds_config = get_deepspeed_config(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip_val=self.gradient_clip_val,
            train_vlm=self.train_vlm,
            batch_size=self.batch_size,
            world_size=self.world_size,
            reduce_bucket_size=self.deepspeed_reduce_bucket_size,
            allgather_bucket_size=self.deepspeed_allgather_bucket_size,
            zero_stage=self.deepspeed_zero_stage,
        )
        if self.deepspeed_config_path is not None:
            import json
            with open(self.deepspeed_config_path) as f:
                custom_config = json.load(f)
            ds_config.update(custom_config)

        # v07: Separate parameter groups for DeepSpeed
        encoder_lr_mult = self.config.encoder_lr_mult
        base_lr = self.learning_rate
        trainable_param_groups = [
            {"params": list(self.policy.model.dit.parameters()), "lr": base_lr},
            {"params": list(self.policy.model.vlm_bridge.parameters()), "lr": base_lr},
            {"params": list(self.policy.model.action_encoder.parameters()), "lr": base_lr * encoder_lr_mult},
            {"params": list(self.policy.model.arm_dit_to_latent.parameters()), "lr": base_lr * encoder_lr_mult},
            {"params": list(self.policy.model.grip_dit_to_latent.parameters()), "lr": base_lr * encoder_lr_mult},
        ]
        if self.policy.model.state_encoder is not None:
            trainable_param_groups.append({"params": list(self.policy.model.state_encoder.parameters()), "lr": base_lr * encoder_lr_mult})
        if self.train_vlm and hasattr(self.policy, "vlm"):
            trainable_param_groups.append({"params": list(self.policy.vlm.parameters()), "lr": self.vlm_lr})
        # Filter out params that don't require grad, then remove empty groups
        for group in trainable_param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        trainable_param_groups = [g for g in trainable_param_groups if g["params"]]

        # DeepSpeed passes the basic (unwrapped) optimizer to this callable,
        # so OneCycleLR's isinstance(optimizer, Optimizer) check passes.
        def lr_scheduler_callable(optimizer):
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                optimizer,
                max_lr=[group["lr"] for group in optimizer.param_groups],
                total_steps=self.total_steps,
                pct_start=min(self.warmup_ratio, 0.1),
                anneal_strategy="cos",
            )

        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.policy,
            model_parameters=trainable_param_groups,
            config=ds_config,
            lr_scheduler=lr_scheduler_callable,
            dist_init_required=False,
        )

        # Dummy Cuda Memory Allocation to avoid mem segmentation
        # torch.cuda.empty_cache()
        # free_mem, total_mem = torch.cuda.mem_get_info()
        # _log(f"Free GPU memory: {free_mem / 1024 ** 2:.2f}MB / {total_mem / 1024 ** 2:.2f}MB")

        # allocate_ratio = 0.9
        # dummy_size = int(allocate_ratio * free_mem)
        # dummy_tensor = torch.empty(dummy_size, dtype=torch.int8, device="cuda")

        # del dummy_tensor
        # _log(f"Allocated {allocate_ratio * 100:.0f}% of GPU memory for dummy tensor")

        # Initialize the model engine
        self.model = model_engine
        self.model_engine = model_engine
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        self._configure_deepspeed_checkpointing()

        trainable_count = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        _log(f"DeepSpeed ZeRO-{self.deepspeed_zero_stage} initialized: {trainable_count:,} trainable params")

    def _configure_deepspeed_checkpointing(self):
        """Replace PyTorch checkpointing with DeepSpeed's in LoLA model."""
        if not self.config.gradient_checkpointing:
            return

        _log("Configuring DeepSpeed activation checkpointing for DiT...")
        self.policy.model.set_deepspeed_checkpointing()

        if self.train_vlm:
            _log("Configuring DeepSpeed activation checkpointing for VLM...")
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

    def setup_optimizer(self, total_steps: int | None = None):
        """设置优化器 - v07: Separate parameter groups with encoder LR multiplier"""
        if total_steps is None:
            total_steps = self.max_steps
        if total_steps is None:
            raise ValueError("Either max_steps or max_epochs must be provided")

        self.total_steps = total_steps

        if self.strategy == "deepspeed":
            _log("Skipping optimizer creation - DeepSpeed will create from config")
            return

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
        if self.train_vlm and hasattr(self.policy, "vlm"):
            param_groups.append({"params": list(self.policy.vlm.parameters()), "lr": self.vlm_lr})

        # Filter out params that don't require grad, then remove empty groups
        for group in param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        param_groups = [g for g in param_groups if g["params"]]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Diagnostic: verify optimizer param group coverage
        opt_total = sum(len(g["params"]) for g in self.optimizer.param_groups)
        opt_params = sum(p.numel() for g in self.optimizer.param_groups for p in g["params"])
        all_params = sum(p.numel() for p in self.policy.parameters())
        _log(f"Optimizer param groups: {len(self.optimizer.param_groups)}, "
             f"total params in optimizer: {opt_params:,} / {all_params:,}")
        for i, g in enumerate(self.optimizer.param_groups):
            n = sum(p.numel() for p in g["params"])
            _log(f"  Group {i}: lr={g['lr']:.2e}, params={n:,}")

        # Diagnostic: check optimizer state dtype after first step
        self._pending_optimizer_dtype_check = True

        from torch.optim.lr_scheduler import OneCycleLR
        warmup_ratio = min(self.warmup_ratio, 0.1)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[group["lr"] for group in self.optimizer.param_groups],
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
        )

    def _extract_special_fields(self, batch):
        """提取特殊字段"""
        special_data = {}
        keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length",
                           "hist_states_full", "hist_states_mask", "hist_states_length"]
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
        """单步训练 - v07: warmup t-truncation and v-loss alarm"""
        t0 = time.monotonic()

        # 移动数据到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        t_device = time.monotonic() - t0

        t1 = time.monotonic()
        # 提取特殊字段
        special_data = self._extract_special_fields(batch)

        # 预处理
        batch = self.preprocessor(batch)
        batch = self._restore_special_fields(batch, special_data)
        t_preprocess = time.monotonic() - t1

        # v07: Warmup t-truncation
        warmup_steps = int(self.total_steps * self.config.warmup_pct)
        time_param = None
        if self.global_step < warmup_steps:
            b = batch["action"].shape[0]
            t_raw = torch.distributions.Beta(
                self.config.time_sampling_beta_alpha,
                self.config.time_sampling_beta_beta,
            ).sample((b,)).to(self.device)
            time_param = t_raw * (self.config.warmup_t_trunc_high - self.config.warmup_t_trunc_low) + self.config.warmup_t_trunc_low

        # 前向传播（混合精度）
        t2 = time.monotonic()
        if self.strategy == "deepspeed":
            # DeepSpeed handles BF16 autocast internally when bf16.enabled=True
            loss, loss_dict = self.model(batch, time=time_param)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = self.model(batch, time=time_param)

        # v07: v-loss alarm
        if loss_dict.get("v_loss", 0) > 1.0:
            _log(f"[WARNING] v_loss = {loss_dict['v_loss']:.4f} > 1.0 at step {self.global_step}")
        t_model_fwd = time.monotonic() - t2

        if timing_dict is not None:
            timing_dict["device_s"] = t_device
            timing_dict["preprocess_s"] = t_preprocess
            timing_dict["model_fwd_s"] = t_model_fwd

        return loss, loss_dict

    def train(self, train_loader, start_step: int = 0, start_epoch: int = 0):
        """训练循环，增强 wandb 日志（throughput / timing / GPU metrics）"""
        self.global_step = start_step
        self.model.train()

        # 创建 checkpoint 目录
        # Rank 0 generates timestamp and broadcasts to all ranks to ensure consistency
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.world_size > 1:
            time_str_list = [time_str]
            dist.broadcast_object_list(time_str_list, src=0)
            time_str = time_str_list[0]
        ckpt_dir = os.path.join(self.ckpt_dir, f"lola-v07-azure-{time_str}")
        if self.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            _log(f"Checkpoint directory: {ckpt_dir}")

            # Save all training configurations as JSON
            import json
            import dataclasses
            from pathlib import Path

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
                "lola_config": self.config,
                "distributed": self.dist_info,
                "training_args": self.training_args,
                "dataset_metadata": self.dataset_metadata,
            })

            config_path = os.path.join(ckpt_dir, "training_config.json")
            with open(config_path, "w") as f:
                json.dump(full_config, f, indent=2, default=str)
            _log(f"Training config saved to {config_path}")

        _log(f"Starting training from step {start_step}, epoch {start_epoch}")

        # 计算 resume 时需要跳过的 batch 数
        try:
            batches_per_epoch = len(train_loader)
            _log(f"Total batches per epoch: {batches_per_epoch}")
        except TypeError:
            batches_per_epoch = None
            _log("IterableDataset detected: cannot determine batches per epoch")

        # resume skip logic: use start_epoch for epoch-based skip
        if start_epoch > 0 and batches_per_epoch is not None:
            skip_epochs = start_epoch
            skip_batches = start_step % batches_per_epoch if start_step > 0 else 0
            _log(f"Resuming: skipping {skip_epochs} epochs + {skip_batches} batches")
        elif start_step > 0 and batches_per_epoch is not None:
            skip_epochs = start_step // batches_per_epoch
            skip_batches = start_step % batches_per_epoch
            _log(f"Resuming: skipping {skip_epochs} epochs + {skip_batches} batches")
        elif start_step > 0 and batches_per_epoch is None:
            skip_epochs = 0
            skip_batches = 0
            _log(
                f"Resuming from step {start_step} with IterableDataset: "
                "data will restart from the beginning (model/optimizer/scheduler states are restored). "
                "For precise data resume, use map-style dataset or add start_index to IterableDataset."
            )
        else:
            skip_epochs = 0
            skip_batches = 0

        epoch = start_epoch
        while True:
            # 终止条件
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break
            if self.max_steps is not None and self.global_step >= self.total_steps:
                break
            epoch += 1
            self.current_epoch = epoch
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader):
                if self.max_steps is not None and self.global_step >= self.total_steps:
                    break

                # Map-style 数据集：跳过已训练的 batch
                if skip_epochs > 0 or skip_batches > 0:
                    if skip_epochs > 0:
                        skip_epochs -= 1
                        break  # 跳过整个 epoch
                    skip_batches -= 1
                    continue

                step_start = time.monotonic()

                if self.strategy != "deepspeed":
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
                if self.strategy == "deepspeed":
                    self.model.backward(loss)
                elif self.use_bf16:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()
                bwd_s = time.monotonic() - bwd_start

                # ── Gradient clipping ─────────────────────────────────
                clip_start = time.monotonic()
                if self.strategy == "deepspeed":
                    grad_norm = None  # DeepSpeed clips from config
                elif self.gradient_clip_val > 0:
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
                if self.strategy == "deepspeed":
                    self.model.step()
                elif self.use_bf16:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                opt_s = time.monotonic() - opt_start

                # 学习率调度（DeepSpeed engine.step() 已内置 scheduler.step()）
                if self.strategy != "deepspeed":
                    self.scheduler.step()

                self.global_step += 1

                # Diagnostic: check optimizer state dtype after first step
                if self._pending_optimizer_dtype_check and self.global_step == 1:
                    self._pending_optimizer_dtype_check = False
                    dtypes_found = {}
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            if p in self.optimizer.state:
                                state = self.optimizer.state[p]
                                for key, val in state.items():
                                    if isinstance(val, torch.Tensor):
                                        dtypes_found[str(val.dtype)] = dtypes_found.get(str(val.dtype), 0) + val.numel()
                    if self.is_main_process:
                        _log(f"Optimizer state dtype breakdown:")
                        for dtype, count in dtypes_found.items():
                            size_gb = count * (2 if 'bf16' in dtype or 'half' in dtype else 4) / 1e9
                            _log(f"  {dtype}: {count:,} elements, ~{size_gb:.1f} GB")

                update_s = time.monotonic() - step_start
                batch_per_s = 1.0 / update_s if update_s > 0 else 0

                # ── CUDA Memory Clear and possible GC ─────────────────
                if self.global_step % 1000 == 0:
                    gc_start = time.monotonic()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    gc_s = time.monotonic() - gc_start
                    _log(f"[Step {self.global_step}/{self.total_steps}] GC took {gc_s:.3f}s")

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
                            f"[Step {self.global_step}/{self.total_steps}] "
                            f"Epoch {epoch}/{self.max_epochs or '-'} "
                            f"Loss={loss.item():.4f} LR={lr:.2e} "
                            f"Update={update_s:.2f}s Throughput={batch_per_s:.2f}batch/s"
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
                                "train/epoch": epoch,
                                "train/batch_per_s": batch_per_s,
                                "timing/step_s": update_s,
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

                # 保存 checkpoint
                should_save = False
                if self.save_every_n_steps is not None and self.global_step % self.save_every_n_steps == 0:
                    should_save = True
                if self.save_every_n_epochs is not None and batch_idx == 0 and epoch % self.save_every_n_epochs == 0:
                    should_save = True
                if self.strategy == "deepspeed":
                    if should_save:
                        self.save_checkpoint(ckpt_dir, self.global_step)
                elif should_save and self.is_main_process:
                    self.save_checkpoint(ckpt_dir, self.global_step)

        # 保存最终 checkpoint
        if self.strategy == "deepspeed":
            self.save_checkpoint(ckpt_dir, self.global_step, is_final=True)
        elif self.is_main_process:
            self.save_checkpoint(ckpt_dir, self.global_step, is_final=True)
            _log(f"Training completed! Final checkpoint saved at step {self.global_step}")

        # 关闭 InterconnectMonitor
        if self.interconnect_monitor:
            self.interconnect_monitor.close()

        # 关闭 Wandb
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, ckpt_dir: str, step: int, is_final: bool = False):
        """保存 checkpoint"""
        if self.strategy == "deepspeed":
            tag = f"step_{step:06d}" if not is_final else "final"
            client_state = {
                "step": step,
                "epoch": self.current_epoch,
            }
            self.model.save_checkpoint(
                save_dir=ckpt_dir,
                tag=tag,
                client_state=client_state,
                exclude_frozen_parameters=True,
            )
            ckpt_path = f"{ckpt_dir}/{tag}"
        elif self.strategy == "fsdp":
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
                    "epoch": [self.current_epoch],
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
                "epoch": self.current_epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }, ckpt_path)

        _log(f"Checkpoint saved: {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str):
        """加载 checkpoint"""
        if self.strategy == "deepspeed":
            load_path, client_state = self.model.load_checkpoint(
                load_dir=ckpt_path,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            if load_path is None:
                raise ValueError(f"Failed to load DeepSpeed checkpoint from {ckpt_path}")
            self.global_step = client_state.get("step", 0)
            self.current_epoch = client_state.get("epoch", 0)
        elif self.strategy == "fsdp":
            from torch.distributed.checkpoint import load as load_fsdp_checkpoint
            from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

            # FSDP checkpoint 加载：先获取空 state_dict 容器，再 load 填充，最后 set 回模型/优化器
            model_sd, optimizer_sd = get_state_dict(self.model, self.optimizer)
            # 用 list 包装 step，因为 int 是不可变对象，load 无法原地修改
            step_container = [0]
            epoch_container = [0]
            load_fsdp_checkpoint(
                {"model": model_sd, "optimizer": optimizer_sd, "step": step_container, "epoch": epoch_container},
                checkpoint_id=ckpt_path,
            )
            set_state_dict(self.model, self.optimizer, model_state_dict=model_sd, optim_state_dict=optimizer_sd)
            self.global_step = step_container[0]
            self.current_epoch = epoch_container[0]
            # 恢复 scheduler 状态
            scheduler_path = os.path.join(ckpt_path, "scheduler.pt")
            if os.path.exists(scheduler_path):
                scheduler_ckpt = torch.load(scheduler_path, map_location=self.device)
                self.scheduler.load_state_dict(scheduler_ckpt["scheduler_state_dict"])
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if self.is_distributed:
                self.model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.global_step = checkpoint.get("step", 0)
            self.current_epoch = checkpoint.get("epoch", 0)

        _log(f"Checkpoint loaded from: {ckpt_path}, starting from step {self.global_step}")


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    # 初始化分布式
    dist_info = setup_distributed()

    # 参数解析
    parser = argparse.ArgumentParser(description="LoLA V07 Azure Distributed Training")

    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)

    # 训练参数
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp", "deepspeed"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (mutually exclusive with --max_epochs)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max training epochs (mutually exclusive with --max_steps)")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="Save checkpoint every N steps (mutually exclusive with --save_every_n_epochs)")
    parser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every N epochs (mutually exclusive with --save_every_n_steps)")
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
    parser.add_argument("--history_type", type=str, default="action", choices=["action", "state"],
                        help="History type: 'action' uses historical actions, 'state' uses historical observation states")
    parser.add_argument("--state_dim", type=int, default=None,
                        help="State dimension (auto-detected from dataset if not provided)")
    parser.add_argument("--state_encoder_mode", type=str, default="unified", choices=["unified", "separated"],
                        help="State encoder mode: 'unified' (single MLP → 2*hidden, split) or 'separated' (arm/grip separate MLPs)")

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

    # Wandb 参数
    parser.add_argument("--wandb_project", type=str, default="lola-azure", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run id (for resume)")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")

    # DeepSpeed 参数
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to custom DeepSpeed config JSON. Default: ZeRO config tuned for B200.")
    parser.add_argument("--deepspeed_zero_stage", type=int, default=2, choices=[1, 2],
                        help="DeepSpeed ZeRO stage: 1 (optimizer partitioning) or 2 (optimizer+gradient partitioning). Default: 2")
    parser.add_argument("--deepspeed_reduce_bucket_size", type=float, default=5e7,
                        help="DeepSpeed ZeRO reduce bucket size (default: 5e7 for B200 NVLink)")
    parser.add_argument("--deepspeed_allgather_bucket_size", type=float, default=5e7,
                        help="DeepSpeed ZeRO allgather bucket size (default: 5e7 for B200 NVLink)")

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4)

    # Static padding parameters
    parser.add_argument("--static_collate_padding", action="store_true", default=True,
                        help="Use static max_history_length padding in collate (default: enabled)")
    parser.add_argument("--no_static_collate_padding", action="store_true",
                        help="Disable static padding, use dynamic per-batch padding")
    parser.add_argument("--static_vlm_padding", action="store_true",
                        help="Pad VLM tokens to fixed max_length for consistent tensor shapes")
    parser.add_argument("--vlm_max_length", type=int, default=None,
                        help="Override tokenizer max_length for static VLM padding; auto-compute if None")

    # 归一化参数
    parser.add_argument("--norm_mode", type=str, default="default",
                        choices=["default", "robovlm", "zscore"],
                        help="归一化模式: default(LoLA默认MEAN_STD), robovlm(min-max→[-1,1],全IDENTITY), zscore(arm=z-score,gripper=二值化{0,1})")
    parser.add_argument("--norm_min", type=float, default=-0.65,
                        help="RoboVLM 归一化下界")
    parser.add_argument("--norm_max", type=float, default=0.65,
                        help="RoboVLM 归一化上界")

    args = parser.parse_args()

    # 检查数据集参数
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

    # 提前初始化 Wandb（在数据加载/模型设置之前，以便记录所有日志）
    use_wandb = HAS_WANDB and not args.disable_wandb and dist_info["world_rank"] == 0

    # broadcast 是集体操作，必须所有 rank 同时参与，不能放在 if use_wandb 里
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if dist_info["world_size"] > 1:
        time_str_list = [time_str]
        dist.broadcast_object_list(time_str_list, src=0)
        time_str = time_str_list[0]

    if use_wandb:
        wandb_run_name = args.wandb_name or f"lola-{args.strategy}-{time_str}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            entity=args.wandb_entity,
            id=args.wandb_id,
            resume="allow" if args.wandb_id else None,
            config={
                "learning_rate": args.learning_rate,
                "weight_decay": 0.0,
                "max_steps": args.max_steps,
                "max_epochs": args.max_epochs,
                "batch_size": args.batch_size,
                "strategy": args.strategy,
                "world_size": dist_info["world_size"],
                "train_vlm": args.train_vlm,
                "gradient_clip_val": args.gradient_clip_val,
            },
        )
        _log(f"Wandb initialized: {wandb_run_name}")

    # 所有 rank 等待 rank 0 完成 wandb 初始化后再继续
    if dist_info["world_size"] > 1:
        dist.barrier()

    # 打印配置
    if dist_info["world_rank"] == 0:
        _log("=" * 60)
        _log("LoLA V07 Azure Distributed Training")
        _log("=" * 60)
        _log(f"Dataset: {args.dataset_repo_id or args.dataset_root}")
        _log(f"Strategy: {args.strategy}")
        _log(f"World Size: {dist_info['world_size']}")
        _log(f"Batch Size: {args.batch_size}")
        _log(f"Learning Rate: {args.learning_rate}")
        _log(f"Max Steps: {args.max_steps or 'N/A (epoch-based)'}")
        _log(f"Max Epochs: {args.max_epochs or 'N/A (step-based)'}")
        _log(f"VLM Path: {args.vlm_path}")
        _log(f"Train VLM: {args.train_vlm}")
        _log("=" * 60)

    # 获取数据集元数据
    _log(f"Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    if "observation.state" in features:
        state_dim = features["observation.state"].shape[0]
    elif args.state_dim is not None:
        state_dim = args.state_dim
    else:
        state_dim = action_dim  # fallback

    _log(f"Dataset: {dataset_metadata.total_episodes} episodes, {dataset_metadata.total_frames} frames")
    _log(f"Action dim: {action_dim}")

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
        gradient_checkpointing=gradient_checkpointing,
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

    # 创建数据集
    _log("Creating dataset...")
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
    _log(f"Dataset size: {len(train_dataset)}")

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

    # Static padding for consistent tensor shapes across steps
    use_static_padding = not args.no_static_collate_padding and args.load_full_history
    static_max_len = args.max_history_length if use_static_padding else None
    if static_max_len is not None:
        _log(f"Using static collate padding to max_history_length={static_max_len}")
    collate = make_collate_fn(static_max_len=static_max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,  # 分布式训练建议 drop_last
    )

    # 创建训练器
    trainer = LoLAV07Trainer(
        config=config,
        dataset_stats=dataset_metadata.stats,
        dist_info=dist_info,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        train_vlm=args.train_vlm,
        strategy=args.strategy,
        gradient_clip_val=args.gradient_clip_val,
        batch_size=args.batch_size,
        ckpt_dir=args.ckpt_dir,
        save_every_n_steps=args.save_every_n_steps,
        save_every_n_epochs=args.save_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        # Wandb 参数
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        wandb_id=args.wandb_id,
        deepspeed_config_path=args.deepspeed_config,
        deepspeed_reduce_bucket_size=args.deepspeed_reduce_bucket_size,
        deepspeed_allgather_bucket_size=args.deepspeed_allgather_bucket_size,
        deepspeed_zero_stage=args.deepspeed_zero_stage,
        # Config saving
        training_args=vars(args),
        dataset_metadata={
            "total_episodes": dataset_metadata.total_episodes,
            "total_frames": dataset_metadata.total_frames,
            "fps": dataset_metadata.fps,
            "features": {k: {"shape": list(v.shape), "type": str(v.type)} for k, v in features.items()},
        },
    )

    # Wandb 已在 main() 开头提前初始化，同步 trainer 的 use_wandb 标记
    trainer.use_wandb = use_wandb

    # 设置模型
    trainer.setup_model()

    # 计算 total_steps
    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        batches_per_epoch = len(train_loader)
        total_steps = args.max_epochs * batches_per_epoch
        _log(f"Epoch-based training: {args.max_epochs} epochs × {batches_per_epoch} batches = {total_steps} total steps")

    # 设置优化器
    trainer.setup_optimizer(total_steps=total_steps)

    # DeepSpeed 初始化（必须在 setup_model() 和 setup_optimizer() 之后）
    if args.strategy == "deepspeed":
        trainer._setup_deepspeed()

    # 加载 checkpoint
    start_step = 0
    start_epoch = 0
    if args.resume:
        trainer.load_checkpoint(args.resume)
        start_step = trainer.global_step
        start_epoch = trainer.current_epoch

    # 开始训练
    trainer.train(train_loader, start_step=start_step, start_epoch=start_epoch)

    # 清理
    cleanup_distributed()
    _log("Training completed!")


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = "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"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    main()
