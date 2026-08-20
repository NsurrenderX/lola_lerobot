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

# 调试开关: LOLA_DETECT_ANOMALY=1 时开启 autograd anomaly detection (定位 backward 错误来源)
if os.environ.get("LOLA_DETECT_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True)

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

# resume 搜索 (同目录模块; run 集合目录 → 按训练配置匹配并选步数最多者)
from resume_search import (
    build_current_snapshot,
    diff_snapshot,
    make_serializable,
    resolve_merge_history_stream,
    resolve_resume_auto,
)

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
    frames_per_cam: int = 1,
) -> int:
    """Auto-compute vlm_max_length from dataset info for static VLM padding.

    Computes: visual_tokens + structural_tokens + max_task_text_tokens + 1 (empty)

    Args:
        dataset_metadata: LeRobotDatasetMetadata with camera_keys, features, tasks info.
        vlm_path: Path to Qwen3.5 model for tokenization.
        min_image_pixels: min_pixels for Qwen smart_resize.
        max_image_pixels: max_pixels for Qwen smart_resize.
        frames_per_cam: 每相机送入 VLM 的帧数 (obs_prev_chunk_frame=2 等),
            直接乘进 visual token 计数。

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
        visual_tokens_total += tokens * frames_per_cam
        num_images += frames_per_cam

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
    reduce_bucket_size: float = 5e8,
    allgather_bucket_size: float = 5e7,
    zero_stage: int = 2,
    stage3_prefetch_bucket_size: float = 5e8,
    stage3_max_live_parameters: float = 2e9,
    stage3_max_reuse_distance: float = 2e9,
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
    if zero_stage == 2:
        zero_optimization.update({
            "overlap_comm": False,
            "reduce_scatter": True,
            "contiguous_gradients": False,
            "round_robin_gradients": True,
        })
    # ZeRO-3: 通信参数针对 A100-40G PCIe 拓扑调优 (2026-08-07, 依据 log_multiepoch.txt
    # 实测: 解冻后 6.5s/step, 通信为瓶颈)。
    #   - prefetch_bucket 5e7→5e8: 默认值在 ~8GB VLM 参数上产生大量小 allgather,
    #     预取与计算重叠差; 放大后单次传输更大、次数更少。
    #   - max_live_parameters / max_reuse_distance 1e9→2e9: 放宽后 fwd 已 gather 的
    #     参数更可能保留到 bwd 复用, 减少重复 gather。代价是 gather 态 bf16 参数驻留
    #     上限 ~4GB (2e9 参数), 实测 40G 卡 reserved 余量 ~16GB, 可承受。
    #   - reduce_bucket_size 默认 5e7→5e8 (见函数签名): ZeRO-3 用它做梯度
    #     reduce-scatter 分桶 (stage3.py:1351), DS 自带默认就是 5e8, 之前的 5e7
    #     是给 ZeRO-2/NVLink 调的值。
    #   - gather_16bit_weights_on_model_save=False (默认) 即按 rank 分片保存,
    #     resume 需相同 world size。
    # ZeRO-2 专属的 round_robin_gradients 等键不能传给 stage 3。
    # 例外: param_persistence_threshold 必须为 0 — DS 0.18 的 ZeRO-3 load_state_dict
    # 末尾会对 persistent(小)参数执行 partition(), 破坏其梯度账目, resume 后第一次
    # backward 即崩 ("size of tensor a (0) ... at AccumulateGrad"); 置 0 禁用该路径。
    if zero_stage >= 3:
        zero_optimization["param_persistence_threshold"] = 0
        zero_optimization["stage3_prefetch_bucket_size"] = stage3_prefetch_bucket_size
        zero_optimization["stage3_max_live_parameters"] = stage3_max_live_parameters
        zero_optimization["stage3_max_reuse_distance"] = stage3_max_reuse_distance

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


def peek_deepspeed_checkpoint_vlm_unfrozen(ckpt_path: str) -> bool:
    """Peek a DeepSpeed checkpoint's client_state for vlm_unfrozen WITHOUT loading the engine.

    用于 resume 场景: 如果 checkpoint 保存时 VLM 已解冻, 训练引擎必须在加载前就以
    "VLM 可训练" 的结构构建 (否则优化器分组不匹配, 加载会失败或退化为丢失 Adam 矩
    的重建)。client_state 由 save_checkpoint 合并进每个 rank 的 model_states 文件,
    读任意一个即可。

    Returns:
        True if the checkpoint was saved with VLM unfrozen, False otherwise (or on any error).
    """
    try:
        # ckpt_path 可能是 tag 目录本身或其父目录
        if os.path.exists(os.path.join(ckpt_path, "latest")):
            tag = open(os.path.join(ckpt_path, "latest")).read().strip()
            tag_dir = os.path.join(ckpt_path, tag)
        else:
            tag_dir = ckpt_path
        if not os.path.isdir(tag_dir):
            return False
        for fname in sorted(os.listdir(tag_dir)):
            if fname.endswith("model_states.pt"):
                state = torch.load(os.path.join(tag_dir, fname), map_location="cpu", weights_only=False)
                return bool(state.get("vlm_unfrozen", False))
    except Exception as e:
        _log(f"peek_deepspeed_checkpoint_vlm_unfrozen({ckpt_path}) failed: {e}")
    return False


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
    # 方案B: 合并 transition + task 为连续历史流
    merge_history_stream: bool = False,
    # Stats mode for z-score normalization
    stats_mode: str = "original",
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
            merge_history_stream=merge_history_stream,
            stats_mode=stats_mode,
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
            action_key = "action_incremental" if stats_mode == "incremental" else "action"
            _mean = dataset_stats[action_key]["mean"]
            _std = dataset_stats[action_key]["std"]
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


def make_collate_fn(static_max_len: int | None = None, chunk_size: int | None = None):
    """Create a collate function with optional static padding length.

    If static_max_len is provided, hist_actions_full and hist_actions_mask
    are always padded to this fixed length, producing constant-size tensors
    every step. This eliminates CUDA memory fragmentation and stabilizes
    DeepSpeed ZeRO-2 reduce-scatter timing.

    If static_max_len is None, falls back to dynamic per-batch padding.

    Padding scheme ("middle padding", requires chunk_size):
    Each item's history is laid out as [transition_padded | task_padded] with
    the transition block starting at index 0 (see LoLADataset). The model
    locates the transition/task boundary by slicing the FIRST max_n_tc chunks,
    so blanket left-padding of the merged sequence would shift the boundary
    and feed zero-padding as "transition" context. Instead, each segment is
    left-padded INDEPENDENTLY:

        [t_left_pad | transition | k_left_pad | task]
         └─ to batch max_n_tc ──┘  └─ right-aligned ──┘

    In static mode, total length = static_max_len every step:
    t_target = batch_max_n_tc * chunk_size, k_target = static_max_len - t_target
    (the boundary moves per step but the tensor SHAPE is constant, and the
    model reads max_n_tc from the batch so the two stay consistent).
    """
    variable_length_keys = {"hist_actions_full", "hist_actions_mask", "hist_states_full", "hist_states_mask"}

    def collate_fn(batch):
        # Per-sample transition chunk counts (0 when key absent, e.g. LeRobotDataset)
        n_tc_list = []
        for item in batch:
            n_tc = item.get("n_transition_chunks", 0)
            n_tc_list.append(int(n_tc.item()) if isinstance(n_tc, torch.Tensor) else int(n_tc))
        has_transition_info = any("n_transition_chunks" in item for item in batch)
        # Middle padding applies only when we know chunk_size and the dataset
        # provides transition boundaries; otherwise fall back to legacy left pad.
        use_middle_padding = chunk_size is not None and has_transition_info
        t_target = (max(n_tc_list) * chunk_size) if use_middle_padding else 0

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
                padded_values = []
                if use_middle_padding:
                    # Per-segment targets; k_target absorbs the rest of the budget
                    if static_max_len is not None:
                        k_target = max(static_max_len - t_target, 0)
                    else:
                        k_target = max(
                            v.shape[0] - n_tc * chunk_size for v, n_tc in zip(values, n_tc_list)
                        )
                    for v, n_tc in zip(values, n_tc_list):
                        t_len = n_tc * chunk_size
                        t_part, k_part = v[:t_len], v[t_len:]

                        # Transition segment: left-pad to t_target (never longer by construction)
                        t_pad = t_target - t_part.shape[0]
                        if t_pad > 0:
                            pad_shape = (t_pad,) + tuple(t_part.shape[1:])
                            t_part = torch.cat([torch.zeros(pad_shape, dtype=v.dtype), t_part], dim=0)

                        # Task segment: left-pad (or left-truncate, keep most recent) to k_target
                        if k_part.shape[0] < k_target:
                            k_pad = k_target - k_part.shape[0]
                            pad_shape = (k_pad,) + tuple(k_part.shape[1:])
                            k_part = torch.cat([torch.zeros(pad_shape, dtype=v.dtype), k_part], dim=0)
                        elif k_part.shape[0] > k_target:
                            k_part = k_part[-k_target:] if k_target > 0 else k_part[:0]

                        padded_values.append(torch.cat([t_part, k_part], dim=0))
                else:
                    # Legacy: left-pad the merged sequence
                    max_len = static_max_len if static_max_len is not None else max(v.shape[0] for v in values)
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
# BF16 Optimizer Wrapper: FP32 master weights for DDP bf16 training
# ----------------------------------------------------------------------
class BF16OptimizerWrapper:
    """Maintains fp32 master params alongside bf16 model params for DDP training.

    Same pattern as DeepSpeed's BF16_Optimizer and FSDP's MixedPrecision:
    optimizer operates entirely in fp32 on master params, updates are copied
    back to bf16 model params after each step.

    Compatible with DDP gradient_as_bucket_view=True and static_graph=True.
    """

    def __init__(self, optimizer: torch.optim.AdamW):
        self.optimizer = optimizer
        self.bf16_param_groups = []   # original bf16 param lists per group
        self.fp32_master_param_groups = []  # fp32 master params per group
        self.param_to_master = {}     # bf16 param -> fp32 master param

        for i, group in enumerate(optimizer.param_groups):
            bf16_params = list(group['params'])
            self.bf16_param_groups.append(bf16_params)

            fp32_masters = []
            for p in bf16_params:
                master = p.detach().clone().float().requires_grad_(True)
                self.param_to_master[p] = master
                fp32_masters.append(master)

            self.fp32_master_param_groups.append(fp32_masters)
            # Swap optimizer's param list to fp32 masters (keep all group metadata)
            group['params'] = fp32_masters

    @torch.no_grad()
    def copy_grads_to_fp32(self):
        """Cast bf16 DDP bucket-view grads to fp32 master param grads."""
        for bf16_list in self.bf16_param_groups:
            for p in bf16_list:
                master = self.param_to_master[p]
                if p.grad is not None:
                    master.grad = p.grad.float()
                else:
                    master.grad = None

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Clip gradient norm on fp32 master params."""
        fp32_params = [m for masters in self.fp32_master_param_groups for m in masters]
        return torch.nn.utils.clip_grad_norm_(fp32_params, max_norm)

    @torch.no_grad()
    def step(self):
        """Optimizer step on fp32 masters, then copy updates to bf16 model params."""
        self.optimizer.step()
        for bf16_list in self.bf16_param_groups:
            for p in bf16_list:
                p.data.copy_(self.param_to_master[p])

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        """Zero bf16 DDP bucket-view grads and fp32 master grads."""
        for bf16_list in self.bf16_param_groups:
            for p in bf16_list:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
        for masters in self.fp32_master_param_groups:
            for m in masters:
                m.grad = None

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        """Save optimizer state + fp32 master param values."""
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fp32_master_params': {
                f"g{i}_p{j}": master.detach().cpu()
                for i, masters in enumerate(self.fp32_master_param_groups)
                for j, master in enumerate(masters)
            },
        }

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state + fp32 master params, sync to bf16 model."""
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        for i, masters in enumerate(self.fp32_master_param_groups):
            for j, master in enumerate(masters):
                key = f"g{i}_p{j}"
                if key in state_dict['fp32_master_params']:
                    master.data.copy_(state_dict['fp32_master_params'][key].to(master.device))
        for bf16_list in self.bf16_param_groups:
            for p in bf16_list:
                p.data.copy_(self.param_to_master[p])


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
        save_every_n_seconds: float | None = None,
        log_every_n_steps: int = 10,
        # Wandb 参数
        wandb_project: str = "lola-azure",
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_id: str | None = None,
        deepspeed_config_path: str | None = None,
        deepspeed_reduce_bucket_size: float = 5e8,
        deepspeed_allgather_bucket_size: float = 5e7,
        deepspeed_zero_stage: int = 2,
        resume_vlm_unfrozen: bool = False,
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
        self.save_every_n_seconds = save_every_n_seconds
        self.log_every_n_steps = log_every_n_steps
        self.current_epoch = 0
        # 原地续训 (方案 B): main() 判定 resume 配置匹配后赋值为被续训的 run 目录,
        # train() 将后续 checkpoint 直接写回该目录; None = 新建时间戳目录
        self.resume_save_dir = None
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
        self.bf16_optimizer = None
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

        # VLM dynamic unfreezing state
        self._vlm_unfrozen = False
        self._vlm_delayed_unfreeze = False
        # For DeepSpeed: engine rebuild mid-step invalidates activation checkpointing
        # closures, so unfreeze is deferred to the step boundary
        self._pending_deepspeed_unfreeze = False
        if self.train_vlm and self.config.vlm_unfreeze_v_loss_threshold > 0:
            self._vlm_delayed_unfreeze = True

        # Resume 一个 VLM 已解冻的 checkpoint: 引擎必须从一开始就以 VLM 可训练的结构
        # 构建 (优化器分组匹配, Adam 矩无损恢复), 跳过延迟解冻的冻结阶段
        self._resume_vlm_unfrozen = resume_vlm_unfrozen
        if resume_vlm_unfrozen and self.train_vlm:
            self._vlm_unfrozen = True
            self._vlm_delayed_unfreeze = False

        # EMA (2026-08-12): {param_name: 本地 shard 副本}。ZeRO-3 下每 rank 只维护
        # 自己分片的 EMA, 零额外通信; None = 关闭 (config.ema_decay <= 0)。
        # 冻结 VLM requires_grad=False 不跟踪 (权重不变, EMA 恒等); 动态解冻时
        # _ema_rebind 以当前权重初始化 VLM 分片并继续跟踪。
        self._ema_state = None

        # Wall-clock checkpoint timer is rank-0-owned. The trigger is broadcast
        # so every rank enters DeepSpeed's collective save on the same step.
        self._last_checkpoint_time = None
        self._checkpoint_timer_signal = None

    # ────────────────────────────────────────────────────────────────
    # EMA (全模型, ZeRO-3 分片本地维护)
    # ────────────────────────────────────────────────────────────────
    def _ema_register(self):
        """注册 EMA: 以当前权重初始化所有 requires_grad 参数的本地副本。"""
        self._ema_state = {}
        with torch.no_grad():
            for name, p in self.policy.named_parameters():
                if p.requires_grad:
                    self._ema_state[name] = p.data.detach().clone()
        n_params = sum(t.numel() for t in self._ema_state.values())
        _log(f"[EMA] 注册 {len(self._ema_state)} 个参数分片, "
             f"本 rank 共 {n_params/1e6:.1f}M 元素 (decay={self.config.ema_decay})")

    @torch.no_grad()
    def _ema_update(self):
        """每步 optimizer.step 之后调用: 本地 shard 原地 EMA 更新。"""
        if self._ema_state is None:
            return
        d = self.config.ema_decay
        for name, p in self.policy.named_parameters():
            ema = self._ema_state.get(name)
            if ema is not None:
                ema.mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def _ema_rebind(self):
        """engine 重建 (VLM 动态解冻) 后重绑: 已有 name 保留 EMA 历史值
        (ZeRO-3 分片布局不随 param group 新增而变化, shard 语义不变),
        新增 name (解冻的 VLM) 以当前权重初始化 — 冻结期权重未变, 无漂移。"""
        if self._ema_state is None:
            return
        new_state = {}
        n_kept, n_new, n_reinit = 0, 0, 0
        for name, p in self.policy.named_parameters():
            if not p.requires_grad:
                continue
            old = self._ema_state.get(name)
            if old is None:
                new_state[name] = p.data.detach().clone()
                n_new += 1
            elif old.shape == p.data.shape:
                new_state[name] = old
                n_kept += 1
            else:
                # 分片布局意外变化 — 响亮降级: 以当前权重重启该参数的 EMA
                _log(f"[EMA] ⚠️ {name} 分片形状变化 {tuple(old.shape)} → {tuple(p.data.shape)}, "
                     f"该参数 EMA 以当前权重重启")
                new_state[name] = p.data.detach().clone()
                n_reinit += 1
        self._ema_state = new_state
        _log(f"[EMA] engine 重建后重绑: 保留 {n_kept}, 新增 {n_new} (VLM), 重启 {n_reinit}")

    def _timer_checkpoint_due(self, now: float | None = None) -> bool:
        """Return a rank-consistent wall-clock checkpoint decision."""
        if self.save_every_n_seconds is None:
            return False

        timer_due = False
        if self.is_main_process and self._last_checkpoint_time is not None:
            current_time = time.monotonic() if now is None else now
            timer_due = current_time - self._last_checkpoint_time >= self.save_every_n_seconds

        if self.is_distributed:
            if self._checkpoint_timer_signal is None:
                self._checkpoint_timer_signal = torch.zeros(1, dtype=torch.uint8, device=self.device)
            if self.is_main_process:
                self._checkpoint_timer_signal.fill_(timer_due)
            dist.broadcast(self._checkpoint_timer_signal, src=0)
            timer_due = bool(self._checkpoint_timer_signal.item())

        return timer_due

    def _checkpoint_reasons(self, batch_idx: int, epoch: int, now: float | None = None) -> list[str]:
        """Collect periodic checkpoint triggers, coalescing overlaps into one save."""
        reasons = []
        if self.save_every_n_steps is not None and self.global_step % self.save_every_n_steps == 0:
            reasons.append("step")
        if self.save_every_n_epochs is not None and batch_idx == 0 and epoch % self.save_every_n_epochs == 0:
            reasons.append("epoch")
        if self._timer_checkpoint_due(now):
            reasons.append("timer")
        return reasons

    def _mark_checkpoint_saved(self, now: float | None = None):
        """Restart the rank-0 wall-clock interval after any successful periodic save."""
        if self.is_main_process and self.save_every_n_seconds is not None:
            self._last_checkpoint_time = time.monotonic() if now is None else now

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

        # VLM delayed unfreeze: start frozen when threshold is set
        if self.train_vlm and self._vlm_delayed_unfreeze:
            _log(f"VLM starts frozen (delayed unfreeze enabled, threshold={self.config.vlm_unfreeze_v_loss_threshold})")
            for param in self.policy.vlm.parameters():
                param.requires_grad = False
            self.policy.vlm.eval()

        # Resume from a checkpoint saved with VLM unfrozen: engine must be built with
        # VLM trainable from the start so optimizer groups match the checkpoint
        if self.train_vlm and self._resume_vlm_unfrozen:
            _log("Resume with VLM unfrozen: VLM trainable from engine construction")
            for param in self.policy.vlm.parameters():
                param.requires_grad = True
            self.policy.vlm.train()

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
        from lerobot.policies.lola.vlm_backbone import get_vlm_backbone
        from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock, Flux2SingleTransformerBlock
        from lerobot.policies.lola.modeling_lola import LolaVLMFeatureExtractor, LoLADualExpertDoubleBlock, LoLADualExpertSingleBlock
        from lerobot.policies.lola_v07.modeling_lola_v07 import LolaV07ActionEncoder, LolaV07StateEncoder

        # VLM layer classes depend on the backbone (Qwen3_5DecoderLayer/Qwen3_5VisionBlock
        # for qwen3_5, Qwen3VLTextDecoderLayer/Qwen3VLVisionBlock for cosmos3_nano)
        vlm_wrap_classes = get_vlm_backbone(self.config.vlm_backbone).get_fsdp_wrap_classes()

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
            module, recurse, nonwrapped_numel,
            transformer_layer_cls={
                *vlm_wrap_classes,
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
        if self.train_vlm and hasattr(self.policy, "vlm") and not self._vlm_delayed_unfreeze:
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
        """Replace PyTorch checkpointing with DeepSpeed's in LoLA model.

        注意: DeepSpeed 的 non_reentrant_checkpoint 与 ZeRO-3 的参数分片不兼容
        (backward recompute 时 param.grad 视图失配: "size of tensor a (0) ..."),
        ZeRO-3 下必须保留 PyTorch 默认 checkpoint (实测 fwd/bwd/step 正常)。
        """
        if not self.config.gradient_checkpointing:
            return
        if self.deepspeed_zero_stage >= 3:
            _log("ZeRO-3: keeping PyTorch default activation checkpointing "
                 "(deepspeed non_reentrant_checkpoint is incompatible with ZeRO-3 sharding)")
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

    def _unfreeze_vlm(self):
        """Unfreeze VLM parameters and rebuild optimizer/scheduler for DDP/FSDP.

        This method is called when v_loss drops below vlm_unfreeze_v_loss_threshold.
        It:
        1. Sets _vlm_unfrozen = True
        2. Unfreezes VLM parameters (requires_grad=True, .train())
        3. Switches VLM forward mode from hook to output_hidden_states
        4. Enables VLM gradient checkpointing if configured
        5. Rebuilds the ENTIRE optimizer + scheduler with VLM group included
        6. Rebuilds DDP if strategy is DDP
        """
        _log(f"Unfreezing VLM at step {self.global_step} (v_loss < {self.config.vlm_unfreeze_v_loss_threshold})")
        self._vlm_unfrozen = True
        self._vlm_delayed_unfreeze = False

        # 1. Unfreeze VLM parameters
        for param in self.policy.vlm.parameters():
            param.requires_grad = True
        self.policy.vlm.train()

        # 2. Switch VLM forward mode: hook → output_hidden_states
        # When VLM is unfrozen, we must use output_hidden_states mode for proper
        # gradient flow through gradient checkpointing (hook mode captures output
        # without gradients, which is fine for frozen VLM but breaks when training)
        if hasattr(self.policy, '_vlm_forward_mode') and self.policy._vlm_forward_mode == "hook":
            # Remove existing hooks
            if hasattr(self.policy, '_hook_handles'):
                for handle in self.policy._hook_handles:
                    handle.remove()
                self.policy._hook_handles = []
            self.policy._vlm_forward_mode = "output_hidden_states"
            _log("VLM forward mode switched from hook to output_hidden_states")

        # 3. Enable VLM gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.policy.vlm.gradient_checkpointing_enable()
            _log("VLM gradient checkpointing enabled")

        # 4. Rebuild optimizer + scheduler with VLM group included
        encoder_lr_mult = self.config.encoder_lr_mult
        vlm_lr_mult = self.config.vlm_lr_mult
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
        # VLM group now included since unfrozen
        param_groups.append({"params": list(self.policy.vlm.parameters()), "lr": base_lr * vlm_lr_mult})

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

        # Rebuild BF16OptimizerWrapper if needed
        self.bf16_optimizer = None
        if self.strategy == "ddp" and self.use_bf16:
            self.bf16_optimizer = BF16OptimizerWrapper(self.optimizer)
            _log("Rebuilt BF16OptimizerWrapper with VLM params included")

        # Rebuild scheduler with remaining steps
        remaining_steps = self.total_steps - self.global_step
        if remaining_steps <= 0:
            remaining_steps = 1  # minimum to avoid scheduler crash
        from torch.optim.lr_scheduler import OneCycleLR
        warmup_ratio = min(self.config.warmup_pct, 0.5)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[group["lr"] for group in self.optimizer.param_groups],
            total_steps=remaining_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
        )
        _log(f"Rebuilt optimizer + scheduler with VLM group (remaining_steps={remaining_steps})")

        # 5. Rebuild DDP if strategy is DDP (needed because param graph changed)
        if self.strategy == "ddp":
            self.model = DDP(
                self.policy,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=False,  # Must use static_graph=False after unfreezing new params
            )
            _log("Rebuilt DDP with VLM params included (static_graph=False for dynamic param graph)")

        # EMA 重绑: 新解冻的 VLM 以当前权重初始化 EMA 并开始跟踪
        self._ema_rebind()

        trainable_count = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        _log(f"VLM unfrozen: {trainable_count:,} trainable params")

    def _exchange_zero3_roundtrip_shards(self, roundtrip_dir: str, tag: str):
        """本地化 IO + 多节点下, ZeRO-3 解冻回环的跨节点分片互换。

        背景: 本地化 IO 后每节点只把本机 ranks 的分片写到本地 NVMe, 但回环 load
        (load_checkpoint → get_fp32_state_dict_from_zero_checkpoint) 要求目录下
        齐 world_size 个 model/optim 分片 + latest 指针 (共享文件系统假设)。
        本方法在各节点 local_rank==0 上:
          1. azcopy 上传本机 tag 目录 (只含本机 ranks 分片) + latest 到 blob 临时目录
          2. 上传 _done_node{i} 标记并轮询所有节点的标记 (blob 即汇合点)
          3. azcopy 按分片过滤只下载缺失的远端 ranks 分片, 补齐本地目录
          4. 校验 world_size 个分片齐全, 缺失则全量下载兜底
        全程无 blobfuse 大文件写 (16 rank 并发 FUSE 写曾致 torch.save IO 崩溃),
        azcopy 自带重试/断点续传; 失败即响亮报错, job 重试后 resume 会幂等重触发解冻。
        由 LOLA_CKPT_BLOB_BASE 环境变量激活 (launcher 本地化 ckpt 时透传); 未设置
        (非本地化运行, ckpt 在共享挂载点上) 或单节点时直接返回。
        """
        blob_base = os.environ.get("LOLA_CKPT_BLOB_BASE", "").rstrip("/")
        if not blob_base:
            return

        import glob
        import re
        import subprocess

        def _barrier():
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        tag_dir = os.path.join(roundtrip_dir, tag)
        # 本机分片 → ranks_per_node / 节点数 / 本节点编号 (无需额外环境变量)
        local_ranks = sorted(
            int(m.group(1))
            for f in glob.glob(os.path.join(tag_dir, "zero_pp_rank_*_model_states.pt"))
            if (m := re.search(r"zero_pp_rank_(\d+)_", f))
        )
        ranks_per_node = len(local_ranks)
        if ranks_per_node == 0 or self.world_size % ranks_per_node != 0:
            raise RuntimeError(
                f"无法从 {tag_dir} 的本机分片推断节点拓扑 "
                f"(found ranks={local_ranks}, world_size={self.world_size})")
        nnodes = self.world_size // ranks_per_node
        if nnodes <= 1:
            return
        node_idx = self.world_rank // ranks_per_node

        if self.local_rank != 0:
            _barrier()  # 等本节点 rank0 完成互换
            return

        # ---- 以下仅每节点 local_rank==0 ----
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from download_azure_azcopy import install_azcopy, run_azcopy_transfer

        azcopy_bin = os.environ.get("LOLA_AZCOPY_BIN") or install_azcopy("/tmp/lola_azcopy/azcopy")
        subprocess.run([azcopy_bin, "login", "--identity"], check=True)
        blob_rt = f"{blob_base}/{os.path.basename(roundtrip_dir)}"
        _log(f"[unfreeze] 跨节点分片互换: node {node_idx}/{nnodes}, "
             f"本机 ranks={local_ranks} -> {blob_rt}")

        # 1. 上传本机分片 (tag 目录整体) + latest 指针 (仅 global rank0 节点有)
        ok = run_azcopy_transfer(azcopy_bin, tag_dir, f"{blob_rt}/{tag}", dir_transfer=True)
        latest_local = os.path.join(roundtrip_dir, "latest")
        if ok and os.path.isfile(latest_local):
            ok = run_azcopy_transfer(azcopy_bin, latest_local, f"{blob_rt}/latest",
                                     overwrite="true", max_retries=3)
        # 2. done 标记 (必须在分片与 latest 之后上传, 作为对端可见的完成信号)
        marker_local = os.path.join(roundtrip_dir, f"_done_node{node_idx}")
        open(marker_local, "w").close()
        ok = ok and run_azcopy_transfer(azcopy_bin, marker_local,
                                        f"{blob_rt}/_done_node{node_idx}",
                                        overwrite="true", max_retries=3)
        if not ok:
            raise RuntimeError(f"[unfreeze] 本机分片上传失败: {tag_dir} -> {blob_rt}")

        # 3. 轮询全部节点的 done 标记 (单次尝试, 失败即未就绪)
        poll_timeout = int(os.environ.get("LOLA_EXCHANGE_TIMEOUT", "1800"))
        deadline = time.time() + poll_timeout
        for i in range(nnodes):
            murl = f"{blob_rt}/_done_node{i}"
            dst = os.path.join(roundtrip_dir, f".poll_node{i}")
            while subprocess.run([azcopy_bin, "copy", murl, dst, "--overwrite=true"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL).returncode != 0:
                if time.time() > deadline:
                    raise RuntimeError(
                        f"[unfreeze] 等待节点 {i} 分片超时 ({poll_timeout}s): {murl}")
                time.sleep(10)

        # 4. 分片过滤下载远端 ranks (省一半流量) + latest 指针
        remote_ranks = [r for r in range(self.world_size) if r not in set(local_ranks)]
        pats = ";".join(f"*zero_pp_rank_{r}_mp_rank_00_*" for r in remote_ranks)
        ok = run_azcopy_transfer(azcopy_bin, f"{blob_rt}/{tag}", tag_dir, dir_transfer=True,
                                 extra_copy_args=[f"--include-pattern={pats}"])
        if ok:
            ok = run_azcopy_transfer(azcopy_bin, f"{blob_rt}/latest", latest_local,
                                     overwrite="true", max_retries=3)

        # 5. 校验 world_size 分片齐全, 缺失全量下载兜底
        def _shards_complete():
            return (len(glob.glob(os.path.join(tag_dir, "*_model_states.pt"))) >= self.world_size
                    and len(glob.glob(os.path.join(tag_dir, "*_optim_states.pt"))) >= self.world_size)

        if ok and not _shards_complete():
            _log("[unfreeze] WARN: 分片过滤下载不完整, 全量下载兜底")
            ok = run_azcopy_transfer(azcopy_bin, f"{blob_rt}/{tag}", tag_dir, dir_transfer=True)
        if not ok or not _shards_complete():
            raise RuntimeError(
                f"[unfreeze] 分片互换后 {tag_dir} 仍不完整 "
                f"(world_size={self.world_size}), 回环 load 必失败, 终止")
        _log(f"[unfreeze] 分片互换完成: {tag_dir} 已齐 {self.world_size} 个分片")
        _barrier()

    def _unfreeze_vlm_deepspeed(self):
        """Unfreeze VLM for DeepSpeed: extract weights, destroy old hooks, rebuild engine, restore weights.

        DeepSpeed ZeRO partitioning is set at engine initialization and cannot be
        dynamically changed. To add VLM optimizer states, we must rebuild the
        entire engine. This method:
        1. Extracts all model weights (including frozen VLM)
           - ZeRO-2: clone to CPU pinned memory (avoids a +16GB GPU transient spike)
           - ZeRO-3: params are sharded, clone/copy_ won't work → save/load checkpoint
             roundtrip with exclude_frozen_parameters=False (frozen VLM shards included)
        2. Destroys old engine (removes ZeRO gradient hooks from parameters)
        3. Unfreezes VLM parameters (requires_grad=True, .train())
        4. Switches VLM forward mode and enables gradient checkpointing
        5. Rebuilds DeepSpeed engine with VLM group included
        6. Restores model weights (ZeRO-2: copy_ from CPU; ZeRO-3: load_checkpoint weights-only)
        7. Replaces trainer references and re-configures checkpointing

        For ZeRO-2 the save/load roundtrip breaks when exclude_frozen_parameters=True
        (the saved checkpoint lacks VLM params, so the new engine can't load them),
        and parameters are NOT partitioned across ranks, so named_parameters() returns
        complete weights on each rank — no disk I/O needed.

        The old engine must be destroyed before rebuilding to remove ZeRO-2's
        register_post_accumulate_grad_hook from parameters; otherwise both old
        and new hooks fire during backward, causing double gradient reduction.
        """
        import deepspeed
        _log(f"Unfreezing VLM for DeepSpeed at step {self.global_step}")

        # 1. Extract all model weights BEFORE any changes (includes frozen VLM params)
        zero3_roundtrip_dir = None
        if self.deepspeed_zero_stage >= 3:
            # ZeRO-3: parameters are partitioned (ds_tensor shards), so clone()/copy_()
            # of full weights is impossible. Save a temporary full checkpoint
            # (exclude_frozen_parameters=False → frozen VLM shards are included via
            # frozen_param_fragments) and reload weights into the rebuilt engine.
            zero3_roundtrip_dir = os.path.join(self.ckpt_dir, f"_unfreeze_tmp_step{self.global_step}")
            self.model_engine.save_checkpoint(
                save_dir=zero3_roundtrip_dir,
                tag="unfreeze",
                exclude_frozen_parameters=False,
            )
            state_dict = None
            _log(f"Saved temporary ZeRO-3 roundtrip checkpoint: {zero3_roundtrip_dir}")
            # 本地化 IO + 多节点: 本机只有本节点 ranks 的分片, 而回环 load 需要
            # world_size 全量分片 → 经 blob 互换补齐 (无共享文件系统时)
            self._exchange_zero3_roundtrip_shards(zero3_roundtrip_dir, tag="unfreeze")
        else:
            # ZeRO-2: clone to CPU pinned memory. The previous GPU-side clone caused a
            # +P*2 bytes transient spike per rank (16GB+ for cosmos) at the unfreeze step.
            state_dict = {
                k: v.detach().to("cpu", copy=True).pin_memory()
                for k, v in self.model_engine.module.named_parameters()
            }
            _log(f"Extracted {len(state_dict)} param tensors from old engine for weight restoration (CPU pinned)")

        # 2. Destroy old engine to remove ZeRO-2 gradient hooks from parameters.
        #    Without this, the old hooks remain on parameters and the new engine
        #    adds its own hooks, causing double gradient reduction:
        #    "The parameter X has already been reduced. Gradient computed twice."
        self.model_engine.destroy()
        _log("Destroyed old DeepSpeed engine (removed gradient hooks)")

        # 2b. Release the old engine's optimizer memory BEFORE building the new
        #     engine. destroy() only removes hooks — ~P*12 bytes of GPU state
        #     (fp32 masters, Adam m/v, grad buffers) stays pinned by reference
        #     chains gc won't reclaim on its own:
        #     - param._z3_optimizer (stage3._link_all_hp_params): module params
        #       pin the old stage3 optimizer (new engine re-links at init)
        #     - param.grad views pin the flat grad buffer
        #     - engine.optimizer/basic_optimizer (FusedAdam.state holds m/v)
        #     - the graph loss tensor's backward-hook manager (loss detached at
        #       the call site before this method runs)
        for p in self.policy.parameters():
            if hasattr(p, "_z3_optimizer"):
                p._z3_optimizer = None
            if p.grad is not None:
                p.grad = None
        self.model_engine.optimizer = None
        self.model_engine.lr_scheduler = None
        self.model_engine.basic_optimizer = None
        self.model_engine = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        _log("Released old engine optimizer memory")

        self._vlm_unfrozen = True
        self._vlm_delayed_unfreeze = False

        # 3. Unfreeze VLM parameters
        for param in self.policy.vlm.parameters():
            param.requires_grad = True
        self.policy.vlm.train()

        # 4. Switch VLM forward mode: hook → output_hidden_states
        if hasattr(self.policy, '_vlm_forward_mode') and self.policy._vlm_forward_mode == "hook":
            if hasattr(self.policy, '_hook_handles'):
                for handle in self.policy._hook_handles:
                    handle.remove()
                self.policy._hook_handles = []
            self.policy._vlm_forward_mode = "output_hidden_states"
            _log("VLM forward mode switched from hook to output_hidden_states")

        # 5. Enable VLM gradient checkpointing
        if self.config.gradient_checkpointing:
            self.policy.vlm.gradient_checkpointing_enable()
            if self.deepspeed_zero_stage < 3:
                # ZeRO-2 下可用 DeepSpeed checkpoint func; ZeRO-3 必须保留 PyTorch 默认
                # (ds non_reentrant_checkpoint + ZeRO-3 分片 → backward param.grad 视图失配)
                ds_fn = deepspeed.checkpointing.non_reentrant_checkpoint
                vlm = self.policy.vlm
                if hasattr(vlm, '_gradient_checkpointing_func'):
                    vlm._gradient_checkpointing_func = ds_fn
                for module in vlm.modules():
                    if hasattr(module, '_gradient_checkpointing_func'):
                        module._gradient_checkpointing_func = ds_fn
                _log("VLM gradient checkpointing enabled with DeepSpeed checkpointing func")
            else:
                _log("VLM gradient checkpointing enabled (PyTorch default, ZeRO-3 compatible)")

        # 6. Rebuild DeepSpeed engine with VLM group included
        encoder_lr_mult = self.config.encoder_lr_mult
        vlm_lr_mult = self.config.vlm_lr_mult
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
        # VLM group now included since unfrozen
        trainable_param_groups.append({"params": list(self.policy.vlm.parameters()), "lr": base_lr * vlm_lr_mult})
        # Filter out params that don't require grad, then remove empty groups
        for group in trainable_param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        trainable_param_groups = [g for g in trainable_param_groups if g["params"]]

        remaining_steps = self.total_steps - self.global_step
        if remaining_steps <= 0:
            remaining_steps = 1

        ds_config = get_deepspeed_config(
            learning_rate=base_lr,
            weight_decay=self.weight_decay,
            gradient_clip_val=self.gradient_clip_val,
            train_vlm=True,  # Now VLM is trainable
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

        def lr_scheduler_callable(optimizer):
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                optimizer,
                max_lr=[group["lr"] for group in optimizer.param_groups],
                total_steps=remaining_steps,
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

        # 7. Restore model weights directly (no disk I/O, no checkpoint format mismatch)
        #    DeepSpeed.initialize() may modify parameter objects (e.g. flatten for ZeRO-3),
        #    so we restore by name matching on the module's state_dict.
        # 7. Restore model weights (no permanent disk I/O, no checkpoint format mismatch)
        #    DeepSpeed.initialize() may modify parameter objects (e.g. flatten for ZeRO-3),
        #    so ZeRO-2 restores by name matching on the module's state_dict;
        #    ZeRO-3 reloads the temporary roundtrip checkpoint (weights only).
        if self.deepspeed_zero_stage >= 3:
            load_path, _ = model_engine.load_checkpoint(
                load_dir=zero3_roundtrip_dir,
                tag="unfreeze",
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_strict=False,
            )
            if load_path is None:
                raise RuntimeError(f"ZeRO-3 unfreeze roundtrip reload failed from {zero3_roundtrip_dir}")
            _log("ZeRO-3 roundtrip weights reloaded into rebuilt engine")
            # cleanup temporary checkpoint
            import shutil
            dist.barrier()
            if self.is_main_process:
                shutil.rmtree(zero3_roundtrip_dir, ignore_errors=True)
            dist.barrier()
            # 分片互换在 blob 上的临时目录也一并清掉 (尽力而为, 失败无碍)
            blob_base = os.environ.get("LOLA_CKPT_BLOB_BASE", "").rstrip("/")
            if blob_base and self.local_rank == 0:
                try:
                    import subprocess
                    azcopy_bin = os.environ.get("LOLA_AZCOPY_BIN") or "azcopy"
                    subprocess.run(
                        [azcopy_bin, "remove", f"{blob_base}/{os.path.basename(zero3_roundtrip_dir)}",
                         "--recursive=true"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
                except Exception:
                    pass
        else:
            current_state = dict(model_engine.module.named_parameters())
            restored, missing, unexpected = 0, 0, 0
            for name, saved_tensor in state_dict.items():
                if name in current_state:
                    current_state[name].data.copy_(saved_tensor.data)
                    restored += 1
                else:
                    missing += 1
            unexpected = len(current_state) - restored
            _log(f"Weight restoration: {restored} params restored, {missing} missing, {unexpected} unexpected")
            del state_dict

        # 8. Replace trainer references
        self.model = model_engine
        self.model_engine = model_engine
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        # Re-configure DeepSpeed checkpointing
        self._configure_deepspeed_checkpointing()

        # EMA 重绑: 已有参数保留 EMA 历史 (分片布局不变), 新解冻的 VLM 以当前
        # 权重初始化 EMA 并开始跟踪
        self._ema_rebind()

        trainable_count = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        _log(f"DeepSpeed VLM unfrozen: {trainable_count:,} trainable params, engine rebuilt")

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
        if self.train_vlm and hasattr(self.policy, "vlm") and not self._vlm_delayed_unfreeze:
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

        # DDP: wrap optimizer with FP32 master weights (bf16 optimizer states have
        # only ~3.3 decimal digits vs ~7 in fp32, degrading Adam numerical stability)
        self.bf16_optimizer = None
        if self.strategy == "ddp" and self.use_bf16:
            self.bf16_optimizer = BF16OptimizerWrapper(self.optimizer)
            # After wrapping, optimizer.param_groups now reference fp32 masters
            # foreach=True is safe (all params same dtype), scheduler metadata unchanged
            _log(f"DDP: BF16OptimizerWrapper enabled, optimizer operates on fp32 master params")

        # Verify optimizer param group coverage
        opt_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.policy.parameters())
        _log(f"Optimizer: {opt_params:,} / {all_params:,} params in optimizer")

        from torch.optim.lr_scheduler import OneCycleLR
        warmup_ratio = min(self.config.warmup_pct, 0.5)
        # Scheduler reads param_groups metadata (lr, initial_lr, betas) which is unchanged
        # regardless of whether BF16OptimizerWrapper swapped params to fp32 masters
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
                           "hist_states_full", "hist_states_mask", "hist_states_length",
                           "n_transition", "n_transition_chunks"]
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

        # Conditional VLM unfreezing detection (requires train_vlm=True)
        if self.train_vlm and not self._vlm_unfrozen and self.config.vlm_unfreeze_v_loss_threshold > 0:
            v_loss_val = loss_dict.get("v_loss", float("inf"))
            if self.is_distributed:
                v_loss_tensor = torch.tensor([v_loss_val], device=self.device)
                dist.all_reduce(v_loss_tensor, op=dist.ReduceOp.AVG)
                v_loss_val = v_loss_tensor.item()
            if v_loss_val < self.config.vlm_unfreeze_v_loss_threshold:
                if self.strategy == "deepspeed":
                    # Defer unfreeze to after this step's backward+optimizer.
                    # Engine rebuild mid-step invalidates DeepSpeed's activation
                    # checkpointing closures (checkpoint_pack/checkpoint_unpack
                    # captured by autograd graph reference old engine's parameter
                    # objects). Rebuilding after step completes avoids stale graph.
                    self._pending_deepspeed_unfreeze = True
                    _log(f"VLM unfreeze triggered at step {self.global_step}, deferred to step boundary")
                else:
                    self._unfreeze_vlm()
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
        # 原地续训 (2026-08-09 方案 B): main() 已判定 resume 配置匹配并把被续训的 run
        # 目录赋给 self.resume_save_dir → 后续 checkpoint 直接写回该目录 (一个 config
        # 一条血统线一个目录, latest 指针即血统末端)。原 training_config.json 保留
        # 不覆盖 — 配置已校验一致, 且原始 vlm_path 等字段不被本次运行的本地化现场值
        # 覆盖; 续训事件追加到 resume_history.jsonl (由 watcher 同步到 blob)。
        import json

        if self.resume_save_dir is not None:
            ckpt_dir = self.resume_save_dir
            if self.is_main_process:
                _log(f"Checkpoint directory (原地续训): {ckpt_dir}")
                try:
                    with open(os.path.join(ckpt_dir, "resume_history.jsonl"), "a") as f:
                        f.write(json.dumps({
                            "time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "resumed_from": getattr(self, "_resume_loaded_from", None),
                            "from_step": start_step,
                            "from_epoch": start_epoch,
                        }) + "\n")
                except OSError as e:
                    _log(f"WARNING: resume_history.jsonl 写入失败 ({e}), 不影响训练")
        else:
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
                # (make_serializable 与 resume_search 共享, 保证写出的 training_config.json
                #  与 resume 搜索重建的快照序列化方式一致, 可按值比较)
                full_config = make_serializable({
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

        # EMA 注册: fresh 训练以初始权重注册; resume 场景 _ema_state 已在
        # load_checkpoint 中恢复 (或以其加载的权重兜底注册), 此处不再覆盖
        if self.config.ema_decay > 0 and self._ema_state is None:
            self._ema_register()

        # 计算 resume 时需要跳过的 batch 数
        try:
            batches_per_epoch = len(train_loader)
            _log(f"Total batches per epoch: {batches_per_epoch}")
        except TypeError:
            batches_per_epoch = None
            _log("IterableDataset detected: cannot determine batches per epoch")

        # resume 定位: 以 start_step (已完成 batch 数) 为唯一基准推导 epoch 起点。
        # checkpoint 里的 current_epoch 是【进行中】的 epoch (1-indexed; save_every_n_epochs
        # 在每个 epoch batch_idx==0 训完后触发, 见下方保存逻辑), 不能直接当循环起点 ——
        # 2026-08-09 生产事故: 旧逻辑 epoch=start_epoch 已把 while 循环定位到
        # start_epoch+1, skip_epochs=start_epoch 又把剩余每个 epoch 的首个 batch 全部
        # break (双重跳过), resume 到后半程时一步未训直接存 final。勿回退。
        if start_step > 0 and batches_per_epoch is not None:
            epoch = start_step // batches_per_epoch  # 已完成的 epoch 数 (循环开头 epoch+=1)
            skip_batches = start_step % batches_per_epoch  # 进行中 epoch 已训的 batch 数
            _log(f"Resuming: start at epoch {epoch + 1}, skipping {skip_batches} batches")
        elif start_step > 0:
            # IterableDataset: 无法按步数定位数据, 数据从头开始 (model/optimizer/scheduler 已恢复)
            epoch = start_epoch
            skip_batches = 0
            _log(
                f"Resuming from step {start_step} with IterableDataset: "
                "data will restart from the beginning (model/optimizer/scheduler states are restored). "
                "For precise data resume, use map-style dataset or add start_index to IterableDataset."
            )
        else:
            epoch = start_epoch
            skip_batches = 0
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

                # Map-style 数据集：跳过 resume 前当前 epoch 已训的 batch
                if skip_batches > 0:
                    skip_batches -= 1
                    continue

                step_start = time.monotonic()

                if self.strategy != "deepspeed":
                    if self.bf16_optimizer is not None:
                        self.bf16_optimizer.zero_grad()
                    else:
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
                elif self.bf16_optimizer is not None:
                    # DDP+BF16OptimizerWrapper: cast bf16 grads -> fp32, clip on fp32 masters
                    self.bf16_optimizer.copy_grads_to_fp32()
                    if self.gradient_clip_val > 0:
                        grad_norm = self.bf16_optimizer.clip_grad_norm(self.gradient_clip_val)
                    else:
                        grad_norm = None
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
                elif self.bf16_optimizer is not None:
                    self.bf16_optimizer.step()
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

                # Start the wall-clock interval at the first productive update.
                # Resume batch skipping and model setup time do not consume the interval.
                if (self.is_main_process and self.save_every_n_seconds is not None
                        and self._last_checkpoint_time is None):
                    self._last_checkpoint_time = time.monotonic()

                # EMA 更新 (optimizer.step 之后, 本地 shard 原地更新, 无通信)
                if self._ema_state is not None:
                    self._ema_update()

                # Deferred DeepSpeed VLM unfreeze: engine rebuild happens at step
                # boundary (after backward+optimizer, before next forward) to avoid
                # invalidating the current step's autograd graph.
                if self._pending_deepspeed_unfreeze:
                    self._pending_deepspeed_unfreeze = False
                    # Detach loss outputs to break the autograd-graph → backward-hook
                    # manager → bound-method → old-engine reference chain. The graph
                    # loss tensor pins the entire old engine + optimizer GPU state
                    # (register_output_backward_hooks binds engine methods to it),
                    # and the rebuilt engine would OOM while both coexist.
                    loss = loss.detach()
                    loss_dict = {
                        k: (v.detach() if torch.is_tensor(v) else v)
                        for k, v in loss_dict.items()
                    }
                    self._unfreeze_vlm_deepspeed()

                # Optimizer state dtype diagnostic (all strategies, step 10)
                if self.global_step == 10 and self.is_main_process:
                    dtypes_found = {}
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            if p in self.optimizer.state:
                                for key, val in self.optimizer.state[p].items():
                                    if isinstance(val, torch.Tensor):
                                        d = str(val.dtype)
                                        dtypes_found[d] = dtypes_found.get(d, 0) + val.numel()
                    _log(f"[Step 10] Optimizer state dtype stats:")
                    for dtype, count in dtypes_found.items():
                        bytes_per = 4 if 'float32' in dtype else 2 if 'bfloat16' in dtype or 'half' in dtype else 1
                        size_gb = count * bytes_per / 1e9
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

                # 保存 checkpoint: step/epoch 是固定评估点, timer 是独立的最大保存间隔兜底。
                # 任意触发成功后都会重置 timer; 同一步多条件命中只保存一次。
                checkpoint_reasons = self._checkpoint_reasons(batch_idx, epoch)
                should_save = bool(checkpoint_reasons)
                checkpoint_start = None
                if should_save and self.is_main_process:
                    checkpoint_start = time.monotonic()
                    _log(f"Checkpoint triggered at step {self.global_step}: "
                         f"reasons={'+'.join(checkpoint_reasons)}")
                if self.strategy == "deepspeed":
                    if should_save:
                        self.save_checkpoint(
                            ckpt_dir, self.global_step, checkpoint_reasons=checkpoint_reasons
                        )
                elif should_save and self.is_main_process:
                    self.save_checkpoint(
                        ckpt_dir, self.global_step, checkpoint_reasons=checkpoint_reasons
                    )
                if should_save:
                    checkpoint_completed = time.monotonic()
                    self._mark_checkpoint_saved(checkpoint_completed)
                    if self.is_main_process:
                        _log(f"Checkpoint completed at step {self.global_step}: "
                             f"duration={checkpoint_completed - checkpoint_start:.1f}s, "
                             f"timer reset")

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

    def save_checkpoint(
        self,
        ckpt_dir: str,
        step: int,
        is_final: bool = False,
        checkpoint_reasons: list[str] | None = None,
    ):
        """保存 checkpoint"""
        checkpoint_reasons = checkpoint_reasons or (["final"] if is_final else [])
        extra_state = {
            "vlm_unfrozen": self._vlm_unfrozen,
            "global_step": self.global_step,
            "checkpoint_reasons": checkpoint_reasons,
        }
        if self.strategy == "deepspeed":
            tag = f"step_{step:06d}" if not is_final else "final"
            client_state = {
                "step": step,
                "epoch": self.current_epoch,
                "vlm_unfrozen": self._vlm_unfrozen,
                "checkpoint_reasons": checkpoint_reasons,
            }
            self.model.save_checkpoint(
                save_dir=ckpt_dir,
                tag=tag,
                client_state=client_state,
                exclude_frozen_parameters=True,
            )
            ckpt_path = f"{ckpt_dir}/{tag}"
            # EMA 分片随 tag 目录落盘 (每 rank 写自己的 shard; watcher 递归上传
            # tag 目录会自动带上)。必须在 .upload_ready 之前写完 — 标记一落 watcher
            # 即开始上传; barrier 保证所有 rank 的 ema_rank_*.pt 就位。
            if self._ema_state is not None:
                torch.save(
                    self._ema_state,
                    os.path.join(ckpt_path, f"ema_rank_{self.world_rank}.pt"),
                )
                if self.is_distributed:
                    dist.barrier()
            # 上传就绪标记: DS save_checkpoint 末尾有 dist.barrier(), 返回时本节点
            # 分片已完整落盘。各节点 local_rank==0 落标记, 后台 watchdog
            # (checkpoint_upload_watcher.py) 据此把 tag 目录异步上传至 blob,
            # 训练本身只读写节点本地盘, 不触碰 blobfuse 挂载点。
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                try:
                    open(os.path.join(ckpt_path, ".upload_ready"), "a").close()
                except OSError:
                    pass
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
                    "vlm_unfrozen": [self._vlm_unfrozen],
                },
                checkpoint_id=ckpt_path,
            )
            # scheduler 不支持 torch.distributed.checkpoint，单独用 torch.save 保存
            if self.is_main_process:
                torch.save(
                    {"scheduler_state_dict": self.scheduler.state_dict(), **extra_state},
                    os.path.join(ckpt_path, "scheduler.pt"),
                )
                # 上传就绪标记 (见 deepspeed 分支注释)
                try:
                    open(os.path.join(ckpt_path, ".upload_ready"), "a").close()
                except OSError:
                    pass
        else:
            # DDP checkpoint 保存
            state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
            ckpt_name = f"lola-step-{step:06d}.pt" if not is_final else "lola-final.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save({
                "step": step,
                "epoch": self.current_epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.bf16_optimizer.state_dict() if self.bf16_optimizer else self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "vlm_unfrozen": self._vlm_unfrozen,
                "global_step": self.global_step,
                "ema_state": self._ema_state,
                "checkpoint_reasons": checkpoint_reasons,
            }, ckpt_path)

        _log(f"Checkpoint saved: {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str):
        """加载 checkpoint"""
        # 记录实际加载来源 (原地续训时写入 resume_history.jsonl)
        self._resume_loaded_from = ckpt_path
        vlm_unfrozen = False
        if self.strategy == "deepspeed":
            # --resume 支持两种形式: 含 latest 文件的 run 目录, 或具体的 tag 目录 (step_XXXXXX)
            if os.path.exists(os.path.join(ckpt_path, "latest")):
                load_dir, load_tag = ckpt_path, None
            else:
                load_dir, load_tag = os.path.dirname(ckpt_path), os.path.basename(ckpt_path)
            load_path, client_state = self.model.load_checkpoint(
                load_dir=load_dir,
                tag=load_tag,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
                # exclude_frozen_parameters=True 保存的 checkpoint 不含冻结 VLM 权重
                # (VLM 权重已由 policy 初始化时从 vlm_path 加载), 必须非严格加载
                load_module_strict=False,
            )
            if load_path is None:
                raise ValueError(f"Failed to load DeepSpeed checkpoint from {ckpt_path}")
            # ZeRO-3 load_checkpoint 会在部分参数上留下 0 尺寸的过期 .grad 视图,
            # 导致下一次 backward 的 AccumulateGrad 广播失配
            # ("size of tensor a (0) ...")。清空为 None, 让首次 backward 走标准新建路径。
            if self.deepspeed_zero_stage >= 3:
                for param in self.model_engine.module.parameters():
                    param.grad = None
            self.global_step = client_state.get("step", 0)
            self.current_epoch = client_state.get("epoch", 0)
            vlm_unfrozen = client_state.get("vlm_unfrozen", False)
            # EMA 恢复: 随 tag 目录的 ema_rank_<world_rank>.pt; 旧 checkpoint 没有
            # 该文件时以刚加载的权重兜底注册 (EMA 从 resume 点重新开始累计)
            if self.config.ema_decay > 0:
                if load_tag is None:
                    with open(os.path.join(ckpt_path, "latest")) as f:
                        _tag = f.read().strip()
                    _tag_dir = os.path.join(ckpt_path, _tag)
                else:
                    _tag_dir = ckpt_path
                ema_path = os.path.join(_tag_dir, f"ema_rank_{self.world_rank}.pt")
                if os.path.isfile(ema_path):
                    self._ema_state = torch.load(ema_path, map_location=self.device)
                    _log(f"[EMA] 从 {ema_path} 恢复 ({len(self._ema_state)} 个分片)")
                else:
                    _log(f"[EMA] {ema_path} 不存在, 以 resume 加载的权重重新注册")
                    self._ema_register()
        elif self.strategy == "fsdp":
            from torch.distributed.checkpoint import load as load_fsdp_checkpoint
            from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

            # FSDP checkpoint 加载：先获取空 state_dict 容器，再 load 填充，最后 set 回模型/优化器
            model_sd, optimizer_sd = get_state_dict(self.model, self.optimizer)
            # 用 list 包装 step，因为 int 是不可变对象，load 无法原地修改
            step_container = [0]
            epoch_container = [0]
            vlm_unfrozen_container = [False]
            load_fsdp_checkpoint(
                {"model": model_sd, "optimizer": optimizer_sd, "step": step_container, "epoch": epoch_container, "vlm_unfrozen": vlm_unfrozen_container},
                checkpoint_id=ckpt_path,
            )
            set_state_dict(self.model, self.optimizer, model_state_dict=model_sd, optim_state_dict=optimizer_sd)
            self.global_step = step_container[0]
            self.current_epoch = epoch_container[0]
            vlm_unfrozen = vlm_unfrozen_container[0]
            # 恢复 scheduler 状态
            scheduler_path = os.path.join(ckpt_path, "scheduler.pt")
            if os.path.exists(scheduler_path):
                scheduler_ckpt = torch.load(scheduler_path, map_location=self.device)
                self.scheduler.load_state_dict(scheduler_ckpt["scheduler_state_dict"])
                # vlm_unfrozen may also be stored in scheduler.pt for FSDP
                if "vlm_unfrozen" in scheduler_ckpt:
                    vlm_unfrozen = scheduler_ckpt["vlm_unfrozen"]
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if self.is_distributed:
                self.model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if self.bf16_optimizer is not None:
                self.bf16_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.global_step = checkpoint.get("step", 0)
            self.current_epoch = checkpoint.get("epoch", 0)
            vlm_unfrozen = checkpoint.get("vlm_unfrozen", False)
            if self.config.ema_decay > 0:
                if checkpoint.get("ema_state") is not None:
                    self._ema_state = checkpoint["ema_state"]
                    _log(f"[EMA] 从 checkpoint 恢复 ({len(self._ema_state)} 个分片)")
                else:
                    self._ema_register()

        # Restore VLM unfreezing state: if VLM was unfrozen before checkpoint,
        # we need to unfreeze it again (optimizer/scheduler rebuilt with VLM group).
        # 仅在引擎仍按"冻结 VLM"结构构建时才需要重建 (即 resume 时未经 peek 预知);
        # 若已通过 resume_vlm_unfrozen 以 VLM 可训练结构构建, 则优化器状态已随
        # load_checkpoint 无损恢复, 只需同步标志位。
        if vlm_unfrozen and self.train_vlm and self._vlm_delayed_unfreeze:
            _log("Checkpoint has vlm_unfrozen=True but engine was built with frozen VLM; "
                 "rebuilding engine (VLM optimizer moments will restart)")
            if self.strategy == "deepspeed":
                self._unfreeze_vlm_deepspeed()
            else:
                self._unfreeze_vlm()
        elif vlm_unfrozen and self.train_vlm:
            self._vlm_unfrozen = True

        _log(f"Checkpoint loaded from: {ckpt_path}, starting from step {self.global_step}")


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def build_lola_config(args, dataset_metadata):
    """从 args + 数据集元数据构建 LoLAV07Config。

    提取为独立函数, 供 trainer main() 与 resume_search CLI 共享 — resume 搜索
    重建当前配置快照时, 必须与实际训练进程走同一条构建路径。

    Returns:
        (config, features, action_dim, state_dim)
    """
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
        # obs_prev_chunk_frame / n_obs_steps>1 时每相机多帧进 VLM, 序列变长
        frames_per_cam = 2 if args.obs_prev_chunk_frame else max(1, args.n_obs_steps)
        args.vlm_max_length = compute_vlm_max_length(
            dataset_metadata,
            vlm_path=args.vlm_path,
            min_image_pixels=args.min_image_pixels,
            max_image_pixels=args.max_image_pixels,
            frames_per_cam=frames_per_cam,
        )

    gradient_checkpointing = not args.no_gradient_checkpointing
    config = LoLAV07Config(
        vlm_backbone=args.vlm_backbone,
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
        use_state_condition=args.use_state_condition,
        gradient_checkpointing=gradient_checkpointing,
        dit_gradient_checkpointing=args.dit_gradient_checkpointing,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        vlm_lr=args.vlm_lr,
        vlm_extract_layers=tuple(args.vlm_extract_layers),
        vlm_bridge_mode=args.vlm_bridge_mode,
        vlm_bridge_width=args.vlm_bridge_width,
        vlm_bridge_layers=args.vlm_bridge_layers,
        vlm_unfreeze_v_loss_threshold=args.vlm_unfreeze_v_loss_threshold,
        vlm_lr_mult=args.vlm_lr_mult,
        use_special_tokens=args.use_special_tokens,
        use_previous_task_end=not args.no_previous_task_end,
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
        # 2026-08-12: EMA / 图像增强 / visual token drop / chunk 帧观测
        ema_decay=args.ema_decay,
        image_aug_brightness=args.image_aug_brightness,
        image_aug_contrast=args.image_aug_contrast,
        image_aug_saturation=args.image_aug_saturation,
        image_aug_translate=args.image_aug_translate,
        image_aug_scale_min=args.image_aug_scale_min,
        image_aug_scale_max=args.image_aug_scale_max,
        visual_token_drop_rate=args.visual_token_drop_rate,
        obs_prev_chunk_frame=args.obs_prev_chunk_frame,
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

    return config, features, action_dim, state_dim


def build_dataset_metadata_snapshot(dataset_metadata, features):
    """训练配置快照的数据集身份部分 (写入 training_config.json 与 resume 搜索共用)。"""
    return {
        "total_episodes": dataset_metadata.total_episodes,
        "total_frames": dataset_metadata.total_frames,
        "fps": dataset_metadata.fps,
        "features": {k: {"shape": list(v.shape), "type": str(v.type)} for k, v in features.items()},
    }


def build_arg_parser():
    """构建 trainer 的 argparse parser。

    提取为函数供 resume_search CLI 复用 — resume 搜索重建当前配置快照时,
    必须与实际训练进程使用同一套参数定义 (含默认值), 保证快照严格同源。
    """
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
    parser.add_argument("--save_every_n_seconds", type=float, default=None,
                        help="Maximum wall-clock seconds between periodic checkpoints; independent of step/epoch saves")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # 模型参数
    parser.add_argument("--vlm_backbone", type=str, default="qwen3_5",
                        choices=["qwen3_5", "cosmos3_nano"],
                        help="VLM backbone: 'qwen3_5' (Qwen3.5-4B) or 'cosmos3_nano' (Cosmos3-Nano Reasoner)")
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
    parser.add_argument("--use_state_condition", action="store_true", default=False,
                        help="Add observation.state to DiT modulation signal (temb) as conditioning (effective with both history_type='action' and 'state')")

    # LoLA 模型配置参数
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="启用梯度检查点（默认开启）")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="关闭梯度检查点")
    parser.add_argument("--dit_gradient_checkpointing", action="store_true",
                        help="对 DiT 也启用梯度检查点 (默认关闭: DiT 激活仅 ~1GB, GC 重算不划算; "
                             "VLM 的 GC 不受此开关影响, 显存 OOM 时再打开)")
    parser.add_argument("--compile_model", action="store_true",
                        help="启用 torch.compile 优化")
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                        help="torch.compile 模式")
    parser.add_argument("--vlm_lr", type=float, default=1e-6,
                        help="VLM 学习率（仅 train_vlm=True 时生效）")
    parser.add_argument("--vlm_extract_layers", type=int, nargs="+", default=[8, 16, 24],
                        help="VLM 提取层索引")
    parser.add_argument("--vlm_bridge_mode", type=str, default="legacy",
                        choices=["legacy", "transformer"],
                        help="VLM 桥接器: 'legacy' (concat 方阵, 兼容旧 checkpoint) 或 'transformer' (LolaVLMContextBridge, 降维+多层Transformer, ~0.63B)")
    parser.add_argument("--vlm_bridge_width", type=int, default=2048,
                        help="transformer 桥接器宽度 (仅 vlm_bridge_mode=transformer 生效)")
    parser.add_argument("--vlm_bridge_layers", type=int, default=8,
                        help="transformer 桥接器层数 (仅 vlm_bridge_mode=transformer 生效)")
    parser.add_argument("--vlm_unfreeze_v_loss_threshold", type=float, default=0.3,
                        help="v_loss threshold for dynamic VLM unfreezing (0 = disabled, only works with train_vlm=True)")
    parser.add_argument("--vlm_lr_mult", type=float, default=1.5,
                        help="VLM LR multiplier after unfreezing: peak_lr = base_lr * vlm_lr_mult")
    parser.add_argument("--use_special_tokens", action="store_true",
                        help="Insert 5 special tokens in DiT sequence (vlm_start/end, hist_start/end, previous_task_end)")
    parser.add_argument("--no_previous_task_end", action="store_true",
                        help="方案B: 禁用 transition/task 拆分与 previous_task_end token, "
                             "历史走连续序列 (hist_start + all chunks + hist_end), 边界语义交给 completed-task 文本")
    parser.add_argument("--merge_history_stream", action="store_true", default=None,
                        help="方案B 数据集侧: 合并 transition+task 为连续历史流 (默认跟随 --no_previous_task_end)")
    parser.add_argument("--no_merge_history_stream", action="store_true",
                        help="显式关闭合并历史流 (用于消融: 方案B模型 + 两段式数据)")
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
    parser.add_argument("--deepspeed_zero_stage", type=int, default=2, choices=[1, 2, 3],
                        help="DeepSpeed ZeRO stage: 1 (optimizer partitioning) or 2 (optimizer+gradient partitioning). Default: 2")
    parser.add_argument("--deepspeed_reduce_bucket_size", type=float, default=5e8,
                        help="DeepSpeed ZeRO reduce bucket size (default: 5e8; ZeRO-3 用它做梯度 "
                             "reduce-scatter 分桶, 5e7 是早年给 ZeRO-2/NVLink 调的值)")
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
    parser.add_argument("--stats_mode", type=str, default="original",
                        choices=["original", "incremental"],
                        help="Stats模式: 'original'使用annotation-only stats, 'incremental'使用包含所有Calvin帧(含transition)的增量stats")

    # 2026-08-12: EMA / 图像增强 / visual token drop / chunk 帧观测
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="全模型 EMA decay (ZeRO-3 分片本地维护, 含解冻后的 VLM); 0=关闭, 建议 0.999")
    parser.add_argument("--image_aug_brightness", type=float, default=0.0,
                        help="亮度 jitter 幅度 (0.2 → U(0.8,1.2)); 样本内所有相机/帧共享参数, 0=关闭")
    parser.add_argument("--image_aug_contrast", type=float, default=0.0, help="对比度 jitter 幅度")
    parser.add_argument("--image_aug_saturation", type=float, default=0.0, help="饱和度 jitter 幅度 (不碰 hue)")
    parser.add_argument("--image_aug_translate", type=float, default=0.0,
                        help="mild affine 平移幅度 (相对图幅, 如 0.1=±10%; reflection 填充保内容完整)")
    parser.add_argument("--image_aug_scale_min", type=float, default=1.0, help="mild affine 缩放下界")
    parser.add_argument("--image_aug_scale_max", type=float, default=1.0, help="mild affine 缩放上界")
    parser.add_argument("--visual_token_drop_rate", type=float, default=0.0,
                        help="visual token 特征置零概率 (bridge 输入侧, training-only); 0=关闭")
    parser.add_argument("--obs_prev_chunk_frame", action="store_true",
                        help="观测扩展为 [上一 action chunk 起始帧, 当前帧] (chunk 尺度动作-场景反馈)")

    return parser


def _resolve_inplace_save_dir(resume_path, current_snapshot, log):
    """原地续训判定 (2026-08-09 方案 B): resume 目标配置与当前一致时返回应写回的 run 目录。

    resume 后后续 checkpoint 直接写回被续训的 run 目录, 而不是另开时间戳目录 —
    一个 config 一条血统线一个目录, latest 指针即血统末端 (resume_search /
    upload watcher / 本地化下载 / eval 全都只认 run 目录 + latest, 下游零改动)。

    Returns:
        run 目录路径 (原地续训); 以下情况返回 None (调用方退回新建时间戳目录,
        fork 语义, 原目录不受污染): 无 resume / 目标无 training_config.json
        (无法校验) / 配置与当前不匹配 (配置漂移)。
    """
    import json

    if not resume_path:
        return None
    base = os.path.basename(resume_path.rstrip("/"))
    if base == "final" or (base.startswith("step_") and base[5:].isdigit()):
        run_dir = os.path.dirname(resume_path.rstrip("/"))  # tag 目录 → 父 run 目录
    else:
        run_dir = resume_path
    cfg_path = os.path.join(run_dir, "training_config.json")
    if not os.path.isfile(cfg_path):
        log(f"[inplace-resume] {run_dir} 无 training_config.json, 无法校验配置 — "
            f"后续 checkpoint 写入新 run 目录 (fork)")
        return None
    try:
        with open(cfg_path) as f:
            cand_json = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(f"[inplace-resume] {cfg_path} 不可读 ({e}) — 后续 checkpoint 写入新 run 目录 (fork)")
        return None
    diffs = diff_snapshot(current_snapshot, cand_json)
    if diffs:
        log(f"[inplace-resume] ⚠️ 当前配置与 {run_dir} 的训练配置不匹配 (差异 {len(diffs)} 处) — "
            f"后续 checkpoint 写入新 run 目录 (fork), 原目录不受污染:")
        for d in diffs[:5]:
            log(f"[inplace-resume]     {d}")
        return None
    log(f"[inplace-resume] 配置匹配, 后续 checkpoint 原地写回: {run_dir}")
    return run_dir


def main():
    # 初始化分布式
    dist_info = setup_distributed()

    # 参数解析
    args = build_arg_parser().parse_args()

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
    if args.save_every_n_seconds is not None and args.save_every_n_seconds <= 0:
        raise ValueError("--save_every_n_seconds must be positive when provided.")

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
        _log(f"VLM Backbone: {args.vlm_backbone}")
        _log(f"VLM Bridge: {args.vlm_bridge_mode}" + (f" (width={args.vlm_bridge_width}, layers={args.vlm_bridge_layers})" if args.vlm_bridge_mode == "transformer" else ""))
        _log(f"VLM Path: {args.vlm_path}")
        _log(f"Train VLM: {args.train_vlm}")
        _log("=" * 60)

    # 获取数据集元数据
    _log(f"Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )

    # 创建 LoLA 配置 (build_lola_config 与 resume_search CLI 共享, 保证 resume
    # 搜索用的配置快照与实际训练进程严格同源)
    config, features, action_dim, state_dim = build_lola_config(args, dataset_metadata)

    # Resume 解析: --resume 支持三形态 — 具体 tag 目录 (step_XXXXXX/final) /
    # 含 latest 指针的 run 目录 / run 集合目录。集合目录进入搜索模式: 读取各 run 的
    # training_config.json, 与当前训练配置做语义匹配, 在匹配的 run 中选 latest
    # 步数最多者 (匹配规则详见 resume_search.py 模块 docstring)
    dataset_meta_snapshot = build_dataset_metadata_snapshot(dataset_metadata, features)
    inplace_save_dir = None
    if args.resume and args.strategy == "deepspeed":
        _resume_snapshot = build_current_snapshot(
            args, config, dataset_meta_snapshot, dist_info["world_size"]
        )
        args.resume = resolve_resume_auto(args.resume, _resume_snapshot, log=_log)
        if args.resume is None:
            _log("Resume disabled: 搜索模式未找到匹配的 checkpoint, 从头开始训练")
        else:
            # 原地续训判定: 配置匹配 → 后续 checkpoint 写回被续训的 run 目录;
            # 配置漂移/无法校验 → 响亮告警并退回新建时间戳目录 (fork)
            inplace_save_dir = _resolve_inplace_save_dir(args.resume, _resume_snapshot, log=_log)

    # 创建数据集
    _log("Creating dataset...")
    if args.norm_mode == "robovlm":
        norm_action = True
    elif args.norm_mode == "zscore":
        norm_action = "zscore"
    else:
        norm_action = False
    # 2026-08-12: 图像增强 (train-only; LoLADataset 内样本级共享参数应用,
    # eval/val 数据集不要传此 transform)
    image_aug_transform = None
    if (args.image_aug_brightness > 0 or args.image_aug_contrast > 0
            or args.image_aug_saturation > 0 or args.image_aug_translate > 0
            or args.image_aug_scale_min < 1.0 or args.image_aug_scale_max > 1.0):
        from lerobot.datasets.lola_dataset import LolaImageAugment
        image_aug_transform = LolaImageAugment(
            brightness=args.image_aug_brightness,
            contrast=args.image_aug_contrast,
            saturation=args.image_aug_saturation,
            translate=args.image_aug_translate,
            scale_min=args.image_aug_scale_min,
            scale_max=args.image_aug_scale_max,
        )
        _log(f"Image augmentation enabled: b={args.image_aug_brightness} "
             f"c={args.image_aug_contrast} s={args.image_aug_saturation} "
             f"t={args.image_aug_translate} scale=[{args.image_aug_scale_min}, {args.image_aug_scale_max}]")

    # 方案B 数据集侧: 合并历史流默认跟随 --no_previous_task_end 旋钮,
    # 可用 --merge_history_stream / --no_merge_history_stream 显式覆盖 (消融用)。
    # 解析与 resume 快照同源 (resume_search.resolve_merge_history_stream),
    # 保证"实际训练行为"与"resume 匹配判定"一致
    merge_history_stream = resolve_merge_history_stream(vars(args))
    if merge_history_stream:
        _log("merge_history_stream enabled: transition+task merged into one continuous history stream")

    train_dataset = create_lola_dataset(
        repo_id=args.dataset_repo_id,
        config=config,
        root=args.dataset_root,
        episodes=args.episodes,
        image_transforms=image_aug_transform,
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
        merge_history_stream=merge_history_stream,
        stats_mode=args.stats_mode,
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
    collate = make_collate_fn(static_max_len=static_max_len, chunk_size=args.action_chunk_size)

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
        save_every_n_seconds=args.save_every_n_seconds,
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
        # Resume 预检: checkpoint 保存时 VLM 已解冻 → 引擎从一开始就以 VLM 可训练构建,
        # 优化器分组与 checkpoint 匹配, Adam 矩无损恢复
        resume_vlm_unfrozen=(
            peek_deepspeed_checkpoint_vlm_unfrozen(args.resume)
            if args.resume and args.strategy == "deepspeed" else False
        ),
        # Config saving (dataset_meta_snapshot 与 resume 搜索的匹配快照同源)
        training_args=vars(args),
        dataset_metadata=dataset_meta_snapshot,
    )

    # Wandb 已在 main() 开头提前初始化，同步 trainer 的 use_wandb 标记
    trainer.use_wandb = use_wandb
    # 原地续训写回目录 (None = 新建时间戳目录)
    trainer.resume_save_dir = inplace_save_dir

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
