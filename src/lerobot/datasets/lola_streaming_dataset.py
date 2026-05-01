#!/usr/bin/env python

# Copyright 2025 LoLA Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoLA 流式数据集，支持从 Azure Blob 等远程存储流式加载数据。

与 LoLADataset 的区别：
- LoLADataset 继承 LeRobotDataset（map-style），通过索引随机访问
- LoLAStreamingDataset 继承 StreamingLeRobotDataset（iterable），通过迭代器顺序访问
- LoLAStreamingDataset 使用 Backtrackable 迭代器的 peek_back 功能加载历史 action

使用方法：
    dataset = LoLAStreamingDataset(
        repo_id="lerobot/pusht",
        max_history_length=100,
        action_chunk_size=10,
        delta_timestamps={...},
        streaming=True,
    )

    for item in dataset:
        # item["hist_actions_full"]: [padded_length, action_dim]
        # item["hist_actions_mask"]: [padded_length] (1=真实, 0=padding)
"""

import importlib
import importlib.util
import concurrent.futures
import pickle
import queue
import threading
from dataclasses import dataclass

import fsspec
import numpy as np
import polars as pl
import torch

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    Backtrackable,
    LookAheadError,
    LookBackError,
    check_version_compatibility,
    find_float_index,
    get_delta_indices,
    is_float_in_list,
    item_to_torch,
)
from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec
from lerobot.utils.constants import HF_LEROBOT_HOME


class BoundedVideoDecoderCache(VideoDecoderCache):
    """带容量上限的 VideoDecoderCache，避免缓存过多解码器占用内存。

    当缓存数量超过 max_size 时，自动淘汰最早加入的解码器。
    对于有大量视频文件的场景（如 17 个 mp4），限制缓存大小可显著
    降低每个 worker 的内存占用（每个 VideoDecoder 约 200MB）。

    Inherits resolution-validation logic from VideoDecoderCache: on cache hit
    the stored resolution is checked against the decoder's current metadata.
    If it diverges (dynamic-resolution AV1), the stale entry is evicted.
    """

    def __init__(self, max_size: int = 4):
        super().__init__()
        self._max_size = max_size
        self._key_order: list[str] = []

    def get_decoder(self, video_path: str):
        video_path = str(video_path)

        with self._lock:
            if video_path in self._cache:
                decoder, file_handle, cached_res = self._cache[video_path]
                meta = decoder.metadata
                current_res = (meta.height, meta.width)
                if current_res != cached_res:
                    try:
                        file_handle.close()
                    except Exception:
                        pass
                    del self._cache[video_path]
                    self._key_order.remove(video_path)

            if video_path not in self._cache:
                # 超过容量时淘汰最早的解码器
                while len(self._cache) >= self._max_size and self._key_order:
                    oldest_key = self._key_order.pop(0)
                    if oldest_key in self._cache:
                        _, old_handle, _ = self._cache.pop(oldest_key)
                        old_handle.close()

                decoder, file_handle, resolution = self._make_decoder(video_path)
                self._cache[video_path] = (decoder, file_handle, resolution)
                self._key_order.append(video_path)

            return self._cache[video_path][0]

    def clear(self):
        with self._lock:
            for _, file_handle, _ in self._cache.values():
                file_handle.close()
            self._cache.clear()
            self._key_order.clear()


class _DecodeError:
    """Wrapper to propagate exceptions from decode thread to main thread."""

    def __init__(self, exc: Exception):
        self.exception = exc


@dataclass
class DecodeProcessConfig:
    """可 pickle 的解码配置，传递给解码子进程。

    将解码所需的所有数据集属性提取为纯数据对象，
    避免传递整个 LoLAStreamingDataset（含 fsspec 文件句柄等，不可 pickle）。
    """

    root: str
    streaming_from_local: bool
    tolerance_s: float
    camera_keys: list
    image_transforms: object  # Callable or None, must be picklable
    delta_indices: object  # dict or None
    video_path_template: str  # meta.video_path format string
    url_root: str
    # 预提取的 episode 视频路径映射: ep_idx -> video_key -> (chunk_index, file_index)
    episode_video_map: dict
    camera_shapes: dict  # camera_key -> shape list, for padding frame
    decode_device: str
    decode_num_threads: int
    cache_size_per_thread: int


def _resolve_video_path(config: DecodeProcessConfig, ep_idx: int, video_key: str) -> str:
    """从 config 重建视频文件路径（等价于 meta.get_video_file_path）。"""
    chunk_idx, file_idx = config.episode_video_map[ep_idx][video_key]
    fpath = config.video_path_template.format(
        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
    )
    root = config.url_root if not config.streaming_from_local else config.root
    return f"{root}/{fpath}"


def _make_padding_frame(camera_shapes: dict, camera_key: str) -> torch.Tensor:
    """创建全零 padding 帧（等价于 _make_padding_camera_frame）。"""
    return torch.zeros(camera_shapes[camera_key]).permute(-1, 0, 1)


def _compute_padding_mask(config: DecodeProcessConfig, video_frames, query_timestamps, original_timestamps):
    """计算视频帧 padding mask（等价于 _get_video_frame_padding_mask）。"""
    padding_mask = {}
    for video_key, timestamps in original_timestamps.items():
        if video_key not in video_frames:
            continue
        frames = []
        mask = []
        padding_frame = _make_padding_frame(config.camera_shapes, video_key)
        for ts in timestamps:
            if is_float_in_list(ts, query_timestamps[video_key]):
                idx = find_float_index(ts, query_timestamps[video_key])
                frames.append(video_frames[video_key][idx, :])
                mask.append(False)
            else:
                frames.append(padding_frame)
                mask.append(True)
        padding_mask[f"{video_key}_is_pad"] = torch.BoolTensor(mask)
    return padding_mask


def _decode_process_main(
    config: DecodeProcessConfig,
    light_queue,
    result_queue,
    shutdown_event,
):
    """解码子进程入口函数。

    在子进程中重建解码基础设施（ThreadPoolExecutor + per-thread cache），
    然后进入与旧 AsyncDecodePipeline 相同的解码循环。

    Args:
        config: 解码配置（可 pickle 的纯数据对象）
        light_queue: torch.multiprocessing.Queue，接收轻量级帧
        result_queue: torch.multiprocessing.Queue，返回解码后的帧
        shutdown_event: torch.multiprocessing.Event，关闭信号
    """
    # 1. 创建 ThreadPoolExecutor（batch 内并行解码）
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=config.decode_num_threads,
        thread_name_prefix="DecodeWorker",
    )

    # 2. Per-thread VideoDecoder cache（与旧 AsyncDecodePipeline 相同机制）
    tls = threading.local()
    all_caches = []
    caches_lock = threading.Lock()

    def get_thread_cache():
        if not hasattr(tls, "decoder_cache"):
            cache = BoundedVideoDecoderCache(max_size=config.cache_size_per_thread)
            tls.decoder_cache = cache
            with caches_lock:
                all_caches.append(cache)
        return tls.decoder_cache

    def get_thread_cuda_cache():
        if not hasattr(tls, "cuda_decoder_cache"):
            cuda_cache_size = max(4, config.cache_size_per_thread // 2)
            cache = BoundedVideoDecoderCache(max_size=cuda_cache_size)
            tls.cuda_decoder_cache = cache
            with caches_lock:
                all_caches.append(cache)
        return tls.cuda_decoder_cache

    # 3. 视频查询函数（等价于旧 AsyncDecodePipeline._query_videos_cached）
    def query_videos(query_timestamps, ep_idx):
        item = {}
        for video_key, query_ts in query_timestamps.items():
            video_path = _resolve_video_path(config, ep_idx, video_key)

            if config.decode_device == "cuda":
                frames = _decode_video_cuda_in_process(
                    config, video_path, query_ts, get_thread_cuda_cache
                )
            else:
                frames = decode_video_frames_torchcodec(
                    video_path, query_ts, config.tolerance_s,
                    decoder_cache=get_thread_cache(),
                )

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames
        return item

    # 4. 单 item 解码函数
    def decode_one(item):
        if "_video_lookup" not in item:
            return item

        item_copy = item.copy()
        video_lookup = item_copy.pop("_video_lookup", None)

        if video_lookup is None:
            for cam_key in config.camera_keys:
                if cam_key not in item_copy:
                    item_copy[cam_key] = _make_padding_frame(config.camera_shapes, cam_key)
                    item_copy[f"{cam_key}_is_pad"] = torch.BoolTensor([True])
            return item_copy

        ep_idx = video_lookup["ep_idx"]
        q_timestamps = video_lookup["query_timestamps"]
        original_timestamps = video_lookup["original_timestamps"]

        video_frames = query_videos(q_timestamps, ep_idx)

        if config.image_transforms is not None:
            for cam in config.camera_keys:
                video_frames[cam] = config.image_transforms(video_frames[cam])

        item_copy.update(video_frames)

        if config.delta_indices is not None:
            padding_mask = _compute_padding_mask(
                config, video_frames, q_timestamps, original_timestamps
            )
            item_copy.update(padding_mask)

        return item_copy

    # 5. 主循环
    while not shutdown_event.is_set():
        try:
            items = light_queue.get(block=True, timeout=0.5)
        except Exception:
            # queue.Empty or other queue exceptions
            continue

        if items is None:  # Shutdown sentinel
            break

        try:
            decoded = list(executor.map(decode_one, items))
            result_queue.put(decoded, block=True, timeout=5.0)
        except Exception as e:
            try:
                result_queue.put(_DecodeError(e), block=True, timeout=5.0)
            except Exception:
                pass

    # 6. 清理
    executor.shutdown(wait=False)
    with caches_lock:
        for cache in all_caches:
            try:
                cache.clear()
            except Exception:
                pass


def _decode_video_cuda_in_process(config, video_path, timestamps, get_cuda_cache):
    """在解码子进程中使用 CUDA 解码视频帧。"""
    from torchcodec.decoders import VideoDecoder

    cache = get_cuda_cache()
    video_path_str = str(video_path)

    with cache._lock:
        if video_path_str in cache._cache:
            decoder, file_handle, cached_res = cache._cache[video_path_str]
            meta = decoder.metadata
            current_res = (meta.height, meta.width)
            if current_res != cached_res:
                try:
                    file_handle.close()
                except Exception:
                    pass
                del cache._cache[video_path_str]
                cache._key_order.remove(video_path_str)

        if video_path_str not in cache._cache:
            file_handle = fsspec.open(video_path_str).__enter__()
            decoder = VideoDecoder(file_handle, seek_mode="approximate", device="cuda")
            meta = decoder.metadata
            resolution = (meta.height, meta.width)
            cache._cache[video_path_str] = (decoder, file_handle, resolution)
            cache._key_order.append(video_path_str)
            # Evict if over capacity
            while len(cache._cache) > cache._max_size:
                oldest_key = cache._key_order.pop(0)
                if oldest_key in cache._cache:
                    _, old_handle, _ = cache._cache.pop(oldest_key)
                    try:
                        old_handle.close()
                    except Exception:
                        pass
        else:
            # Update access order (LRU)
            if video_path_str in cache._key_order:
                cache._key_order.remove(video_path_str)
            cache._key_order.append(video_path_str)

    decoder = cache._cache[video_path_str][0]

    # Decode frames
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    num_frames = metadata.num_frames
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    clamped_mask = [idx >= num_frames or idx < 0 for idx in frame_indices]
    frame_indices = [max(0, min(idx, num_frames - 1)) for idx in frame_indices]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    # GPU decode → CPU
    loaded_frames = [frame.cpu() for frame in frames_batch.data]
    loaded_ts = [pts.item() for pts in frames_batch.pts_seconds]

    # Tolerance check
    query_ts_tensor = torch.tensor(timestamps)
    loaded_ts_tensor = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
    min_, argmin_ = dist.min(1)
    clamped_mask_tensor = torch.tensor(clamped_mask)
    is_within_tol = (min_ < config.tolerance_s) | clamped_mask_tensor
    assert is_within_tol.all(), (
        f"Timestamp tolerance violated: {min_[~is_within_tol]} > {config.tolerance_s=}. "
        f"video: {video_path}"
    )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_frames = (closest_frames / 255.0).type(torch.float32)
    return closest_frames


class DecodeProcessPipeline:
    """独立子进程视频解码管线，带持久化 per-thread VideoDecoder cache。

    与旧 AsyncDecodePipeline（线程）的区别：
    - 使用 torch.multiprocessing.Process 而非 threading.Thread
    - 独立 GIL，Python 层代码也可真并行
    - torch.multiprocessing.Queue 零拷贝传输 tensor（Linux fd sharing）
    - 内存由队列深度控制：result_queue(maxsize=1) 最多缓冲 1 个解码 batch

    Pipeline timeline:
        主进程:       [fetch N] [consume decoded N-1] [train N-1] [consume decoded N] [train N] ...
        解码子进程:           [decode N (parallel)]                  [decode N+1 (parallel)] ...

    子进程内部:
        - ThreadPoolExecutor(decode_num_threads) 并行解码 batch 内多个 item
        - 每个线程有自己的 BoundedVideoDecoderCache（threading.local）
        - torchcodec 释放 GIL，C 层 FFmpeg 真并行
    """

    def __init__(
        self,
        config: DecodeProcessConfig,
        light_queue_depth: int = 2,
        result_queue_depth: int = 1,
    ):
        import torch.multiprocessing as mp

        self._config = config
        self._light_queue = mp.Queue(maxsize=light_queue_depth)
        self._result_queue = mp.Queue(maxsize=result_queue_depth)
        self._shutdown_event = mp.Event()
        self._shutdown_called = False

        self._process = mp.Process(
            target=_decode_process_main,
            args=(self._config, self._light_queue, self._result_queue, self._shutdown_event),
            name="DecodeProcessPipeline",
            daemon=True,
        )
        self._process.start()

        # 注册 atexit 清理，防止主进程 crash 留下僵尸进程
        import atexit
        atexit.register(self.shutdown)

    def submit(self, items: list[dict]) -> None:
        """Submit lightweight items for decoding. Blocks if queue is full."""
        self._light_queue.put(items, block=True)

    def consume(self) -> list[dict]:
        """Get decoded items. Blocks until available. Raises on decode error."""
        result = self._result_queue.get(block=True)
        if isinstance(result, _DecodeError):
            raise result.exception
        return result

    def shutdown(self) -> None:
        """Signal shutdown, drain queues, and wait for process to exit."""
        if self._shutdown_called:
            return
        self._shutdown_called = True

        self._shutdown_event.set()
        # Drain light_queue to unblock the process if it's waiting on put
        try:
            while not self._light_queue.empty():
                self._light_queue.get_nowait()
        except Exception:
            pass
        # Put a None sentinel to unblock the process if it's waiting on get
        try:
            self._light_queue.put(None, block=False)
        except Exception:
            pass
        # Wait for process to finish
        self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)



class PolarsRowIterator:
    """
    使用 polars 读取 parquet 文件并逐行生成 dict，替代 HF IterableDataset。

    采用文件级懒加载策略：只读取当前 worker 行范围覆盖的那些 parquet 文件，
    跳过不相关的文件。对于 2M+ episode 的大规模数据集（~15GB parquet，
    数十个文件），每个 worker 只需加载 ~1/N 的 parquet 数据到内存，
    避免所有 worker 都加载全量数据导致内存爆炸。

    生成的 dict 格式与 HF IterableDataset 一致，确保 item_to_torch、
    make_frame、_add_history_actions 等下游逻辑无需修改。
    """

    def __init__(self, parquet_paths: list[str], episodes: list[int] | None = None,
                 row_offset: int = 0, row_limit: int | None = None):
        """
        Args:
            parquet_paths: parquet 文件路径列表（按顺序读取）
            episodes: 如果指定，只迭代这些 episode 的行
            row_offset: 全局行偏移（跳过前 N 行）
            row_limit: 最多读取 N 行（在偏移之后）
        """
        import pyarrow.parquet as pq

        # ── episodes 过滤模式：使用 polars lazy scan + filter ──────────
        if episodes is not None:
            dfs = []
            for path in parquet_paths:
                df = pl.scan_parquet(path).filter(
                    pl.col("episode_index").is_in(episodes)
                ).collect()
                if df.height > 0:
                    dfs.append(df)
            if dfs:
                self._df = pl.concat(dfs, how="vertical_relaxed")
                if self._df.height > 0:
                    total = self._df.height
                    start = min(row_offset, total)
                    end = total if row_limit is None else min(start + row_limit, total)
                    self._df = self._df.slice(start, end - start)
            else:
                self._df = pl.DataFrame()
            self._pos = 0
            self._len = self._df.height
            return

        # ── 无 episodes 过滤：文件级懒加载 ──────────────────────────────
        # Step 1: 快速扫描元数据，获取每个文件的行数（不读取数据）
        file_infos: list[tuple[str, int, int]] = []  # (path, num_rows, global_start)
        global_offset = 0
        for path in parquet_paths:
            pf = pq.ParquetFile(path)
            n = pf.metadata.num_rows
            file_infos.append((path, n, global_offset))
            global_offset += n
        total_global_rows = global_offset

        if total_global_rows == 0:
            self._df = pl.DataFrame()
            self._pos = 0
            self._len = 0
            return

        # Step 2: 确定目标行范围
        target_start = row_offset
        target_end = total_global_rows if row_limit is None else min(row_offset + row_limit, total_global_rows)

        if target_start >= total_global_rows:
            self._df = pl.DataFrame()
            self._pos = 0
            self._len = 0
            return

        # Step 3: 只读取与目标行范围有交集的 parquet 文件
        dfs = []
        for path, file_rows, file_global_start in file_infos:
            file_global_end = file_global_start + file_rows
            # 该文件是否与目标范围有交集？
            if file_global_end <= target_start or file_global_start >= target_end:
                continue

            # 计算该文件内的局部切片范围
            local_start = max(0, target_start - file_global_start)
            local_end = min(file_rows, target_end - file_global_start)

            # 使用 lazy scan + slice 下推读取
            df = pl.scan_parquet(path).slice(local_start, local_end - local_start).collect()
            if df.height > 0:
                dfs.append(df)

        if not dfs:
            self._df = pl.DataFrame()
        else:
            self._df = pl.concat(dfs, how="vertical_relaxed")

        self._pos = 0
        self._len = self._df.height

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= self._len:
            raise StopIteration
        row = self._df.row(self._pos, named=True)
        self._pos += 1
        return self._row_to_dict(row)

    def _row_to_dict(self, row: dict) -> dict:
        """将 polars 行转为与 HF IterableDataset 输出一致的 dict 格式。"""
        result = {}
        for key, val in row.items():
            if isinstance(val, list):
                # list columns (observation.state, action) -> numpy array
                result[key] = np.array(val, dtype=np.float32)
            elif isinstance(val, (int, np.integer)):
                result[key] = int(val)
            elif isinstance(val, (float, np.floating)):
                result[key] = float(val)
            else:
                result[key] = val
        return result


def _is_valid_parquet_file(path) -> bool:
    """Filter out hidden and temporary files (e.g. Azure .azDownload-* partial downloads)."""
    return not path.name.startswith(".")


def _discover_parquet_files(root: str) -> list[str]:
    """发现数据目录下所有 chunk/file parquet 文件，按顺序排列。"""
    from pathlib import Path
    data_dir = Path(root) / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    files = sorted(f for f in data_dir.glob("*/*.parquet") if _is_valid_parquet_file(f))
    return [str(f) for f in files]


class LoLAStreamingDataset(torch.utils.data.IterableDataset):
    """
    支持流式加载完整历史 action 的 LoLA 数据集。

    不继承 StreamingLeRobotDataset，而是直接继承 IterableDataset，
    避免 super().__init__() 调用 load_dataset(streaming=True) 创建
    HF IterableDataset，该对象在 DataLoader fork 子进程中会导致死锁。

    数据加载使用 polars 读取 parquet 文件，视频解码使用 torchcodec。
    make_frame 和 _add_history_actions 逻辑与原 StreamingLeRobotDataset 兼容。
    """

    def __init__(
        self,
        repo_id: str,
        max_history_length: int = 100,
        action_chunk_size: int = 10,
        history_padding_side: str = "left",
        root: str | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng=None,
        shuffle: bool = True,
        deferred_video_decode: bool = True,
        decode_device: str = "cpu",
        decode_num_threads: int = 1,
        async_decode: bool = False,
        num_dataloader_workers: int = 0,
        start_index: int = 0,
    ):
        """
        Args:
            repo_id: 数据集仓库 ID
            max_history_length: 历史 action 最大长度，超过则截断
            action_chunk_size: action 块大小，历史长度补齐到该值的整数倍
            history_padding_side: padding 方向，"left" 或 "right"
            root: 本地数据集根目录（挂载路径）
            episodes: 指定加载的 episode 列表
            image_transforms: 图像变换
            delta_timestamps: 时间戳偏移配置
            tolerance_s: 时间戳容差
            revision: 版本
            force_cache_sync: 是否强制同步缓存
            streaming: 是否流式加载（保留接口兼容，实际始终用 polars）
            buffer_size: 流式 shuffle 缓冲区大小
            max_num_shards: 最大分片数（保留接口兼容，实际按行范围分配）
            seed: 随机种子
            rng: 随机数生成器
            shuffle: 是否 shuffle
            deferred_video_decode: 是否延迟视频解码。默认 True，shuffle buffer
                只存储轻量级数据（~4KB/帧），yield 时再解码视频帧，内存占用
                远低于在 make_frame 中解码（~3.5MB/帧 × buffer_size × num_workers）。
                设为 False 则在 make_frame 中立即解码视频帧存入 buffer，适用于
                大内存低 IO 场景（如本地 NVMe + 充足内存）。
            decode_device: 视频解码设备，"cpu" 或 "cuda"。当 deferred_video_decode=True
                时，指定主进程中视频解码使用的设备。"cuda" 使用 NVDEC 硬件加速解码，
                可显著提升解码速度。"cpu" 使用 CPU 解码（默认）。
            decode_num_threads: 主进程中并行解码的线程数。当 decode_device="cpu" 时，
                使用 ThreadPoolExecutor 并行解码多个 item。设为 1 表示串行解码，
                设为 >1 表示多线程并行（视频解码是 IO 密集型，线程效果较好）。
                当 decode_device="cuda" 时此参数无效（CUDA 解码本身已并行）。
            async_decode: 是否启用异步解码管线。启用后，视频解码在专用子进程中
                执行，与训练前向传播重叠（训练 batch N 时解码 batch N+1）。
                子进程内部使用 ThreadPoolExecutor + per-thread BoundedVideoDecoderCache
                并行解码，缓存容量为 2 * num_dataloader_workers * n_cameras。
                内存由队列深度控制：result_queue(maxsize=1) 最多缓冲 1 个解码 batch。
            num_dataloader_workers: DataLoader 的 worker 数量，用于计算异步解码
                管线的 VideoDecoder 缓存容量。仅在 async_decode=True 时需要。
            start_index: Resume 时跳过的全局样本数。每个 worker 按 start_index // total_parallel
                跳过对应数量的 yield item，rng/buffer 状态正常推进但不输出，实现数据级断点续训。
        """
        super().__init__()  # torch.utils.data.IterableDataset.__init__

        self.repo_id = repo_id
        self.root = __import__("pathlib").Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.shuffle = shuffle

        self.buffer_size = buffer_size
        self.video_decoder_cache = None
        self.deferred_video_decode = deferred_video_decode
        self.decode_device = decode_device
        self.decode_num_threads = decode_num_threads

        # CUDA 解码器缓存（仅 decode_device="cuda" 时使用）
        self._cuda_decoder_cache = None

        # 异步解码管线（仅 async_decode=True 时使用）
        self.async_decode = async_decode
        self._num_dataloader_workers = num_dataloader_workers
        self._decode_pipeline = None
        self.start_index = start_index

        # 加载元数据（不加载 HF dataset）
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        self.delta_timestamps = None
        self.delta_indices = None

        if delta_timestamps is not None:
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.max_history_length = max_history_length
        self.action_chunk_size = action_chunk_size
        self.history_padding_side = history_padding_side

        # 获取 action 维度
        if "action" in self.meta.features:
            self.action_dim = self.meta.features["action"]["shape"][0]
        else:
            self.action_dim = 1

        # 发现 parquet 文件（polars 模式用）
        self._parquet_files = _discover_parquet_files(str(self.root))

        # 预计算总行数用于 shard 分配
        self._total_rows = self.meta.total_frames

        print(f"[LoLAStreamingDataset] max_history_length: {max_history_length}")
        print(f"[LoLAStreamingDataset] action_chunk_size: {action_chunk_size}")
        print(f"[LoLAStreamingDataset] history_padding_side: {history_padding_side}")
        print(f"[LoLAStreamingDataset] action_dim: {self.action_dim}")
        print(f"[LoLAStreamingDataset] parquet_files: {len(self._parquet_files)}")
        print(f"[LoLAStreamingDataset] total_rows: {self._total_rows}")

    # ── 从 StreamingLeRobotDataset 继承的属性 ──────────────────────────────

    @property
    def num_frames(self):
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def fps(self):
        return self.meta.fps

    # ── 从 StreamingLeRobotDataset 继承的静态方法 ──────────────────────────

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator, buffer_size: int, random_batch_size=100
    ):
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    @staticmethod
    def _infinite_generator_over_elements(rng: np.random.Generator, elements: list[int]):
        while True:
            yield rng.choice(elements)

    # ── 核心方法 ──────────────────────────────────────────────────────────

    def _get_window_steps(self, delta_timestamps=None, dynamic_bounds=False):
        """计算 lookback/lookahead 窗口大小"""
        from lerobot.utils.constants import LOOKAHEAD_BACKTRACKTABLE, LOOKBACK_BACKTRACKTABLE

        if delta_timestamps is None:
            return 1, 1

        if not dynamic_bounds:
            lookback = LOOKBACK_BACKTRACKTABLE
            lookahead = LOOKAHEAD_BACKTRACKTABLE
        else:
            all_timestamps = sum(delta_timestamps.values(), [])
            lookback = min(all_timestamps) * self.fps
            lookahead = max(all_timestamps) * self.fps
            lookback = 0 if lookback >= 0 else (lookback * -1)

        # lookback 需要至少 max_history_length 以支持完整历史加载
        lookback = max(lookback, self.max_history_length)

        return lookback, lookahead

    def _make_polars_backtrackable(self, row_offset: int = 0, row_limit: int | None = None) -> Backtrackable:
        """创建基于 polars 的 Backtrackable 迭代器。"""
        lookback, lookahead = self._get_window_steps(self.delta_timestamps)

        row_iter = PolarsRowIterator(
            self._parquet_files,
            episodes=self.episodes,
            row_offset=row_offset,
            row_limit=row_limit,
        )

        if row_iter._len == 0:
            raise StopIteration("No data available")

        return Backtrackable(row_iter, history=lookback, lookahead=lookahead)

    def __iter__(self):
        """使用 polars 读取数据并支持分布式数据分片。

        用 polars 读取 parquet 替代 HF IterableDataset.shard()。
        polars 读取是 stateless 的，在 DataLoader fork 的子进程中安全运行。
        不调用 load_dataset，避免 HF IterableDataset 在 fork 后导致死锁。

        shard 分配逻辑：将总行数按 parallel_id 分配不重叠的行范围，
        每个 worker 处理 [row_offset, row_offset + row_limit) 范围内的行。

        三种视频解码模式：
        1. deferred_video_decode=True（默认）：
           worker yield 轻量级帧（~4KB），主进程 decode_items_batch() 解码，
           或通过 async_decode=True 启用独立解码子进程管线。

        2. deferred_video_decode=False：
           make_frame 中立即解码，buffer 存完整帧（~3.5MB/帧）。
           速度快但 flush 阶段内存飙升（10 workers × 1000 帧 × 3.5MB ≈ 35GB）。
        """
        # 初始化视频解码缓存，容量按视角数调整（每视角缓存 2 个 decoder）
        if self.video_decoder_cache is None:
            n_cameras = max(1, len(self.meta.video_keys))
            cache_size = max(4, n_cameras * 2)
            self.video_decoder_cache = BoundedVideoDecoderCache(max_size=cache_size)

        # 获取分布式信息
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # 获取 DataLoader worker 信息
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # 总并行度 = world_size × num_workers
        total_parallel = world_size * num_workers
        parallel_id = rank * num_workers + worker_id

        # 按行范围分配 shard：每个 worker 处理不重叠的行范围
        rows_per_worker = self._total_rows // total_parallel
        remainder = self._total_rows % total_parallel

        # 前 remainder 个 worker 多处理一行
        if parallel_id < remainder:
            row_offset = parallel_id * (rows_per_worker + 1)
            row_limit = rows_per_worker + 1
        else:
            row_offset = remainder * (rows_per_worker + 1) + (parallel_id - remainder) * rows_per_worker
            row_limit = rows_per_worker

        if row_offset >= self._total_rows:
            print(f"[LoLAStreamingDataset] Worker {parallel_id} has no rows assigned "
                  f"(total={self._total_rows}, parallel={total_parallel})", flush=True)
            return

        print(f"[LoLAStreamingDataset] Worker {parallel_id} assigned rows "
              f"[{row_offset}, {row_offset + row_limit})", flush=True)

        # 创建 polars backtrackable 迭代器
        try:
            backtrack_dataset = self._make_polars_backtrackable(row_offset, row_limit)
        except StopIteration:
            print(f"[LoLAStreamingDataset] Worker {parallel_id} no data available", flush=True)
            return

        # 随机数生成器
        rng = np.random.default_rng(self.seed + rank) if not self.shuffle else np.random.default_rng(self.rng.integers(0, 2**31) + rank)

        # 选择帧生成方式：
        # - deferred + async: buffer 存轻量级帧（~4KB），yield 轻量帧，子进程解码
        # - deferred + 非 async: buffer 存轻量级帧（~4KB），yield 时解码
        #   （速度和内存的最佳平衡：buffer 省内存，worker 进程解码快于主进程）
        # - 非 deferred: make_frame 中立即解码，buffer 存完整帧（~3.5MB）
        if self.deferred_video_decode:
            frame_generator = self._make_frame_lightweight
        else:
            frame_generator = self.make_frame

        # deferred + 非 async 时，yield 前解码视频（在 worker 进程中执行）
        decode_on_yield = self.deferred_video_decode and not self.async_decode

        # 单 shard 模式：直接迭代，使用 buffer shuffle
        buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)
        frames_buffer = []
        yield_count = 0

        # Resume: fast-forward through start_index items per worker
        skip_remaining = self.start_index // total_parallel
        if skip_remaining > 0:
            print(f"[LoLAStreamingDataset] Worker {parallel_id} skipping {skip_remaining} items for resume...", flush=True)

        while True:
            try:
                for frame in frame_generator(backtrack_dataset):
                    if len(frames_buffer) == self.buffer_size:
                        i = next(buffer_indices_generator)
                        to_yield = frames_buffer[i]
                        if skip_remaining > 0:
                            skip_remaining -= 1
                            frames_buffer[i] = frame
                            continue  # fast-forward: advance rng/buffer state but don't yield
                        yield_count += 1
                        if decode_on_yield and "_video_lookup" in to_yield:
                            to_yield = self._decode_videos(to_yield)
                        yield to_yield
                        frames_buffer[i] = frame
                    else:
                        frames_buffer.append(frame)
                    break
            except (RuntimeError, StopIteration) as e:
                print(
                    f"[LoLAStreamingDataset] Worker {parallel_id} "
                    f"finished after yielding {yield_count} items: {type(e).__name__}: {e}",
                    flush=True,
                )
                break

        # Flush remaining buffer
        rng.shuffle(frames_buffer)
        yield_count += len(frames_buffer)
        print(f"[LoLAStreamingDataset] Worker {parallel_id} finished, "
              f"total yielded: {yield_count}, buffer: {len(frames_buffer)}", flush=True)
        if decode_on_yield:
            for frame in frames_buffer:
                if "_video_lookup" in frame:
                    frame = self._decode_videos(frame)
                yield frame
        else:
            yield from frames_buffer

    def make_frame(self, dataset_iterator):
        """生成帧数据，包含完整历史 action 和视频帧。"""
        item = next(dataset_iterator)
        item = item_to_torch(item)

        updates = []

        ep_idx = item["episode_index"]
        current_ts = item["index"] / self.fps

        episode_boundaries_ts = {
            key: (
                self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        # Apply delta querying logic if necessary
        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.append(query_result)
            updates.append(padding)

        # Load video frames
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)

            if self.image_transforms is not None:
                for cam in self.meta.camera_keys:
                    video_frames[cam] = self.image_transforms(video_frames[cam])

            updates.append(video_frames)

            if self.delta_indices is not None:
                padding_mask = self._get_video_frame_padding_mask(
                    video_frames, query_timestamps, original_timestamps
                )
                updates.append(padding_mask)

        result = item.copy()
        for update in updates:
            result.update(update)

        result["task"] = self.meta.tasks.iloc[item["task_index"]].name

        # 添加历史 action
        result = self._add_history_actions(result, dataset_iterator)

        yield result

    def _make_frame_lightweight(self, dataset_iterator):
        """生成轻量级帧数据（不含解码后的视频帧），用于低内存 shuffle buffer。

        与 make_frame 的区别：
        - 不调用 _query_videos 解码视频帧
        - 将视频查找信息存入 _video_lookup 字段
        - 返回的帧不含视频图像数据（~4KB vs ~3.5MB）

        视频帧在 yield 时通过 _decode_videos 按需解码。
        """
        item = next(dataset_iterator)
        item = item_to_torch(item)

        updates = []

        ep_idx = item["episode_index"]
        current_ts = item["index"] / self.fps

        episode_boundaries_ts = {
            key: (
                self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        # Apply delta querying logic if necessary
        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.append(query_result)
            updates.append(padding)

        # 不解码视频帧，改为存储视频查找信息
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            # 存储查找信息，yield 时再解码
            item["_video_lookup"] = {
                "ep_idx": ep_idx,
                "query_timestamps": query_timestamps,
                "original_timestamps": original_timestamps,
            }

        result = item.copy()
        for update in updates:
            result.update(update)

        result["task"] = self.meta.tasks.iloc[item["task_index"]].name

        # 添加历史 action
        result = self._add_history_actions(result, dataset_iterator)

        yield result

    def _decode_videos(self, lightweight_frame):
        """对轻量级帧执行延迟的视频解码。

        从 _video_lookup 中读取视频查找信息，解码视频帧，
        添加到结果中并删除 _video_lookup 字段。
        """
        video_lookup = lightweight_frame.pop("_video_lookup", None)

        if video_lookup is None:
            # 无视频（非视频数据集或无 camera keys）
            # 为 camera key 添加 placeholder（保持接口一致）
            for cam_key in self.meta.camera_keys:
                if cam_key not in lightweight_frame:
                    lightweight_frame[cam_key] = self._make_padding_camera_frame(cam_key)
                    lightweight_frame[f"{cam_key}_is_pad"] = torch.BoolTensor([True])
            return lightweight_frame

        ep_idx = video_lookup["ep_idx"]
        query_timestamps = video_lookup["query_timestamps"]
        original_timestamps = video_lookup["original_timestamps"]

        # 执行视频解码
        video_frames = self._query_videos(query_timestamps, ep_idx)

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                video_frames[cam] = self.image_transforms(video_frames[cam])

        lightweight_frame.update(video_frames)

        if self.delta_indices is not None:
            padding_mask = self._get_video_frame_padding_mask(
                video_frames, query_timestamps, original_timestamps
            )
            lightweight_frame.update(padding_mask)

        return lightweight_frame

    def decode_item(self, item):
        """在主进程中解码轻量级帧的视频数据。

        当 deferred_video_decode=True 时，worker 只 yield 轻量级帧（~4KB），
        通过 DataLoader 的 multiprocessing.Queue 传输到主进程。
        此方法在主进程中对轻量级帧执行视频解码，避免 decoded frames
        在 worker 进程中堆积导致内存飙升（flush 阶段可达 50GB+）。

        重要：PyTorch DataLoader 的 collate_fn 在 worker 进程中运行，
        因此不能在 collate_fn 中调用此方法。正确的用法是在主进程中
        通过 decode_and_collate() 辅助函数调用：

            def passthrough_collate(batch):
                return batch  # 直接传递，不做解码

            loader = DataLoader(dataset, collate_fn=passthrough_collate, ...)

            for items in loader:
                # items 是 list of dicts
                decoded = [dataset.decode_item(item) for item in items]
                # ... collate decoded items
        """
        if "_video_lookup" not in item:
            return item
        return self._decode_videos(item)

    def decode_items_batch(self, items):
        """批量解码多个轻量级帧的视频数据（主进程调用）。

        支持三种解码模式：
        1. CUDA 解码 (decode_device="cuda"): 使用 NVDEC 硬件加速，串行调用
        2. 多 decoder 并行 CPU 解码 (decode_device="cpu", decode_num_threads > 1):
           为每个 item 创建独立的 VideoDecoder(num_ffmpeg_threads=2)，
           使用 ThreadPoolExecutor 并行解码。多个 decoder 在 C 层面真正并行，
           不受 Python GIL 限制，比单 decoder 多 ffmpeg 线程更高效。
        3. 串行 CPU 解码 (decode_device="cpu", decode_num_threads=1): 默认方案

        Args:
            items: 轻量级帧列表（从 DataLoader 获取）

        Returns:
            解码后的帧列表
        """
        if not self.deferred_video_decode:
            return items

        # 过滤出需要解码的 items
        need_decode = ["_video_lookup" in item for item in items]

        if not any(need_decode):
            return items

        if self.decode_device == "cuda":
            # CUDA 解码：串行调用（NVDEC 本身已高效）
            return [self.decode_item(item) for item in items]

        if self.decode_num_threads <= 1:
            # 串行 CPU 解码：使用共享 cache
            return [self.decode_item(item) for item in items]

        # 多 decoder 并行 CPU 解码：每个 item 独立 decoder，C 层面并行
        return self._decode_items_parallel(items)

    def _decode_items_parallel(self, items):
        """多 decoder 并行解码：每个 item 独立 VideoDecoder，真正 C 层并行。

        与共享 cache 的 ThreadPoolExecutor 方案不同：
        - 共享 cache: 多线程争抢同一把锁，GIL 限制了 Python 层并行
        - 独立 decoder: 每个 decoder 是独立的 C 对象，FFmpeg 线程不受 GIL 限制
        - 每个 decoder 使用 num_ffmpeg_threads=2（经验最优值）
        - ThreadPoolExecutor 只负责调度，实际解码在 C 层并行
        """
        from concurrent.futures import ThreadPoolExecutor
        from torchcodec.decoders import VideoDecoder as _VideoDecoder
        import fsspec as _fsspec

        num_ffmpeg_threads = 2  # 每个 decoder 的 ffmpeg 线程数

        def _decode_one_item(item):
            """在独立线程中解码单个 item，使用独立的 VideoDecoder。"""
            if "_video_lookup" not in item:
                return item

            video_lookup = item.get("_video_lookup", None)
            if video_lookup is None:
                for cam_key in self.meta.camera_keys:
                    if cam_key not in item:
                        item[cam_key] = self._make_padding_camera_frame(cam_key)
                        item[f"{cam_key}_is_pad"] = torch.BoolTensor([True])
                return item

            ep_idx = video_lookup["ep_idx"]
            query_timestamps = video_lookup["query_timestamps"]
            original_timestamps = video_lookup["original_timestamps"]

            # 使用独立的 VideoDecoder（不经过 cache）
            video_frames = self._query_videos_independent(
                query_timestamps, ep_idx, num_ffmpeg_threads
            )

            result = item.copy()
            if self.image_transforms is not None:
                for cam in self.meta.camera_keys:
                    video_frames[cam] = self.image_transforms(video_frames[cam])

            result.update(video_frames)
            del result["_video_lookup"]

            if self.delta_indices is not None:
                padding_mask = self._get_video_frame_padding_mask(
                    video_frames, query_timestamps, original_timestamps
                )
                result.update(padding_mask)

            return result

        with ThreadPoolExecutor(max_workers=self.decode_num_threads) as executor:
            decoded = list(executor.map(_decode_one_item, items))

        return decoded

    def _query_videos_independent(self, query_timestamps, ep_idx, num_ffmpeg_threads=2):
        """使用独立的 VideoDecoder 解码视频（不使用缓存，线程安全）。

        每个 decoder 使用独立的文件句柄和 FFmpeg 上下文，
        多个线程可以真正并行解码而不争抢锁。
        """
        from torchcodec.decoders import VideoDecoder as _VideoDecoder
        import fsspec as _fsspec

        item = {}
        for video_key, query_ts in query_timestamps.items():
            root = self.meta.url_root if hasattr(self.meta, 'url_root') and self.streaming_from_local is False else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"

            fh = _fsspec.open(video_path).__enter__()
            try:
                decoder = _VideoDecoder(fh, seek_mode="approximate", num_ffmpeg_threads=num_ffmpeg_threads)
                metadata = decoder.metadata
                average_fps = metadata.average_fps
                num_frames = metadata.num_frames
                frame_indices = [round(ts * average_fps) for ts in query_ts]
                clamped_mask = [idx >= num_frames or idx < 0 for idx in frame_indices]
                frame_indices = [max(0, min(idx, num_frames - 1)) for idx in frame_indices]
                frames_batch = decoder.get_frames_at(indices=frame_indices)

                loaded_frames = [frame for frame in frames_batch.data]
                loaded_ts = [pts.item() for pts in frames_batch.pts_seconds]

                # Tolerance check
                query_ts_tensor = torch.tensor(query_ts)
                loaded_ts_tensor = torch.tensor(loaded_ts)
                dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
                min_, argmin_ = dist.min(1)
                clamped_mask_tensor = torch.tensor(clamped_mask)
                is_within_tol = (min_ < self.tolerance_s) | clamped_mask_tensor
                assert is_within_tol.all(), (
                    f"Timestamp tolerance violated: {min_[~is_within_tol]} > {self.tolerance_s=}. "
                    f"video: {video_path}"
                )

                closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
                closest_frames = (closest_frames / 255.0).type(torch.float32)
                item[video_key] = closest_frames.squeeze(0) if len(query_ts) == 1 else closest_frames
            finally:
                try:
                    fh.close()
                except Exception:
                    pass
        return item

    # ── 异步解码管线 ──────────────────────────────────────────────────

    def _ensure_decode_pipeline(self):
        """Lazily create the DecodeProcessPipeline if not already created."""
        if self._decode_pipeline is None:
            n_cameras = max(1, len(self.meta.video_keys))
            cache_size = max(4, 2 * self._num_dataloader_workers * n_cameras)
            num_threads = self.decode_num_threads if self.decode_num_threads > 1 else self._num_dataloader_workers

            # 预提取 episode 视频路径映射（避免传递不可 pickle 的 HF datasets.Dataset）
            episode_video_map = {}
            for ep_idx in range(len(self.meta.episodes)):
                ep = self.meta.episodes[ep_idx]
                episode_video_map[ep_idx] = {}
                for vid_key in self.meta.video_keys:
                    episode_video_map[ep_idx][vid_key] = (
                        ep[f"videos/{vid_key}/chunk_index"],
                        ep[f"videos/{vid_key}/file_index"],
                    )

            camera_shapes = {}
            for k in self.meta.camera_keys:
                camera_shapes[k] = list(self.meta.info["features"][k]["shape"])

            config = DecodeProcessConfig(
                root=str(self.root),
                streaming_from_local=self.streaming_from_local,
                tolerance_s=self.tolerance_s,
                camera_keys=list(self.meta.camera_keys),
                image_transforms=self.image_transforms,
                delta_indices=self.delta_indices,
                video_path_template=self.meta.video_path,
                url_root=self.meta.url_root,
                episode_video_map=episode_video_map,
                camera_shapes=camera_shapes,
                decode_device=self.decode_device,
                decode_num_threads=num_threads,
                cache_size_per_thread=cache_size,
            )

            # 预检查 config 可 pickle 性
            try:
                pickle.dumps(config)
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                raise ValueError(
                    f"DecodeProcessConfig 包含不可 pickle 的属性，无法启动解码子进程: {e}. "
                    f"请确保 image_transforms 是可 pickle 的（如 torchvision.transforms.Compose 或 None），"
                    f"而非 lambda 或闭包。"
                ) from e

            self._decode_pipeline = DecodeProcessPipeline(config)

    def shutdown_decode_pipeline(self):
        """Clean up the async decode pipeline. Call at end of training."""
        if self._decode_pipeline is not None:
            self._decode_pipeline.shutdown()
            self._decode_pipeline = None

    def decode_iter(self, dataloader):
        """Async decode iterator: overlaps video decode with training.

        While the main process trains on batch N, the decode subprocess
        decodes batch N+1 using a persistent per-thread VideoDecoder cache.
        This hides decode latency behind training compute.

        Usage:
            for decoded_items in dataset.decode_iter(loader):
                batch = collate_decoded(decoded_items)
                loss = model(batch)  # decode process works on N+1 here

        Timeline:
            Main:    [fetch N] [consume N-1 + submit N] [train N-1] [consume N + submit N+1] ...
            Decode:            [decode N]                              [decode N+1]            ...
        """
        items_iter = iter(dataloader)

        # Create pipeline lazily
        self._ensure_decode_pipeline()

        # Prefetch: submit first batch for decoding
        try:
            first_items = next(items_iter)
        except StopIteration:
            return

        # If no video decode needed, pass through directly
        if not self.deferred_video_decode or not any(
            "_video_lookup" in item for item in first_items
        ):
            yield first_items
            yield from items_iter
            return

        self._decode_pipeline.submit(first_items)

        for items in items_iter:
            # Get decoded PREVIOUS batch (blocks until ready)
            decoded = self._decode_pipeline.consume()

            # Submit CURRENT items for decoding (non-blocking if queue has space)
            self._decode_pipeline.submit(items)

            yield decoded

        # Get the last decoded batch
        decoded = self._decode_pipeline.consume()
        yield decoded

    def _make_timestamps_from_indices(self, start_ts, indices=None):
        if indices is not None:
            return {
                key: (start_ts + torch.tensor(indices[key]) / self.fps).tolist()
                for key in self.delta_timestamps
            }
        else:
            return dict.fromkeys(self.meta.video_keys, [start_ts])

    def _make_padding_camera_frame(self, camera_key):
        return torch.zeros(self.meta.info["features"][camera_key]["shape"]).permute(-1, 0, 1)

    def _get_video_frame_padding_mask(self, video_frames, query_timestamps, original_timestamps):
        padding_mask = {}
        for video_key, timestamps in original_timestamps.items():
            if video_key not in video_frames:
                continue
            frames = []
            mask = []
            padding_frame = self._make_padding_camera_frame(video_key)
            for ts in timestamps:
                if is_float_in_list(ts, query_timestamps[video_key]):
                    idx = find_float_index(ts, query_timestamps[video_key])
                    frames.append(video_frames[video_key][idx, :])
                    mask.append(False)
                else:
                    frames.append(padding_frame)
                    mask.append(True)
            padding_mask[f"{video_key}_is_pad"] = torch.BoolTensor(mask)
        return padding_mask

    def _get_query_timestamps(self, current_ts, query_indices=None, episode_boundaries_ts=None):
        query_timestamps = {}
        keys_to_timestamps = self._make_timestamps_from_indices(current_ts, query_indices)
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = keys_to_timestamps[key]
                query_timestamps[key] = torch.clamp(
                    torch.tensor(timestamps), *episode_boundaries_ts[key]
                ).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_videos(self, query_timestamps, ep_idx):
        item = {}
        for video_key, query_ts in query_timestamps.items():
            root = self.meta.url_root if hasattr(self.meta, 'url_root') and self.streaming_from_local is False else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"

            if self.decode_device == "cuda":
                # CUDA 解码路径：使用 GPU 硬件加速 (NVDEC)
                frames = self._decode_video_cuda(video_path, query_ts)
            else:
                # CPU 解码路径：使用缓存的 decoder
                frames = decode_video_frames_torchcodec(
                    video_path, query_ts, self.tolerance_s, decoder_cache=self.video_decoder_cache
                )

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames
        return item

    def _decode_video_cuda(self, video_path, timestamps):
        """使用 CUDA (NVDEC) 解码视频帧。

        在主进程中使用 device="cuda" 的 VideoDecoder，
        利用 GPU 硬件加速视频解码，解码后转回 CPU tensor。
        """
        from torchcodec.decoders import VideoDecoder
        import fsspec

        # 使用 CUDA 解码器缓存
        if self._cuda_decoder_cache is None:
            self._cuda_decoder_cache = BoundedVideoDecoderCache(max_size=4)

        cache = self._cuda_decoder_cache
        video_path_str = str(video_path)

        with cache._lock:
            if video_path_str in cache._cache:
                decoder, file_handle, cached_res = cache._cache[video_path_str]
                meta = decoder.metadata
                current_res = (meta.height, meta.width)
                if current_res != cached_res:
                    try:
                        file_handle.close()
                    except Exception:
                        pass
                    del cache._cache[video_path_str]
                    cache._key_order.remove(video_path_str)

            if video_path_str not in cache._cache:
                file_handle = fsspec.open(video_path_str).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate", device="cuda")
                meta = decoder.metadata
                resolution = (meta.height, meta.width)
                cache._cache[video_path_str] = (decoder, file_handle, resolution)
                cache._key_order.append(video_path_str)
                # 淘汰超出容量的解码器
                while len(cache._cache) > cache._max_size:
                    oldest_key = cache._key_order.pop(0)
                    if oldest_key in cache._cache:
                        _, old_handle, _ = cache._cache.pop(oldest_key)
                        try:
                            old_handle.close()
                        except Exception:
                            pass
            else:
                # 更新访问顺序
                if video_path_str in cache._key_order:
                    cache._key_order.remove(video_path_str)
                cache._key_order.append(video_path_str)

        decoder = cache._cache[video_path_str][0]

        # 解码帧
        metadata = decoder.metadata
        average_fps = metadata.average_fps
        num_frames = metadata.num_frames
        frame_indices = [round(ts * average_fps) for ts in timestamps]
        clamped_mask = [idx >= num_frames or idx < 0 for idx in frame_indices]
        frame_indices = [max(0, min(idx, num_frames - 1)) for idx in frame_indices]
        frames_batch = decoder.get_frames_at(indices=frame_indices)

        # GPU 解码后转回 CPU
        loaded_frames = [frame.cpu() for frame in frames_batch.data]
        loaded_ts = [pts.item() for pts in frames_batch.pts_seconds]

        # Tolerance check (same as decode_video_frames_torchcodec)
        query_ts_tensor = torch.tensor(timestamps)
        loaded_ts_tensor = torch.tensor(loaded_ts)
        dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
        min_, argmin_ = dist.min(1)
        clamped_mask_tensor = torch.tensor(clamped_mask)
        is_within_tol = (min_ < self.tolerance_s) | clamped_mask_tensor
        assert is_within_tol.all(), (
            f"Timestamp tolerance violated: {min_[~is_within_tol]} > {self.tolerance_s=}. "
            f"video: {video_path}"
        )

        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        closest_frames = (closest_frames / 255.0).type(torch.float32)
        return closest_frames

    def _get_delta_frames(self, dataset_iterator, current_item):
        current_episode_idx = current_item["episode_index"]
        query_result = {}
        padding = {}

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue

            target_frames = []
            is_pad = []
            delta_results = {}

            negative_deltas = sorted([d for d in delta_indices if d < 0], reverse=True)
            positive_deltas = sorted([d for d in delta_indices if d > 0])
            zero_deltas = [d for d in delta_indices if d == 0]

            for delta in zero_deltas:
                delta_results[delta] = (current_item[key], False)

            lookback_failed = False
            last_successful_frame = current_item[key]
            for delta in negative_deltas:
                if lookback_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue
                try:
                    steps_back = abs(delta)
                    if dataset_iterator.can_peek_back(steps_back):
                        past_item = dataset_iterator.peek_back(steps_back)
                        past_item = item_to_torch(past_item)
                        if past_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (past_item[key], False)
                            last_successful_frame = past_item[key]
                        else:
                            raise LookBackError("Retrieved frame is from different episode!")
                    else:
                        raise LookBackError("Cannot go back further than the history buffer!")
                except LookBackError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookback_failed = True

            lookahead_failed = False
            last_successful_frame = current_item[key]
            for delta in positive_deltas:
                if lookahead_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue
                try:
                    if dataset_iterator.can_peek_ahead(delta):
                        future_item = dataset_iterator.peek_ahead(delta)
                        future_item = item_to_torch(future_item)
                        if future_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (future_item[key], False)
                            last_successful_frame = future_item[key]
                        else:
                            raise LookAheadError("Retrieved frame is from different episode!")
                    else:
                        raise LookAheadError("Cannot go ahead further than the lookahead buffer!")
                except LookAheadError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookahead_failed = True

            for delta in delta_indices:
                frame, is_padded = delta_results[delta]
                target_frames.append(frame)
                is_pad.append(is_padded)

            if target_frames:
                query_result[key] = torch.stack(target_frames)
                padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

        return query_result, padding

    def _add_history_actions(self, item, dataset_iterator):
        """为当前帧添加完整历史 action。

        优化：使用单次 history() 调用替代 1023 次 peek_back + item_to_torch，
        仅提取 episode_index 和 action 字段，在 episode 边界处提前终止。
        """
        current_episode_idx = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]

        current_action = item.get("action")
        if current_action is None:
            return item

        if current_action.dim() > 1:
            current_action = current_action[0] if current_action.shape[0] == 1 else current_action[-1]

        # 单次获取整个历史缓冲区，替代逐个 peek_back 调用
        hist = dataset_iterator.history()  # 按时间顺序 (oldest first)

        past_actions = []
        past_masks = []

        # 从最近的历史项向前遍历，遇到 episode 边界即终止
        # 因为 episode 是连续的，一旦找到不同 episode 的项，更早的项也属于不同 episode
        for i in range(len(hist) - 1, -1, -1):
            if len(past_actions) >= self.max_history_length - 1:
                break

            past_item = hist[i]

            # 仅提取 episode_index（最小化转换）
            ep_idx = past_item.get("episode_index")
            if isinstance(ep_idx, (np.ndarray, list)):
                ep_idx = torch.tensor(ep_idx)
                past_item["episode_index"] = ep_idx  # 缓存转换结果
            ep_val = ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx

            if ep_val != current_episode_idx:
                break  # Episode 边界，更早的项也属于不同 episode，提前终止

            # 仅提取 action 字段（最小化转换）
            past_action = past_item.get("action")
            if past_action is not None:
                if isinstance(past_action, (np.ndarray, list)):
                    past_action = torch.tensor(past_action)
                    past_item["action"] = past_action  # 缓存转换结果
                if past_action.dim() > 1:
                    past_action = past_action[0] if past_action.shape[0] == 1 else past_action[-1]
                past_actions.append(past_action)
                past_masks.append(True)
            else:
                past_actions.append(torch.zeros(self.action_dim, dtype=current_action.dtype))
                past_masks.append(False)

        # 反转为时间顺序 (oldest first)
        past_actions.reverse()
        past_masks.reverse()

        past_actions.append(current_action)
        past_masks.append(True)

        hist_actions = torch.stack(past_actions)
        hist_actions_mask = torch.BoolTensor(past_masks)

        actual_history_length = len(past_actions)

        padded_length = ((actual_history_length + self.action_chunk_size - 1) // self.action_chunk_size) * self.action_chunk_size

        if padded_length > self.max_history_length:
            padded_length = (self.max_history_length // self.action_chunk_size) * self.action_chunk_size
            if actual_history_length > padded_length:
                truncate_length = actual_history_length - padded_length
                hist_actions = hist_actions[truncate_length:]
                hist_actions_mask = hist_actions_mask[truncate_length:]
                actual_history_length = padded_length

        if actual_history_length < padded_length:
            pad_length = padded_length - actual_history_length
            padding_actions = torch.zeros(pad_length, self.action_dim, dtype=hist_actions.dtype)
            padding_mask = torch.zeros(pad_length, dtype=torch.bool)

            if self.history_padding_side == "left":
                hist_actions = torch.cat([padding_actions, hist_actions], dim=0)
                hist_actions_mask = torch.cat([padding_mask, hist_actions_mask], dim=0)
            else:
                hist_actions = torch.cat([hist_actions, padding_actions], dim=0)
                hist_actions_mask = torch.cat([hist_actions_mask, padding_mask], dim=0)

        item["hist_actions_full"] = hist_actions
        item["hist_actions_mask"] = hist_actions_mask
        item["hist_actions_length"] = torch.tensor(actual_history_length, dtype=torch.long)

        return item


class AsyncDecodeDataLoader:
    """Wraps a DataLoader with video decode + collate for PyTorch Lightning.

    Supports three decode modes based on dataset configuration:

    1. deferred_video_decode=True + async_decode=False (yield-time decode, best balance):
       Workers store lightweight frames in shuffle buffer, decode on yield.
       Items arrive at DataLoader already decoded. Throughput: ~9-10 batch/s.
       Memory: only DataLoader's prefetch queue holds decoded frames (~1.8GB).

    2. deferred_video_decode=True + async_decode=True (subprocess pipeline):
       Workers yield lightweight frames. A dedicated decode subprocess decodes
       them via decode_iter. Memory-safe with bounded queues.

    3. deferred_video_decode=False (worker decode, fastest):
       Workers decode video inside make_frame. Items are already decoded.
       Throughput: ~9-10 batch/s. But flush phase causes memory spike.

    Usage:
        raw_loader = DataLoader(dataset, ...)
        async_loader = AsyncDecodeDataLoader(raw_loader, dataset, collate_fn=collate_decoded)

        for batch in async_loader:
            # batch is already decoded and collated
            loss = model(batch)

    For PyTorch Lightning:
        trainer.fit(model, train_dataloaders=async_loader)
    """

    VARIABLE_LENGTH_KEYS = {"hist_actions_full", "hist_actions_mask"}

    def __init__(self, dataloader, dataset, collate_fn=None):
        self._loader = dataloader
        self._dataset = dataset
        self._collate_fn = collate_fn

    @staticmethod
    def make_collate_fn():
        """Create a collate function that handles variable-length tensors.

        This collate fn handles hist_actions_full and hist_actions_mask by
        left-padding shorter sequences to the max length in the batch,
        then stacking. All other tensors are stacked normally.

        Use as the DataLoader's collate_fn when items already contain
        decoded video (deferred_video_decode=True + async_decode=False,
        or deferred_video_decode=False).
        """
        variable_length_keys = AsyncDecodeDataLoader.VARIABLE_LENGTH_KEYS

        def collate_fn(batch):
            result = {}
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

        return collate_fn

    def __iter__(self):
        if self._dataset.async_decode and self._dataset.deferred_video_decode:
            # Mode 2: Subprocess decode pipeline (memory-safe).
            for decoded_items in self._dataset.decode_iter(self._loader):
                if self._collate_fn is not None:
                    yield self._collate_fn(decoded_items)
                else:
                    yield decoded_items
        else:
            # Mode 1 & 3: Items already decoded (yield-time or worker decode).
            for batch in self._loader:
                if self._collate_fn is not None:
                    batch = self._collate_fn(batch)
                yield batch

    def close(self):
        """Shut down DataLoader workers and decode pipeline.

        Must be called before discarding an AsyncDecodeDataLoader instance
        to ensure clean shutdown of persistent workers and background threads.
        """
        if hasattr(self._dataset, 'shutdown_decode_pipeline'):
            self._dataset.shutdown_decode_pipeline()
        if hasattr(self._loader, '_iterator') and self._loader._iterator is not None:
            self._loader._iterator._shutdown_workers()
            self._loader._iterator = None

    def __len__(self):
        return len(self._loader)

    @property
    def batch_size(self):
        return self._loader.batch_size

    @property
    def num_workers(self):
        return self._loader.num_workers

    @property
    def dataset(self):
        return self._loader.dataset
