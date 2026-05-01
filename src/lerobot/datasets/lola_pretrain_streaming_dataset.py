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
LoLA Pretrain Streaming Dataset -- episode-aligned chunk loading version.

Key design: load contiguous chunks of episodes from parquet, process with
vectorized numpy operations (no Backtrackable), and shuffle at chunk level
with strided worker distribution for global diversity + I/O efficiency.

Performance vs old streaming version:
- Memory: ~20-40MB/worker (1 episode + buffer) vs 15-30GB (full DataFrame)
- Row access: np_array[i] O(1) vs df.row(pos) O(n)
- History/delta: array slice/index O(1) vs peek_back/peek_ahead O(n)
- Shuffle: episode chunk shuffle + buffer (5000) vs buffer-only (1000)
"""

import bisect
import concurrent.futures
import json
import logging
import os
import pickle
import threading
from dataclasses import dataclass, field

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    EPISODES_DIR,
    check_version_compatibility,
    get_delta_indices,
    is_float_in_list,
    item_to_torch,
    load_info,
    load_stats,
    load_tasks,
)
from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec, scan_video_seek_modes
from lerobot.utils.constants import HF_LEROBOT_HOME


def _safe_stack_frames(frames: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    try:
        return torch.stack(frames)
    except RuntimeError:
        return frames


def _frames_get(frames: torch.Tensor | list[torch.Tensor], idx: int) -> torch.Tensor:
    if isinstance(frames, list):
        return frames[idx]
    return frames[idx]


def _frames_len(frames: torch.Tensor | list[torch.Tensor]) -> int:
    if isinstance(frames, list):
        return len(frames)
    return frames.shape[0]


def _maybe_squeeze(frames: torch.Tensor | list[torch.Tensor], n_ts: int):
    if n_ts == 1:
        if isinstance(frames, list):
            return frames[0]
        return frames.squeeze(0)
    return frames


logger = logging.getLogger(__name__)


class BoundedVideoDecoderCache(VideoDecoderCache):
    """VideoDecoderCache with capacity limit and seek_mode support.

    On cache hit, checks stored seek_mode and resolution. Evicts stale entries
    when they diverge from current requirements.
    """

    def __init__(self, max_size: int = 4):
        super().__init__()
        self._max_size = max_size
        self._key_order: list[str] = []

    def get_decoder(self, video_path: str, seek_mode: str = "approximate"):
        video_path = str(video_path)

        with self._lock:
            if video_path in self._cache:
                decoder, file_handle, cached_res, cached_seek = self._cache[video_path]
                # Evict if seek_mode changed
                if cached_seek != seek_mode:
                    try:
                        file_handle.close()
                    except Exception:
                        pass
                    del self._cache[video_path]
                    self._key_order.remove(video_path)
                else:
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
                while len(self._cache) >= self._max_size and self._key_order:
                    oldest_key = self._key_order.pop(0)
                    if oldest_key in self._cache:
                        _, old_handle, _, _ = self._cache.pop(oldest_key)
                        old_handle.close()

                decoder, file_handle, resolution, seek_mode = self._make_decoder(video_path, seek_mode)
                self._cache[video_path] = (decoder, file_handle, resolution, seek_mode)
                self._key_order.append(video_path)

            return self._cache[video_path][0]

    def evict_and_rebuild(self, video_path: str, seek_mode: str = "approximate"):
        video_path = str(video_path)
        with self._lock:
            if video_path in self._cache:
                _, file_handle, _, _ = self._cache[video_path]
                try:
                    file_handle.close()
                except Exception:
                    pass
                del self._cache[video_path]
                self._key_order.remove(video_path)
            while len(self._cache) >= self._max_size and self._key_order:
                oldest_key = self._key_order.pop(0)
                if oldest_key in self._cache:
                    _, old_handle, _, _ = self._cache.pop(oldest_key)
                    old_handle.close()
            decoder, file_handle, resolution, seek_mode = self._make_decoder(video_path, seek_mode)
            self._cache[video_path] = (decoder, file_handle, resolution, seek_mode)
            self._key_order.append(video_path)
            return decoder

    def clear(self):
        with self._lock:
            for _, file_handle, _, _ in self._cache.values():
                file_handle.close()
            self._cache.clear()
            self._key_order.clear()


class _DecodeError:
    def __init__(self, exc: Exception):
        self.exception = exc


@dataclass
class DecodeProcessConfig:
    """Picklable decode config for subprocess pipeline."""
    root: str
    streaming_from_local: bool
    tolerance_s: float
    tolerance_frames: int | None
    camera_keys: list
    delta_indices: object  # dict or None
    video_path_template: str
    url_root: str
    episode_video_map: dict
    camera_shapes: dict
    decode_device: str
    decode_num_threads: int
    cache_size_per_thread: int
    episode_is_valid_map: dict = field(default_factory=dict)
    video_seek_modes: dict = field(default_factory=dict)


def _resolve_video_path(config: DecodeProcessConfig, ep_idx: int, video_key: str) -> str:
    chunk_idx, file_idx = config.episode_video_map[ep_idx][video_key]
    fpath = config.video_path_template.format(
        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
    )
    root = config.url_root if not config.streaming_from_local else config.root
    return f"{root}/{fpath}"


def _make_padding_frame(camera_shapes: dict, camera_key: str) -> torch.Tensor:
    return torch.zeros(camera_shapes[camera_key]).permute(-1, 0, 1)


def _compute_padding_mask(config: DecodeProcessConfig, video_frames, query_timestamps, original_timestamps):
    padding_mask = {}
    for video_key, timestamps in original_timestamps.items():
        if video_key not in video_frames:
            continue
        mask = []
        for ts in timestamps:
            if is_float_in_list(ts, query_timestamps[video_key]):
                mask.append(False)
            else:
                mask.append(True)
        padding_mask[f"{video_key}_is_pad"] = torch.BoolTensor(mask)
    return padding_mask


def _decode_process_main(
    config: DecodeProcessConfig,
    light_queue,
    result_queue,
    shutdown_event,
):
    """Decode subprocess entry point."""
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=config.decode_num_threads,
        thread_name_prefix="DecodeWorker",
    )

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

    def query_videos(query_timestamps, ep_idx):
        item = {}
        for video_key, query_ts in query_timestamps.items():
            video_path = _resolve_video_path(config, ep_idx, video_key)

            # Look up seek_mode
            video_rel = video_path
            if "/videos/" in video_rel:
                video_rel = video_rel[video_rel.index("/videos/") + len("/videos/"):]
            seek_mode = config.video_seek_modes.get(video_rel, "approximate")

            if config.decode_device == "cuda":
                frames = _decode_video_cuda_in_process(
                    config, video_path, query_ts, get_thread_cuda_cache, seek_mode
                )
            else:
                frames = decode_video_frames_torchcodec(
                    video_path, query_ts, config.tolerance_s,
                    tolerance_frames=config.tolerance_frames,
                    decoder_cache=get_thread_cache(),
                    seek_mode=seek_mode,
                )

            item[video_key] = _maybe_squeeze(frames, len(query_ts))
        return item

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
        camera_valid_mask = video_lookup.get("camera_valid_mask", {})

        video_frames = query_videos(q_timestamps, ep_idx)

        item_copy.update(video_frames)

        is_valid_map = config.episode_is_valid_map.get(ep_idx, {})
        for cam_key in config.camera_keys:
            if not is_valid_map.get(cam_key, True):
                item_copy[cam_key] = _make_padding_frame(config.camera_shapes, cam_key)

        if config.delta_indices is not None:
            padding_mask = _compute_padding_mask(
                config, video_frames, q_timestamps, original_timestamps
            )
            item_copy.update(padding_mask)

        return item_copy

    while not shutdown_event.is_set():
        try:
            items = light_queue.get(block=True, timeout=0.5)
        except Exception:
            continue

        if items is None:
            break

        try:
            decoded = list(executor.map(decode_one, items))
            result_queue.put(decoded, block=True, timeout=5.0)
        except Exception as e:
            try:
                result_queue.put(_DecodeError(e), block=True, timeout=5.0)
            except Exception:
                pass

    executor.shutdown(wait=False)
    with caches_lock:
        for cache in all_caches:
            try:
                cache.clear()
            except Exception:
                pass


def _decode_video_cuda_in_process(config, video_path, timestamps, get_cuda_cache, seek_mode="approximate"):
    """CUDA video decode in subprocess."""
    from torchcodec.decoders import VideoDecoder

    cache = get_cuda_cache()
    video_path_str = str(video_path)

    decoder = cache.get_decoder(video_path_str, seek_mode)

    metadata = decoder.metadata
    average_fps = metadata.average_fps
    num_frames = metadata.num_frames

    effective_tol_s = config.tolerance_s
    if config.tolerance_frames is not None:
        effective_tol_s = (config.tolerance_frames + 0.5) / average_fps

    frame_indices = [round(ts * average_fps) for ts in timestamps]
    clamped_mask = [idx >= num_frames or idx < 0 for idx in frame_indices]
    frame_indices = [max(0, min(idx, num_frames - 1)) for idx in frame_indices]

    try:
        frames_batch = decoder.get_frames_at(indices=frame_indices)
    except RuntimeError as e:
        if "Expected pre-allocated tensor" in str(e):
            logging.warning(f"CUDA pre-allocated tensor mismatch for {video_path_str}. Evicting and rebuilding.")
            decoder = cache.evict_and_rebuild(video_path_str, seek_mode)
            frames_batch = decoder.get_frames_at(indices=frame_indices)
        else:
            raise

    loaded_frames = [frame.cpu() for frame in frames_batch.data]
    loaded_ts = [pts.item() for pts in frames_batch.pts_seconds]

    query_ts_tensor = torch.tensor(timestamps)
    loaded_ts_tensor = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
    min_, argmin_ = dist.min(1)
    clamped_mask_tensor = torch.tensor(clamped_mask)
    is_within_tol = (min_ < effective_tol_s) | clamped_mask_tensor
    assert is_within_tol.all(), (
        f"Timestamp tolerance violated: {min_[~is_within_tol]} > {effective_tol_s=}. "
        f"video: {video_path}"
    )

    closest_frames = _safe_stack_frames([loaded_frames[idx] for idx in argmin_])
    if isinstance(closest_frames, torch.Tensor):
        closest_frames = (closest_frames / 255.0).type(torch.float32)
    else:
        closest_frames = [(f / 255.0).type(torch.float32) for f in closest_frames]
    return closest_frames


class DecodeProcessPipeline:
    """Independent subprocess video decode pipeline with per-thread cache."""

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

        import atexit
        atexit.register(self.shutdown)

    def submit(self, items: list[dict]) -> None:
        self._light_queue.put(items, block=True)

    def consume(self) -> list[dict]:
        result = self._result_queue.get(block=True)
        if isinstance(result, _DecodeError):
            raise result.exception
        return result

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True

        self._shutdown_event.set()
        try:
            while not self._light_queue.empty():
                self._light_queue.get_nowait()
        except Exception:
            pass
        try:
            self._light_queue.put(None, block=False)
        except Exception:
            pass
        self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)


def _safe_concat_tables(tables: list[pa.Table]) -> pa.Table:
    """Safely concatenate pyarrow Tables with different columns/types.
    Handles missing columns (null-fill) and type mismatches (promotion).
    Replaces polars _safe_concat to avoid Rayon fork deadlock."""
    if len(tables) == 1:
        return tables[0]

    all_cols: list[str] = []
    seen: set[str] = set()
    for tbl in tables:
        for c in tbl.column_names:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)

    unified_types: dict[str, pa.DataType] = {}
    for tbl in tables:
        for c in tbl.column_names:
            col_type = tbl.schema.field(c).type
            if c not in unified_types:
                unified_types[c] = col_type
            else:
                existing = unified_types[c]
                if existing != col_type:
                    if pa.types.is_floating(existing) and pa.types.is_floating(col_type):
                        unified_types[c] = pa.float64()
                    elif pa.types.is_integer(existing) and pa.types.is_integer(col_type):
                        unified_types[c] = pa.int64()
                    elif pa.types.is_list(existing) and pa.types.is_list(col_type):
                        existing_val = existing.value_type
                        new_val = col_type.value_type
                        if existing_val != new_val:
                            if pa.types.is_floating(existing_val) and pa.types.is_floating(new_val):
                                unified_types[c] = pa.list_(pa.float64())
                            elif pa.types.is_integer(existing_val) and pa.types.is_integer(new_val):
                                unified_types[c] = pa.list_(pa.int64())

    aligned: list[pa.Table] = []
    for tbl in tables:
        existing_cols = set(tbl.column_names)

        cast_fields: dict[str, pa.DataType] = {}
        for c in tbl.column_names:
            current_type = tbl.schema.field(c).type
            target_type = unified_types[c]
            if current_type != target_type:
                cast_fields[c] = target_type

        if cast_fields:
            new_schema = pa.schema([
                pa.field(c, cast_fields.get(c, tbl.schema.field(c).type))
                for c in tbl.column_names
            ])
            tbl = tbl.cast(new_schema)

        null_cols = [c for c in all_cols if c not in existing_cols]
        if null_cols:
            for c in null_cols:
                null_arr = pa.nulls(tbl.num_rows, type=unified_types[c])
                tbl = tbl.append_column(c, null_arr)

        tbl = tbl.select(all_cols)
        aligned.append(tbl)

    return pa.concat_tables(aligned)


def _is_valid_parquet_file(path) -> bool:
    """Filter out hidden and temporary files (e.g. Azure .azDownload-* partial downloads)."""
    return not path.name.startswith(".")


def _discover_parquet_files(root: str) -> list[str]:
    from pathlib import Path
    data_dir = Path(root) / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(f for f in data_dir.glob("*/*.parquet") if _is_valid_parquet_file(f))
    return [str(f) for f in files]


def _load_episodes_polars(root) -> list[dict]:
    """Load episodes metadata using pyarrow to avoid initializing polars Rust
    Rayon thread pool in the parent process (which causes fork deadlock in
    DataLoader workers that later fork)."""
    from pathlib import Path
    import pandas as pd

    episodes_dir = Path(root) / EPISODES_DIR
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    parquet_files = sorted(f for f in episodes_dir.glob("*/*.parquet") if _is_valid_parquet_file(f))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {episodes_dir}")

    tables = []
    for path in parquet_files:
        table = pq.read_table(str(path))
        non_stats_cols = [c for c in table.column_names if not c.startswith("stats/")]
        table = table.select(non_stats_cols)
        if table.num_rows > 0:
            tables.append(table)

    if not tables:
        return []

    combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    combined = combined.sort_by("episode_index")

    df = combined.to_pandas()
    for col in df.columns:
        if col.endswith("/is_valid"):
            df[col] = df[col].fillna(0)
        elif df[col].dtype in ("int64", "float64", "int32", "float32"):
            df[col] = df[col].fillna(0)

    episodes = []
    for _, row in df.iterrows():
        ep_dict = {}
        for key, val in row.items():
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (np.ndarray,)):
                val = val.tolist()
            ep_dict[key] = val
        episodes.append(ep_dict)

    return episodes


class _EpisodeAccessor:
    """Dict-style episode access compatible with HF Dataset rows."""

    def __init__(self, episodes: list[dict]):
        self._episodes = episodes

    def __getitem__(self, idx: int) -> dict:
        return self._episodes[idx]

    def __len__(self) -> int:
        return len(self._episodes)


class EpisodeChunkReader:
    """Load contiguous ranges of episodes from parquet into dict-of-numpy-arrays.

    For each episode, returns {col_name: np.ndarray} where list columns
    (action, state) become 2D arrays [N, dim]. Uses pyarrow read_table + slice
    for efficient row-level access, reading only the parquet files containing
    the requested episodes. Avoids polars to prevent Rayon fork deadlock.
    """

    def __init__(self, parquet_files: list[str], file_cumsum: list[int],
                 episode_starts: np.ndarray, episode_ends: np.ndarray,
                 episode_file_ranges: list[tuple[int, int]]):
        self._parquet_files = parquet_files
        self._file_cumsum = file_cumsum
        self._episode_starts = episode_starts
        self._episode_ends = episode_ends
        self._episode_file_ranges = episode_file_ranges

    def load_episode(self, ep_idx: int) -> dict[str, np.ndarray]:
        """Load a single episode as dict of numpy arrays."""
        return self.load_episode_range(ep_idx, ep_idx + 1)[0]

    def load_episode_range(self, start_ep_idx: int, end_ep_idx: int) -> list[dict[str, np.ndarray]]:
        """Load episodes [start_ep_idx, end_ep_idx) as list of per-episode dicts.

        Reads only the parquet files containing the requested episode range,
        typically 1 file for small ranges. Uses pyarrow to avoid initializing
        polars Rayon thread pool (which causes fork deadlocks).
        """
        if start_ep_idx >= end_ep_idx:
            return []

        # Determine file range for the entire episode range
        first_file, _ = self._episode_file_ranges[start_ep_idx]
        _, last_file = self._episode_file_ranges[end_ep_idx - 1]

        # Global row range
        global_start = int(self._episode_starts[start_ep_idx])
        global_end = int(self._episode_ends[end_ep_idx - 1])

        # Read relevant parquet files with row slicing
        tables: list[pa.Table] = []
        for file_idx in range(first_file, last_file + 1):
            path = self._parquet_files[file_idx]
            file_global_start = self._file_cumsum[file_idx]
            file_global_end = self._file_cumsum[file_idx + 1]

            # Compute local slice within this file
            local_start = max(0, global_start - file_global_start)
            local_end = min(file_global_end - file_global_start, global_end - file_global_start)

            if local_start >= local_end:
                continue

            tbl = pq.read_table(path).slice(local_start, local_end - local_start)
            if tbl.num_rows > 0:
                tables.append(tbl)

        if not tables:
            return []

        combined = _safe_concat_tables(tables) if len(tables) > 1 else tables[0]

        # Convert to numpy arrays (columnar)
        col_arrays: dict[str, np.ndarray] = {}
        for col_name in combined.column_names:
            col = combined.column(col_name)
            col_type = col.type

            if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
                value_type = col_type.value_type
                if pa.types.is_floating(value_type) or pa.types.is_integer(value_type):
                    pylist = col.to_pylist()
                    inner_dim = 0
                    for v in pylist:
                        if v is not None:
                            inner_dim = len(v)
                            break
                    col_arrays[col_name] = np.array([
                        np.array(v, dtype=np.float32) if v is not None
                        else np.zeros(inner_dim, dtype=np.float32)
                        for v in pylist
                    ])
                else:
                    col_arrays[col_name] = np.array([
                        np.array(v) if v is not None else np.array([])
                        for v in col.to_pylist()
                    ])

            elif pa.types.is_floating(col_type):
                col_arrays[col_name] = col.to_numpy(zero_copy_only=False).astype(np.float32)

            elif pa.types.is_integer(col_type):
                filled = pa.compute.fill_null(col, 0)
                col_arrays[col_name] = filled.to_numpy().astype(np.int64)

            else:
                col_arrays[col_name] = col.to_numpy(zero_copy_only=False)

        # Split into per-episode dicts
        result: list[dict[str, np.ndarray]] = []
        for ep_idx in range(start_ep_idx, end_ep_idx):
            ep_start = int(self._episode_starts[ep_idx])
            ep_end = int(self._episode_ends[ep_idx])
            # Local offset within combined Table
            local_start = ep_start - global_start
            local_end = ep_end - global_start

            ep_dict: dict[str, np.ndarray] = {}
            for col, arr in col_arrays.items():
                ep_dict[col] = arr[local_start:local_end]
            result.append(ep_dict)

        return result


class LoLAPretrainStreamingDataset(torch.utils.data.IterableDataset):
    """Episode-aligned streaming dataset for LoLA pretraining.

    Loads contiguous chunks of episodes from parquet, processes with
    vectorized numpy operations (no Backtrackable), and shuffles at
    chunk level with strided worker distribution.
    """

    def __init__(
        self,
        repo_id: str,
        max_history_length: int = 100,
        action_chunk_size: int = 10,
        history_padding_side: str = "left",
        root: str | None = None,
        sub_root: str | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        tolerance_frames: int | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        buffer_size: int = 5000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng=None,
        shuffle: bool = True,
        deferred_video_decode: bool = True,
        decode_device: str = "cpu",
        decode_num_threads: int = 1,
        async_decode: bool = False,
        num_dataloader_workers: int = 0,
        dataset_to_episodes_path: str | None = None,
        temp_process: bool = False,
        episode_chunk_size: int = 8,
        start_index: int = 0,
        tier_config_path: str | None = None,
        yield_tier: int | None = None,
        pred_chunk_size: int = 50,
        max_image_pixels: int = 230400,
        min_image_pixels: int = 65536,
        vision_tower_multiplier: float = 4.0,
        action_token_weight: float = 1.0,
    ):
        super().__init__()

        self.repo_id = repo_id
        self.root = __import__("pathlib").Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.sub_root = sub_root
        self.streaming_from_local = root is not None

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.tolerance_frames = tolerance_frames
        self.revision = revision if revision else CODEBASE_VERSION
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.shuffle = shuffle

        self.buffer_size = buffer_size
        self.video_decoder_cache = None
        self.deferred_video_decode = deferred_video_decode
        self.decode_device = decode_device
        self.decode_num_threads = decode_num_threads

        self._cuda_decoder_cache = None

        self.async_decode = async_decode
        self._num_dataloader_workers = num_dataloader_workers
        self._decode_pipeline = None
        self.temp_process = temp_process
        self.episode_chunk_size = episode_chunk_size
        self.start_index = start_index
        self.pred_chunk_size = pred_chunk_size
        self._max_image_pixels = max_image_pixels
        self._min_image_pixels = min_image_pixels
        self._vision_tower_multiplier = vision_tower_multiplier
        self._action_token_weight = action_token_weight
        self.tier_config_path = tier_config_path
        self.yield_tier = yield_tier

        # Build metadata
        self.meta = self._build_metadata_polars(
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

        if "action" in self.meta.features:
            self.action_dim = self.meta.features["action"]["shape"][0]
        else:
            self.action_dim = 1

        self._parquet_files = _discover_parquet_files(str(self.root))

        # ── Episode boundary arrays ────────────────────────────────────
        num_episodes = len(self.meta.episodes)
        self._episode_starts = np.empty(num_episodes, dtype=np.int64)
        self._episode_ends = np.empty(num_episodes, dtype=np.int64)
        for ep_idx in range(num_episodes):
            ep = self.meta.episodes[ep_idx]
            self._episode_starts[ep_idx] = ep["dataset_from_index"]
            self._episode_ends[ep_idx] = ep["dataset_to_index"]

        # ── File cumulative-sum and per-episode file ranges ────────────
        self._file_cumsum = [0]
        for path in self._parquet_files:
            pf = pq.ParquetFile(path)
            self._file_cumsum.append(self._file_cumsum[-1] + pf.metadata.num_rows)

        self._total_rows = self._file_cumsum[-1]

        self._episode_file_ranges: list[tuple[int, int]] = []
        for ep_idx in range(num_episodes):
            first_file = bisect.bisect_right(self._file_cumsum, self._episode_starts[ep_idx]) - 1
            last_file = bisect.bisect_right(self._file_cumsum, self._episode_ends[ep_idx] - 1) - 1
            self._episode_file_ranges.append((first_file, last_file))

        # ── Chunk-to-file mapping (primary file only, computed early) ────
        # Full chunk cost + tier classification is deferred until after tier
        # config is loaded (see below).
        chunk_size = self.episode_chunk_size
        num_chunks = (num_episodes + chunk_size - 1) // chunk_size
        self._chunk_primary_file = []
        for c in range(num_chunks):
            ep_start = c * chunk_size
            self._chunk_primary_file.append(self._episode_file_ranges[ep_start][0])

        # ── EpisodeChunkReader ─────────────────────────────────────────
        self._chunk_reader = EpisodeChunkReader(
            self._parquet_files, self._file_cumsum,
            self._episode_starts, self._episode_ends,
            self._episode_file_ranges,
        )

        # ── Fast membership sets ───────────────────────────────────────
        self._video_keys_set = set(self.meta.video_keys)
        self._cached_video_keys_list = self.meta.video_keys
        self._cached_camera_keys_list = self.meta.camera_keys

        # ── Task name lookup ───────────────────────────────────────────
        self._task_names = [self.meta.tasks.iloc[i].name for i in range(len(self.meta.tasks))]

        # ── Seek-mode mapping + tier config loading ────────────────────
        self._video_seek_modes: dict[str, str] = {}
        self._video_resolution_map: dict[str, tuple[int, int]] = {}
        self._episode_visual_cost: list[float] | None = None
        self._episode_tier: list[int] | None = None
        self._tier_boundaries: tuple[float, ...] | None = None
        self._tier_stats: dict | None = None

        if self.tier_config_path is not None:
            # Load pre-computed tier config from Phase 1b JSON (no scanning needed)
            import json as _json
            with open(self.tier_config_path) as f:
                tier_config = _json.load(f)

            # Video resolution + seek mode from JSON
            for rel_path, vmeta in tier_config["video_resolution_map"].items():
                self._video_resolution_map[rel_path] = (vmeta["height"], vmeta["width"])
                self._video_seek_modes[rel_path] = vmeta.get("seek_mode", "approximate")

            # Episode costs and tiers from JSON
            self._episode_visual_cost = tier_config["episode_costs"]
            self._episode_tier = tier_config["episode_tiers"]
            self._tier_boundaries = tuple(tier_config["tier_boundaries"][:-1]) + (float("inf"),)
            self._tier_stats = tier_config["tier_stats"]

            # Use calibrated coefficients from JSON
            self._vision_tower_multiplier = tier_config["params"]["vision_tower_multiplier"]
            self._action_token_weight = tier_config["params"]["action_token_weight"]

            num_tiers = len(self._tier_stats)
            print(f"[LoLAPretrainStreamingDataset] Loaded tier config from {self.tier_config_path}: "
                  f"{len(self._video_resolution_map)} videos, "
                  f"{len(self._episode_visual_cost)} episode costs, "
                  f"{num_tiers} tiers, boundaries={self._tier_boundaries}")
        elif self.streaming_from_local and os.path.isdir(os.path.join(str(self.root), "videos")):
            # Original behavior: scan videos at init (for non-tier-aware mode)
            self._video_seek_modes = scan_video_seek_modes(str(self.root), num_workers=8)
            exact_count = sum(1 for v in self._video_seek_modes.values() if v == "exact")
            print(f"[LoLAPretrainStreamingDataset] seek-mode scan: {len(self._video_seek_modes)} videos, "
                  f"{exact_count} require exact mode")

        # ── Per-sub-dataset normalization setup ──────────────────────────
        self._episode_to_ds_idx = np.full(num_episodes, -1, dtype=np.int16)
        self._sub_dataset_names: list[str] = []
        self._sub_dataset_paths: list[str] = []
        self._sub_dataset_norm_params: list[dict | None] = []
        self._sub_dataset_dims: list[tuple[int, int]] = []

        if dataset_to_episodes_path is not None:
            self._load_dataset_to_episodes(dataset_to_episodes_path)

        print(f"[LoLAPretrainStreamingDataset] max_history_length: {max_history_length}")
        print(f"[LoLAPretrainStreamingDataset] action_chunk_size: {action_chunk_size}")
        print(f"[LoLAPretrainStreamingDataset] history_padding_side: {history_padding_side}")
        print(f"[LoLAPretrainStreamingDataset] action_dim: {self.action_dim}")
        print(f"[LoLAPretrainStreamingDataset] parquet_files: {len(self._parquet_files)}")
        print(f"[LoLAPretrainStreamingDataset] total_rows: {self._total_rows}")
        print(f"[LoLAPretrainStreamingDataset] total_episodes: {num_episodes}")
        print(f"[LoLAPretrainStreamingDataset] episode_chunk_size: {episode_chunk_size}")
        print(f"[LoLAPretrainStreamingDataset] sub_datasets: {len(self._sub_dataset_names)}")

        # ── Chunk memory cost + tier classification (deferred after tier config) ──
        # This must run after _episode_visual_cost and _episode_tier are populated
        # from tier_config_path (if provided), otherwise the fallback is used.
        self._chunk_memory_cost: list[float] = []
        self._chunk_tier: list[int] = []
        # Per-tier frame count per chunk (for yield_tier worker balancing)
        self._chunk_tier_frame_counts: dict[int, list[int]] = {}

        num_tiers = len(self._tier_boundaries) - 1 if self._tier_boundaries else 0
        chunk_size = self.episode_chunk_size
        num_chunks = (num_episodes + chunk_size - 1) // chunk_size

        # Initialize per-tier frame count lists
        for t in range(num_tiers):
            self._chunk_tier_frame_counts[t] = [0] * num_chunks

        for c in range(num_chunks):
            ep_start = c * chunk_size
            ep_end = min(ep_start + chunk_size, num_episodes)

            cost = 0.0
            tier_counts = [0] * max(num_tiers, 1)
            tier_frame_counts = [0] * max(num_tiers, 1)

            for ep_i in range(ep_start, ep_end):
                # Use pre-computed episode cost if available (from tier config JSON)
                if self._episode_visual_cost is not None:
                    ep_cost = self._episode_visual_cost[ep_i]
                else:
                    # Fallback: count valid cameras (original behavior)
                    ep_meta = self.meta.episodes[ep_i]
                    valid_cameras = 0
                    for cam_key in self.meta.camera_keys:
                        is_valid = ep_meta.get(f"videos/{cam_key}/is_valid", 1)
                        if is_valid == 1:
                            valid_cameras += 1
                    ep_cost = valid_cameras * self._vision_tower_multiplier

                    # Add action cost
                    ds_idx = int(self._episode_to_ds_idx[ep_i])
                    action_dim, _ = self._sub_dataset_dims[ds_idx] if ds_idx >= 0 and ds_idx < len(self._sub_dataset_dims) else (self.action_dim, 0)
                    pred_tokens = self.pred_chunk_size // self.action_chunk_size
                    hist_tokens = self.max_history_length // self.action_chunk_size
                    action_cost = (pred_tokens + hist_tokens) * self._action_token_weight * action_dim
                    ep_cost += action_cost

                cost += ep_cost

                # Track tier counts and frame counts for this chunk
                if self._episode_tier is not None and num_tiers > 0:
                    ep_tier = self._episode_tier[ep_i]
                    tier_counts[ep_tier] += 1
                    ep_frame_count = self._episode_ends[ep_i] - self._episode_starts[ep_i]
                    tier_frame_counts[ep_tier] += ep_frame_count

            self._chunk_memory_cost.append(cost)
            # Dominant tier = most frequent tier in chunk
            if num_tiers > 0 and sum(tier_counts) > 0:
                self._chunk_tier.append(tier_counts.index(max(tier_counts)))
            else:
                self._chunk_tier.append(0)

            # Store per-tier frame counts for this chunk
            for t in range(num_tiers):
                self._chunk_tier_frame_counts[t][c] = tier_frame_counts[t]

    @staticmethod
    def _build_metadata_polars(repo_id, root, revision, force_cache_sync=False):
        from pathlib import Path

        meta_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        _revision = revision if revision else CODEBASE_VERSION

        if force_cache_sync:
            from lerobot.datasets.lerobot_dataset import is_valid_version, get_safe_version
            if is_valid_version(_revision):
                _revision = get_safe_version(repo_id, _revision)
            (meta_root / "meta").mkdir(exist_ok=True, parents=True)
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id, repo_type="dataset", revision=_revision,
                local_dir=meta_root, allow_patterns="meta/",
            )

        meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
        meta.repo_id = repo_id
        meta.revision = _revision
        meta.root = meta_root
        meta.writer = None
        meta.latest_episode = None
        meta.metadata_buffer = []
        meta.metadata_buffer_size = 10

        meta.info = load_info(meta_root)
        meta.tasks = load_tasks(meta_root)
        meta.stats = load_stats(meta_root)

        episodes_list = _load_episodes_polars(meta_root)
        meta.episodes = _EpisodeAccessor(episodes_list)

        print(f"[LoLAPretrainStreamingDataset] Loaded {len(episodes_list)} episodes via polars")
        return meta

    def _load_dataset_to_episodes(self, dataset_to_episodes_path: str):
        with open(dataset_to_episodes_path, "r") as f:
            dataset_map = json.load(f)

        ds_idx = 0
        for ds_name, ds_info in dataset_map.items():
            ds_path = ds_info["path"]
            start_ep = ds_info["start_episode_index"]
            end_ep = ds_info["end_episode_index"]

            for ep_idx in range(start_ep, end_ep + 1):
                if ep_idx < len(self._episode_to_ds_idx):
                    self._episode_to_ds_idx[ep_idx] = ds_idx

            self._sub_dataset_names.append(ds_name)
            self._sub_dataset_paths.append(ds_path)

            if self.sub_root is not None:
                stats_path = os.path.join(str(self.sub_root), ds_path, "meta", "stats.json")
            else:
                stats_path = None
            norm_params = None
            action_dim = self.action_dim
            state_dim = 0

            try:
                with open(stats_path, "r") as sf:
                    raw_stats = json.load(sf)

                norm_params = {}
                for key in ("observation.state", "action"):
                    if key in raw_stats:
                        mean = torch.tensor(raw_stats[key]["mean"], dtype=torch.float32)
                        std = torch.tensor(raw_stats[key]["std"], dtype=torch.float32)
                        norm_params[key] = {"mean": mean, "std": std}

                if "action" in norm_params:
                    action_dim = len(norm_params["action"]["mean"])
                if "observation.state" in norm_params:
                    state_dim = len(norm_params["observation.state"]["mean"])

            except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
                logger.warning(
                    f"[LoLAPretrainStreamingDataset] Could not load stats for "
                    f"sub-dataset '{ds_name}' from {stats_path}: {e}. "
                    f"Skipping per-dataset normalization for this sub-dataset."
                )
                norm_params = None
                action_dim = self.action_dim
                state_dim = 0

            self._sub_dataset_norm_params.append(norm_params)
            self._sub_dataset_dims.append((action_dim, state_dim))
            ds_idx += 1

    @staticmethod
    def _make_translation_norm_mask(action_dim: int) -> torch.Tensor:
        mask = torch.zeros(action_dim, dtype=torch.bool)
        arm_dim = 10
        num_arms = action_dim // arm_dim
        for arm in range(num_arms):
            offset = arm * arm_dim
            mask[offset:offset + 3] = True
        return mask

    def _normalize_per_subdataset(self, item, temp_process=False):
        ep_idx = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]
        if ep_idx >= len(self._episode_to_ds_idx) or self._episode_to_ds_idx[ep_idx] < 0:
            return item

        ds_idx = self._episode_to_ds_idx[ep_idx]
        stats = self._sub_dataset_norm_params[ds_idx]

        if stats is None:
            return item

        padded_stats = {}
        for key in ("observation.state", "action"):
            if key not in stats:
                continue
            mean, std = stats[key]["mean"], stats[key]["std"]

            if key == "action" and mean.shape[0] != self.action_dim:
                if not temp_process:
                    raise ValueError(
                        f"Sub-dataset {self._sub_dataset_names[ds_idx]} has action dim "
                        f"{mean.shape[0]} but global action_dim is {self.action_dim}. "
                        f"Set temp_process=True to pad stats, or update the sub-dataset's stats.json."
                    )
                pad_len = self.action_dim - mean.shape[0]
                mean = torch.cat([mean, torch.zeros(pad_len)])
                std = torch.cat([std, torch.ones(pad_len)])
                logger.warning(
                    f"[LoLAPretrainStreamingDataset] Padded stats for '{key}' in "
                    f"sub-dataset '{self._sub_dataset_names[ds_idx]}' from "
                    f"{stats[key]['mean'].shape[0]} to {self.action_dim} dims (temp_process mode)"
                )

            padded_stats[key] = {"mean": mean, "std": std}

        if "observation.state" in item and "observation.state" in padded_stats:
            mean, std = padded_stats["observation.state"]["mean"], padded_stats["observation.state"]["std"]
            item["observation.state"] = (item["observation.state"] - mean) / (std + 1e-8)

        if "action" in item and "action" in padded_stats:
            mean, std = padded_stats["action"]["mean"], padded_stats["action"]["std"]
            norm_mask = self._make_translation_norm_mask(mean.shape[0])
            action = item["action"]
            normalized = (action - mean) / (std + 1e-8)
            item["action"] = torch.where(norm_mask, normalized, action)

        if "hist_actions_full" in item and "action" in padded_stats:
            mean, std = padded_stats["action"]["mean"], padded_stats["action"]["std"]
            norm_mask = self._make_translation_norm_mask(mean.shape[0])
            mask = item["hist_actions_mask"]
            normalized = (item["hist_actions_full"] - mean) / (std + 1e-8)
            mask_expanded = mask.unsqueeze(-1).expand_as(normalized)
            norm_mask_expanded = norm_mask.unsqueeze(0).expand_as(normalized)
            should_normalize = mask_expanded & norm_mask_expanded
            item["hist_actions_full"] = torch.where(should_normalize, normalized, item["hist_actions_full"])

        return item

    def _tensor_to_pil(self, tensor):
        from PIL import Image
        if isinstance(tensor, list):
            return [self._tensor_to_pil(t) for t in tensor]
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.permute(1, 2, 0)
        if img.dtype in [torch.float32, torch.float64]:
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(img.cpu().numpy())

    def _apply_camera_valid_mask(self, item, ep_idx):
        from PIL import Image

        camera_valid_mask = {}
        ep_meta = self.meta.episodes[ep_idx]

        for cam_key in self.meta.camera_keys:
            is_valid_key = f"videos/{cam_key}/is_valid"
            is_valid = ep_meta.get(is_valid_key, 1)
            camera_valid_mask[cam_key] = (is_valid == 1)

            if cam_key in item:
                if is_valid == 0:
                    item[cam_key] = None
                elif isinstance(item[cam_key], (torch.Tensor, list)):
                    item[cam_key] = self._tensor_to_pil(item[cam_key])

        item["camera_valid_mask"] = camera_valid_mask
        return item

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def num_frames(self):
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def fps(self):
        return self.meta.fps

    # ── Core: process one episode ───────────────────────────────────────

    def _process_episode_frames(self, ep_data: dict[str, np.ndarray], ep_idx: int) -> list[dict]:
        """Process all frames of an episode into list of item dicts.

        Uses vectorized numpy operations for history actions and delta frames
        instead of Backtrackable peek_back/peek_ahead.
        """
        ep_length = len(ep_data["action"])
        ep_start = int(self._episode_starts[ep_idx])

        # Pre-extract action array as numpy for vectorized history
        action_array = ep_data["action"]  # [N, action_dim] or [N]
        if action_array.ndim == 1:
            action_array = action_array.reshape(-1, 1)

        # Pre-compute delta frame arrays (non-video keys only)
        delta_arrays = {}
        delta_padding = {}
        if self.delta_indices is not None:
            for key, delta_indices in self.delta_indices.items():
                if key in self._video_keys_set:
                    continue
                if key not in ep_data:
                    continue
                arr = ep_data[key]
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                delta_arrays[key] = (arr, delta_indices)

        items = []
        for frame_in_ep in range(ep_length):
            global_idx = ep_start + frame_in_ep

            # Build base item from row data
            item = {}
            for key, arr in ep_data.items():
                val = arr[frame_in_ep]
                if isinstance(val, np.ndarray):
                    item[key] = torch.tensor(val, dtype=torch.float32)
                elif isinstance(val, (np.integer,)):
                    item[key] = int(val)
                elif isinstance(val, (np.floating,)):
                    item[key] = float(val)
                else:
                    item[key] = val

            # ── Delta frames (vectorized) ──────────────────────────
            if self.delta_indices is not None:
                query_result = {}
                padding = {}
                for key, (arr, delta_indices) in delta_arrays.items():
                    target_frames = []
                    is_pad = []
                    for delta in delta_indices:
                        target_pos = frame_in_ep + delta
                        if 0 <= target_pos < ep_length:
                            val = arr[target_pos]
                            target_frames.append(torch.tensor(val, dtype=torch.float32))
                            is_pad.append(False)
                        else:
                            # Zero-fill for out-of-bounds frames (semantic: "stop action")
                            # rather than repeating the last frame which is semantically wrong.
                            target_frames.append(torch.zeros(arr.shape[1], dtype=torch.float32))
                            is_pad.append(True)
                    if target_frames:
                        query_result[key] = torch.stack(target_frames)
                        padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)
                item.update(query_result)
                item.update(padding)

            # ── History actions (vectorized) ───────────────────────
            current_action = item.get("action")
            if current_action is not None:
                if current_action.dim() > 1:
                    current_action = current_action[0] if current_action.shape[0] == 1 else current_action[-1]

                max_lookback = min(self.max_history_length - 1, frame_in_ep)

                past_actions = []
                past_masks = []

                if max_lookback > 0:
                    past_actions_np = action_array[frame_in_ep - max_lookback : frame_in_ep]
                    for j in range(past_actions_np.shape[0]):
                        pa = torch.tensor(past_actions_np[j], dtype=torch.float32)
                        if pa.dim() > 1:
                            pa = pa[0] if pa.shape[0] == 1 else pa[-1]
                        past_actions.append(pa)
                        past_masks.append(True)

                past_actions.append(current_action)
                past_masks.append(True)

                hist_actions = torch.stack(past_actions)
                hist_actions_mask = torch.BoolTensor(past_masks)

                actual_history_length = len(past_actions)
                padded_length = (
                    ((actual_history_length + self.action_chunk_size - 1) // self.action_chunk_size)
                    * self.action_chunk_size
                )

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
                    padding_mask_val = torch.zeros(pad_length, dtype=torch.bool)
                    if self.history_padding_side == "left":
                        hist_actions = torch.cat([padding_actions, hist_actions], dim=0)
                        hist_actions_mask = torch.cat([padding_mask_val, hist_actions_mask], dim=0)
                    else:
                        hist_actions = torch.cat([hist_actions, padding_actions], dim=0)
                        hist_actions_mask = torch.cat([hist_actions_mask, padding_mask_val], dim=0)

                item["hist_actions_full"] = hist_actions
                item["hist_actions_mask"] = hist_actions_mask
                item["hist_actions_length"] = torch.tensor(actual_history_length, dtype=torch.long)

            # ── Task name ──────────────────────────────────────────
            task_index = item.get("task_index", 0)
            if isinstance(task_index, torch.Tensor):
                task_index = task_index.item()
            item["task"] = self._task_names[task_index] if task_index < len(self._task_names) else ""

            # ── Video lookup (deferred) or immediate decode ────────
            ep_meta = self.meta.episodes[ep_idx]
            current_ts = global_idx / self.fps

            if len(self.meta.video_keys) > 0:
                episode_boundaries_ts = {
                    key: (
                        ep_meta[f"videos/{key}/from_timestamp"],
                        ep_meta[f"videos/{key}/to_timestamp"],
                    )
                    for key in self.meta.video_keys
                }

                original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)
                query_timestamps = self._get_query_timestamps(
                    current_ts, self.delta_indices, episode_boundaries_ts
                )

                if self.deferred_video_decode:
                    item["_video_lookup"] = {
                        "ep_idx": ep_idx,
                        "query_timestamps": query_timestamps,
                        "original_timestamps": original_timestamps,
                        "camera_valid_mask": {
                            cam_key: ep_meta.get(f"videos/{cam_key}/is_valid", 1) == 1
                            for cam_key in self.meta.video_keys
                        },
                    }
                else:
                    video_frames = self._query_videos(query_timestamps, ep_idx)
                    item.update(video_frames)
                    if self.delta_indices is not None:
                        padding_mask = self._get_video_frame_padding_mask(
                            video_frames, query_timestamps, original_timestamps
                        )
                        item.update(padding_mask)

            # ── Per-sub-dataset normalization ──────────────────────
            item = self._normalize_per_subdataset(item, temp_process=self.temp_process)

            # ── Camera valid mask + PIL conversion ─────────────────
            if not self.deferred_video_decode:
                item = self._apply_camera_valid_mask(item, ep_idx)
            else:
                # For deferred mode, add camera_valid_mask but don't decode
                camera_valid_mask = {}
                for cam_key in self.meta.camera_keys:
                    is_valid_key = f"videos/{cam_key}/is_valid"
                    is_valid = ep_meta.get(is_valid_key, 1)
                    camera_valid_mask[cam_key] = (is_valid == 1)
                item["camera_valid_mask"] = camera_valid_mask

            # ── Dimension info ─────────────────────────────────────
            ds_idx_val = self._episode_to_ds_idx[ep_idx] if ep_idx < len(self._episode_to_ds_idx) and self._episode_to_ds_idx[ep_idx] >= 0 else 0
            action_dim, state_dim = self._sub_dataset_dims[ds_idx_val] if ds_idx_val < len(self._sub_dataset_dims) else (self.action_dim, 0)
            item["action_dim"] = action_dim
            item["state_dim"] = state_dim

            # ── Tier metadata (for per-tier DataLoader filtering) ────
            if self._episode_visual_cost is not None:
                item["_memory_cost"] = float(self._episode_visual_cost[ep_idx])
            else:
                item["_memory_cost"] = 0.0
            if self._episode_tier is not None:
                item["_tier"] = int(self._episode_tier[ep_idx])
            else:
                item["_tier"] = 0

            items.append(item)

        return items

    # ── __iter__: episode-aware shuffle ─────────────────────────────────

    def __iter__(self):
        """Episode-aware streaming with chunk-level strided sharding."""
        # Initialize video decoder cache
        if self.video_decoder_cache is None:
            n_cameras = max(1, len(self.meta.video_keys))
            cache_size = max(4, n_cameras * self.episode_chunk_size)
            self.video_decoder_cache = BoundedVideoDecoderCache(max_size=cache_size)

        # Distributed info
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        total_parallel = world_size * num_workers
        parallel_id = rank * num_workers + worker_id

        # ── File-aligned + visual-cost-balanced chunk assignment ─────
        num_episodes = len(self.meta.episodes)
        chunk_size = self.episode_chunk_size
        num_chunks = (num_episodes + chunk_size - 1) // chunk_size

        # Group chunks by primary parquet file
        file_to_chunks: dict[int, list[int]] = {}
        for c in range(num_chunks):
            pf = self._chunk_primary_file[c]
            file_to_chunks.setdefault(pf, []).append(c)

        # Sort file groups by their maximum chunk memory cost (descending)
        file_groups = sorted(
            file_to_chunks.values(),
            key=lambda group: max(self._chunk_memory_cost[c] for c in group),
            reverse=True,
        )

        # Greedy round-robin: assign each file group to the worker with lowest current cost
        # When yield_tier is set, balance by target-tier frame count instead of memory cost
        worker_indices: list[list[int]] = [[] for _ in range(total_parallel)]
        worker_costs: list[float] = [0.0] * total_parallel
        use_tier_frame_balance = (
            self.yield_tier is not None
            and self.yield_tier in self._chunk_tier_frame_counts
        )
        for group in file_groups:
            min_worker = worker_costs.index(min(worker_costs))
            worker_indices[min_worker].extend(group)
            for c in group:
                if use_tier_frame_balance:
                    worker_costs[min_worker] += self._chunk_tier_frame_counts[self.yield_tier][c]
                else:
                    worker_costs[min_worker] += self._chunk_memory_cost[c]

        worker_chunk_indices = worker_indices[parallel_id]

        # Note: tier filtering is done at per-episode level inside the episode loop
        # (see below), not at chunk level. This is more precise than chunk-level
        # filtering because episodes in a mixed-tier chunk can be correctly routed.

        if not worker_chunk_indices:
            print(f"[LoLAPretrainStreamingDataset] Worker {parallel_id} has no chunks assigned "
                  f"(num_chunks={num_chunks}, total_parallel={total_parallel})", flush=True)
            return

        # Shuffle chunk order
        rng = np.random.default_rng(
            self.seed + rank if not self.shuffle else self.rng.integers(0, 2**31) + rank
        )
        chunk_indices_shuffled = np.array(worker_chunk_indices)
        if self.shuffle:
            rng.shuffle(chunk_indices_shuffled)

        print(f"[LoLAPretrainStreamingDataset] Worker {parallel_id} processing "
              f"{len(worker_chunk_indices)} chunks (chunk_size={chunk_size}, "
              f"total_episodes={num_episodes}, visual_cost={worker_costs[parallel_id]:.0f})", flush=True)

        # ── Decode mode ─────────────────────────────────────────────
        # Decode video inside DataLoader workers using BoundedVideoDecoderCache.
        # Safe because this file no longer uses polars at all (replaced with
        # pyarrow), so Rayon is never initialized in the parent process.
        decode_on_yield = self.deferred_video_decode and not self.async_decode

        # ── Shuffle buffer ──────────────────────────────────────────
        buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)
        frames_buffer = []
        yield_count = 0
        buffer_full = False

        # Resume: fast-forward through start_index items per worker
        skip_remaining = self.start_index // total_parallel
        if skip_remaining > 0:
            print(f"[LoLAPretrainStreamingDataset] Worker {parallel_id} skipping {skip_remaining} items for resume...", flush=True)

        print(f"[LoLAPretrainStreamingDataset] Worker {parallel_id} filling buffer "
              f"(target={self.buffer_size}, decode_on_yield={decode_on_yield})", flush=True)

        for chunk_idx in chunk_indices_shuffled:
            ep_start_idx = int(chunk_idx) * chunk_size
            ep_end_idx = min(ep_start_idx + chunk_size, num_episodes)

            # Load contiguous episode range
            try:
                ep_data_list = self._chunk_reader.load_episode_range(ep_start_idx, ep_end_idx)
            except Exception as e:
                logger.warning(f"Failed to load episodes [{ep_start_idx}, {ep_end_idx}): {e}")
                continue

            # Optional: shuffle episodes within chunk
            ep_order = list(range(len(ep_data_list)))
            if self.shuffle:
                rng.shuffle(ep_order)

            for local_ep_idx in ep_order:
                ep_idx = ep_start_idx + local_ep_idx
                ep_data = ep_data_list[local_ep_idx]

                # Per-episode tier filter (more precise than chunk-level filtering)
                if self.yield_tier is not None and self._episode_tier is not None:
                    if self._episode_tier[ep_idx] != self.yield_tier:
                        continue

                # Process all frames in this episode
                try:
                    frame_items = self._process_episode_frames(ep_data, ep_idx)
                except Exception as e:
                    logger.warning(f"Failed to process episode {ep_idx}: {e}")
                    continue

                # Optional: shuffle frames within episode
                if self.shuffle and len(frame_items) > 1:
                    rng.shuffle(frame_items)

                # Feed into shuffle buffer
                for frame in frame_items:
                    if len(frames_buffer) == self.buffer_size:
                        if not buffer_full:
                            buffer_full = True
                            print(f"[LoLAPretrainStreamingDataset] Worker {parallel_id} buffer full, starting yield", flush=True)
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

        # Flush remaining buffer
        rng.shuffle(frames_buffer)
        yield_count += len(frames_buffer)
        print(f"[LoLAPretrainStreamingDataset] Worker {parallel_id} finished, "
              f"total yielded: {yield_count}, buffer: {len(frames_buffer)}", flush=True)
        if decode_on_yield:
            for frame in frames_buffer:
                if "_video_lookup" in frame:
                    frame = self._decode_videos(frame)
                yield frame
        else:
            yield from frames_buffer

    # ── Video decode ────────────────────────────────────────────────────

    def _decode_videos(self, lightweight_frame):
        video_lookup = lightweight_frame.pop("_video_lookup", None)

        if video_lookup is None:
            for cam_key in self.meta.camera_keys:
                if cam_key not in lightweight_frame:
                    lightweight_frame[cam_key] = self._make_padding_camera_frame(cam_key)
                    lightweight_frame[f"{cam_key}_is_pad"] = torch.BoolTensor([True])
            return lightweight_frame

        ep_idx = video_lookup["ep_idx"]
        query_timestamps = video_lookup["query_timestamps"]
        original_timestamps = video_lookup["original_timestamps"]
        camera_valid_mask = video_lookup.get("camera_valid_mask", {})

        video_frames = self._query_videos(query_timestamps, ep_idx)

        lightweight_frame.update(video_frames)

        for cam_key in self.meta.camera_keys:
            if not camera_valid_mask.get(cam_key, True):
                lightweight_frame[cam_key] = None

        if self.delta_indices is not None:
            padding_mask = self._get_video_frame_padding_mask(
                video_frames, query_timestamps, original_timestamps
            )
            lightweight_frame.update(padding_mask)

        lightweight_frame = self._apply_camera_valid_mask(lightweight_frame, ep_idx)

        return lightweight_frame

    def decode_item(self, item):
        if "_video_lookup" not in item:
            return item
        return self._decode_videos(item)

    def decode_items_batch(self, items):
        if not self.deferred_video_decode:
            return items

        need_decode = ["_video_lookup" in item for item in items]
        if not any(need_decode):
            return items

        if self.decode_device == "cuda":
            return [self.decode_item(item) for item in items]

        if self.decode_num_threads <= 1:
            return [self.decode_item(item) for item in items]

        return self._decode_items_parallel(items)

    def _decode_items_parallel(self, items):
        from concurrent.futures import ThreadPoolExecutor

        num_ffmpeg_threads = 2

        def _decode_one_item(item):
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
            camera_valid_mask = video_lookup.get("camera_valid_mask", {})

            video_frames = self._query_videos_independent(
                query_timestamps, ep_idx, num_ffmpeg_threads
            )

            result = item.copy()
            result.update(video_frames)
            del result["_video_lookup"]

            for cam_key in self.meta.camera_keys:
                if not camera_valid_mask.get(cam_key, True):
                    result[cam_key] = None

            if self.delta_indices is not None:
                padding_mask = self._get_video_frame_padding_mask(
                    video_frames, query_timestamps, original_timestamps
                )
                result.update(padding_mask)

            result = self._apply_camera_valid_mask(result, ep_idx)

            return result

        with ThreadPoolExecutor(max_workers=self.decode_num_threads) as executor:
            decoded = list(executor.map(_decode_one_item, items))

        return decoded

    def _query_videos_independent(self, query_timestamps, ep_idx, num_ffmpeg_threads=2):
        from torchcodec.decoders import VideoDecoder as _VideoDecoder
        import fsspec as _fsspec

        item = {}
        for video_key, query_ts in query_timestamps.items():
            root = self.meta.url_root if hasattr(self.meta, 'url_root') and self.streaming_from_local is False else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"

            # Look up seek_mode
            video_rel = str(self.meta.get_video_file_path(ep_idx, video_key))
            if video_rel.startswith("videos/"):
                video_rel = video_rel[len("videos/"):]
            seek_mode = self._video_seek_modes.get(video_rel, "approximate")

            fh = _fsspec.open(video_path).__enter__()
            try:
                decoder = _VideoDecoder(fh, seek_mode=seek_mode, num_ffmpeg_threads=num_ffmpeg_threads)
                metadata = decoder.metadata
                average_fps = metadata.average_fps
                num_frames = metadata.num_frames

                effective_tol_s = self.tolerance_s
                if self.tolerance_frames is not None:
                    effective_tol_s = (self.tolerance_frames + 0.5) / average_fps

                frame_indices = [round(ts * average_fps) for ts in query_ts]
                clamped_mask = [idx >= num_frames or idx < 0 for idx in frame_indices]
                frame_indices = [max(0, min(idx, num_frames - 1)) for idx in frame_indices]
                frames_batch = decoder.get_frames_at(indices=frame_indices)

                loaded_frames = [frame for frame in frames_batch.data]
                loaded_ts = [pts.item() for pts in frames_batch.pts_seconds]

                query_ts_tensor = torch.tensor(query_ts)
                loaded_ts_tensor = torch.tensor(loaded_ts)
                dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
                min_, argmin_ = dist.min(1)
                clamped_mask_tensor = torch.tensor(clamped_mask)
                is_within_tol = (min_ < effective_tol_s) | clamped_mask_tensor
                assert is_within_tol.all(), (
                    f"Timestamp tolerance violated: {min_[~is_within_tol]} > {effective_tol_s=}. "
                    f"video: {video_path}"
                )

                closest_frames = _safe_stack_frames([loaded_frames[idx] for idx in argmin_])
                if isinstance(closest_frames, torch.Tensor):
                    closest_frames = (closest_frames / 255.0).type(torch.float32)
                else:
                    closest_frames = [(f / 255.0).type(torch.float32) for f in closest_frames]
                item[video_key] = _maybe_squeeze(closest_frames, len(query_ts))
            finally:
                try:
                    fh.close()
                except Exception:
                    pass
        return item

    # ── Async decode pipeline ───────────────────────────────────────────

    def _ensure_decode_pipeline(self):
        if self._decode_pipeline is None:
            n_cameras = max(1, len(self.meta.video_keys))
            cache_size = max(4, 2 * self._num_dataloader_workers * n_cameras)
            num_threads = self.decode_num_threads if self.decode_num_threads > 1 else self._num_dataloader_workers

            episode_video_map = {}
            episode_is_valid_map = {}
            for ep_idx in range(len(self.meta.episodes)):
                ep = self.meta.episodes[ep_idx]
                episode_video_map[ep_idx] = {}
                for vid_key in self.meta.video_keys:
                    episode_video_map[ep_idx][vid_key] = (
                        ep[f"videos/{vid_key}/chunk_index"],
                        ep[f"videos/{vid_key}/file_index"],
                    )
                episode_is_valid_map[ep_idx] = {
                    cam_key: ep.get(f"videos/{cam_key}/is_valid", 1) == 1
                    for cam_key in self.meta.camera_keys
                }

            camera_shapes = {}
            for k in self.meta.camera_keys:
                camera_shapes[k] = list(self.meta.info["features"][k]["shape"])

            config = DecodeProcessConfig(
                root=str(self.root),
                streaming_from_local=self.streaming_from_local,
                tolerance_s=self.tolerance_s,
                tolerance_frames=self.tolerance_frames,
                camera_keys=list(self.meta.camera_keys),
                delta_indices=self.delta_indices,
                video_path_template=self.meta.video_path,
                url_root=self.meta.url_root,
                episode_video_map=episode_video_map,
                camera_shapes=camera_shapes,
                decode_device=self.decode_device,
                decode_num_threads=num_threads,
                cache_size_per_thread=cache_size,
                episode_is_valid_map=episode_is_valid_map,
                video_seek_modes=self._video_seek_modes,
            )

            try:
                pickle.dumps(config)
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                raise ValueError(
                    f"DecodeProcessConfig contains unpicklable attributes: {e}. "
                    f"Ensure delta_indices and other attributes are picklable."
                ) from e

            self._decode_pipeline = DecodeProcessPipeline(config)

    def shutdown_decode_pipeline(self):
        if self._decode_pipeline is not None:
            self._decode_pipeline.shutdown()
            self._decode_pipeline = None

    def decode_iter(self, dataloader):
        items_iter = iter(dataloader)

        self._ensure_decode_pipeline()

        try:
            first_items = next(items_iter)
        except StopIteration:
            return

        if not self.deferred_video_decode or not any(
            "_video_lookup" in item for item in first_items
        ):
            yield first_items
            yield from items_iter
            return

        self._decode_pipeline.submit(first_items)

        for items in items_iter:
            decoded = self._decode_pipeline.consume()
            self._decode_pipeline.submit(items)
            yield decoded

        decoded = self._decode_pipeline.consume()
        yield decoded

    # ── Video helpers ───────────────────────────────────────────────────

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
            mask = []
            for ts in timestamps:
                if is_float_in_list(ts, query_timestamps[video_key]):
                    mask.append(False)
                else:
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

    def _query_videos(self, query_timestamps, ep_idx, skip_invalid=True):
        item = {}
        ep_meta = self.meta.episodes[ep_idx]

        for video_key, query_ts in query_timestamps.items():
            is_valid_key = f"videos/{video_key}/is_valid"
            is_valid = ep_meta.get(is_valid_key, 1)

            if skip_invalid and is_valid == 0:
                item[video_key] = self._make_padding_camera_frame(video_key)
                continue

            root = (
                self.meta.url_root
                if hasattr(self.meta, "url_root") and not self.streaming_from_local
                else self.root
            )
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"

            # Look up seek_mode
            video_rel = str(self.meta.get_video_file_path(ep_idx, video_key))
            if video_rel.startswith("videos/"):
                video_rel = video_rel[len("videos/"):]
            seek_mode = self._video_seek_modes.get(video_rel, "approximate")

            try:
                if self.decode_device == "cuda":
                    frames = self._decode_video_cuda(video_path, query_ts, seek_mode=seek_mode)
                else:
                    frames = decode_video_frames_torchcodec(
                        video_path, query_ts, self.tolerance_s,
                        tolerance_frames=self.tolerance_frames,
                        decoder_cache=self.video_decoder_cache,
                        seek_mode=seek_mode,
                    )
            except Exception as e:
                logging.error(
                    f"Video decode failed for ep_idx={ep_idx}, video_key={video_key}: "
                    f"video_path={video_path}, query_ts={query_ts}, err={e}"
                )
                raise

            item[video_key] = _maybe_squeeze(frames, len(query_ts))
        return item

    def _decode_video_cuda(self, video_path, timestamps, seek_mode="approximate"):
        from torchcodec.decoders import VideoDecoder

        video_path_str = str(video_path)

        if self._cuda_decoder_cache is None:
            self._cuda_decoder_cache = BoundedVideoDecoderCache(max_size=4)

        decoder = self._cuda_decoder_cache.get_decoder(video_path_str, seek_mode)

        metadata = decoder.metadata
        average_fps = metadata.average_fps
        num_frames = metadata.num_frames

        effective_tol_s = self.tolerance_s
        if self.tolerance_frames is not None:
            effective_tol_s = (self.tolerance_frames + 0.5) / average_fps

        frame_indices = [round(ts * average_fps) for ts in timestamps]
        clamped_mask = [idx >= num_frames or idx < 0 for idx in frame_indices]
        frame_indices = [max(0, min(idx, num_frames - 1)) for idx in frame_indices]

        try:
            frames_batch = decoder.get_frames_at(indices=frame_indices)
        except RuntimeError as e:
            if "Expected pre-allocated tensor" in str(e):
                logging.warning(f"CUDA pre-allocated tensor mismatch for {video_path_str}. Evicting and rebuilding.")
                decoder = self._cuda_decoder_cache.evict_and_rebuild(video_path_str, seek_mode)
                frames_batch = decoder.get_frames_at(indices=frame_indices)
            else:
                raise

        loaded_frames = [frame.cpu() for frame in frames_batch.data]
        loaded_ts = [pts.item() for pts in frames_batch.pts_seconds]

        query_ts_tensor = torch.tensor(timestamps)
        loaded_ts_tensor = torch.tensor(loaded_ts)
        dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
        min_, argmin_ = dist.min(1)
        clamped_mask_tensor = torch.tensor(clamped_mask)
        is_within_tol = (min_ < effective_tol_s) | clamped_mask_tensor
        assert is_within_tol.all(), (
            f"Timestamp tolerance violated: {min_[~is_within_tol]} > {effective_tol_s=}. "
            f"video: {video_path}"
        )

        closest_frames = _safe_stack_frames([loaded_frames[idx] for idx in argmin_])
        if isinstance(closest_frames, torch.Tensor):
            closest_frames = (closest_frames / 255.0).type(torch.float32)
        else:
            closest_frames = [(f / 255.0).type(torch.float32) for f in closest_frames]
        return closest_frames

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=100):
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))


class AsyncDecodeDataLoader:
    """Wraps a DataLoader with video decode + collate for PyTorch Lightning.

    Supports optional background-thread prefetch for smoother data delivery
    during training. When prefetch_queue_size > 0, a producer thread reads
    batches from the DataLoader ahead of the training loop and buffers them
    in a queue.Queue, eliminating data-yield stalls caused by chunk boundary
    I/O or worker stalls.

    When preprocess_fn is provided alongside prefetch_queue_size > 0, the
    producer thread also runs CPU-only preprocessing (skipping GPU transfer)
    before enqueuing, so the main training thread only needs to move tensors
    to GPU — eliminating the ~1.0-1.5s CPU preprocess bottleneck.
    """

    VARIABLE_LENGTH_KEYS = {"hist_actions_full", "hist_actions_mask"}

    def __init__(self, dataloader, dataset, collate_fn=None, prefetch_queue_size=0, preprocess_fn=None):
        self._loader = dataloader
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._prefetch_queue_size = prefetch_queue_size
        self._preprocess_fn = preprocess_fn

    @staticmethod
    def make_collate_fn():
        variable_length_keys = AsyncDecodeDataLoader.VARIABLE_LENGTH_KEYS

        def collate_fn(batch):
            result = {}
            for key in batch[0].keys():
                values = [item[key] for item in batch]
                if key == "task":
                    result[key] = values
                elif key.startswith("observation.images."):
                    result[key] = values
                elif key == "camera_valid_mask":
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
                elif key == "action_dim" or key == "state_dim" or key == "_tier":
                    result[key] = torch.tensor(values)
                elif key == "_memory_cost":
                    result[key] = torch.tensor(values, dtype=torch.float32)
                elif isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values)
                else:
                    result[key] = values
            return result

        return collate_fn

    def _prefetch_iter(self):
        """Background-thread prefetch: producer reads from DataLoader,
        applies collate_fn and optionally preprocess_fn, then buffers
        batches in queue.Queue.

        IMPORTANT: The DataLoader iterator is created on the *main* thread
        before the producer thread starts.  This is critical because
        ``iter(loader)`` with ``persistent_workers=True`` forks worker
        processes, and forking inside a background thread of a
        multi-threaded process is unsafe — only the calling thread
        survives the fork, leaving locks held by other threads permanently
        locked in the child, which causes deadlocks.
        """
        import queue as queue_mod
        import threading

        data_queue = queue_mod.Queue(maxsize=self._prefetch_queue_size)
        shutdown_event = threading.Event()
        error_holder = [None]  # shared mutable container for error propagation

        decode_on_yield = self._dataset.deferred_video_decode and not self._dataset.async_decode

        # Create the DataLoader iterator on the main thread *before* starting
        # the producer thread.  This ensures that fork() happens while only
        # the main thread is alive, avoiding the multi-threaded fork deadlock.
        if self._dataset.async_decode and self._dataset.deferred_video_decode:
            loader_iter = self._dataset.decode_iter(self._loader)
        else:
            loader_iter = iter(self._loader)

        def producer():
            try:
                for batch in loader_iter:
                    if shutdown_event.is_set():
                        break
                    if not (self._dataset.async_decode and self._dataset.deferred_video_decode):
                        if self._dataset.deferred_video_decode and not decode_on_yield:
                            batch = self._dataset.decode_items_batch(batch)
                    batch = self._collate_fn(batch) if self._collate_fn else batch
                    if self._preprocess_fn:
                        batch = self._preprocess_fn(batch)
                    # Retry put until success or shutdown (don't discard preprocessed batches)
                    while not shutdown_event.is_set():
                        try:
                            data_queue.put(batch, block=True, timeout=0.5)
                            break
                        except queue_mod.Full:
                            continue
            except Exception as e:
                error_holder[0] = e
            finally:
                # Non-blocking sentinel: if queue is full the consumer will
                # see StopIteration from the generator exit anyway.
                try:
                    data_queue.put(None, block=False)
                except queue_mod.Full:
                    pass

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                item = data_queue.get(block=True)
                if item is None:
                    # Sentinel reached — iterator exhausted or producer stopped
                    if error_holder[0] is not None:
                        raise error_holder[0]
                    break
                yield item
        finally:
            shutdown_event.set()
            # Drain queue so producer thread can exit if blocked on put
            while not data_queue.empty():
                try:
                    data_queue.get_nowait()
                except queue_mod.Empty:
                    break
            # Wait for producer thread to fully exit before returning.
            # A longer timeout is needed because the thread may be blocked
            # inside the DataLoader's next() call with persistent_workers,
            # which has no timeout.  If this returns with the thread still
            # alive, a subsequent iter() on the same DataLoader could race.
            thread.join(timeout=30)
            if thread.is_alive():
                import warnings
                warnings.warn(
                    "AsyncDecodeDataLoader: producer thread did not exit "
                    "within 30s.  This may cause issues with "
                    "persistent_workers restart."
                )

    def __iter__(self):
        if self._prefetch_queue_size > 0:
            yield from self._prefetch_iter()
        elif self._dataset.async_decode and self._dataset.deferred_video_decode:
            for decoded_items in self._dataset.decode_iter(self._loader):
                if self._collate_fn is not None:
                    yield self._collate_fn(decoded_items)
                else:
                    yield decoded_items
        else:
            # In decode_on_yield mode, workers already decoded items (no _video_lookup).
            # Only call decode_items_batch when items still have _video_lookup
            # (deferred + async path, where DecodeProcessPipeline handles decode).
            decode_on_yield = self._dataset.deferred_video_decode and not self._dataset.async_decode
            for batch in self._loader:
                if self._dataset.deferred_video_decode and not decode_on_yield:
                    batch = self._dataset.decode_items_batch(batch)
                if self._collate_fn is not None:
                    batch = self._collate_fn(batch)
                yield batch

    def close(self):
        """Shut down prefetch thread, DataLoader workers, and decode pipeline.

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
