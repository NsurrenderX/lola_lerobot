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
LoLA Pretrain Dataset -- Map-style Dataset for LoLA pretraining.

Refactored from LoLAPretrainStreamingDataset (IterableDataset) to a standard
torch.utils.data.Dataset with __len__ and __getitem__.

Key design choices:
- Ultra-low memory init: only reads parquet metadata (row counts), never data.
  Builds a cumulative-sum index for O(log N) global_idx -> file + local_row mapping.
- On-demand row reading via PyArrow memory-mapped I/O with per-worker table cache.
- Episode boundary arrays enable delta_frames and history_actions without Backtrackable.
- Video decode always in __getitem__ (DataLoader workers parallelize decode).

Usage:
    dataset = LoLAPretrainDataset(
        repo_id="lerobot/pusht",
        max_history_length=100,
        action_chunk_size=10,
        delta_timestamps={...},
        root="/path/to/data",
        dataset_to_episodes_path="/path/to/dataset_to_episodes.json",
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        collate_fn=make_collate_fn(),
    )

    for batch in loader:
        # batch["observation.state"]: normalized per sub-dataset
        # batch["action"]: normalized per sub-dataset
        # batch["hist_actions_full"]: normalized (mask=True parts only)
        # batch["camera_valid_mask"]: list of {cam_key: bool}
        # batch["action_dim"], batch["state_dim"]: tensor
        # batch[cam_key]: list of PIL Image (valid) or None (invalid)
"""

import bisect
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    EPISODES_DIR,
    check_version_compatibility,
    find_float_index,
    get_delta_indices,
    is_float_in_list,
    item_to_torch,
    load_info,
    load_stats,
    load_tasks,
)
from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)


# ── Reusable helpers from streaming version ──────────────────────────────


class BoundedVideoDecoderCache(VideoDecoderCache):
    """VideoDecoderCache with capacity limit to bound per-worker memory.

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


def _discover_parquet_files(root: str) -> list[str]:
    """Discover all chunk/file parquet files under root/data/, sorted."""
    data_dir = Path(root) / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(data_dir.glob("*/*.parquet"))
    return [str(f) for f in files]


def _safe_concat(dfs: list) -> "pl.DataFrame":
    """Safely concatenate multiple polars DataFrames with different columns/types."""
    import polars as pl

    if len(dfs) == 1:
        return dfs[0]

    all_cols: list[str] = []
    seen = set()
    for df in dfs:
        for c in df.columns:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)

    aligned = []
    for df in dfs:
        existing = set(df.columns)
        null_cols = [c for c in all_cols if c not in existing]
        if null_cols:
            df = df.with_columns([pl.lit(None).alias(c) for c in null_cols])
        try:
            df = df.select(all_cols)
        except pl.exceptions.SchemaError:
            selected = []
            for col in all_cols:
                if col in existing:
                    selected.append(pl.col(col))
                else:
                    selected.append(pl.lit(None).alias(col))
            df = df.select(selected)
        aligned.append(df)

    return pl.concat(aligned, how="vertical_relaxed")


def _load_episodes_polars(root) -> list[dict]:
    """Load episodes metadata via polars to avoid HF datasets CastError.

    Handles schema-inconsistent parquet files (e.g. files 000-045 that only
    have stats/* columns and lack episode_index) by using diagonal concat,
    then forcefully overwriting episode_index with a continuous 0..N-1 index
    based on physical row count.
    """
    import polars as pl

    episodes_dir = Path(root) / EPISODES_DIR
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {episodes_dir}")

    dfs = []
    for path in parquet_files:
        df = pl.scan_parquet(str(path)).collect()
        if df.height > 0:
            dfs.append(df)

    if not dfs:
        return []

    # Cast Int-type stats/ columns to Float64 to avoid SchemaError on concat
    processed_dfs = []
    for df in dfs:
        cast_exprs = []
        for c, t in zip(df.columns, df.dtypes):
            if c.startswith("stats/"):
                type_str = str(t)

                # 情况 A：如果这是一个列表类型 (List)
                if "List" in type_str:
                    # 如果列表内部装的是整数或低精度浮点
                    if "Int" in type_str or "Float32" in type_str:
                        cast_exprs.append(pl.col(c).cast(pl.List(pl.Float64)))
                # 情况 B：如果这是一个整数或低精度浮点
                elif "Int" in type_str or "Float32" in type_str:
                    cast_exprs.append(pl.col(c).cast(pl.Float64))
        if cast_exprs:
            df = df.with_columns(cast_exprs)
        processed_dfs.append(df)

    combined = pl.concat(processed_dfs, how="diagonal")

    # Forcefully overwrite episode_index with continuous 0..N-1 range
    if "episode_index" in combined.columns:
        combined = combined.drop("episode_index")
    combined = combined.with_row_index("episode_index")

    for col in combined.columns:
        if col == "episode_index":
            continue
        if col.endswith("/is_valid"):
            combined = combined.with_columns(pl.col(col).fill_null(0))
        elif combined[col].dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
            combined = combined.with_columns(pl.col(col).fill_null(0))

    episodes = []
    for i in range(combined.height):
        row = combined.row(i, named=True)
        ep_dict = {}
        for key, val in row.items():
            if val is None:
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
    """Dict-style access to episodes list, compatible with HF Dataset row interface."""

    def __init__(self, episodes: list[dict]):
        self._episodes = episodes

    def __getitem__(self, idx: int) -> dict:
        return self._episodes[idx]

    def __len__(self) -> int:
        return len(self._episodes)


# ── Parquet row reader with PyArrow mmap ────────────────────────────────


class _ParquetRowReader:
    """Memory-efficient row reader using PyArrow memory-mapped I/O + per-worker cache.

    Each DataLoader worker gets its own fork() copy, so the _table_cache is
    naturally per-worker. With persistent_workers=True, the cache persists
    across epochs -- avoiding repeated mmap setup overhead.
    """

    def __init__(self, parquet_files: list[str], file_cumsum: list[int]):
        self._parquet_files = parquet_files
        self._file_cumsum = file_cumsum
        self._table_cache: dict[int, "pyarrow.Table"] = {}

    def _get_table(self, file_idx: int):
        if file_idx not in self._table_cache:
            path = self._parquet_files[file_idx]
            self._table_cache[file_idx] = pq.read_table(path, memory_map=True)
            # Evict oldest entries if cache exceeds 3 files (~60MB per file for typical sizes)
            while len(self._table_cache) > 3:
                oldest = next(iter(self._table_cache))
                del self._table_cache[oldest]
        return self._table_cache[file_idx]

    def read_row(self, global_idx: int) -> dict:
        """Read a single row by global index."""
        file_idx = bisect.bisect_right(self._file_cumsum, global_idx) - 1
        local_idx = global_idx - self._file_cumsum[file_idx]
        table = self._get_table(file_idx)
        row = table.slice(local_idx, 1).to_pydict()
        return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in row.items()}

    def read_rows_range(self, start_idx: int, end_idx: int) -> list[dict]:
        """Batch-read rows [start_idx, end_idx). Handles cross-file ranges."""
        rows = []
        current = start_idx
        while current < end_idx:
            file_idx = bisect.bisect_right(self._file_cumsum, current) - 1
            file_start = self._file_cumsum[file_idx]
            file_end = self._file_cumsum[file_idx + 1]
            local_start = current - file_start
            local_end = min(end_idx - file_start, file_end - file_start)
            table = self._get_table(file_idx)
            slice_table = table.slice(local_start, local_end - local_start)
            for i in range(slice_table.num_rows):
                row = slice_table.slice(i, 1).to_pydict()
                rows.append({k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in row.items()})
            current = file_start + local_end
        return rows


# ── Collate function ────────────────────────────────────────────────────


VARIABLE_LENGTH_KEYS = {"hist_actions_full", "hist_actions_mask"}


def make_collate_fn():
    """Create a collate function that handles variable-length tensors and dynamic resolution.

    Handles:
    - hist_actions_full and hist_actions_mask by left-padding shorter sequences
      to the max length in the batch, then stacking.
    - Camera keys (observation.images.*) as lists of PIL Images or None.
    - camera_valid_mask as a list of dicts.
    - action_dim and state_dim as tensors.
    - All other tensors are stacked normally.
    """

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
            elif key in VARIABLE_LENGTH_KEYS and isinstance(values[0], torch.Tensor):
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
            elif key == "action_dim" or key == "state_dim":
                result[key] = torch.tensor(values)
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result

    return collate_fn


# ── Main Dataset class ─────────────────────────────────────────────────


class LoLAPretrainDataset(torch.utils.data.Dataset):
    """Map-style Dataset for LoLA pretraining with per-sub-dataset normalization.

    Key features:
    - Ultra-low memory init: only reads parquet metadata (row counts), not data.
    - O(log N) index mapping via cumulative sum + bisect.
    - On-demand row reading via PyArrow mmap with per-worker cache.
    - Episode boundary arrays for delta_frames and history_actions.
    - Video decode in __getitem__ with per-worker BoundedVideoDecoderCache.
    - Per-sub-dataset normalization (observation.state, action, hist_actions).
    - Camera validity handling (is_valid=0 -> None + camera_valid_mask).
    - Heterogeneous action/state spaces via action_dim and state_dim per item.
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
        decode_device: str = "cpu",
        dataset_to_episodes_path: str | None = None,
        temp_process: bool = False,
    ):
        """
        Args:
            repo_id: Dataset repository ID.
            max_history_length: Max history action length, truncated if exceeded.
            action_chunk_size: Action chunk size, history padded to its multiples.
            history_padding_side: Padding direction, "left" or "right".
            root: Local dataset root directory.
            sub_root: Sub-dataset directory for per-sub-dataset stats.
            episodes: Optional list of episode indices to load (not yet implemented).
            image_transforms: Image transforms (unused, Qwen3.5 processor handles).
            delta_timestamps: Timestamp offset config for lookback/lookahead.
            tolerance_s: Fallback timestamp tolerance in seconds (used when tolerance_frames is None).
            tolerance_frames: Max allowed frame offset for video decode. When set,
                tolerance_s is computed as (tolerance_frames + 0.5) / average_fps per-video.
            revision: Dataset version.
            force_cache_sync: Force sync from HuggingFace Hub.
            decode_device: Video decode device, "cpu" or "cuda".
            dataset_to_episodes_path: JSON file mapping episodes to sub-datasets.
            temp_process: When True, zero-pad mismatched sub-dataset stats dims.
        """
        super().__init__()

        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.sub_root = sub_root
        self.streaming_from_local = root is not None

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.tolerance_frames = tolerance_frames
        self.revision = revision if revision else CODEBASE_VERSION
        self.decode_device = decode_device
        self.temp_process = temp_process

        # Load metadata via polars (avoids HF datasets CastError)
        self.meta = self._build_metadata_polars(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        # Delta timestamps
        self.delta_timestamps = None
        self.delta_indices = None
        if delta_timestamps is not None:
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.max_history_length = max_history_length
        self.action_chunk_size = action_chunk_size
        self.history_padding_side = history_padding_side

        # Action dimension
        if "action" in self.meta.features:
            self.action_dim = self.meta.features["action"]["shape"][0]
        else:
            self.action_dim = 1

        # ── Lightweight index mapping (no data read) ──────────────────
        self._parquet_files = _discover_parquet_files(str(self.root))

        # Build cumulative sum from parquet metadata only
        self._file_row_counts: list[int] = []
        self._file_cumsum: list[int] = [0]
        for path in self._parquet_files:
            pf = pq.ParquetFile(path)
            n = pf.metadata.num_rows
            self._file_row_counts.append(n)
            self._file_cumsum.append(self._file_cumsum[-1] + n)

        self._total_rows = self._file_cumsum[-1]

        # Episode boundary arrays for delta_frames and history_actions
        self._episode_starts = np.empty(len(self.meta.episodes), dtype=np.int64)
        self._episode_ends = np.empty(len(self.meta.episodes), dtype=np.int64)
        for ep_idx in range(len(self.meta.episodes)):
            ep = self.meta.episodes[ep_idx]
            self._episode_starts[ep_idx] = ep["dataset_from_index"]
            self._episode_ends[ep_idx] = ep["dataset_to_index"]

        # Row reader (will be shared per-worker via fork)
        self._row_reader = _ParquetRowReader(self._parquet_files, self._file_cumsum)

        # Video decoder caches (initialized lazily in __getitem__)
        self.video_decoder_cache = None
        self._cuda_decoder_cache = None

        # ── Per-sub-dataset normalization setup ──────────────────────
        self._episode_to_ds_idx = np.full(self.meta.total_episodes, -1, dtype=np.int16)
        self._sub_dataset_names: list[str] = []
        self._sub_dataset_paths: list[str] = []
        self._sub_dataset_norm_params: list[dict | None] = []
        self._sub_dataset_dims: list[tuple[int, int]] = []  # (action_dim, state_dim)

        if dataset_to_episodes_path is not None:
            self._load_dataset_to_episodes(dataset_to_episodes_path)

        print(f"[LoLAPretrainDataset] max_history_length: {max_history_length}")
        print(f"[LoLAPretrainDataset] action_chunk_size: {action_chunk_size}")
        print(f"[LoLAPretrainDataset] history_padding_side: {history_padding_side}")
        print(f"[LoLAPretrainDataset] action_dim: {self.action_dim}")
        print(f"[LoLAPretrainDataset] parquet_files: {len(self._parquet_files)}")
        print(f"[LoLAPretrainDataset] total_rows: {self._total_rows}")
        print(f"[LoLAPretrainDataset] total_episodes: {len(self.meta.episodes)}")
        print(f"[LoLAPretrainDataset] sub_datasets: {len(self._sub_dataset_names)}")

    # ── Metadata loading ─────────────────────────────────────────────

    @staticmethod
    def _build_metadata_polars(repo_id, root, revision, force_cache_sync=False):
        """Build metadata using polars for episodes (avoids HF datasets CastError)."""
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

        print(f"[LoLAPretrainDataset] Loaded {len(episodes_list)} episodes via polars")
        return meta

    # ── Sub-dataset normalization setup ──────────────────────────────

    def _load_dataset_to_episodes(self, dataset_to_episodes_path: str):
        """Load dataset_to_episodes.json and build per-sub-dataset normalization data."""
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
                    f"[LoLAPretrainDataset] Could not load stats for "
                    f"sub-dataset '{ds_name}' from {stats_path}: {e}. "
                    f"Skipping per-dataset normalization for this sub-dataset."
                )
                norm_params = None
                action_dim = self.action_dim
                state_dim = 0

            self._sub_dataset_norm_params.append(norm_params)
            self._sub_dataset_dims.append((action_dim, state_dim))
            ds_idx += 1

    # ── Properties ───────────────────────────────────────────────────

    @property
    def num_frames(self):
        return self._total_rows

    @property
    def num_episodes(self):
        return len(self.meta.episodes)

    @property
    def fps(self):
        return self.meta.fps

    # ── Map-style interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return self._total_rows

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += self._total_rows
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(idx)

        # Ensure decoder caches are initialized (lazily, per-worker)
        self._ensure_decoder_cache()

        # 1. Read the primary row
        row = self._row_reader.read_row(idx)
        item = item_to_torch(self._row_to_item(row))

        # Map global row idx → list-level episode index via bisect on _episode_starts
        # (item["episode_index"] is the raw global episode ID from parquet, NOT a list index)
        ep_idx = bisect.bisect_right(self._episode_starts, idx) - 1

        # 2. Video decode (always in getitem)
        if len(self.meta.video_keys) > 0:
            current_ts = item["index"] / self.fps
            episode_boundaries_ts = {
                key: (
                    self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                    self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
                )
                for key in self.meta.video_keys
            }
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item.update(video_frames)
            if self.delta_indices is not None:
                item.update(
                    self._get_video_frame_padding_mask(
                        video_frames, query_timestamps, original_timestamps
                    )
                )

        # 3. Delta timestamps for non-video keys
        if self.delta_indices is not None:
            delta_result, padding = self._get_delta_frames_map(idx, ep_idx, item)
            item.update(delta_result)
            item.update(padding)

        # 4. History actions
        item = self._add_history_actions_map(idx, ep_idx, item)

        # 5. Post-processing
        item["task"] = self.meta.tasks.iloc[item["task_index"]].name
        item = self._apply_camera_valid_mask(item, ep_idx)
        item = self._normalize_per_subdataset(item, temp_process=self.temp_process)
        ds_idx = (
            self._episode_to_ds_idx[ep_idx]
            if ep_idx < len(self._episode_to_ds_idx) and self._episode_to_ds_idx[ep_idx] >= 0
            else 0
        )
        action_dim, state_dim = (
            self._sub_dataset_dims[ds_idx]
            if ds_idx < len(self._sub_dataset_dims)
            else (self.action_dim, 0)
        )
        item["action_dim"] = action_dim
        item["state_dim"] = state_dim

        return item

    # ── Row conversion ───────────────────────────────────────────────

    @staticmethod
    def _row_to_item(row: dict) -> dict:
        """Convert a raw parquet row dict to the format expected by item_to_torch."""
        result = {}
        for key, val in row.items():
            if isinstance(val, list):
                result[key] = np.array(val, dtype=np.float32)
            elif isinstance(val, (int, np.integer)):
                result[key] = int(val)
            elif isinstance(val, (float, np.floating)):
                result[key] = float(val)
            else:
                result[key] = val
        return result

    # ── Delta frames (map-style, replaces Backtrackable) ─────────────

    def _get_delta_frames_map(self, idx: int, ep_idx: int, current_item: dict):
        """Compute delta frames using direct index math instead of Backtrackable."""
        ep_start = int(self._episode_starts[ep_idx])
        ep_end = int(self._episode_ends[ep_idx])

        query_result = {}
        padding = {}

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue

            target_frames = []
            is_pad = []

            for delta in delta_indices:
                target_idx = idx + delta
                if ep_start <= target_idx < ep_end:
                    row = self._row_reader.read_row(target_idx)
                    row_item = item_to_torch(self._row_to_item(row))
                    target_frames.append(row_item[key])
                    is_pad.append(False)
                else:
                    target_frames.append(current_item[key])
                    is_pad.append(True)

            if target_frames:
                query_result[key] = torch.stack(target_frames)
                padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

        return query_result, padding

    # ── History actions (map-style, replaces Backtrackable.history()) ─

    def _add_history_actions_map(self, idx: int, ep_idx: int, item: dict):
        """Add full history actions using batch row reading instead of Backtrackable."""
        current_action = item.get("action")
        if current_action is None:
            return item
        if current_action.dim() > 1:
            current_action = current_action[0] if current_action.shape[0] == 1 else current_action[-1]

        ep_start = int(self._episode_starts[ep_idx])
        max_lookback = min(self.max_history_length - 1, idx - ep_start)

        # Batch-read history rows for efficiency
        history_start = idx - max_lookback
        if max_lookback > 0:
            history_rows = self._row_reader.read_rows_range(history_start, idx)
        else:
            history_rows = []

        past_actions = []
        past_masks = []

        for row in history_rows:
            past_action = row.get("action")
            if past_action is not None:
                if isinstance(past_action, (np.ndarray, list)):
                    past_action = torch.tensor(past_action, dtype=torch.float32)
                if past_action.dim() > 1:
                    past_action = past_action[0] if past_action.shape[0] == 1 else past_action[-1]
                past_actions.append(past_action)
                past_masks.append(True)
            else:
                past_actions.append(torch.zeros(self.action_dim, dtype=current_action.dtype))
                past_masks.append(False)

        # Already in time order (oldest first) from read_rows_range

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

        return item

    # ── Video decode (always in getitem) ─────────────────────────────

    def _ensure_decoder_cache(self):
        """No-op — decoders are no longer cached to avoid torchcodec
        pre-allocated tensor shape mismatch across different-resolution videos.
        Kept for backward-compatible call sites."""
        pass

    def _query_videos(self, query_timestamps, ep_idx, skip_invalid=True):
        """Query video frames, optionally skipping invalid cameras."""
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

            try:
                if self.decode_device == "cuda":
                    frames = self._decode_video_cuda(video_path, query_ts)
                else:
                    frames = decode_video_frames_torchcodec(
                        video_path, query_ts, self.tolerance_s,
                        tolerance_frames=self.tolerance_frames,
                        decoder_cache=self.video_decoder_cache,
                    )
            except Exception as e:
                logging.error(
                    f"Video decode failed for ep_idx={ep_idx}, video_key={video_key}: "
                    f"video_path={video_path}, query_ts={query_ts}, "
                    f"ep_meta_video_keys={list(ep_meta.keys())}, "
                    f"from_timestamp={ep_meta.get(f'videos/{video_key}/from_timestamp')}, "
                    f"to_timestamp={ep_meta.get(f'videos/{video_key}/to_timestamp')}, "
                    f"is_valid={is_valid}, err={e}"
                )
                raise

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames
        return item

    def _decode_video_cuda(self, video_path, timestamps):
        """Decode video frames using CUDA (NVDEC).

        Always creates a fresh decoder per call to avoid torchcodec's internal
        pre-allocated tensor shape mismatch across videos of different resolutions.
        """
        from torchcodec.decoders import VideoDecoder

        video_path_str = str(video_path)
        file_handle = fsspec.open(video_path_str).__enter__()

        try:
            decoder = VideoDecoder(file_handle, seek_mode="approximate", device="cuda")

            metadata = decoder.metadata
            average_fps = metadata.average_fps
            num_frames = metadata.num_frames

            # Compute tolerance_s from tolerance_frames if provided
            effective_tol_s = self.tolerance_s
            if self.tolerance_frames is not None:
                effective_tol_s = (self.tolerance_frames + 0.5) / average_fps

            frame_indices = [round(ts * average_fps) for ts in timestamps]
            clamped_mask = [idx >= num_frames or idx < 0 for idx in frame_indices]
            frame_indices = [max(0, min(idx, num_frames - 1)) for idx in frame_indices]

            # Batch decode — safe because decoder is fresh, no stale pre-allocated buffer
            frame_batch = decoder.get_frames_at(indices=frame_indices)
            loaded_frames = frame_batch.data      # (T, C, H, W) uint8 on cuda
            loaded_ts = frame_batch.pts_seconds   # (T,) float

            query_ts_tensor = torch.tensor(timestamps)
            loaded_ts_tensor = torch.tensor(loaded_ts.tolist()) if not isinstance(loaded_ts, torch.Tensor) else loaded_ts
            dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
            min_, argmin_ = dist.min(1)
            clamped_mask_tensor = torch.tensor(clamped_mask)
            is_within_tol = (min_ < effective_tol_s) | clamped_mask_tensor
            assert is_within_tol.all(), (
                f"Timestamp tolerance violated: {min_[~is_within_tol]} > {effective_tol_s=}. "
                f"video: {video_path}"
            )

            closest_frames = loaded_frames[argmin_].cpu()
            closest_frames = (closest_frames / 255.0).type(torch.float32)
            return closest_frames
        finally:
            try:
                file_handle.close()
            except Exception:
                pass

    # ── Timestamp and padding helpers ────────────────────────────────

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

    # ── Camera validity ──────────────────────────────────────────────

    def _tensor_to_pil(self, tensor):
        """Convert [C, H, W] float32 tensor to PIL Image."""
        from PIL import Image

        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.permute(1, 2, 0)
        if img.dtype in [torch.float32, torch.float64]:
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(img.cpu().numpy())

    def _apply_camera_valid_mask(self, item, ep_idx):
        """Add camera_valid_mask and convert video frames to PIL Image / None."""
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
                elif isinstance(item[cam_key], torch.Tensor):
                    item[cam_key] = self._tensor_to_pil(item[cam_key])

        item["camera_valid_mask"] = camera_valid_mask
        return item

    # ── Per-sub-dataset normalization ────────────────────────────────

    @staticmethod
    def _make_translation_norm_mask(action_dim: int) -> torch.Tensor:
        """Build normalization mask: only translation dims need normalization."""
        mask = torch.zeros(action_dim, dtype=torch.bool)
        arm_dim = 10
        num_arms = action_dim // arm_dim
        for arm in range(num_arms):
            offset = arm * arm_dim
            mask[offset : offset + 3] = True
        return mask

    def _normalize_per_subdataset(self, item, temp_process=False):
        """Per-sub-dataset normalization for observation.state, action, hist_actions_full."""
        ep_idx = (
            item["episode_index"].item()
            if isinstance(item["episode_index"], torch.Tensor)
            else item["episode_index"]
        )
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

            if mean.shape[0] != self.action_dim:
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
                    f"[LoLAPretrainDataset] Padded stats for '{key}' in "
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
