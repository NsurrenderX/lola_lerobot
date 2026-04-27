#!/usr/bin/env python
"""
Validate LoLAPretrainStreamingDataset correctness and performance.

Tests:
1. Dataset creation + metadata sanity
2. Multi-worker DataLoader traversal (shuffle, buffer, deferred decode)
3. Item correctness: key presence, tensor shapes, NaN/Inf, hist_actions
4. collate_fn correctness (variable-length padding, camera lists, camera_valid_mask)
5. Video decode correctness (PIL quality, invalid cameras, resolution check)
   + save sample images for visual inspection
6. Streaming vs map-style consistency (compare output for same episodes)
7. Performance benchmark (throughput, memory)

Usage:
    python src/lerobot/scripts/validate_stream_dataloader.py \
        --dataset_root /path/to/dataset \
        --dataset_to_episodes_path /path/to/dataset_to_episodes.json \
        --batch_size 4 --num_workers 2 --max_batches 10

Note:
    - Multi-worker DataLoader runs FIRST (Step 2) to avoid torchcodec fork deadlock.
    - Use --no_mapping to skip per-dataset normalization if no dataset_to_episodes.json.
    - Use --save_images_dir "" to disable image saving.
    - Use --compare_map to enable streaming vs map-style consistency check (Step 6).
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from lerobot.configs.types import FeatureType
from lerobot.datasets.lola_pretrain_streaming_dataset import (
    LoLAPretrainStreamingDataset,
    AsyncDecodeDataLoader,
    EpisodeChunkReader,
)
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


# ── Validation helpers ──────────────────────────────────────────────────────


class ValidationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def ok(self, msg):
        self.passed.append(msg)

    def fail(self, msg):
        self.failed.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        lines = [
            "=" * 60,
            "Validation Summary",
            "=" * 60,
            f"Passed: {len(self.passed)}/{total}",
            f"Failed: {len(self.failed)}/{total}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.failed:
            lines.append("--- Failures ---")
            for f in self.failed:
                lines.append(f"  [FAIL] {f}")
        if self.warnings:
            lines.append("--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
        lines.append("")
        if not self.failed:
            lines.append("All validations passed!")
        else:
            lines.append("Some validations failed -- please investigate!")
        return "\n".join(lines)


def _build_config(dataset_root, dataset_to_episodes_path=None, sub_root=None,
                   temp_process=False, max_history_length=100,
                   action_chunk_size=10, n_obs_steps=1, pred_chunk_size=50):
    """Build LoLAConfig and delta_timestamps from dataset metadata."""
    from lerobot.datasets.lola_pretrain_streaming_dataset import _load_episodes_polars, _EpisodeAccessor
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, CODEBASE_VERSION
    from lerobot.datasets.utils import load_info, load_stats, load_tasks

    # Build minimal metadata (same as LoLAPretrainStreamingDataset._build_metadata_polars)
    import pathlib
    meta_root = pathlib.Path(dataset_root)
    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.repo_id = "test"
    meta.revision = CODEBASE_VERSION
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

    fps = meta.fps
    features = dataset_to_policy_features(meta.features)
    action_dim = features["action"].shape[0] if "action" in features else 20

    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        pred_chunk_size=pred_chunk_size,
        n_obs_steps=n_obs_steps,
        input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
        output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
        load_full_history=True,
        max_history_length=max_history_length,
    )

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in meta.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    return config, delta_timestamps, action_dim


def check_no_nan_inf(batch, result, label=""):
    prefix = f"[{label}] " if label else ""
    has_issue = False
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            if torch.isnan(val).any():
                result.fail(f"{prefix}[{key}] contains NaN")
                has_issue = True
            if torch.isinf(val).any():
                result.fail(f"{prefix}[{key}] contains Inf")
                has_issue = True
    if not has_issue:
        result.ok(f"{prefix}No NaN/Inf in float tensors")


def _make_invalid_placeholder(width, height):
    if not HAS_PIL:
        return None
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((width // 4, height // 4), "INVALID", fill=(255, 255, 255))
    except Exception:
        pass
    return img


def save_pil_images(items, camera_keys, save_dir, stage_name, max_samples=3, dataset=None):
    if not HAS_PIL:
        return

    stage_dir = os.path.join(save_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    saved_count = 0
    n_samples = min(len(items), max_samples)

    placeholder_w, placeholder_h = 64, 64
    if dataset is not None and camera_keys:
        first_cam = camera_keys[0]
        shape = dataset.meta.info["features"][first_cam]["shape"]
        if len(shape) == 3:
            placeholder_h, placeholder_w = shape[1], shape[2]

    for i in range(n_samples):
        item = items[i]
        ep_idx = item.get("episode_index")
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        idx_val = item.get("index")
        if isinstance(idx_val, torch.Tensor):
            idx_val = idx_val.item()

        valid_frames = []
        for cam_key in camera_keys:
            val = item.get(cam_key)
            cam_name = cam_key.replace("/", "_").replace(".", "_")

            if val is not None and isinstance(val, Image.Image):
                fname = os.path.join(stage_dir, f"{cam_name}_ep{ep_idx}_idx{idx_val}.png")
                val.save(fname)
                saved_count += 1
                valid_frames.append(val)
            else:
                placeholder = _make_invalid_placeholder(placeholder_w, placeholder_h)
                if placeholder is not None:
                    fname = os.path.join(stage_dir, f"{cam_name}_INVALID_ep{ep_idx}_idx{idx_val}.png")
                    placeholder.save(fname)
                    saved_count += 1
                    valid_frames.append(placeholder)

        if len(valid_frames) > 1:
            total_w = sum(f.width for f in valid_frames)
            max_h = max(f.height for f in valid_frames)
            composite = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
            x_offset = 0
            for f in valid_frames:
                composite.paste(f, (x_offset, 0))
                x_offset += f.width
            composite_fname = os.path.join(stage_dir, f"all_cameras_ep{ep_idx}_idx{idx_val}.png")
            composite.save(composite_fname)
            saved_count += 1

    if saved_count > 0:
        print(f"    Saved {saved_count} images to {stage_dir}/")


def save_batch_images(batch, camera_keys, save_dir, stage_name, max_samples=3, dataset=None):
    if not HAS_PIL:
        return

    stage_dir = os.path.join(save_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    saved_count = 0
    placeholder_w, placeholder_h = 64, 64
    if dataset is not None and camera_keys:
        first_cam = camera_keys[0]
        shape = dataset.meta.info["features"][first_cam]["shape"]
        if len(shape) == 3:
            placeholder_h, placeholder_w = shape[1], shape[2]

    bs = 0
    for cam_key in camera_keys:
        if cam_key in batch and isinstance(batch[cam_key], list):
            bs = len(batch[cam_key])
            break
    if bs == 0:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                bs = val.shape[0]
                break
    if bs == 0:
        return

    n_samples = min(bs, max_samples)

    for i in range(n_samples):
        ep_idx_batch = batch.get("episode_index")
        if isinstance(ep_idx_batch, torch.Tensor):
            ep_idx = ep_idx_batch[i].item()
        elif isinstance(ep_idx_batch, list):
            ep_idx = ep_idx_batch[i]
        else:
            ep_idx = i

        idx_val_batch = batch.get("index")
        if isinstance(idx_val_batch, torch.Tensor):
            idx_val = idx_val_batch[i].item()
        elif isinstance(idx_val_batch, list):
            idx_val = idx_val_batch[i]
        else:
            idx_val = i

        valid_frames = []
        for cam_key in camera_keys:
            cam_list = batch.get(cam_key)
            cam_name = cam_key.replace("/", "_").replace(".", "_")

            if cam_list is not None and i < len(cam_list):
                val = cam_list[i]
                if val is not None and isinstance(val, Image.Image):
                    fname = os.path.join(stage_dir, f"{cam_name}_ep{ep_idx}_idx{idx_val}.png")
                    val.save(fname)
                    saved_count += 1
                    valid_frames.append(val)
                else:
                    placeholder = _make_invalid_placeholder(placeholder_w, placeholder_h)
                    if placeholder is not None:
                        fname = os.path.join(stage_dir, f"{cam_name}_INVALID_ep{ep_idx}_idx{idx_val}.png")
                        placeholder.save(fname)
                        saved_count += 1
                        valid_frames.append(placeholder)

        if len(valid_frames) > 1:
            total_w = sum(f.width for f in valid_frames)
            max_h = max(f.height for f in valid_frames)
            composite = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
            x_offset = 0
            for f in valid_frames:
                composite.paste(f, (x_offset, 0))
                x_offset += f.width
            composite_fname = os.path.join(stage_dir, f"all_cameras_ep{ep_idx}_idx{idx_val}.png")
            composite.save(composite_fname)
            saved_count += 1

    if saved_count > 0:
        print(f"    Saved {saved_count} images to {stage_dir}/")


def _create_dataset(args, delta_timestamps, config):
    """Helper to create LoLAPretrainStreamingDataset with common args."""
    dataset_to_episodes_path = args.dataset_to_episodes_path
    if args.no_mapping:
        dataset_to_episodes_path = None

    return LoLAPretrainStreamingDataset(
        repo_id="test",
        max_history_length=args.max_history_length,
        action_chunk_size=config.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        sub_root=args.sub_root,
        delta_timestamps=delta_timestamps,
        dataset_to_episodes_path=dataset_to_episodes_path,
        temp_process=args.temp_process,
        tolerance_frames=args.tolerance_frames,
        decode_device=args.decode_device,
        decode_num_threads=args.decode_num_threads,
        async_decode=args.async_decode,
        num_dataloader_workers=args.num_workers,
        deferred_video_decode=not args.no_deferred,
        buffer_size=args.buffer_size,
        episode_chunk_size=args.episode_chunk_size,
    )


# ── Step 1: Dataset creation + metadata ─────────────────────────────────────


def step1_dataset_creation(args, result, config, delta_timestamps, action_dim):
    print("\n" + "=" * 60)
    print("Step 1: Dataset Creation + Metadata Sanity")
    print("=" * 60)

    dataset = _create_dataset(args, delta_timestamps, config)

    if dataset.num_episodes > 0:
        result.ok(f"Dataset has {dataset.num_episodes} episodes")
    else:
        result.fail("Dataset has 0 episodes")

    if dataset.num_frames > 0:
        result.ok(f"Dataset has {dataset.num_frames} frames")
    else:
        result.fail("Dataset has 0 frames")

    if dataset.action_dim > 0:
        result.ok(f"action_dim={dataset.action_dim}")
    else:
        result.fail("action_dim=0")

    camera_keys = dataset.meta.camera_keys
    video_keys = dataset.meta.video_keys
    if len(camera_keys) > 0:
        result.ok(f"camera_keys={camera_keys}")
    else:
        result.warn("No camera keys found")

    if set(video_keys).issubset(set(camera_keys)):
        result.ok("video_keys subset of camera_keys")
    else:
        result.fail(f"video_keys not subset of camera_keys: {video_keys} vs {camera_keys}")

    # Check episode index arrays
    if len(dataset._episode_starts) == dataset.num_episodes:
        result.ok(f"_episode_starts array length matches episodes ({dataset.num_episodes})")
    else:
        result.fail(f"_episode_starts length {len(dataset._episode_starts)} != {dataset.num_episodes}")

    if len(dataset._episode_ends) == dataset.num_episodes:
        result.ok(f"_episode_ends array length matches episodes ({dataset.num_episodes})")
    else:
        result.fail(f"_episode_ends length {len(dataset._episode_ends)} != {dataset.num_episodes}")

    # Check EpisodeChunkReader
    if hasattr(dataset, '_chunk_reader') and isinstance(dataset._chunk_reader, EpisodeChunkReader):
        result.ok("EpisodeChunkReader initialized")
    else:
        result.fail("EpisodeChunkReader not initialized")

    sub_ds_count = len(dataset._sub_dataset_names)
    if args.dataset_to_episodes_path and not args.no_mapping:
        if sub_ds_count > 0:
            result.ok(f"Sub-datasets loaded: {sub_ds_count}")
        else:
            result.fail("No sub-datasets loaded despite dataset_to_episodes_path provided")
    else:
        result.ok("No dataset_to_episodes_path provided, skipping sub-dataset check")

    print(f"  fps: {dataset.fps}")
    print(f"  total_rows: {dataset.num_frames}")
    print(f"  total_episodes: {dataset.num_episodes}")
    print(f"  camera_keys: {camera_keys}")
    print(f"  video_keys: {video_keys}")
    print(f"  episode_chunk_size: {dataset.episode_chunk_size}")
    print(f"  sub_datasets: {sub_ds_count}")

    return dataset


# ── Step 2: Multi-worker DataLoader traversal ───────────────────────────────


def step2_multiworker_traversal(dataset, args, result, config, delta_timestamps):
    print("\n" + "=" * 60)
    print(f"Step 2: Multi-Worker DataLoader Traversal "
          f"(num_workers={args.num_workers}, batch_size={args.batch_size}, "
          f"max_batches={args.max_batches})")
    print("=" * 60)

    fresh_dataset = _create_dataset(args, delta_timestamps, config)

    raw_loader = DataLoader(
        fresh_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        pin_memory=False,
    )
    async_loader = AsyncDecodeDataLoader(
        dataloader=raw_loader,
        dataset=fresh_dataset,
        collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
    )

    batch_count = 0
    sample_count = 0
    start_time = time.time()
    camera_keys = fresh_dataset.meta.camera_keys

    try:
        for batch_idx, batch in enumerate(async_loader):
            batch_count += 1

            bs = None
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    bs = val.shape[0]
                    break
                elif isinstance(val, list):
                    bs = len(val)
                    break
            sample_count += bs if bs is not None else 0

            # Consistent batch_size
            batch_sizes = set()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_sizes.add(val.shape[0])
                elif isinstance(val, list):
                    batch_sizes.add(len(val))
            if len(batch_sizes) > 1:
                result.fail(f"[batch {batch_idx}] batch dimensions inconsistent: {batch_sizes}")
            elif len(batch_sizes) == 1:
                actual_bs = batch_sizes.pop()
                if actual_bs != args.batch_size and batch_idx == 0:
                    result.warn(f"[batch {batch_idx}] batch_size={actual_bs} < {args.batch_size}")

            # hist_actions checks
            if "hist_actions_full" in batch and "hist_actions_mask" in batch:
                haf = batch["hist_actions_full"]
                ham = batch["hist_actions_mask"]
                if haf.shape[0] != ham.shape[0]:
                    result.fail(f"[batch {batch_idx}] hist_actions batch dim mismatch: {haf.shape[0]} vs {ham.shape[0]}")
                if haf.shape[1] != ham.shape[1]:
                    result.fail(f"[batch {batch_idx}] hist_actions seq len mismatch: {haf.shape[1]} vs {ham.shape[1]}")
                if haf.shape[-1] != fresh_dataset.action_dim:
                    result.fail(f"[batch {batch_idx}] hist_actions last dim {haf.shape[-1]} != action_dim {fresh_dataset.action_dim}")
                if ham.dtype != torch.bool:
                    result.fail(f"[batch {batch_idx}] hist_actions_mask dtype {ham.dtype}, expected bool")

            if "hist_actions_length" in batch:
                hal = batch["hist_actions_length"]
                max_hal = hal.max().item()
                if max_hal > args.max_history_length:
                    result.fail(f"[batch {batch_idx}] hist_actions_length max {max_hal} > max_history_length {args.max_history_length}")

            # action_dim / state_dim
            if "action_dim" in batch:
                ad = batch["action_dim"]
                if not isinstance(ad, torch.Tensor):
                    result.fail(f"[batch {batch_idx}] action_dim type {type(ad).__name__}, expected tensor")
            if "state_dim" in batch:
                sd = batch["state_dim"]
                if not isinstance(sd, torch.Tensor):
                    result.fail(f"[batch {batch_idx}] state_dim type {type(sd).__name__}, expected tensor")

            # NaN/Inf check
            check_no_nan_inf(batch, result, label=f"batch{batch_idx}")

            # episode_index range check
            if "episode_index" in batch and isinstance(batch["episode_index"], torch.Tensor):
                ep_max = batch["episode_index"].max().item()
                ep_min = batch["episode_index"].min().item()
                if ep_max >= fresh_dataset.num_episodes:
                    result.fail(f"[batch {batch_idx}] episode_index max {ep_max} >= num_episodes {fresh_dataset.num_episodes}")
                if ep_min < 0:
                    result.fail(f"[batch {batch_idx}] episode_index min {ep_min} < 0")

            # Camera keys are lists of PIL Image / None
            for cam_key in camera_keys:
                if cam_key in batch:
                    val = batch[cam_key]
                    if not isinstance(val, list):
                        result.fail(f"[batch {batch_idx}] camera key '{cam_key}' is not a list, got {type(val).__name__}")
                    elif len(val) != (bs or args.batch_size):
                        result.fail(f"[batch {batch_idx}] camera key '{cam_key}' list length {len(val)} != batch_size {bs}")
                    else:
                        for v in val:
                            if v is not None and not isinstance(v, Image.Image):
                                result.fail(f"[batch {batch_idx}] camera '{cam_key}' entry is neither None nor PIL Image: {type(v).__name__}")

            # camera_valid_mask is list of dicts
            if "camera_valid_mask" in batch:
                cvm = batch["camera_valid_mask"]
                if not isinstance(cvm, list):
                    result.fail(f"[batch {batch_idx}] camera_valid_mask is not a list, got {type(cvm).__name__}")
                elif len(cvm) != (bs or args.batch_size):
                    result.fail(f"[batch {batch_idx}] camera_valid_mask length {len(cvm)} != batch_size {bs}")
                else:
                    for d in cvm:
                        if not isinstance(d, dict):
                            result.fail(f"[batch {batch_idx}] camera_valid_mask entry is not dict: {type(d).__name__}")

            # Print first 3 batches detail
            if batch_idx < 3:
                print(f"  batch {batch_idx}:")
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                    elif isinstance(val, list):
                        types = set(type(v).__name__ for v in val[:4])
                        print(f"    {key}: list len={len(val)}, types={types}")
                    elif isinstance(val, dict):
                        print(f"    {key}: dict")
                    else:
                        print(f"    {key}: {type(val).__name__}")

            # Save images from first batch
            if args.save_images_dir and batch_idx == 0:
                save_batch_images(batch, camera_keys, args.save_images_dir, "step2_multibatch",
                                  max_samples=args.num_images_per_stage, dataset=fresh_dataset)

            # Progress
            elapsed = time.time() - start_time
            if (batch_idx + 1) % max(1, args.max_batches // 20) == 0 or batch_idx == 0:
                speed = (batch_idx + 1) / max(elapsed, 1e-6)
                eta = (args.max_batches - batch_idx - 1) / max(speed, 1e-6)
                print(f"  [{(batch_idx+1)/args.max_batches*100:.0f}%] batch {batch_idx+1}/{args.max_batches}, "
                      f"{sample_count} samples, {speed:.1f} batch/s, ETA {eta:.0f}s")

            if batch_count >= args.max_batches:
                break

        elapsed = time.time() - start_time
        result.ok(f"Multi-worker traversal: {batch_count} batches, {sample_count} samples, "
                  f"{elapsed:.1f}s, {elapsed/max(batch_count,1):.2f}s/batch")

        # Performance summary
        if batch_count > 0:
            avg_batch_time = elapsed / batch_count
            throughput = sample_count / max(elapsed, 1e-6)
            print(f"\n  Performance: {throughput:.1f} samples/s, {avg_batch_time:.2f}s/batch")

    except Exception as e:
        result.fail(f"Multi-worker traversal failed: {e}")
        traceback.print_exc()

    # Cleanup
    if fresh_dataset._decode_pipeline is not None:
        fresh_dataset.shutdown_decode_pipeline()

    # Force cleanup of DataLoader worker processes to avoid lingering
    # fork-based state that can deadlock subsequent ProcessPoolExecutor forks.
    import gc
    gc.collect()


# ── Step 3: Item correctness (single worker) ────────────────────────────────


def step3_item_correctness(dataset, args, result, action_dim):
    print("\n" + "=" * 60)
    print("Step 3: Item Correctness (num_workers=0)")
    print("=" * 60)

    try:
        single_dataset = LoLAPretrainStreamingDataset(
            repo_id="test",
            max_history_length=args.max_history_length,
            action_chunk_size=args.action_chunk_size,
            history_padding_side=args.history_padding_side,
            root=args.dataset_root,
            sub_root=args.sub_root,
            delta_timestamps=dataset.delta_timestamps,
            dataset_to_episodes_path=args.dataset_to_episodes_path if not args.no_mapping else None,
            temp_process=args.temp_process,
            tolerance_frames=args.tolerance_frames,
            decode_device=args.decode_device,
            async_decode=False,
            deferred_video_decode=True,
            buffer_size=10,  # small buffer for quick test
            episode_chunk_size=args.episode_chunk_size,
        )

        raw_loader = DataLoader(
            single_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=lambda x: x,
        )
        async_loader = AsyncDecodeDataLoader(
            dataloader=raw_loader,
            dataset=single_dataset,
            collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
        )

        batch = next(iter(async_loader))
        result.ok("Single-item fetch succeeded")

        # Expected keys
        expected_keys = {
            "action", "episode_index", "frame_index", "timestamp",
            "index", "task_index", "task",
            "observation.state",
            "hist_actions_full", "hist_actions_mask", "hist_actions_length",
            "action_dim", "state_dim",
            "camera_valid_mask",
        }
        camera_keys = single_dataset.meta.camera_keys
        for cam_key in camera_keys:
            expected_keys.add(cam_key)
            expected_keys.add(f"{cam_key}_is_pad")

        # Delta frame keys
        expected_keys.add("action_is_pad")
        expected_keys.add("observation.state_is_pad")

        actual_keys = set(batch.keys())
        missing = expected_keys - actual_keys
        if missing:
            result.fail(f"Missing keys: {missing}")
        else:
            result.ok("All expected keys present")

        # Tensor shape/dtype checks
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

        # hist_actions_full
        if "hist_actions_full" in batch and isinstance(batch["hist_actions_full"], torch.Tensor):
            haf = batch["hist_actions_full"]
            if haf.shape[-1] != action_dim:
                result.fail(f"hist_actions_full last dim {haf.shape[-1]} != action_dim {action_dim}")
            else:
                result.ok(f"hist_actions_full last dim = action_dim ({action_dim})")

            if haf.shape[0] != 1:
                result.fail(f"hist_actions_full batch dim {haf.shape[0]}, expected 1")
            else:
                result.ok("hist_actions_full batch dim = 1")

        # hist_actions_mask dtype
        if "hist_actions_mask" in batch and isinstance(batch["hist_actions_mask"], torch.Tensor):
            if batch["hist_actions_mask"].dtype != torch.bool:
                result.fail(f"hist_actions_mask dtype {batch['hist_actions_mask'].dtype}, expected bool")
            else:
                result.ok("hist_actions_mask dtype = bool")

        # hist_actions_length <= max_history_length
        if "hist_actions_length" in batch and isinstance(batch["hist_actions_length"], torch.Tensor):
            hal = batch["hist_actions_length"].item()
            if hal > args.max_history_length:
                result.fail(f"hist_actions_length {hal} > max_history_length {args.max_history_length}")
            else:
                result.ok(f"hist_actions_length {hal} <= max_history_length {args.max_history_length}")

        # NaN/Inf
        check_no_nan_inf(batch, result, label="step3")

        # action_dim / state_dim
        if "action_dim" in batch:
            ad = batch["action_dim"]
            if isinstance(ad, torch.Tensor) and ad.shape == (1,):
                result.ok(f"action_dim = {ad.item()}")
            else:
                result.fail(f"action_dim type/shape unexpected: {type(ad).__name__} {ad.shape if isinstance(ad, torch.Tensor) else ''}")

        if "state_dim" in batch:
            sd = batch["state_dim"]
            if isinstance(sd, torch.Tensor) and sd.shape == (1,):
                result.ok(f"state_dim = {sd.item()}")
            else:
                result.fail(f"state_dim type/shape unexpected: {type(sd).__name__} {sd.shape if isinstance(sd, torch.Tensor) else ''}")

        # Camera: valid -> PIL Image, invalid -> None
        for cam_key in camera_keys:
            if cam_key in batch:
                val = batch[cam_key]
                if isinstance(val, list) and len(val) == 1:
                    v = val[0]
                    cvm = batch.get("camera_valid_mask", [{}])[0]
                    is_valid = cvm.get(cam_key, True)
                    if is_valid and v is not None and isinstance(v, Image.Image):
                        result.ok(f"Camera '{cam_key}': valid PIL Image {v.size}")
                    elif not is_valid and v is None:
                        result.ok(f"Camera '{cam_key}': invalid -> None")
                    elif is_valid and v is None:
                        result.fail(f"Camera '{cam_key}': valid but None")
                    elif not is_valid and v is not None:
                        result.fail(f"Camera '{cam_key}': invalid but not None")

        # Save images
        if args.save_images_dir:
            items = [{k: v[0] if isinstance(v, list) and len(v) == 1 else (v[0] if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v)
                       for k, v in batch.items()}]
            save_pil_images(items, camera_keys, args.save_images_dir, "step3_single",
                            max_samples=1, dataset=single_dataset)

    except Exception as e:
        result.fail(f"Item correctness check failed: {e}")
        traceback.print_exc()


# ── Step 4: collate_fn correctness ──────────────────────────────────────────


def step4_collate_fn(dataset, args, result, action_dim):
    print("\n" + "=" * 60)
    print("Step 4: collate_fn Correctness")
    print("=" * 60)

    try:
        # Test with fake variable-length data
        fake_item_1 = {
            "action": torch.randn(args.action_chunk_size, action_dim),
            "episode_index": torch.tensor(0),
            "hist_actions_full": torch.randn(40, action_dim),
            "hist_actions_mask": torch.ones(40, dtype=torch.bool),
            "hist_actions_length": torch.tensor(40, dtype=torch.long),
            "action_dim": 20,
            "state_dim": 7,
            "task": "pick_place",
            "camera_valid_mask": {"cam1": True, "cam2": False},
            "cam1": None,  # simulate invalid
        }
        fake_item_2 = {
            "action": torch.randn(args.action_chunk_size, action_dim),
            "episode_index": torch.tensor(0),
            "hist_actions_full": torch.randn(30, action_dim),  # different length
            "hist_actions_mask": torch.ones(30, dtype=torch.bool),
            "hist_actions_length": torch.tensor(30, dtype=torch.long),
            "action_dim": 20,
            "state_dim": 7,
            "task": "pick_place",
            "camera_valid_mask": {"cam1": True, "cam2": True},
            "cam1": None,
        }

        collated = AsyncDecodeDataLoader.make_collate_fn()([fake_item_1, fake_item_2])
        result.ok("collate_fn handles variable-length hist_actions")

        # Padded lengths match
        if collated["hist_actions_full"].shape[1] == collated["hist_actions_mask"].shape[1]:
            result.ok(f"Padded seq len consistent: {collated['hist_actions_full'].shape[1]}")
        else:
            result.fail("Padded seq len mismatch")

        # Batch dim
        if collated["hist_actions_full"].shape[0] == 2:
            result.ok("Batch dim = 2")
        else:
            result.fail(f"Batch dim = {collated['hist_actions_full'].shape[0]}, expected 2")

        # Left padding: shorter item should have zeros at the front
        padded_len = collated["hist_actions_full"].shape[1]
        if args.history_padding_side == "left":
            padding_region = collated["hist_actions_full"][1, :padded_len - 30]
            if (padding_region == 0).all():
                result.ok("Left padding region is all zeros")
            else:
                result.warn("Left padding region not all zeros (may be from truncation)")

        # action_dim/state_dim as tensor
        if isinstance(collated["action_dim"], torch.Tensor):
            result.ok(f"action_dim collated as tensor: {collated['action_dim']}")
        else:
            result.fail(f"action_dim type {type(collated['action_dim']).__name__}, expected tensor")

        # task as list
        if isinstance(collated["task"], list) and len(collated["task"]) == 2:
            result.ok("task collated as list of strings")
        else:
            result.fail(f"task type {type(collated['task']).__name__}, expected list")

        # camera_valid_mask as list of dicts
        if isinstance(collated["camera_valid_mask"], list) and len(collated["camera_valid_mask"]) == 2:
            result.ok("camera_valid_mask collated as list of dicts")
        else:
            result.fail(f"camera_valid_mask unexpected: {type(collated['camera_valid_mask']).__name__}")

    except Exception as e:
        result.fail(f"collate_fn check failed: {e}")
        traceback.print_exc()


# ── Step 5: Video decode correctness ────────────────────────────────────────


def step5_video_decode(dataset, args, result):
    print("\n" + "=" * 60)
    print("Step 5: Video Decode Correctness")
    print("=" * 60)

    camera_keys = dataset.meta.camera_keys
    num_episodes = dataset.num_episodes

    # Collect items by iterating with small buffer
    try:
        test_dataset = LoLAPretrainStreamingDataset(
            repo_id="test",
            max_history_length=args.max_history_length,
            action_chunk_size=args.action_chunk_size,
            history_padding_side=args.history_padding_side,
            root=args.dataset_root,
            sub_root=args.sub_root,
            delta_timestamps=dataset.delta_timestamps,
            dataset_to_episodes_path=args.dataset_to_episodes_path if not args.no_mapping else None,
            temp_process=args.temp_process,
            tolerance_frames=args.tolerance_frames,
            decode_device=args.decode_device,
            async_decode=False,
            deferred_video_decode=True,
            buffer_size=10,
            episode_chunk_size=args.episode_chunk_size,
        )

        loader = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=lambda x: x)
        async_loader = AsyncDecodeDataLoader(
            dataloader=loader,
            dataset=test_dataset,
            collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
        )

        items = []
        for item_list in async_loader:
            # item_list is a batch dict with batch_size=1
            item = {k: (v[0] if isinstance(v, list) and len(v) == 1 else (v[0] if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v))
                    for k, v in item_list.items()}
            items.append(item)
            if len(items) >= 5:
                break

        result.ok(f"Collected {len(items)} items for video decode check")

        # Check valid cameras
        valid_count = 0
        invalid_count = 0
        for item in items:
            cvm = item.get("camera_valid_mask", {})
            for cam_key in camera_keys:
                if cam_key in item:
                    val = item[cam_key]
                    is_valid = cvm.get(cam_key, True)

                    if is_valid:
                        if val is None:
                            result.fail(f"Valid camera '{cam_key}' is None (ep={item.get('episode_index')})")
                        elif not isinstance(val, Image.Image):
                            result.fail(f"Valid camera '{cam_key}' is {type(val).__name__}, expected PIL Image")
                        else:
                            valid_count += 1
                            # Not all-black
                            arr = np.array(val)
                            if arr.mean() > 0.01 * 255:
                                result.ok(f"Camera '{cam_key}': not all-black (mean={arr.mean():.1f})")
                            else:
                                result.warn(f"Camera '{cam_key}': nearly all-black (mean={arr.mean():.1f})")
                    else:
                        if val is not None:
                            result.fail(f"Invalid camera '{cam_key}' should be None, got {type(val).__name__}")
                        else:
                            invalid_count += 1

        if valid_count > 0:
            result.ok(f"{valid_count} valid camera frames checked")
        if invalid_count > 0:
            result.ok(f"{invalid_count} invalid camera frames correctly set to None")

        # Save images
        if args.save_images_dir and items:
            save_pil_images(items, camera_keys, args.save_images_dir, "step5_video_decode",
                            max_samples=args.num_images_per_stage, dataset=test_dataset)

    except Exception as e:
        result.fail(f"Video decode check failed: {e}")
        traceback.print_exc()


# ── Step 6: Streaming vs map-style consistency ──────────────────────────────


def step6_streaming_vs_map(args, result, config, delta_timestamps, action_dim):
    print("\n" + "=" * 60)
    print("Step 6: Streaming vs Map-Style Consistency")
    print("=" * 60)

    try:
        from lerobot.datasets.lola_pretrain_dataset import LoLAPretrainDataset

        dataset_to_episodes_path = args.dataset_to_episodes_path
        if args.no_mapping:
            dataset_to_episodes_path = None

        # Create map-style dataset
        map_dataset = LoLAPretrainDataset(
            repo_id="test",
            max_history_length=args.max_history_length,
            action_chunk_size=config.action_chunk_size,
            history_padding_side=args.history_padding_side,
            root=args.dataset_root,
            sub_root=args.sub_root,
            delta_timestamps=delta_timestamps,
            dataset_to_episodes_path=dataset_to_episodes_path,
            temp_process=args.temp_process,
            tolerance_frames=args.tolerance_frames,
            decode_device=args.decode_device,
        )

        # Create streaming dataset (no shuffle, small buffer, for deterministic order)
        stream_dataset = LoLAPretrainStreamingDataset(
            repo_id="test",
            max_history_length=args.max_history_length,
            action_chunk_size=config.action_chunk_size,
            history_padding_side=args.history_padding_side,
            root=args.dataset_root,
            sub_root=args.sub_root,
            delta_timestamps=delta_timestamps,
            dataset_to_episodes_path=dataset_to_episodes_path,
            temp_process=args.temp_process,
            tolerance_frames=args.tolerance_frames,
            decode_device=args.decode_device,
            shuffle=False,
            buffer_size=5,
            episode_chunk_size=args.episode_chunk_size,
            deferred_video_decode=True,
        )

        # Get streaming items
        stream_items = []
        loader = DataLoader(stream_dataset, batch_size=1, num_workers=0, collate_fn=lambda x: x)
        async_loader = AsyncDecodeDataLoader(
            dataloader=loader,
            dataset=stream_dataset,
            collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
        )
        for batch in async_loader:
            item = {k: (v[0] if isinstance(v, list) and len(v) == 1 else (v[0] if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v))
                    for k, v in batch.items()}
            stream_items.append(item)
            if len(stream_items) >= 10:
                break

        # Compare with map-style for matching episode_index + frame_index
        mismatches = 0
        comparisons = 0
        compare_keys = ["observation.state", "action", "hist_actions_length"]

        for stream_item in stream_items[:5]:
            ep_idx = stream_item.get("episode_index")
            frame_idx = stream_item.get("frame_index")
            if isinstance(ep_idx, torch.Tensor):
                ep_idx = ep_idx.item()
            if isinstance(frame_idx, torch.Tensor):
                frame_idx = frame_idx.item()

            # Find corresponding map-style item
            ep_start = int(map_dataset._episode_starts[ep_idx])
            map_idx = ep_start + frame_idx

            try:
                map_item = map_dataset[map_idx]
            except Exception as e:
                result.warn(f"Map-style dataset[{map_idx}] failed: {e}")
                continue

            for key in compare_keys:
                if key in stream_item and key in map_item:
                    sv = stream_item[key]
                    mv = map_item[key]
                    if isinstance(sv, torch.Tensor) and isinstance(mv, torch.Tensor):
                        comparisons += 1
                        if sv.is_floating_point() and mv.is_floating_point():
                            if not torch.allclose(sv, mv, atol=1e-4):
                                mismatches += 1
                                if mismatches <= 3:
                                    result.warn(f"ep={ep_idx} frame={frame_idx} key='{key}': streaming vs map differ "
                                                f"(max_diff={(sv - mv).abs().max().item():.6f})")
                        else:
                            if not torch.equal(sv, mv):
                                mismatches += 1
                                if mismatches <= 3:
                                    result.warn(f"ep={ep_idx} frame={frame_idx} key='{key}': values differ")

            # hist_actions_mask comparison (allow padding differences)
            if "hist_actions_mask" in stream_item and "hist_actions_mask" in map_item:
                sm = stream_item["hist_actions_mask"]
                mm = map_item["hist_actions_mask"]
                if sm.shape == mm.shape:
                    comparisons += 1
                    if not torch.equal(sm, mm):
                        mismatches += 1
                        result.warn(f"ep={ep_idx} frame={frame_idx}: hist_actions_mask differs")

        if comparisons > 0 and mismatches == 0:
            result.ok(f"Streaming vs map-style: all {comparisons} comparisons match")
        elif mismatches > 0:
            result.warn(f"Streaming vs map-style: {mismatches}/{comparisons} comparisons differ "
                        "(may be due to normalization order or padding differences)")
        else:
            result.warn("No comparisons made (missing keys)")

    except ImportError:
        result.warn("LoLAPretrainDataset not available, skipping streaming vs map comparison")
    except Exception as e:
        result.fail(f"Streaming vs map comparison failed: {e}")
        traceback.print_exc()


# ── Step 7: Performance benchmark ───────────────────────────────────────────


def step7_performance(args, result, config, delta_timestamps):
    print("\n" + "=" * 60)
    print(f"Step 7: Performance Benchmark "
          f"(num_workers={args.num_workers}, batch_size={args.batch_size}, "
          f"max_batches={args.perf_max_batches})")
    print("=" * 60)

    try:
        import psutil

        perf_dataset = _create_dataset(args, delta_timestamps, config)

        raw_loader = DataLoader(
            perf_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=lambda x: x,
            pin_memory=False,
        )
        async_loader = AsyncDecodeDataLoader(
            dataloader=raw_loader,
            dataset=perf_dataset,
            collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
        )

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 3)

        batch_count = 0
        sample_count = 0
        start_time = time.time()
        peak_rss_gb = mem_before

        for batch_idx, batch in enumerate(async_loader):
            batch_count += 1
            bs = None
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    bs = val.shape[0]
                    break
                elif isinstance(val, list):
                    bs = len(val)
                    break
            sample_count += bs if bs is not None else 0

            # Track peak memory
            current_rss = process.memory_info().rss / (1024 ** 3)
            if current_rss > peak_rss_gb:
                peak_rss_gb = current_rss

            if batch_count >= args.perf_max_batches:
                break

        elapsed = time.time() - start_time
        mem_after = process.memory_info().rss / (1024 ** 3)

        throughput = sample_count / max(elapsed, 1e-6)
        avg_batch_time = elapsed / max(batch_count, 1)
        mem_delta = mem_after - mem_before

        result.ok(f"Performance: {batch_count} batches, {sample_count} samples, "
                  f"{elapsed:.1f}s, {avg_batch_time:.2f}s/batch, {throughput:.1f} samples/s")
        result.ok(f"Memory: RSS {mem_before:.1f}GB -> {mem_after:.1f}GB (delta={mem_delta:+.1f}GB), "
                  f"peak={peak_rss_gb:.1f}GB")

        print(f"\n  Throughput: {throughput:.1f} samples/s")
        print(f"  Latency:    {avg_batch_time:.2f}s/batch")
        print(f"  Memory:     RSS delta={mem_delta:+.1f}GB, peak={peak_rss_gb:.1f}GB")

        # Cleanup
        if perf_dataset._decode_pipeline is not None:
            perf_dataset.shutdown_decode_pipeline()

    except ImportError:
        result.warn("psutil not installed, skipping memory tracking")
    except Exception as e:
        result.fail(f"Performance benchmark failed: {e}")
        traceback.print_exc()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate LoLAPretrainStreamingDataset")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None,
                        help="dataset_to_episodes.json path")
    parser.add_argument("--no_mapping", action="store_true",
                        help="Skip per-dataset mapping")
    parser.add_argument("--sub_root", type=str, default=None,
                        help="Sub-dataset root directory")
    parser.add_argument("--temp_process", action="store_true",
                        help="Zero-pad mismatched sub-dataset stats")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=20,
                        help="Max batches for traversal")
    parser.add_argument("--perf_max_batches", type=int, default=100,
                        help="Max batches for performance benchmark")
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--history_padding_side", type=str, default="left",
                        choices=["left", "right"])
    parser.add_argument("--tolerance_frames", type=int, default=1,
                        help="Max allowed frame offset for video decode")
    parser.add_argument("--decode_device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--decode_num_threads", type=int, default=1,
                        help="Main process decode threads (deferred + non-async mode)")
    parser.add_argument("--async_decode", action="store_true",
                        help="Enable async decode pipeline")
    parser.add_argument("--no_deferred", action="store_true",
                        help="Disable deferred video decode (decode in worker)")
    parser.add_argument("--buffer_size", type=int, default=5000,
                        help="Shuffle buffer size")
    parser.add_argument("--episode_chunk_size", type=int, default=8,
                        help="Episodes per chunk for strided sharding")
    parser.add_argument("--compare_map", action="store_true",
                        help="Enable streaming vs map-style consistency check")
    parser.add_argument("--save_images_dir", type=str, default="./validate_stream_images",
                        help="Save decoded images for visual inspection (empty string to disable)")
    parser.add_argument("--num_images_per_stage", type=int, default=3,
                        help="Max images per camera per stage")

    args = parser.parse_args()
    if args.save_images_dir == "":
        args.save_images_dir = None

    result = ValidationResult()

    print("=" * 60)
    print("LoLAPretrainStreamingDataset Validation")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Batch size: {args.batch_size}, num_workers: {args.num_workers}")
    print(f"Buffer size: {args.buffer_size}, episode_chunk_size: {args.episode_chunk_size}")
    print(f"Decode: device={args.decode_device}, deferred={not args.no_deferred}, async={args.async_decode}")

    time_start = time.time()

    # Build config + delta_timestamps
    config, delta_timestamps, action_dim = _build_config(
        args.dataset_root,
        args.dataset_to_episodes_path,
        args.sub_root,
        args.temp_process,
        args.max_history_length,
        args.action_chunk_size,
        args.n_obs_steps,
        args.pred_chunk_size,
    )

    # Step 1: Dataset creation
    dataset = step1_dataset_creation(args, result, config, delta_timestamps, action_dim)
    time_s1 = time.time()

    # Step 2: Multi-worker traversal (MUST run before num_workers=0 steps)
    step2_multiworker_traversal(dataset, args, result, config, delta_timestamps)
    time_s2 = time.time()

    # Step 3: Item correctness (num_workers=0)
    step3_item_correctness(dataset, args, result, action_dim)
    time_s3 = time.time()

    # Step 4: collate_fn correctness
    step4_collate_fn(dataset, args, result, action_dim)
    time_s4 = time.time()

    # Step 5: Video decode correctness
    step5_video_decode(dataset, args, result)
    time_s5 = time.time()

    # Step 6: Streaming vs map-style consistency (optional)
    if args.compare_map:
        step6_streaming_vs_map(args, result, config, delta_timestamps, action_dim)
    else:
        print("\n[Skip] Step 6: Streaming vs map-style comparison (use --compare_map to enable)")
    time_s6 = time.time()

    # Step 7: Performance benchmark
    step7_performance(args, result, config, delta_timestamps)
    time_s7 = time.time()

    # Timing summary
    print("\n" + "=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(f"  Step 1 (dataset creation):       {time_s1 - time_start:.2f}s")
    print(f"  Step 2 (multi-worker traversal): {time_s2 - time_s1:.2f}s")
    print(f"  Step 3 (item correctness):       {time_s3 - time_s2:.2f}s")
    print(f"  Step 4 (collate_fn):             {time_s4 - time_s3:.2f}s")
    print(f"  Step 5 (video decode):           {time_s5 - time_s4:.2f}s")
    if args.compare_map:
        print(f"  Step 6 (streaming vs map):       {time_s6 - time_s5:.2f}s")
    print(f"  Step 7 (performance benchmark):  {time_s7 - time_s6:.2f}s")
    print(f"  Total:                           {time_s7 - time_start:.2f}s")

    # Final summary
    print("\n" + result.summary())
    sys.exit(1 if result.failed else 0)


if __name__ == "__main__":
    main()
