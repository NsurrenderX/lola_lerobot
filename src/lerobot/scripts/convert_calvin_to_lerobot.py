#!/usr/bin/env python
"""Convert original CALVIN dataset to LeRobot v3.0 format (parallel version).

Reads the original CALVIN npz files and language annotations, then writes
a LeRobot v3.0 dataset with:
  - observation.images.rgb_static  (200x200x3 video)
  - observation.images.rgb_gripper (84x84x3 video)
  - observation.state (7-dim: xyz + rpy + gripper, NOT normalized)
  - action (7-dim: rel_actions from CALVIN, optionally normalized)

Episode boundaries are determined by language annotation index pairs
from `auto_lang_ann.npy`, matching the original RoboVLM training setup.

Parallelism:
  - Phase 1: NPZ reading + image saving via ProcessPoolExecutor (I/O bound)
  - Phase 2: Sequential save_episode on main thread (parquet + metadata)
  - Phase 3: Parallel video encoding + custom sequential file management

Usage:
    # Validation set
    python convert_calvin_to_lerobot.py \
        --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/validation \
        --output_dir /data_6t_2/lerobot_v30/calvin_task_ABC_D_validation_v3 \
        --repo_id calvin_task_ABC_D_validation

    # Training set with custom workers
    python convert_calvin_to_lerobot.py \
        --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/training \
        --output_dir /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v3 \
        --repo_id calvin_task_ABC_D_training \
        --num_workers 32 --video_encode_workers 16
"""

import argparse
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset, _encode_video_worker
from lerobot.datasets.utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_IMAGE_PATH,
    get_file_size_in_mb,
    load_episodes,
    update_chunk_file_indices,
)
from lerobot.datasets.video_utils import (
    concatenate_video_files,
    get_video_duration_in_s,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FPS = 30

# Video keys in the output dataset (must match build_features)
VIDEO_KEYS = [
    "observation.images.rgb_static",
    "observation.images.rgb_gripper",
]


def build_features() -> dict:
    return {
        "observation.images.rgb_static": {
            "dtype": "video",
            "shape": (200, 200, 3),
            "names": ["height", "width", "rgb"],
            "info": {
                "video.fps": float(FPS),
                "video.height": 200,
                "video.width": 200,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.images.rgb_gripper": {
            "dtype": "video",
            "shape": (84, 84, 3),
            "names": ["height", "width", "rgb"],
            "info": {
                "video.fps": float(FPS),
                "video.height": 84,
                "video.width": 84,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            "fps": FPS,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            "fps": FPS,
        },
    }


def load_lang_annotations(input_dir: str):
    """Load language annotations and return sorted episode definitions."""
    lang_path = os.path.join(input_dir, "lang_annotations", "auto_lang_ann.npy")
    if not os.path.exists(lang_path):
        raise FileNotFoundError(f"Language annotations not found at {lang_path}")

    lang_data = np.load(lang_path, allow_pickle=True).item()
    annotations = lang_data["language"]["ann"]
    tasks = lang_data["language"]["task"]
    indx = lang_data["info"]["indx"]

    episodes = []
    for i, (start, end) in enumerate(indx):
        start = int(start)
        end = int(end)
        episodes.append({
            "start": start,
            "end": end,
            "annotation": annotations[i],
            "task": tasks[i],
        })

    return episodes


def normalize_action(action: np.ndarray, action_min: float = -0.65, action_max: float = 0.65) -> np.ndarray:
    """Clip action to [action_min, action_max] and linearly map to [-1, 1].
    Preserves the last dimension (gripper) as-is.
    """
    last_val = action[..., -1].copy()
    action = np.clip(action, a_min=action_min, a_max=action_max)
    action = 2.0 * (action - action_min) / (action_max - action_min) - 1.0
    action[..., -1] = last_val
    return action


def process_episode_worker(
    ep_idx: int,
    ep_info: dict,
    input_dir: str,
    output_dir: str,
    skip_norm: bool,
    norm_min: float,
    norm_max: float,
    features: dict,
    fps: int,
):
    """Worker function for parallel NPZ reading + image saving.

    Reads npz files for one episode, processes actions/state, saves images
    as PNG, and constructs an episode_data dict matching the format expected
    by LeRobotDataset.save_episode(episode_data=...).

    Returns (ep_idx, episode_data) on success, or (ep_idx, None) on failure.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    start = ep_info["start"]
    end = ep_info["end"]
    task = ep_info["annotation"]

    # Build episode_data dict matching create_episode_buffer() format
    episode_data = {
        "size": 0,
        "task": [],
        "episode_index": ep_idx,
    }
    for key in features:
        if key == "episode_index":
            continue
        episode_data[key] = []
    # Default features that create_episode_buffer includes
    episode_data["frame_index"] = []
    episode_data["timestamp"] = []
    episode_data["index"] = []  # will be overwritten by save_episode
    episode_data["task_index"] = []  # will be overwritten by save_episode

    num_frames = end - start
    for frame_offset in range(num_frames):
        npz_idx = start + frame_offset
        npz_path = input_dir / f"episode_{npz_idx:07d}.npz"

        if not npz_path.exists():
            return (ep_idx, None)

        data = np.load(str(npz_path), allow_pickle=True)

        # Action
        action = data["rel_actions"].astype(np.float32)
        if not skip_norm:
            action = normalize_action(action, norm_min, norm_max)

        # State
        state = data["robot_obs"][:7].astype(np.float32)

        # Frame index and timestamp
        frame_index = frame_offset
        timestamp = frame_offset / fps

        episode_data["frame_index"].append(frame_index)
        episode_data["timestamp"].append(timestamp)
        episode_data["task"].append(task)

        # Scalar features
        episode_data["observation.state"].append(state)
        episode_data["action"].append(action)

        # Video features: save images as PNG
        rgb_static = data["rgb_static"]  # [200, 200, 3] uint8
        rgb_gripper = data["rgb_gripper"]  # [84, 84, 3] uint8

        img_static = Image.fromarray(rgb_static)
        img_gripper = Image.fromarray(rgb_gripper)

        for video_key, img in [
            ("observation.images.rgb_static", img_static),
            ("observation.images.rgb_gripper", img_gripper),
        ]:
            img_path = output_dir / DEFAULT_IMAGE_PATH.format(
                image_key=video_key, episode_index=ep_idx, frame_index=frame_index
            )
            if frame_offset == 0:
                img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(img_path), compress_level=1)
            episode_data[video_key].append(str(img_path))

    episode_data["size"] = num_frames

    return (ep_idx, episode_data)


def encode_videos_parallel(dataset, video_encode_workers: int):
    """Parallel video encoding + batch file management.

    Phase A: Encode all episode videos in parallel using ProcessPoolExecutor.
    Phase B: Scan temp MP4 sizes, plan file boundaries, then concatenate
             per-group (much faster than per-episode concatenation).
    Phase C: Update episode parquet with video metadata.
    """
    num_episodes = dataset.num_episodes
    video_keys = dataset.meta.video_keys

    if num_episodes == 0 or len(video_keys) == 0:
        logger.info("No episodes or video keys — skipping video encoding")
        return

    # Phase A: Parallel encoding
    logger.info(f"Phase A: Encoding {num_episodes * len(video_keys)} videos "
                f"with {video_encode_workers} workers...")
    with ProcessPoolExecutor(max_workers=video_encode_workers) as executor:
        encode_futures = {}
        for vk in video_keys:
            for ei in range(num_episodes):
                f = executor.submit(_encode_video_worker, vk, ei, dataset.root, FPS)
                encode_futures[f] = (vk, ei)

        encode_results = {}
        done_count = 0
        total = len(encode_futures)
        for future in as_completed(encode_futures):
            vk, ei = encode_futures[future]
            try:
                encode_results[(vk, ei)] = future.result()
            except Exception as exc:
                logger.error(f"Video encoding failed for {vk}/episode_{ei}: {exc}")
                raise
            done_count += 1
            if done_count % 1000 == 0 or done_count == total:
                logger.info(f"  Encoded {done_count}/{total} videos")

    logger.info(f"All {total} videos encoded")

    # Phase B: Scan sizes + plan boundaries + batch concatenate
    logger.info("Phase B: Planning file boundaries and assembling videos...")

    video_files_size_mb = dataset.meta.video_files_size_in_mb
    chunks_size = dataset.meta.chunks_size

    # Collect size and duration for all temp MP4s
    ep_sizes = {}   # (vk, ei) -> size_in_mb
    ep_durations = {}  # (vk, ei) -> duration_in_s
    for vk in video_keys:
        for ei in range(num_episodes):
            temp_path = encode_results[(vk, ei)]
            ep_sizes[(vk, ei)] = get_file_size_in_mb(temp_path)
            ep_durations[(vk, ei)] = get_video_duration_in_s(temp_path)

    # Plan file boundaries and assemble per video_key
    video_meta_map = {}

    for vk in video_keys:
        # Step 1: Plan which episodes go into which file
        file_groups = []  # list of (chunk_idx, file_idx, [episode_indices])
        chunk_idx, file_idx = 0, 0
        cumulative_size = 0.0
        group = []

        for ei in range(num_episodes):
            ep_size = ep_sizes[(vk, ei)]
            if cumulative_size + ep_size >= video_files_size_mb and len(group) > 0:
                file_groups.append((chunk_idx, file_idx, group))
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunks_size)
                cumulative_size = 0.0
                group = []
            group.append(ei)
            cumulative_size += ep_size

        if group:
            file_groups.append((chunk_idx, file_idx, group))

        # Step 2: Compute video metadata (from_timestamp, to_timestamp)
        for ck, fi, ep_indices in file_groups:
            from_ts = 0.0
            for gei in ep_indices:
                ep_dur = ep_durations[(vk, gei)]
                video_meta_map[(vk, gei)] = {
                    f"videos/{vk}/chunk_index": ck,
                    f"videos/{vk}/file_index": fi,
                    f"videos/{vk}/from_timestamp": from_ts,
                    f"videos/{vk}/to_timestamp": from_ts + ep_dur,
                }
                from_ts += ep_dur

        # Step 3: Assemble video files — one concatenate per file group
        for ck, fi, ep_indices in file_groups:
            video_path = dataset.root / dataset.meta.video_path.format(
                video_key=vk, chunk_index=ck, file_index=fi
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)

            temp_paths = [encode_results[(vk, ei)] for ei in ep_indices]

            if len(temp_paths) == 1:
                shutil.move(str(temp_paths[0]), str(video_path))
            else:
                # Move first as base, then concatenate all at once
                shutil.move(str(temp_paths[0]), str(video_path))
                concatenate_video_files([video_path] + temp_paths[1:], video_path)

            # Cleanup temp dirs
            for tp in temp_paths:
                td = tp.parent
                if td.exists():
                    shutil.rmtree(str(td))

        logger.info(f"  {vk}: {len(file_groups)} video files for {num_episodes} episodes")

    # Phase C: Update episode parquet with video metadata
    logger.info("Phase C: Updating episode metadata with video info...")

    # Flush and close the ParquetWriter so we can read the episode parquet files
    dataset.meta._close_writer()

    # Group episodes by their parquet file, update in batch
    episodes = load_episodes(dataset.root)
    chunk_file_episodes = {}

    for ei in range(num_episodes):
        ep = episodes[ei]
        ck = ep["data/chunk_index"]
        fi = ep["data/file_index"]
        key = (ck, fi)
        if key not in chunk_file_episodes:
            chunk_file_episodes[key] = []
        chunk_file_episodes[key].append(ei)

    for (ck, fi), ep_indices in chunk_file_episodes.items():
        episode_df_path = dataset.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=ck, file_index=fi
        )
        episode_df = pd.read_parquet(episode_df_path)

        for ei in ep_indices:
            for vk in video_keys:
                meta = video_meta_map.get((vk, ei), {})
                for col, val in meta.items():
                    if col in episode_df.columns:
                        episode_df.at[ei, col] = val
                    else:
                        episode_df[col] = None
                        episode_df.at[ei, col] = val

        episode_df.to_parquet(episode_df_path)

    # Update video info from first episode
    for vk in video_keys:
        dataset.meta.update_video_info(vk)

    # Reload episodes metadata
    dataset.meta.episodes = load_episodes(dataset.root)
    logger.info("Video encoding complete")


def convert_dataset(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    skip_norm: bool = True,
    norm_min: float = -0.65,
    norm_max: float = 0.65,
    num_workers: int | None = None,
    video_encode_workers: int | None = None,
):
    """Convert CALVIN dataset to LeRobot v3.0 format with parallel processing."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if num_workers is None:
        num_workers = os.cpu_count() or 4
    if video_encode_workers is None:
        video_encode_workers = max(1, (os.cpu_count() or 4) // 2)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Skip normalization: {skip_norm}")
    logger.info(f"NPZ workers: {num_workers}, Video encode workers: {video_encode_workers}")

    # Load episode definitions from language annotations
    episodes = load_lang_annotations(str(input_dir))
    logger.info(f"Found {len(episodes)} episodes from language annotations")

    # Create LeRobot dataset with deferred video encoding
    features = build_features()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=str(output_dir),
        robot_type="franka",
        use_videos=True,
        batch_encoding_size=len(episodes) + 1,  # defer ALL video encoding
    )

    # Phase 1+2: Parallel NPZ reading + sequential save_episode
    logger.info(f"Phase 1+2: Processing {len(episodes)} episodes "
                f"({num_workers} NPZ workers, sequential save)...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for ep_idx, ep_info in enumerate(episodes):
            f = executor.submit(
                process_episode_worker,
                ep_idx,
                ep_info,
                str(input_dir),
                str(output_dir),
                skip_norm,
                norm_min,
                norm_max,
                features,
                FPS,
            )
            futures[f] = ep_idx

        # Collect results and save in episode_index order
        pending = {}
        next_ep = 0
        total_saved = 0
        for future in as_completed(futures):
            ep_idx, episode_data = future.result()
            pending[ep_idx] = episode_data

            # Save any episodes that are ready (in order)
            while next_ep in pending:
                episode_data = pending.pop(next_ep)
                if episode_data is not None:
                    dataset.save_episode(episode_data=episode_data, parallel_encoding=False)
                    total_saved += 1
                next_ep += 1
                if next_ep % 1000 == 0:
                    logger.info(f"  Saved {next_ep}/{len(episodes)} episodes (parquet)")

    logger.info(f"Phase 1+2 complete: {total_saved} episodes saved to parquet")

    # Phase 3: Parallel video encoding
    encode_videos_parallel(dataset, video_encode_workers)

    # Finalize
    dataset.stop_image_writer()
    dataset.meta._close_writer()
    logger.info(f"Done! Total: {total_saved} episodes")


def main():
    parser = argparse.ArgumentParser(description="Convert CALVIN dataset to LeRobot v3.0 format")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to original CALVIN directory (containing episode_*.npz and lang_annotations/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for LeRobot v3.0 dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Dataset repo ID (e.g. calvin_task_ABC_D_training)",
    )
    parser.add_argument(
        "--skip_norm",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        help="Skip action normalization (default: True, preserves original rel_actions range)",
    )
    parser.add_argument(
        "--norm_min",
        type=float,
        default=-0.65,
        help="Action normalization min (used when skip_norm=False)",
    )
    parser.add_argument(
        "--norm_max",
        type=float,
        default=0.65,
        help="Action normalization max (used when skip_norm=False)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers for NPZ reading (default: os.cpu_count())",
    )
    parser.add_argument(
        "--video_encode_workers",
        type=int,
        default=None,
        help="Number of parallel workers for video encoding (default: os.cpu_count() // 2)",
    )
    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        skip_norm=args.skip_norm,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
        num_workers=args.num_workers,
        video_encode_workers=args.video_encode_workers,
    )


if __name__ == "__main__":
    main()
