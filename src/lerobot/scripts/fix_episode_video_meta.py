#!/usr/bin/env python3
"""Fix episode video metadata and generate calvin_episode_metadata.json after a crashed Phase C.

This script:
1. Reconstructs ALL video metadata (chunk_index, file_index, from/to timestamps)
   from scratch using video file durations and episode lengths
2. Generates calvin_episode_metadata.json by re-running the V2 annotation pipeline
3. Updates info.json with video info

Usage:
    python fix_episode_video_meta.py \
        --dataset_dir /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v4 \
        --calvin_input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/training/
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FPS = 30


def get_video_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def fix_video_metadata(dataset_dir: Path):
    """Reconstruct video metadata in episode parquet from scratch."""
    ep_path = dataset_dir / "meta/episodes/chunk-000/file-000.parquet"
    if not ep_path.exists():
        logger.error(f"Episode parquet not found: {ep_path}")
        sys.exit(1)

    ep_df = pd.read_parquet(ep_path)
    num_episodes = len(ep_df)
    logger.info(f"Loaded {num_episodes} episodes")

    video_keys = sorted(set(
        col.split("/")[1] for col in ep_df.columns
        if col.startswith("videos/") and "/chunk_index" in col
    ))
    logger.info(f"Video keys: {video_keys}")

    ep_duration_s = ep_df["length"] / FPS

    for vk in video_keys:
        video_dir = dataset_dir / "videos" / vk / "chunk-000"
        if not video_dir.exists():
            logger.error(f"Video directory not found: {video_dir}")
            sys.exit(1)

        video_files = sorted(video_dir.glob("file-*.mp4"))
        logger.info(f"  {vk}: {len(video_files)} video files")

        file_durations = [get_video_duration(str(vf)) for vf in video_files]
        total_video_dur = sum(file_durations)
        total_ep_dur = ep_duration_s.sum()
        logger.info(f"  {vk}: total video = {total_video_dur:.1f}s, total episodes = {total_ep_dur:.1f}s")

        if abs(total_video_dur - total_ep_dur) > 1.0:
            logger.warning(f"  {vk}: duration mismatch! video={total_video_dur:.1f}s vs episodes={total_ep_dur:.1f}s")

        file_idx = 0
        from_ts = 0.0

        for i in range(num_episodes):
            ep_dur = ep_duration_s.iloc[i]
            to_ts = from_ts + ep_dur

            ep_df.at[ep_df.index[i], f"videos/{vk}/chunk_index"] = 0
            ep_df.at[ep_df.index[i], f"videos/{vk}/file_index"] = file_idx
            ep_df.at[ep_df.index[i], f"videos/{vk}/from_timestamp"] = float(from_ts)
            ep_df.at[ep_df.index[i], f"videos/{vk}/to_timestamp"] = float(to_ts)

            from_ts = to_ts

            if file_idx < len(file_durations) and from_ts >= file_durations[file_idx] - 0.01:
                file_idx += 1
                from_ts = 0.0

        still_missing = ep_df[f"videos/{vk}/from_timestamp"].isna().sum()
        last_ep_fi = int(ep_df.iloc[-1][f"videos/{vk}/file_index"])
        logger.info(f"  {vk}: missing = {still_missing}, last episode in file-{last_ep_fi}")

    # Ensure chunk_index and file_index are int, not float
    for col in ep_df.columns:
        if col.startswith("videos/") and ("chunk_index" in col or "file_index" in col):
            ep_df[col] = ep_df[col].astype(int)

    ep_df.to_parquet(ep_path)
    logger.info(f"Saved updated episode metadata to {ep_path}")

    # Update info.json
    info_path = dataset_dir / "meta/info.json"
    with open(info_path) as f:
        info = json.load(f)

    if "videos" not in info:
        info["videos"] = {}

    for vk in video_keys:
        video_dir = dataset_dir / "videos" / vk / "chunk-000"
        video_files = sorted(video_dir.glob("file-*.mp4"))
        total_duration = sum(get_video_duration(str(vf)) for vf in video_files)

        info["videos"][vk] = {
            "chunk_index": 0,
            "file_indices": list(range(len(video_files))),
            "total_duration": total_duration,
        }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Updated {info_path} with video info")


def generate_calvin_metadata(dataset_dir: Path, calvin_input_dir: str, max_transition_len: int = 64):
    """Generate calvin_episode_metadata.json by re-running the V2 annotation pipeline."""
    from convert_calvin_to_lerobot_v2 import (
        load_trajectory_boundaries,
        load_lang_annotations,
        assign_annotations_to_trajectories,
        resolve_overlaps,
        build_episode_definitions,
    )

    logger.info(f"Loading Calvin annotations from {calvin_input_dir}...")
    trajectories = load_trajectory_boundaries(calvin_input_dir)
    logger.info(f"  Found {len(trajectories)} trajectories")

    annotations = load_lang_annotations(calvin_input_dir)
    logger.info(f"  Found {len(annotations)} annotations")

    annotations = assign_annotations_to_trajectories(annotations, trajectories)
    annotations = resolve_overlaps(annotations)
    episodes = build_episode_definitions(annotations, max_transition_len)
    logger.info(f"  Built {len(episodes)} episode definitions")

    metadata = {}
    for ep_idx, ep in enumerate(episodes):
        metadata[str(ep_idx)] = {
            "transition_len": ep["transition_len"],
            "annotation_len": ep["annotation_len"],
            "completed_tasks": ep["completed_tasks"],
            "completed_tasks_ann_choices": ep["completed_tasks_ann_choices"],
            "calvin_trajectory_index": ep["calvin_trajectory_index"],
        }

    metadata_path = dataset_dir / "calvin_episode_metadata.json"
    with open(str(metadata_path), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote {len(metadata)} episode entries to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix episode video metadata after crashed Phase C")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the LeRobot dataset directory")
    parser.add_argument("--calvin_input_dir", type=str, default=None,
                        help="Path to original Calvin data (needed for calvin_episode_metadata.json). "
                             "If not provided, only video metadata is fixed.")
    parser.add_argument("--max_transition_len", type=int, default=64,
                        help="Max transition frames (must match original conversion)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Step 1: Fix video metadata
    logger.info("=== Step 1: Reconstructing video metadata ===")
    fix_video_metadata(dataset_dir)

    # Step 2: Generate calvin_episode_metadata.json
    if args.calvin_input_dir:
        logger.info("=== Step 2: Generating calvin_episode_metadata.json ===")
        generate_calvin_metadata(dataset_dir, args.calvin_input_dir, args.max_transition_len)
    else:
        logger.info("=== Step 2: Skipping calvin_episode_metadata.json (no --calvin_input_dir) ===")

    logger.info("All fixes complete!")


if __name__ == "__main__":
    main()
