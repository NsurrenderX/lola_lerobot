#!/usr/bin/env python
"""Resume CALVIN dataset conversion — finish video assembly and update metadata.

Handles the case where Phase 1+2 (parquet + PNG) completed but Phase 3 (video
encoding/assembly) was interrupted. Uses one-shot PyAV remux for fast assembly
instead of slow incremental concatenation.

Phases:
  A: Verify parquet integrity
  B: Assemble remaining videos (only for keys with un-assembled temp MP4s)
  C: Reconstruct video metadata for ALL keys from assembled files
  D: Update episode parquet metadata for both keys
  E: Final verification (load full LeRobotDataset)

Usage:
    python resume_calvin_assembly.py \
        --output_dir /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v3 \
        --repo_id calvin_task_ABC_D_training
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import av
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
    get_file_size_in_mb,
    load_episodes,
    update_chunk_file_indices,
    write_info,
)
from lerobot.datasets.video_utils import get_video_duration_in_s

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FPS = 30

VIDEO_KEYS = [
    "observation.images.rgb_static",
    "observation.images.rgb_gripper",
]


# ─── Core merge function ───


def one_shot_merge(input_paths: list[Path], output_path: Path, show_progress=True):
    """Merge multiple MP4 files into one using PyAV concat demuxer (stream copy).

    Uses ffmpeg's concat demuxer which automatically adjusts DTS/PTS across
    inputs — avoids "non monotonically increasing dts" error.
    """
    import tempfile

    tmp_output = Path(tempfile.mktemp(suffix=".mp4", dir=str(output_path.parent)))

    # Create .ffconcat file listing all inputs
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as f:
        f.write("ffconcat version 1.0\n")
        for input_path in input_paths:
            f.write(f"file '{str(input_path.resolve())}'\n")
        f.flush()
        concat_path = f.name

    input_container = av.open(concat_path, mode="r", format="concat", options={"safe": "0"})
    output_container = av.open(str(tmp_output), mode="w", options={"movflags": "faststart"})

    stream_map = {}
    for input_stream in input_container.streams:
        if input_stream.type in ("video", "audio", "subtitle"):
            out_stream = output_container.add_stream_from_template(
                template=input_stream, opaque=True
            )
            out_stream.time_base = input_stream.time_base
            stream_map[input_stream.index] = out_stream

    total_packets = 0
    pbar = tqdm(desc=f"Merging {output_path.name}", leave=False, unit="pkt") if show_progress else None
    for packet in input_container.demux():
        if packet.stream.index not in stream_map:
            continue
        if packet.dts is None:
            continue
        packet.stream = stream_map[packet.stream.index]
        output_container.mux(packet)
        total_packets += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    input_container.close()
    output_container.close()
    Path(concat_path).unlink()

    if output_path.exists():
        output_path.unlink()
    shutil.move(str(tmp_output), str(output_path))

    logger.info(f"  {output_path.name}: {total_packets} packets from {len(input_paths)} inputs")


# ─── Phase A: Parquet integrity ───


def verify_parquet_integrity(root: Path) -> bool:
    """Check all parquet files readable. Return True if all OK."""
    root = Path(root)
    ok_count = 0
    corrupt = []
    for label, pattern in [("data", "data/**/*.parquet"), ("episodes", "meta/episodes/**/*.parquet")]:
        for p in sorted(root.glob(pattern)):
            try:
                pq.read_table(p)
                ok_count += 1
            except Exception as e:
                corrupt.append((str(p), str(e)))

    logger.info(f"Parquet integrity: {ok_count} OK, {len(corrupt)} corrupt")
    for path, err in corrupt:
        logger.error(f"  CORRUPT: {path} — {err}")
    return len(corrupt) == 0


# ─── Helpers ───


def collect_temp_mp4s(root: Path) -> dict:
    """Scan all tmp* directories, collect (video_key, episode_index) -> Path."""
    all_temp = {}
    skipped = 0
    for tmp_dir in sorted(root.glob("tmp*")):
        if not tmp_dir.is_dir():
            continue
        for mp4 in tmp_dir.glob("*.mp4"):
            name = mp4.stem
            parts = name.rsplit("_", 1)
            vk = parts[0]
            ei = int(parts[1])
            if mp4.stat().st_size > 0:
                all_temp[(vk, ei)] = mp4
            else:
                skipped += 1
                logger.warning(f"Empty temp MP4: {mp4} — cleaning up")
                shutil.rmtree(str(tmp_dir), ignore_errors=True)

    logger.info(f"Found {len(all_temp)} valid temp MP4s, {skipped} empty/corrupt skipped")
    return all_temp


def get_episode_lengths(episodes) -> dict:
    """Get episode_index -> length (frames) from HF Dataset."""
    lengths = {}
    for i in range(len(episodes)):
        ei = episodes[i]["episode_index"]
        lengths[ei] = episodes[i]["length"]
    return lengths


# ─── Phase B: Assemble remaining videos ───


def assemble_remaining_videos(meta, all_temp_mp4s: dict) -> dict:
    """Assemble remaining videos for keys with un-assembled temp MP4s.

    Only runs assembly if there are temp MP4s for a key that haven't been
    assembled yet. Returns video metadata map for assembled episodes.
    """
    video_meta_map = {}
    root = meta.root
    num_episodes = meta.total_episodes
    target_size_mb = meta.video_files_size_in_mb
    chunks_size = meta.chunks_size
    video_path_template = meta.video_path

    meta._close_writer()
    episodes = load_episodes(root)
    ep_lengths = get_episode_lengths(episodes)

    for vk in VIDEO_KEYS:
        videos_dir = root / "videos" / vk
        existing_files = sorted(videos_dir.glob("**/*.mp4")) if videos_dir.exists() else []

        # Count how many episodes have temp MP4s for this key
        vk_temp_count = sum(1 for (k, _) in all_temp_mp4s if k == vk)

        if vk_temp_count == 0:
            logger.info(f"{vk}: no remaining temp MP4s — skipping assembly")
            continue

        # Determine which episodes are already assembled
        if existing_files:
            existing_file_dur = get_video_duration_in_s(existing_files[0])
            cum_dur = 0.0
            last_assembled = -1
            for ei in range(num_episodes):
                ep_dur = ep_lengths[ei] / FPS
                cum_dur += ep_dur
                if cum_dur <= existing_file_dur + 0.5:
                    last_assembled = ei
                else:
                    break
            logger.info(f"{vk}: existing file-000 has {last_assembled + 1} episodes")
        else:
            last_assembled = -1

        # Plan file boundaries
        ck, fi = 0, 0
        file_groups = []
        group = list(range(0, last_assembled + 1))
        cumulative_size = get_file_size_in_mb(existing_files[0]) if existing_files else 0.0

        for ei in range(last_assembled + 1, num_episodes):
            if (vk, ei) not in all_temp_mp4s:
                continue
            ep_size = get_file_size_in_mb(all_temp_mp4s[(vk, ei)])
            if cumulative_size + ep_size >= target_size_mb and len(group) > 0:
                file_groups.append((ck, fi, group))
                ck, fi = update_chunk_file_indices(ck, fi, chunks_size)
                cumulative_size = 0.0
                group = []
            group.append(ei)
            cumulative_size += ep_size

        if group:
            file_groups.append((ck, fi, group))

        logger.info(f"{vk}: planned {len(file_groups)} file groups")

        # Assemble each file group
        for ck_i, fi_i, ep_indices in tqdm(file_groups, desc=f"Assembling {vk}"):
            video_path = root / video_path_template.format(
                video_key=vk, chunk_index=ck_i, file_index=fi_i
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)

            input_paths = []
            if ck_i == 0 and fi_i == 0 and existing_files:
                input_paths.append(existing_files[0])
            for ei in ep_indices:
                if ei <= last_assembled:
                    continue
                if (vk, ei) in all_temp_mp4s:
                    input_paths.append(all_temp_mp4s[(vk, ei)])

            if not input_paths:
                continue

            one_shot_merge(input_paths, video_path, show_progress=True)

            from_ts = 0.0
            for ei in ep_indices:
                ep_dur = ep_lengths[ei] / FPS
                video_meta_map[(vk, ei)] = {
                    f"videos/{vk}/chunk_index": ck_i,
                    f"videos/{vk}/file_index": fi_i,
                    f"videos/{vk}/from_timestamp": from_ts,
                    f"videos/{vk}/to_timestamp": from_ts + ep_dur,
                }
                from_ts += ep_dur

        # Clean up temp MP4s for this key
        for (vk_key, ei), mp4_path in all_temp_mp4s.items():
            if vk_key == vk and mp4_path.exists():
                mp4_path.unlink()

    # Clean up empty tmp dirs
    for tmp_dir in sorted(root.glob("tmp*")):
        if tmp_dir.is_dir() and not any(tmp_dir.iterdir()):
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return video_meta_map


# ─── Phase C: Reconstruct video metadata from assembled files ───


def reconstruct_video_metadata(meta) -> dict:
    """Reconstruct video metadata for ALL video keys from assembled files.

    Matches episode durations against assembled video file durations to
    determine which episodes are in which file, then computes
    chunk_index, file_index, from_timestamp, to_timestamp.
    """
    root = meta.root
    num_episodes = meta.total_episodes

    meta._close_writer()
    episodes = load_episodes(root)
    ep_lengths = get_episode_lengths(episodes)

    video_meta_map = {}

    for vk in VIDEO_KEYS:
        videos_dir = root / "videos" / vk
        video_files = sorted(videos_dir.glob("**/*.mp4")) if videos_dir.exists() else []

        if not video_files:
            logger.warning(f"{vk}: no assembled video files found — skipping")
            continue

        file_info = []
        for vf in video_files:
            # Only process files matching expected pattern: file-{fi:03d}.mp4
            if not vf.stem.startswith("file-"):
                logger.warning(f"Skipping non-standard video file: {vf}")
                continue
            ck = int(vf.parent.name.split("-")[1])
            fi = int(vf.stem.split("-")[1])
            dur = get_video_duration_in_s(vf)
            size = get_file_size_in_mb(vf)
            file_info.append((ck, fi, dur, size, vf))

        # Assign episodes to files by cumulative duration matching
        assigned_total = 0
        for ck, fi, file_dur, file_size, vf in file_info:
            cum_dur = 0.0
            assigned = 0
            for ei in range(num_episodes):
                if (vk, ei) in video_meta_map:
                    continue
                ep_dur = ep_lengths[ei] / FPS
                if cum_dur + ep_dur <= file_dur + 0.5:
                    video_meta_map[(vk, ei)] = {
                        f"videos/{vk}/chunk_index": ck,
                        f"videos/{vk}/file_index": fi,
                        f"videos/{vk}/from_timestamp": cum_dur,
                        f"videos/{vk}/to_timestamp": cum_dur + ep_dur,
                    }
                    cum_dur += ep_dur
                    assigned += 1
                else:
                    break
            assigned_total += assigned
            logger.info(f"  {vk} chunk-{ck:03d}/file-{fi:03d}: {assigned} eps, "
                         f"{cum_dur:.1f}s/{file_dur:.1f}s, {file_size:.1f}MB")

        logger.info(f"{vk}: reconstructed metadata for {assigned_total}/{num_episodes} episodes")

        missing = [ei for ei in range(num_episodes) if (vk, ei) not in video_meta_map]
        if missing:
            logger.warning(f"{vk}: {len(missing)} episodes not assigned! "
                           f"Range: {missing[0]}-{missing[-1]}")

    return video_meta_map


# ─── Phase D: Update episode metadata ───


def update_episode_video_metadata(meta, video_meta_map: dict):
    """Write video columns into episode parquet files for both keys."""
    meta._close_writer()

    episodes = load_episodes(meta.root)
    num_episodes = meta.total_episodes

    # Group episodes by their meta/episodes parquet file
    chunk_file_episodes = {}
    for i in range(len(episodes)):
        ei = episodes[i]["episode_index"]
        ck = episodes[i]["meta/episodes/chunk_index"]
        fi = episodes[i]["meta/episodes/file_index"]
        key = (ck, fi)
        if key not in chunk_file_episodes:
            chunk_file_episodes[key] = []
        chunk_file_episodes[key].append(ei)

    logger.info(f"Updating {len(chunk_file_episodes)} episode parquet files")

    for (ck, fi), ep_indices in tqdm(chunk_file_episodes.items(), desc="Updating episode metadata"):
        episode_df_path = meta.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=ck, file_index=fi
        )
        episode_df = pd.read_parquet(episode_df_path)
        episode_df = episode_df.set_index("episode_index")

        # Remove any existing video columns (clean slate)
        video_cols = [col for col in episode_df.columns if col.startswith("videos/")]
        if video_cols:
            episode_df = episode_df.drop(columns=video_cols)

        # Add fresh video columns with values for all episodes
        for ei in ep_indices:
            for vk in VIDEO_KEYS:
                meta_dict = video_meta_map.get((vk, ei), {})
                for col, val in meta_dict.items():
                    episode_df.loc[ei, col] = val

        # Cast chunk_index/file_index to int (avoid float format errors)
        for vk in VIDEO_KEYS:
            for suffix in ["chunk_index", "file_index"]:
                col = f"videos/{vk}/{suffix}"
                if col in episode_df.columns:
                    episode_df[col] = episode_df[col].astype("Int64")

        episode_df = episode_df.reset_index()
        episode_df.to_parquet(episode_df_path)

    # Update video info from first episode
    for vk in VIDEO_KEYS:
        meta.update_video_info(vk)

    write_info(meta.info, meta.root)
    meta.episodes = load_episodes(meta.root)
    logger.info("Episode metadata updated")


# ─── Phase E: Final verification ───


def verify_final_dataset(output_dir: Path, repo_id: str):
    """Load and verify the final dataset."""
    ds = LeRobotDataset(repo_id, root=str(output_dir))
    print(f"Episodes: {ds.num_episodes}, Frames: {ds.num_frames}")
    print(f"Video keys: {ds.meta.video_keys}")

    for ei in [0, ds.num_episodes // 2, ds.num_episodes - 1]:
        ep = ds[ei]
        print(f"Episode {ei}: keys={sorted(ep.keys())}")


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(description="Resume CALVIN video assembly and metadata update")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Phase A: Verify parquet integrity
    logger.info("Phase A: Verifying parquet integrity...")
    if not verify_parquet_integrity(output_dir):
        logger.error("Parquet corruption detected — cannot proceed.")
        sys.exit(1)

    # Load metadata (NOT full LeRobotDataset — video columns may be missing/broken)
    logger.info("Loading dataset metadata...")
    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=str(output_dir))
    num_episodes = meta.total_episodes
    logger.info(f"Dataset: {num_episodes} episodes, {meta.total_frames} frames, "
                 f"video_keys={meta.video_keys}")
    meta._close_writer()

    # Collect all temp MP4s
    logger.info("Scanning temp MP4 files...")
    all_temp_mp4s = collect_temp_mp4s(output_dir)

    # Phase B: Assemble remaining videos (if any un-assembled temp MP4s)
    has_remaining = any(k == vk for (k, _) in all_temp_mp4s for vk in VIDEO_KEYS)
    if has_remaining:
        logger.info("Phase B: Assembling remaining videos...")
        assemble_meta = assemble_remaining_videos(meta, all_temp_mp4s)
    else:
        logger.info("Phase B: No remaining temp MP4s — skipping assembly")
        assemble_meta = {}

    # Phase C: Reconstruct video metadata for ALL keys from assembled files
    logger.info("Phase C: Reconstructing video metadata for all keys...")
    reconstruct_meta = reconstruct_video_metadata(meta)

    # Merge: reconstruct overrides assembly metadata (more complete)
    video_meta_map = {}
    video_meta_map.update(assemble_meta)
    video_meta_map.update(reconstruct_meta)

    # Phase D: Update episode metadata (clean slate — drop old video columns first)
    logger.info("Phase D: Updating episode parquet metadata...")
    update_episode_video_metadata(meta, video_meta_map)

    # Phase E: Verify
    logger.info("Phase E: Final verification...")
    verify_final_dataset(output_dir, args.repo_id)

    # Cleanup
    remaining_tmp = list(output_dir.glob("tmp*"))
    if remaining_tmp:
        logger.info(f"Cleaning up {len(remaining_tmp)} remaining tmp dirs...")
        for tmp_dir in remaining_tmp:
            if tmp_dir.is_dir():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)

    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(str(images_dir), ignore_errors=True)

    logger.info("Resume complete!")


if __name__ == "__main__":
    main()