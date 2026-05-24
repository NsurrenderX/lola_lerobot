#!/usr/bin/env python
"""Convert CALVIN dataset to LeRobot v3.0 format (V2 — no overlap split + pre-computed hist data).

V2 changes over V1 (convert_calvin_to_lerobot.py):
  - NO overlap resolution — each auto_lang_ann.npy entry = 1 episode with original [start, end)
  - Episode parquet stores only annotation frames (no transition frames in parquet)
  - Pre-computed hist_action/hist_state stored in calvin_episode_metadata.npz
  - Per-episode text metadata (completed_tasks, etc.) in calvin_episode_metadata.json
  - Pre-loaded Calvin NPZ data for fast conversion (no repeated NPZ I/O)
  - tqdm progress bar

Usage:
    python convert_calvin_to_lerobot_v2.py \
        --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/training \
        --output_dir /data_6t_2/lerobot_v30/calvin_v2_training \
        --repo_id calvin_v2_training \
        --num_workers 32 --video_encode_workers 16 \
        --max_transition_len 64
"""

import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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

VIDEO_KEYS = [
    "observation.images.rgb_static",
    "observation.images.rgb_gripper",
]

ACTION_DIM = 7
STATE_DIM = 7

# Module-level globals for fork-based shared memory (set by preload_calvin_data)
_shared_all_actions = None
_shared_all_states = None


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


def load_trajectory_boundaries(input_dir: str) -> list[dict]:
    """Load ep_start_end_ids.npy and return sorted trajectory boundaries."""
    path = Path(input_dir) / "ep_start_end_ids.npy"
    data = np.load(str(path), allow_pickle=True)
    trajectories = []
    for i, (start, end) in enumerate(data):
        trajectories.append({"traj_idx": i, "start": int(start), "end": int(end)})
    trajectories.sort(key=lambda x: x["start"])
    return trajectories


def load_lang_annotations(input_dir: str) -> list[dict]:
    """Load language annotations sorted by start index."""
    lang_path = Path(input_dir) / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_path.exists():
        raise FileNotFoundError(f"Language annotations not found at {lang_path}")

    lang_data = np.load(str(lang_path), allow_pickle=True).item()
    annotations = lang_data["language"]["ann"]
    tasks = lang_data["language"]["task"]
    indx = lang_data["info"]["indx"]

    result = []
    for i, (s, e) in enumerate(indx):
        result.append({
            "index": i,
            "start": int(s),
            "end": int(e),
            "task": tasks[i],
            "ann": annotations[i],
        })
    result.sort(key=lambda x: x["start"])
    return result


def assign_annotations_to_trajectories(
    annotations: list[dict],
    trajectories: list[dict],
) -> list[dict]:
    """Assign each annotation to its parent trajectory. Returns annotations with trajectory info."""
    for ann in annotations:
        assigned = False
        for traj in trajectories:
            if ann["start"] >= traj["start"] and ann["end"] <= traj["end"]:
                ann["traj_idx"] = traj["traj_idx"]
                ann["traj_start"] = traj["start"]
                ann["traj_end"] = traj["end"]
                assigned = True
                break
        if not assigned:
            logger.warning(f"Annotation {ann['index']} [{ann['start']}, {ann['end']}) "
                          f"not in any trajectory — skipping")
            ann["traj_idx"] = None
    return [a for a in annotations if a["traj_idx"] is not None]


def compute_hist_start(
    annotation: dict,
    max_transition_len: int = 64,
) -> tuple[int, int]:
    """Compute history frames preceding the annotation within its trajectory.

    Returns (hist_start, hist_len) where hist_start = max(start - max_transition_len, traj_start).
    """
    traj_start = annotation["traj_start"]
    ann_start = annotation["start"]
    hist_start = max(ann_start - max_transition_len, traj_start)
    hist_len = ann_start - hist_start
    return hist_start, hist_len


def compute_completed_tasks(
    annotation: dict,
    all_annotations_in_traj: list[dict],
) -> tuple[list[str], dict[str, list[str]]]:
    """Compute completed tasks before this annotation in the trajectory.

    Completed tasks = annotations whose end <= current annotation's start,
    listed chronologically. Overlapping annotations with the same task
    (i.e. paraphrases of the same execution) are deduplicated to count only once.
    The same task executed at a different time (non-overlapping) is still listed.
    """
    # Collect previous annotations grouped by dedup key
    # Two annotations are the same execution if they overlap AND have the same task
    dedup_segments: list[tuple[int, int, str]] = []  # (merged_start, merged_end, task)
    completed = []
    ann_choices = {}

    for prev_ann in all_annotations_in_traj:
        if prev_ann["end"] <= annotation["start"]:
            task = prev_ann["task"]
            # Collect ann_text regardless (for diversity in ann_choices)
            ann_text = prev_ann["ann"]
            if task not in ann_choices:
                ann_choices[task] = []
            if ann_text not in ann_choices[task]:
                ann_choices[task].append(ann_text)

            # Check if this annotation overlaps with an existing segment of the same task
            s, e = prev_ann["start"], prev_ann["end"]
            merged = False
            for i, (ms, me, mt) in enumerate(dedup_segments):
                if mt == task and s < me and e > ms:
                    # Overlap — merge into this segment
                    dedup_segments[i] = (min(ms, s), max(me, e), mt)
                    merged = True
                    break
            if not merged:
                dedup_segments.append((s, e, task))

    # Sort deduped segments by start and build completed list
    dedup_segments.sort(key=lambda x: x[0])
    for _, _, task in dedup_segments:
        completed.append(task)

    return completed, ann_choices


def build_episode_definitions(
    annotations: list[dict],
    max_transition_len: int = 64,
) -> list[dict]:
    """Build episode definitions with original annotation bounds + hist metadata."""
    traj_groups = {}
    for ann in annotations:
        traj_idx = ann["traj_idx"]
        if traj_idx not in traj_groups:
            traj_groups[traj_idx] = []
        traj_groups[traj_idx].append(ann)

    episodes = []
    for ann in annotations:
        hist_start, hist_len = compute_hist_start(ann, max_transition_len)
        completed_tasks, completed_tasks_ann_choices = compute_completed_tasks(
            ann, traj_groups[ann["traj_idx"]]
        )

        episodes.append({
            "annotation_index": ann["index"],
            "start": ann["start"],
            "end": ann["end"],
            "hist_start": hist_start,
            "hist_len": hist_len,
            "task": ann["task"],
            "ann": ann["ann"],
            "completed_tasks": completed_tasks,
            "completed_tasks_ann_choices": completed_tasks_ann_choices,
            "calvin_trajectory_index": ann["traj_idx"],
        })

    return episodes


def preload_calvin_data(input_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Pre-load all Calvin NPZ action/state data into memory for fast index-based access.

    Returns (all_actions, all_states, total_npz_files).
    """
    npz_files = sorted(input_dir.glob("episode_*.npz"))
    total_npz = len(npz_files)
    logger.info(f"Pre-loading {total_npz} NPZ files into memory...")

    all_actions = np.zeros((total_npz, ACTION_DIM), dtype=np.float32)
    all_states = np.zeros((total_npz, STATE_DIM), dtype=np.float32)

    for i in tqdm(range(total_npz), desc="Loading NPZ"):
        npz_path = input_dir / f"episode_{i:07d}.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            all_actions[i] = data["rel_actions"].astype(np.float32)
            all_states[i] = data["robot_obs"][:STATE_DIM].astype(np.float32)

    logger.info(f"Pre-loaded {total_npz} NPZ files "
                f"(actions: {all_actions.nbytes / 1e6:.1f}MB, states: {all_states.nbytes / 1e6:.1f}MB)")

    # Set module-level globals for fork-based worker access
    global _shared_all_actions, _shared_all_states
    _shared_all_actions = all_actions
    _shared_all_states = all_states

    return all_actions, all_states, total_npz


def process_episode_worker(
    ep_idx: int,
    ep_info: dict,
    input_dir: str,
    output_dir: str,
    features: dict,
    fps: int,
    max_transition_len: int,
) -> tuple[int, dict | None, dict | None]:
    """Worker for processing a single episode.

    Uses module-level _shared_all_actions/_shared_all_states (set by preload_calvin_data)
    via fork-based copy-on-write inheritance.

    Returns (ep_idx, episode_data, hist_metadata).
    - episode_data: parquet data for annotation frames only
    - hist_metadata: pre-computed hist_action/hist_state arrays
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    start = ep_info["start"]
    end = ep_info["end"]
    hist_start = ep_info["hist_start"]
    hist_len = ep_info["hist_len"]
    task = ep_info["ann"]

    # Build episode_data for annotation frames only
    episode_data = {
        "size": 0,
        "task": [],
        "episode_index": ep_idx,
    }
    for key in features:
        if key == "episode_index":
            continue
        episode_data[key] = []
    episode_data["frame_index"] = []
    episode_data["timestamp"] = []
    episode_data["index"] = []
    episode_data["task_index"] = []

    for local_idx in range(end - start):
        npz_idx = start + local_idx
        npz_path = input_dir / f"episode_{npz_idx:07d}.npz"
        if not npz_path.exists():
            return (ep_idx, None, None)

        data = np.load(str(npz_path), allow_pickle=True)

        action = data["rel_actions"].astype(np.float32)
        state = data["robot_obs"][:STATE_DIM].astype(np.float32)

        frame_index = local_idx
        timestamp = local_idx / fps

        episode_data["frame_index"].append(frame_index)
        episode_data["timestamp"].append(timestamp)
        episode_data["task"].append(task)

        episode_data["observation.state"].append(state)
        episode_data["action"].append(action)

        # Save images as PNG
        rgb_static = data["rgb_static"]
        rgb_gripper = data["rgb_gripper"]
        img_static = Image.fromarray(rgb_static)
        img_gripper = Image.fromarray(rgb_gripper)

        for video_key, img in [
            ("observation.images.rgb_static", img_static),
            ("observation.images.rgb_gripper", img_gripper),
        ]:
            img_path = output_dir / DEFAULT_IMAGE_PATH.format(
                image_key=video_key, episode_index=ep_idx, frame_index=frame_index
            )
            if local_idx == 0:
                img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(img_path), compress_level=1)
            episode_data[video_key].append(str(img_path))

    episode_data["size"] = end - start

    # Build hist_action / hist_state from pre-loaded arrays (fork-based shared memory)
    hist_action = np.zeros((max_transition_len, ACTION_DIM), dtype=np.float32)
    hist_state = np.zeros((max_transition_len, STATE_DIM), dtype=np.float32)

    if hist_len > 0 and _shared_all_actions is not None:
        # Clip to available data range
        actual_hist_start = max(hist_start, 0)
        actual_hist_end = min(start, len(_shared_all_actions))
        actual_len = actual_hist_end - actual_hist_start
        if actual_len > 0:
            fill_offset = max_transition_len - actual_len
            hist_action[fill_offset:fill_offset + actual_len] = _shared_all_actions[actual_hist_start:actual_hist_end]
            hist_state[fill_offset:fill_offset + actual_len] = _shared_all_states[actual_hist_start:actual_hist_end]

    hist_metadata = {
        "hist_action": hist_action,
        "hist_state": hist_state,
        "hist_len": hist_len,
    }

    return (ep_idx, episode_data, hist_metadata)


def encode_videos_parallel(dataset, video_encode_workers: int):
    """Parallel video encoding + batch file management. Same as V1."""
    num_episodes = dataset.num_episodes
    video_keys = dataset.meta.video_keys

    if num_episodes == 0 or len(video_keys) == 0:
        logger.info("No episodes or video keys — skipping video encoding")
        return

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

    logger.info("Phase B: Planning file boundaries and assembling videos...")
    video_files_size_mb = dataset.meta.video_files_size_in_mb
    chunks_size = dataset.meta.chunks_size

    ep_sizes = {}
    ep_durations = {}
    for vk in video_keys:
        for ei in range(num_episodes):
            temp_path = encode_results[(vk, ei)]
            ep_sizes[(vk, ei)] = get_file_size_in_mb(temp_path)
            ep_durations[(vk, ei)] = get_video_duration_in_s(temp_path)

    video_meta_map = {}
    for vk in video_keys:
        file_groups = []
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

        for ck, fi, ep_indices in file_groups:
            video_path = dataset.root / dataset.meta.video_path.format(
                video_key=vk, chunk_index=ck, file_index=fi
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            temp_paths = [encode_results[(vk, ei)] for ei in ep_indices]
            if len(temp_paths) == 1:
                shutil.move(str(temp_paths[0]), str(video_path))
            else:
                shutil.move(str(temp_paths[0]), str(video_path))
                concatenate_video_files([video_path] + temp_paths[1:], video_path)
            for tp in temp_paths:
                td = tp.parent
                if td.exists():
                    shutil.rmtree(str(td))
        logger.info(f"  {vk}: {len(file_groups)} video files for {num_episodes} episodes")

    logger.info("Phase C: Updating episode metadata with video info...")
    dataset.meta._close_writer()
    episodes = load_episodes(dataset.root)
    chunk_file_episodes = {}
    for ei in range(num_episodes):
        ep = episodes[ei]
        ck = ep["meta/episodes/chunk_index"]
        fi = ep["meta/episodes/file_index"]
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
        # Ensure chunk_index/file_index are int, not float
        for col in episode_df.columns:
            if col.startswith("videos/") and ("chunk_index" in col or "file_index" in col):
                episode_df[col] = episode_df[col].astype(int)
        episode_df.to_parquet(episode_df_path)

    for vk in video_keys:
        dataset.meta.update_video_info(vk)
    dataset.meta.episodes = load_episodes(dataset.root)
    logger.info("Video encoding complete")


def write_episode_metadata(episodes: list[dict], output_dir: Path):
    """Write calvin_episode_metadata.json alongside the dataset."""
    metadata = {}
    for ep_idx, ep in enumerate(episodes):
        metadata[str(ep_idx)] = {
            "hist_len": ep["hist_len"],
            "annotation_len": ep["end"] - ep["start"],
            "completed_tasks": ep["completed_tasks"],
            "completed_tasks_ann_choices": ep["completed_tasks_ann_choices"],
            "calvin_trajectory_index": ep["calvin_trajectory_index"],
        }

    metadata_path = output_dir / "calvin_episode_metadata.json"
    with open(str(metadata_path), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote episode metadata to {metadata_path}")


def write_hist_metadata_npz(
    hist_metadata_all: dict[int, dict],
    num_episodes: int,
    output_dir: Path,
    max_transition_len: int,
):
    """Write hist_action/hist_state arrays to calvin_episode_metadata.npz."""
    hist_action = np.zeros((num_episodes, max_transition_len, ACTION_DIM), dtype=np.float32)
    hist_state = np.zeros((num_episodes, max_transition_len, STATE_DIM), dtype=np.float32)
    hist_len = np.zeros(num_episodes, dtype=np.int32)

    for ep_idx in range(num_episodes):
        if ep_idx in hist_metadata_all:
            meta = hist_metadata_all[ep_idx]
            hist_action[ep_idx] = meta["hist_action"]
            hist_state[ep_idx] = meta["hist_state"]
            hist_len[ep_idx] = meta["hist_len"]

    npz_path = output_dir / "calvin_episode_metadata.npz"
    np.savez_compressed(str(npz_path), hist_action=hist_action, hist_state=hist_state, hist_len=hist_len)
    logger.info(f"Wrote hist metadata to {npz_path} "
                f"(hist_action: {hist_action.shape}, hist_state: {hist_state.shape})")


def convert_dataset(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    skip_norm: bool = True,
    norm_min: float = -0.65,
    norm_max: float = 0.65,
    num_workers: int | None = None,
    video_encode_workers: int | None = None,
    max_transition_len: int = 64,
):
    """Convert CALVIN dataset to LeRobot v3.0 format (V2) with no overlap split + pre-computed hist data."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if num_workers is None:
        num_workers = os.cpu_count() or 4
    if video_encode_workers is None:
        video_encode_workers = max(1, (os.cpu_count() or 4) // 2)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Skip normalization: {skip_norm}")
    logger.info(f"Max transition length: {max_transition_len}")
    logger.info(f"NPZ workers: {num_workers}, Video encode workers: {video_encode_workers}")

    # Load trajectory boundaries
    trajectories = load_trajectory_boundaries(str(input_dir))
    logger.info(f"Found {len(trajectories)} trajectories from ep_start_end_ids.npy")

    # Load language annotations
    annotations = load_lang_annotations(str(input_dir))
    logger.info(f"Found {len(annotations)} language annotations")

    # Assign annotations to trajectories (NO overlap resolution)
    annotations = assign_annotations_to_trajectories(annotations, trajectories)
    logger.info(f"Assigned {len(annotations)} annotations to trajectories (no overlap resolution)")

    # Build episode definitions
    episodes = build_episode_definitions(annotations, max_transition_len)
    logger.info(f"Built {len(episodes)} episode definitions")

    # Pre-load all Calvin NPZ data into shared arrays
    all_actions, all_states, _ = preload_calvin_data(input_dir)

    # Create LeRobot dataset with deferred video encoding
    features = build_features()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=str(output_dir),
        robot_type="franka",
        use_videos=True,
        batch_encoding_size=len(episodes) + 1,
    )

    # Phase 1+2: Process episodes + save parquet
    logger.info(f"Phase 1+2: Processing {len(episodes)} episodes "
                f"({num_workers} workers, sequential save)...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for ep_idx, ep_info in enumerate(episodes):
            f = executor.submit(
                process_episode_worker,
                ep_idx,
                ep_info,
                str(input_dir),
                str(output_dir),
                features,
                FPS,
                max_transition_len,
            )
            futures[f] = ep_idx

        pending = {}
        next_ep = 0
        total_saved = 0
        hist_metadata_all = {}
        pbar = tqdm(total=len(episodes), desc="Saving episodes")
        for future in as_completed(futures):
            ep_idx, episode_data, hist_meta = future.result()
            pending[ep_idx] = (episode_data, hist_meta)

            while next_ep in pending:
                episode_data, hist_meta = pending.pop(next_ep)
                if episode_data is not None:
                    dataset.save_episode(episode_data=episode_data, parallel_encoding=False)
                    total_saved += 1
                if hist_meta is not None:
                    hist_metadata_all[next_ep] = hist_meta
                next_ep += 1
                pbar.update(1)
        pbar.close()

    logger.info(f"Phase 1+2 complete: {total_saved} episodes saved to parquet")

    # Phase 3: Parallel video encoding
    encode_videos_parallel(dataset, video_encode_workers)

    # Write episode metadata
    write_episode_metadata(episodes, output_dir)
    write_hist_metadata_npz(hist_metadata_all, len(episodes), output_dir, max_transition_len)

    # Finalize
    dataset.stop_image_writer()
    dataset.meta._close_writer()
    logger.info(f"Done! Total: {total_saved} episodes")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CALVIN dataset to LeRobot v3.0 format (V2 — no overlap split + pre-computed hist data)"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to CALVIN directory (containing episode_*.npz and lang_annotations/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LeRobot v3.0 dataset")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Dataset repo ID")
    parser.add_argument("--skip_norm", type=lambda x: x.lower() not in ("false", "0", "no"),
                        default=True, help="Skip action normalization (default: True)")
    parser.add_argument("--norm_min", type=float, default=-0.65)
    parser.add_argument("--norm_max", type=float, default=0.65)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers for episode processing")
    parser.add_argument("--video_encode_workers", type=int, default=None,
                        help="Number of parallel workers for video encoding")
    parser.add_argument("--max_transition_len", type=int, default=64,
                        help="Max history frames preceding each annotation (default: 64)")
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
        max_transition_len=args.max_transition_len,
    )


if __name__ == "__main__":
    main()
