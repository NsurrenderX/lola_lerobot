#!/usr/bin/env python

"""
Visualize Calvin raw dataset annotations — trajectory boundaries, task segments,
overlaps, and gaps. Produces both analysis text/plots and trajectory videos.

Reads original Calvin NPZ data (not converted LeRobot format) and produces:
  1. Text analysis: task sequences, overlap counts, gap statistics per trajectory
  2. Static plots: overlap heatmap, task sequence diagram per trajectory
  3. Trajectory video: camera images + annotation timeline + state/action info

Modes:
  - analysis: text + static plots only (fastest, no video)
  - fast:      PIL-only video + analysis (no matplotlib, much faster rendering)
  - full:      matplotlib video + analysis (detailed line plots, slower rendering)

Usage:
  # Analysis only (fast, no video rendering)
  python visualize_calvin_annotations.py \
      --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/validation \
      --output_dir ./calvin_annotation_vis \
      --mode analysis

  # Fast mode (PIL-only video, no matplotlib — recommended for speed)
  python visualize_calvin_annotations.py \
      --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/validation \
      --output_dir ./calvin_annotation_vis \
      --trajectory-index 0 \
      --mode fast \
      --max-frames 500

  # Full mode (matplotlib line plots, slower)
  python visualize_calvin_annotations.py \
      --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/validation \
      --output_dir ./calvin_annotation_vis \
      --trajectory-index 0 \
      --mode full \
      --max-frames 500
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


# ── Colors ────────────────────────────────────────────────────────────────────

# 34 unique Calvin task labels → assign distinct colors
TASK_COLORS = {}
_CMAP = matplotlib.colormaps.get_cmap("tab20")
_CMAP2 = matplotlib.colormaps.get_cmap("tab20b")
for i, task in enumerate(sorted([
    "close_drawer", "lift_blue_block_drawer", "lift_blue_block_slider",
    "lift_blue_block_table", "lift_pink_block_drawer", "lift_pink_block_slider",
    "lift_pink_block_table", "lift_red_block_drawer", "lift_red_block_slider",
    "lift_red_block_table", "move_slider_left", "move_slider_right",
    "open_drawer", "place_in_drawer", "place_in_slider",
    "push_blue_block_left", "push_blue_block_right", "push_into_drawer",
    "push_pink_block_left", "push_pink_block_right", "push_red_block_left",
    "push_red_block_right", "rotate_blue_block_left", "rotate_blue_block_right",
    "rotate_pink_block_left", "rotate_pink_block_right", "rotate_red_block_left",
    "rotate_red_block_right", "stack_block", "turn_off_led",
    "turn_off_lightbulb", "turn_on_led", "turn_on_lightbulb", "unstack_block",
])):
    if i < 20:
        TASK_COLORS[task] = _CMAP(i)
    else:
        TASK_COLORS[task] = _CMAP2(i - 20)

TASK_COLORS["none"] = (0.85, 0.85, 0.85, 1.0)  # gray for unannotated

COLOR_GRIPPER_OPEN = "#2ca02c"
COLOR_GRIPPER_CLOSED = "#d62728"
COLOR_CURRENT_POS = "#1f77b4"
COLOR_START_MARKER = "#333333"
COLOR_FUTURE = (0.5, 0.5, 0.5, 0.3)


# ── Font & Text Helpers ────────────────────────────────────────────────────────


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try loading a TrueType font; fall back to default if unavailable."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contrast_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose black or white text color based on background luminance."""
    luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


def _sample_region_color(img_np: np.ndarray, x: int, y: int, w: int = 10, h: int = 5) -> tuple[int, int, int]:
    """Sample average color of a rectangular region in an image array."""
    ih, iw = img_np.shape[:2]
    y0 = max(0, min(y, ih - 1))
    y1 = max(0, min(y + h, ih))
    x0 = max(0, min(x, iw - 1))
    x1 = max(0, min(x + w, iw))
    region = img_np[y0:y1, x0:x1]
    if region.size == 0:
        return (0, 0, 0)
    avg = region.mean(axis=(0, 1)).astype(int)
    return (int(avg[0]), int(avg[1]), int(avg[2]))


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_trajectory_boundaries(input_dir: str) -> list[dict]:
    path = Path(input_dir) / "ep_start_end_ids.npy"
    data = np.load(str(path), allow_pickle=True)
    trajectories = []
    for i, (start, end) in enumerate(data):
        trajectories.append({
            "trajectory_index": i,
            "start": int(start),
            "end": int(end),
        })
    # Sort by start index (ascending time order)
    trajectories.sort(key=lambda x: x["start"])
    return trajectories


def load_lang_annotations(input_dir: str) -> list[dict]:
    path = Path(input_dir) / "lang_annotations" / "auto_lang_ann.npy"
    data = np.load(str(path), allow_pickle=True).item()
    annotations = data["language"]["ann"]
    tasks = data["language"]["task"]
    indx = data["info"]["indx"]

    annotations_list = []
    for i, (s, e) in enumerate(indx):
        annotations_list.append({
            "index": i,
            "start": int(s),
            "end": int(e),
            "task": tasks[i],
            "ann": annotations[i],
        })

    # Sort by start index
    annotations_list.sort(key=lambda x: x["start"])
    return annotations_list


def read_npz_frame(input_dir: str, global_idx: int) -> dict | None:
    npz_path = Path(input_dir) / f"episode_{global_idx:07d}.npz"
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path), allow_pickle=True)
    return {
        "rgb_static": data["rgb_static"],      # (200, 200, 3) uint8
        "rgb_gripper": data["rgb_gripper"],     # (84, 84, 3) uint8
        "robot_obs": data["robot_obs"][:7],     # (7,) float64
        "rel_actions": data["rel_actions"],     # (7,) float64
    }


# ── Annotation Analysis ──────────────────────────────────────────────────────


def compute_annotation_segments_for_trajectory(
    annotations: list[dict],
    traj_start: int,
    traj_end: int,
) -> list[dict]:
    """Filter annotations that fall within a trajectory's global index range."""
    segments = []
    for ann in annotations:
        # Annotation overlaps trajectory if it starts before traj_end and ends after traj_start
        if ann["start"] < traj_end and ann["end"] > traj_start:
            # Clip to trajectory boundaries
            clipped_start = max(ann["start"], traj_start)
            clipped_end = min(ann["end"], traj_end)
            segments.append({
                **ann,
                "clipped_start": clipped_start,
                "clipped_end": clipped_end,
            })
    return segments


def compute_overlap_pairs(segments: list[dict]) -> list[dict]:
    """Find all overlapping annotation pairs within a trajectory."""
    overlaps = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            si, sj = segments[i], segments[j]
            overlap_start = max(si["clipped_start"], sj["clipped_start"])
            overlap_end = min(si["clipped_end"], sj["clipped_end"])
            overlap_len = overlap_end - overlap_start
            if overlap_len > 0:
                overlaps.append({
                    "ann_i": si["index"],
                    "ann_j": sj["index"],
                    "task_i": si["task"],
                    "task_j": sj["task"],
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "overlap_len": overlap_len,
                    "midpoint": overlap_start + overlap_len // 2,
                })
    return overlaps


def compute_per_frame_annotation_count(
    segments: list[dict],
    traj_start: int,
    traj_end: int,
) -> np.ndarray:
    """Count how many annotations cover each frame in the trajectory."""
    counts = np.zeros(traj_end - traj_start, dtype=int)
    for seg in segments:
        local_start = seg["clipped_start"] - traj_start
        local_end = seg["clipped_end"] - traj_start
        counts[local_start:local_end] += 1
    return counts


def analyze_trajectory(
    annotations: list[dict],
    traj: dict,
) -> dict:
    """Full analysis of one trajectory: segments, overlaps, gaps, task sequence."""
    traj_start = traj["start"]
    traj_end = traj["end"]
    traj_len = traj_end - traj_start

    segments = compute_annotation_segments_for_trajectory(annotations, traj_start, traj_end)
    overlaps = compute_overlap_pairs(segments)
    annotation_counts = compute_per_frame_annotation_count(segments, traj_start, traj_end)

    # Task sequence (unique task segments in chronological order)
    # Merge overlapping annotations at the same task by grouping them
    task_sequence = []
    seen_segments = set()
    for seg in segments:
        key = (seg["clipped_start"], seg["task"])
        if key not in seen_segments:
            task_sequence.append(seg["task"])
            seen_segments.add(key)

    # Gap statistics
    unannotated_frames = np.sum(annotation_counts == 0)
    annotated_frames = np.sum(annotation_counts > 0)
    multi_annotated_frames = np.sum(annotation_counts > 1)

    # Overlap stats
    max_overlap = max((o["overlap_len"] for o in overlaps), default=0)
    same_task_overlaps = [o for o in overlaps if o["task_i"] == o["task_j"]]
    diff_task_overlaps = [o for o in overlaps if o["task_i"] != o["task_j"]]

    return {
        "trajectory": traj,
        "segments": segments,
        "overlaps": overlaps,
        "annotation_counts": annotation_counts,
        "task_sequence": task_sequence,
        "traj_len": traj_len,
        "unannotated_frames": unannotated_frames,
        "annotated_frames": annotated_frames,
        "multi_annotated_frames": multi_annotated_frames,
        "max_overlap": max_overlap,
        "same_task_overlaps": same_task_overlaps,
        "diff_task_overlaps": diff_task_overlaps,
    }


def print_analysis_summary(analysis: dict):
    """Print text summary of trajectory analysis."""
    traj = analysis["trajectory"]
    print(f"\n{'='*60}")
    print(f"Trajectory {traj['trajectory_index']}: frames [{traj['start']}, {traj['end']})")
    print(f"  Length: {analysis['traj_len']} frames")
    print(f"  Annotations: {len(analysis['segments'])}")
    print(f"  Annotated frames: {analysis['annotated_frames']} ({analysis['annotated_frames']/analysis['traj_len']*100:.1f}%)")
    print(f"  Unannotated frames: {analysis['unannotated_frames']} ({analysis['unannotated_frames']/analysis['traj_len']*100:.1f}%)")
    print(f"  Multi-annotated frames: {analysis['multi_annotated_frames']}")
    print(f"  Overlaps: {len(analysis['overlaps'])} total, "
          f"{len(analysis['same_task_overlaps'])} same-task, "
          f"{len(analysis['diff_task_overlaps'])} different-task")
    print(f"  Max overlap: {analysis['max_overlap']} frames")

    # Deduplicated task sequence (merge consecutive same-task annotations)
    deduped_seq = []
    prev_task = None
    for task in analysis["task_sequence"]:
        if task != prev_task:
            deduped_seq.append(task)
            prev_task = task
    print(f"  Task sequence (deduped): {' → '.join(deduped_seq)}")
    print(f"  Unique tasks in trajectory: {len(set(deduped_seq))}")
    print(f"  Total task segments: {len(analysis['task_sequence'])}")

    if analysis["overlaps"]:
        # Show overlap length distribution
        overlap_lengths = [o["overlap_len"] for o in analysis["overlaps"]]
        print(f"\n  Overlap length stats:")
        print(f"    Min: {min(overlap_lengths)}, Max: {max(overlap_lengths)}, "
              f"Mean: {np.mean(overlap_lengths):.1f}, Median: {np.median(overlap_lengths):.1f}")

        # Show top 5 longest overlaps
        sorted_overlaps = sorted(analysis["overlaps"], key=lambda o: o["overlap_len"], reverse=True)
        print(f"  Top 5 longest overlaps:")
        for o in sorted_overlaps[:5]:
            same = o["task_i"] == o["task_j"]
            print(f"    Ann {o['ann_i']}({o['task_i']}) × Ann {o['ann_j']}({o['task_j']}): "
                  f"overlap={o['overlap_len']} frames {('(same task)' if same else '(diff task)')}")

        # Show top 5 different-task overlaps
        diff_sorted = sorted(analysis["diff_task_overlaps"], key=lambda o: o["overlap_len"], reverse=True)
        if diff_sorted:
            print(f"  Top 5 different-task overlaps:")
            for o in diff_sorted[:5]:
                print(f"    Ann {o['ann_i']}({o['task_i']}) × Ann {o['ann_j']}({o['task_j']}): "
                      f"overlap={o['overlap_len']} frames, midpoint={o['midpoint']}")


# ── Static Plot Generation ────────────────────────────────────────────────────


def plot_overlap_heatmap(
    analysis: dict,
    output_path: Path,
    max_segments: int = 50,
):
    """Plot overlap heatmap: annotation pairs × overlap frame count.
    Skips if there are too many segments (> max_segments) to keep rendering fast.
    """
    segments = analysis["segments"]
    if len(segments) > max_segments:
        print(f"  Skipping overlap heatmap ({len(segments)} segments > {max_segments} limit)")
        return
    if len(segments) < 2:
        print("  Not enough segments for overlap heatmap")
        return

    n = len(segments)
    overlap_matrix = np.zeros((n, n), dtype=int)
    for o in analysis["overlaps"]:
        # Find indices in the segments list
        i_idx = next(j for j, s in enumerate(segments) if s["index"] == o["ann_i"])
        j_idx = next(j for j, s in enumerate(segments) if s["index"] == o["ann_j"])
        overlap_matrix[i_idx, j_idx] = o["overlap_len"]
        overlap_matrix[j_idx, i_idx] = o["overlap_len"]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(8, n * 0.4)))
    im = ax.imshow(overlap_matrix, cmap="YlOrRd", interpolation="nearest")

    # Labels
    labels = [f"{s['task'][:12]}[{s['clipped_start']}-{s['clipped_end']}]" for s in segments]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)

    # Values in cells
    for i in range(n):
        for j in range(n):
            if overlap_matrix[i, j] > 0:
                ax.text(j, i, str(overlap_matrix[i, j]), ha="center", va="center",
                        fontsize=5, color="black" if overlap_matrix[i, j] < 30 else "white")

    ax.set_title(f"Annotation Overlap Heatmap — Trajectory {analysis['trajectory']['trajectory_index']}")
    plt.colorbar(im, ax=ax, label="Overlap (frames)")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_task_sequence_diagram(
    analysis: dict,
    output_path: Path,
    max_segments: int = 50,
):
    """Plot task sequence as horizontal bars with annotation ranges.
    Shows only the first max_segments annotations if there are too many.
    """
    segments = analysis["segments"]
    traj = analysis["trajectory"]
    traj_start = traj["start"]
    traj_len = analysis["traj_len"]

    if not segments:
        print("  No segments for task sequence diagram")
        return

    # Limit segments for rendering performance
    if len(segments) > max_segments:
        print(f"  Showing first {max_segments}/{len(segments)} annotations in task sequence diagram")
        segments = segments[:max_segments]

    fig, ax = plt.subplots(figsize=(16, max(4, len(segments) * 0.3)))

    # Group segments by task (for stacking overlapping annotations of same task)
    # Each unique annotation gets its own row
    for i, seg in enumerate(segments):
        local_start = seg["clipped_start"] - traj_start
        local_end = seg["clipped_end"] - traj_start
        color = TASK_COLORS.get(seg["task"], (0.5, 0.5, 0.5, 1.0))

        ax.barh(i, local_end - local_start, left=local_start, height=0.8,
                color=color, edgecolor="black", linewidth=0.5)

        # Label: task name + annotation text (truncated)
        label = f"{seg['task']}"
        ann_text = seg["ann"][:30] + "..." if len(seg["ann"]) > 30 else seg["ann"]
        ax.text(local_start + 1, i, f"{label}: {ann_text}",
                fontsize=5, va="center", ha="left", color="black")

    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels([f"Ann {s['index']}" for s in segments], fontsize=7)
    ax.set_xlabel("Frame index (local)")
    ax.set_title(f"Task Sequence — Trajectory {traj['trajectory_index']} "
                 f"(frames [{traj_start}, {traj['end']}))")
    ax.set_xlim(0, traj_len)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_annotation_density(
    analysis: dict,
    output_path: Path,
):
    """Plot annotation coverage density over the trajectory timeline."""
    counts = analysis["annotation_counts"]
    traj_len = analysis["traj_len"]
    traj = analysis["trajectory"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4), sharex=True)

    # Top: annotation count per frame
    ax1.fill_between(range(traj_len), counts, step="mid", color="#1f77b4", alpha=0.5)
    ax1.set_ylabel("Annotation count")
    ax1.set_ylim(0, max(counts.max() + 1, 3))
    ax1.set_title(f"Annotation Coverage — Trajectory {traj['trajectory_index']}")

    # Bottom: per-frame task assignment (with midpoint split)
    # Assign each frame to a task using the overlap-split logic
    segments = analysis["segments"]
    overlaps = analysis["overlaps"]

    # Build effective ranges with midpoint split
    effective_ranges = []
    for seg in segments:
        eff_start = seg["clipped_start"] - traj["start"]
        eff_end = seg["clipped_end"] - traj["start"]
        # Apply midpoint splits from overlaps
        for o in overlaps:
            if o["ann_i"] == seg["index"]:
                # This annotation is earlier — truncate at midpoint
                overlap_local_start = o["overlap_start"] - traj["start"]
                overlap_local_end = o["overlap_end"] - traj["start"]
                mid_local = o["midpoint"] - traj["start"]
                # Earlier annotation gets truncated at midpoint
                if seg["index"] == min(o["ann_i"], o["ann_j"]):
                    eff_end = min(eff_end, mid_local)
                else:
                    eff_start = max(eff_start, mid_local)
        effective_ranges.append((eff_start, eff_end, seg["task"]))

    # Draw task bands
    for eff_start, eff_end, task in effective_ranges:
        color = TASK_COLORS.get(task, (0.5, 0.5, 0.5, 1.0))
        ax2.axvspan(eff_start, eff_end, color=color, alpha=0.7)
        if eff_end - eff_start > 20:
            ax2.text((eff_start + eff_end) / 2, 0.5, task,
                     fontsize=6, ha="center", va="center", rotation=0)

    # Mark overlaps with dashed lines at midpoints
    for o in overlaps:
        mid_local = o["midpoint"] - traj["start"]
        ax2.axvline(mid_local, color="red", linestyle="--", linewidth=0.5, alpha=0.5)

    ax2.set_ylabel("Task (after split)")
    ax2.set_xlabel("Frame index (local)")
    ax2.set_xlim(0, traj_len)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Video Rendering ──────────────────────────────────────────────────────────


def compose_camera_row(
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    target_width: int,
    max_height: int = 300,
) -> np.ndarray:
    """Compose two camera images into a single row, filling target_width."""
    images = [rgb_static, rgb_gripper]
    gap = 4
    usable_width = target_width - gap
    per_cam_width = usable_width // 2

    scaled = []
    for img in images:
        h, w = img.shape[:2]
        new_w = per_cam_width
        new_h = int(h * new_w / w)
        pil = PILImage.fromarray(img)
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        scaled.append(np.array(pil))

    row_h = max(img.shape[0] for img in scaled)
    if row_h > max_height:
        scale = max_height / row_h
        rescaled = []
        for img in scaled:
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            pil = PILImage.fromarray(img)
            pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
            rescaled.append(np.array(pil))
        scaled = rescaled
        row_h = max_height

    row = np.full((row_h, target_width, 3), 255, dtype=np.uint8)
    x_off = 0
    for img in scaled:
        h, w = img.shape[:2]
        y_off = (row_h - h) // 2
        row[y_off:y_off + h, x_off:x_off + w, :] = img
        x_off += w + gap

    return row


def preload_trajectory_data(
    input_dir: str,
    traj_start: int,
    render_len: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Pre-load all NPZ data for the rendered range upfront.
    Returns: states (7-dim), actions (7-dim), rgb_static list, rgb_gripper list.
    Much faster than per-frame I/O inside the rendering loop.
    """
    states = np.zeros((render_len, 7), dtype=np.float64)
    actions = np.zeros((render_len, 7), dtype=np.float64)
    rgb_statics = []
    rgb_grippers = []

    for i in range(render_len):
        global_idx = traj_start + i
        npz_path = Path(input_dir) / f"episode_{global_idx:07d}.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            states[i] = data["robot_obs"][:7]
            actions[i] = data["rel_actions"]
            rgb_statics.append(data["rgb_static"])
            rgb_grippers.append(data["rgb_gripper"])
        else:
            rgb_statics.append(np.full((200, 200, 3), 128, dtype=np.uint8))
            rgb_grippers.append(np.full((84, 84, 3), 128, dtype=np.uint8))

    return states, actions, rgb_statics, rgb_grippers


def build_frame_to_task_map(task_segments: list[tuple], traj_len: int) -> list[str]:
    """Build a per-frame task label list from task segments.
    task_segments: [(local_start, local_end, task_label), ...]
    Frames not covered by any segment get "none".
    """
    frame_task = ["none"] * traj_len
    for local_start, local_end, task_label in task_segments:
        for i in range(max(0, local_start), min(traj_len, local_end)):
            frame_task[i] = task_label
    return frame_task


def build_frame_to_tasks_map(task_segments: list[tuple], traj_len: int) -> list[list[str]]:
    """Build per-frame list of active task labels (supports overlaps).
    task_segments: [(local_start, local_end, task_label), ...]
    Frames not covered by any segment get an empty list.
    """
    frame_tasks: list[list[str]] = [[] for _ in range(traj_len)]
    for local_start, local_end, task_label in task_segments:
        for i in range(max(0, local_start), min(traj_len, local_end)):
            if task_label not in frame_tasks[i]:
                frame_tasks[i].append(task_label)
    return frame_tasks


def build_frame_task_info(
    segments: list[dict], traj_start: int, traj_len: int,
) -> list[list[tuple[int, str, str]]]:
    """Build per-frame list of (seq_index, task_label, ann_text) from raw annotations.

    No midpoint splitting, no merging. Each raw annotation is used as-is.
    Seq_index is assigned per unique (task, ann) pair — each distinct annotation
    gets its own unique seq when it first appears, and keeps that seq throughout.
    After a "none" gap, the same (task, ann) reappearing gets a NEW seq.
    "none" frames get empty list.

    Returns list indexed by local frame index.
    """
    # Step 1: raw per-frame task lookup (local coords)
    frame_tasks: list[list[tuple[str, str]]] = [[] for _ in range(traj_len)]
    for seg in segments:
        local_s = seg["clipped_start"] - traj_start
        local_e = seg["clipped_end"] - traj_start
        if local_s >= local_e:
            continue
        task = seg["task"]
        ann = seg["ann"]
        for i in range(max(0, local_s), min(traj_len, local_e)):
            if (task, ann) not in frame_tasks[i]:
                frame_tasks[i].append((task, ann))

    # Step 2: assign seq_index per unique (task, ann) pair
    # Each distinct annotation combination gets its own seq.
    # After a "none" gap, seq map resets — same (task, ann) later gets new seq.
    ann_seq_map: dict[tuple[str, str], int] = {}
    seq = 0
    prev_has_ann = False

    frame_info: list[list[tuple[int, str, str]]] = [[] for _ in range(traj_len)]

    for i in range(traj_len):
        if not frame_tasks[i]:
            frame_info[i] = []
            ann_seq_map = {}
            prev_has_ann = False
            continue

        # Assign new seq for (task, ann) pairs not seen since last gap
        for task, ann in frame_tasks[i]:
            if (task, ann) not in ann_seq_map:
                seq += 1
                ann_seq_map[(task, ann)] = seq

        for task, ann in frame_tasks[i]:
            frame_info[i].append((ann_seq_map[(task, ann)], task, ann))

        prev_has_ann = True

    return frame_info


def setup_video_figure(
    traj_idx: int,
    traj_start: int,
    render_len: int,
    traj_len: int,
    segments: list[dict],
):
    """Create figure with state/action line plots + annotation timeline.
    No 3D plot — uses fast 2D line charts instead.
    Draws raw annotation spans on the timeline (no midpoint splitting).
    """
    fig = plt.figure(figsize=(14, 6), dpi=80, facecolor="white")
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1, 1, 0.6],
        hspace=0.4,
        left=0.08, right=0.96, top=0.92, bottom=0.08,
    )

    ax_state = fig.add_subplot(gs[0])
    ax_action = fig.add_subplot(gs[1], sharex=ax_state)
    ax_ann = fig.add_subplot(gs[2], sharex=ax_state)

    # State panel setup
    ax_state.set_xlim(0, render_len)
    ax_state.set_ylabel("State value", fontsize=8)
    ax_state.set_title("Robot State (xyz + rpy + gripper)", fontsize=9)
    ax_state.tick_params(labelsize=7)

    # Action panel setup
    ax_action.set_xlim(0, render_len)
    ax_action.set_ylabel("Action value", fontsize=8)
    ax_action.set_title("Rel Actions (xyz + rpy + gripper)", fontsize=9)
    ax_action.tick_params(labelsize=7)

    # Annotation timeline setup
    ax_ann.set_xlim(0, traj_len)
    ax_ann.set_ylim(-0.5, 2.5)
    ax_ann.set_xlabel("Frame index (local)", fontsize=8)
    ax_ann.set_title("Annotation Timeline", fontsize=9)
    ax_ann.tick_params(labelsize=7)

    # Draw raw annotation spans as colored bands on the timeline
    for seg in segments:
        local_start = seg["clipped_start"] - traj_start
        local_end = seg["clipped_end"] - traj_start
        task_label = seg["task"]
        if local_start >= local_end:
            continue
        color = TASK_COLORS.get(task_label, (0.5, 0.5, 0.5, 1.0))
        ax_ann.axvspan(local_start, local_end, color=color, alpha=0.6)
        if local_end - local_start > 15:
            ax_ann.text((local_start + local_end) / 2, 1.0, task_label,
                        fontsize=5, ha="center", va="center")

    fig.suptitle(f"Trajectory {traj_idx}: Calvin Annotations", fontsize=12, fontweight="bold")

    return fig, ax_state, ax_action, ax_ann


def render_trajectory_video(
    input_dir: str,
    traj: dict,
    analysis: dict,
    output_path: Path,
    fps_factor: float = 0.2,
    max_frames: int | None = None,
):
    """Render one trajectory as a video with RGB cameras + state/action plots + annotation timeline.

    Layout per frame:
      - Header: frame count + active task labels (separate from camera images)
      - Camera row: RGB camera images (static + gripper, side by side)
      - Matplotlib plots: state/action line plots + annotation timeline
    """
    traj_start = traj["start"]
    traj_end = traj["end"]
    traj_len = traj_end - traj_start
    traj_idx = traj["trajectory_index"]

    font_small = _try_load_font(12)

    # Build per-frame task info from raw annotations (no midpoint split)
    segments = analysis["segments"]
    frame_task_info = build_frame_task_info(segments, traj_start, traj_len)

    # Pre-load all data upfront (much faster than per-frame I/O)
    render_len = min(traj_len, max_frames) if max_frames else traj_len
    print(f"  Pre-loading {render_len} frames for trajectory {traj_idx}...")
    states, actions, rgb_statics, rgb_grippers = preload_trajectory_data(
        input_dir, traj_start, render_len
    )

    # Setup matplotlib figure (no 3D — fast 2D line plots)
    fig, ax_state, ax_action, ax_ann = setup_video_figure(
        traj_idx, traj_start, render_len, traj_len, segments
    )

    # Pre-render first frame to get plot dimensions
    frame_idx_arr = np.arange(render_len)
    for d in range(7):
        ax_state.plot(frame_idx_arr, states[:, d], linewidth=0.5, alpha=0.3)
        ax_action.plot(frame_idx_arr, actions[:, d], linewidth=0.5, alpha=0.3)
    fig.canvas.draw()
    plot_img = extract_plot_frame(fig)
    plot_h, plot_w = plot_img.shape[:2]

    # Reset axes (clear the preview lines)
    ax_state.cla()
    ax_action.cla()
    ax_state.set_xlim(0, render_len)
    ax_action.set_xlim(0, render_len)
    ax_state.set_ylabel("State value", fontsize=8)
    ax_state.set_title("Robot State (xyz + rpy + gripper)", fontsize=9)
    ax_action.set_ylabel("Action value", fontsize=8)
    ax_action.set_title("Rel Actions (xyz + rpy + gripper)", fontsize=9)
    ax_state.tick_params(labelsize=7)
    ax_action.tick_params(labelsize=7)

    # Redraw annotation timeline bands (cla cleared them)
    for seg in segments:
        local_start = seg["clipped_start"] - traj_start
        local_end = seg["clipped_end"] - traj_start
        if local_start >= local_end:
            continue
        color = TASK_COLORS.get(seg["task"], (0.5, 0.5, 0.5, 1.0))
        ax_ann.axvspan(local_start, local_end, color=color, alpha=0.6)
        if local_end - local_start > 15:
            ax_ann.text((local_start + local_end) / 2, 1.0, seg["task"],
                        fontsize=5, ha="center", va="center")

    # Camera row dimensions
    cam_target_w = plot_w
    cam_max_h = 300  # fixed height for camera row

    # Total frame dimensions: header is variable, use max possible height
    max_header_h = 20 + MAX_TASK_ROWS * 14 + 4  # frame line + max task rows
    gap = 4
    total_h = max_header_h + cam_max_h + gap + plot_h
    total_w = plot_w

    output_fps = max(1, int(30 * fps_factor))
    print(f"  Layout: header {header_h_px}px + cameras {cam_max_h}px + plots {plot_h}px = {total_h}px, {total_w}px wide")
    print(f"  Rendering {render_len} frames at {output_fps}fps...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute full state/action sequences for background traces
    state_colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    action_colors = state_colors

    # ── Render all frames as PNGs, then encode with ffmpeg + h264_nvenc ────
    # This is much faster than PyAV: ffmpeg with NVENC uses GPU hardware encoding.
    # We write raw RGB frames as PNGs to a temp directory, then pipe to ffmpeg.
    tmp_dir = tempfile.mkdtemp(prefix="calvin_vis_")
    print(f"  Writing {render_len} raw frames to temp dir...")

    # Ensure even dimensions for h264
    total_w_even = total_w + (total_w % 2)
    total_h_even = total_h + (total_h % 2)

    for i in range(render_len):
        # ── Header: task info with sequential indices ────────────────
        fti = frame_task_info[min(i, len(frame_task_info) - 1)]
        task_rows = _layout_task_lines(fti)
        line_h = 14
        hdr_h = line_h + len(task_rows) * line_h + 4
        hdr_h = max(hdr_h, 20)

        header = np.full((hdr_h, total_w, 3), 30, dtype=np.uint8)
        header_pil = PILImage.fromarray(header)
        draw_hdr = ImageDraw.Draw(header_pil)
        draw_hdr.text((6, 2), f"Frame {i}/{render_len}", fill=(200, 200, 200), font=font_small)
        y = 2 + line_h
        for row in task_rows:
            draw_hdr.text((6, y), row, fill=(255, 255, 255), font=font_small)
            y += line_h
        header = np.array(header_pil)

        # ── Camera row (no text overlay) ──────────────────────────────
        cam_row = compose_camera_row(
            rgb_statics[i], rgb_grippers[i],
            cam_target_w, cam_max_h,
        )

        # ── Matplotlib plots (fast 2D line charts) ──────────────────────
        ax_state.cla()
        ax_action.cla()

        # Background: full sequence as thin gray lines
        for d in range(7):
            ax_state.plot(frame_idx_arr, states[:, d], color=state_colors[d],
                          linewidth=0.3, alpha=0.2)
            ax_action.plot(frame_idx_arr, actions[:, d], color=action_colors[d],
                          linewidth=0.3, alpha=0.2)

        # Progressive: bright lines up to current frame
        for d in range(7):
            ax_state.plot(frame_idx_arr[:i+1], states[:i+1, d],
                          color=state_colors[d], linewidth=1.0, alpha=0.8)
            ax_action.plot(frame_idx_arr[:i+1], actions[:i+1, d],
                          color=action_colors[d], linewidth=1.0, alpha=0.8)

        ax_state.axvline(i, color="red", linewidth=1, linestyle="--")
        ax_action.axvline(i, color="red", linewidth=1, linestyle="--")
        ax_state.set_xlim(0, render_len)
        ax_action.set_xlim(0, render_len)
        ax_state.set_ylabel("State value", fontsize=8)
        ax_state.set_title("Robot State", fontsize=9)
        ax_action.set_ylabel("Action value", fontsize=8)
        ax_action.set_title("Rel Actions", fontsize=9)
        ax_state.tick_params(labelsize=7)
        ax_action.tick_params(labelsize=7)

        # Annotation timeline: redraw bands + current frame marker
        ax_ann.cla()
        for seg in segments:
            local_start = seg["clipped_start"] - traj_start
            local_end = seg["clipped_end"] - traj_start
            if local_start >= local_end:
                continue
            color = TASK_COLORS.get(seg["task"], (0.5, 0.5, 0.5, 1.0))
            ax_ann.axvspan(local_start, local_end, color=color, alpha=0.6)
            if local_end - local_start > 15:
                ax_ann.text((local_start + local_end) / 2, 1.0, seg["task"],
                            fontsize=5, ha="center", va="center")
        ax_ann.axvline(i, color="red", linewidth=1.5, linestyle="--")
        ax_ann.set_xlim(0, traj_len)
        ax_ann.set_ylim(-0.5, 2.5)
        ax_ann.set_xlabel("Frame index (local)", fontsize=8)
        ax_ann.set_title("Annotation Timeline", fontsize=9)
        ax_ann.tick_params(labelsize=7)

        fig.canvas.draw()
        plot_img = extract_plot_frame(fig)

        # ── Compose final frame: header + cameras + plots ────────────
        full = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
        full[:hdr_h, :, :] = header
        # Camera row starts at max_header_h to keep consistent positioning
        full[max_header_h:max_header_h + cam_max_h, :, :] = cam_row
        full[max_header_h + cam_max_h + gap:, :, :] = plot_img

        # Pad to even dimensions if needed
        if total_w_even != total_w or total_h_even != total_h:
            padded = np.full((total_h_even, total_w_even, 3), 255, dtype=np.uint8)
            padded[:total_h, :total_w, :] = full
            full = padded

        # Save as PNG to temp dir (ffmpeg reads these sequentially)
        PILImage.fromarray(full).save(os.path.join(tmp_dir, f"frame_{i:06d}.png"))

        # Progress logging
        if (i + 1) % 100 == 0 or i == render_len - 1:
            print(f"  Rendered {i+1}/{render_len} frames")

    # ── ffmpeg encode with h264_nvenc ────────────────────────────────────
    # Try NVENC first; fall back to libx264 if NVENC is unavailable
    nvenc_available = _check_nvenc()
    codec = "h264_nvenc" if nvenc_available else "libx264"
    preset = "p1" if nvenc_available else "fast"  # p1 = fastest NVENC preset
    print(f"  Encoding with ffmpeg ({codec}, preset={preset})...")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(output_fps),
        "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v", codec,
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(output_path),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # If NVENC failed, retry with libx264
        if nvenc_available:
            print(f"  NVENC failed, falling back to libx264...")
            ffmpeg_cmd_fallback = [
                "ffmpeg", "-y",
                "-framerate", str(output_fps),
                "-i", os.path.join(tmp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-crf", "20",
                str(output_path),
            ]
            result = subprocess.run(ffmpeg_cmd_fallback, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ffmpeg error:\n{result.stderr}")
        else:
            print(f"  ffmpeg error:\n{result.stderr}")

    # Cleanup temp PNGs
    shutil.rmtree(tmp_dir, ignore_errors=True)

    plt.close(fig)
    print(f"  Saved video: {output_path}  (fps={output_fps}, {render_len} frames, codec={codec})")


def _check_nvenc() -> bool:
    """Check if ffmpeg h264_nvenc encoder is available on this system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        )
        return "h264_nvenc" in result.stdout
    except FileNotFoundError:
        return False


def extract_plot_frame(fig: plt.Figure) -> np.ndarray:
    """Extract matplotlib figure as (H, W, 3) uint8 numpy array."""
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()


# ── Fast Mode (PIL-only, no matplotlib) ────────────────────────────────────────

STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_cmd"]


def _format_value_lines(
    names: list[str],
    values: list[float],
    prefix: str,
    max_chars: int = 100,
) -> list[str]:
    """Wrap name=value pairs into text lines (like visualize_lola_dataset)."""
    lines: list[str] = []
    cur = f"{prefix}: "
    for name, val in zip(names, values):
        entry = f"{name}={val:+.3f}  "
        if len(cur) + len(entry) > max_chars:
            lines.append(cur)
            cur = f"    {entry}"
        else:
            cur += entry
    lines.append(cur)
    return lines


MAX_ANN_TEXT_LEN = 25  # max characters for annotation text in header
MAX_TASK_ROWS = 5     # max rows of task text in header
MAX_TASKS_PER_ROW = 3 # max task items per row


def _layout_task_lines(
    frame_task_info: list[tuple[int, str, str]],
) -> list[str]:
    """Layout task items into rows: max MAX_TASKS_PER_ROW per row, max MAX_TASK_ROWS rows.

    Each annotation is shown separately — same task with different ann_text
    are distinct items with their own seq_index. No deduplication.
    Returns list of text lines. If tasks exceed capacity, the last row shows
    a "+N more" summary. Returns ["none"] if no tasks.
    """
    if not frame_task_info:
        return ["none"]

    items: list[str] = []
    for seq_idx, task_label, ann_text in frame_task_info:
        if ann_text and len(ann_text) > MAX_ANN_TEXT_LEN:
            ann_text = ann_text[:MAX_ANN_TEXT_LEN] + ".."
        if ann_text:
            items.append(f"#{seq_idx} {task_label}: \"{ann_text}\"")
        else:
            items.append(f"#{seq_idx} {task_label}")

    if not items:
        return ["none"]

    max_items = MAX_TASK_ROWS * MAX_TASKS_PER_ROW
    truncated = len(items) - max_items if len(items) > max_items else 0
    if truncated > 0:
        items = items[:max_items]

    # Lay out into rows
    rows: list[str] = []
    for r in range(0, len(items), MAX_TASKS_PER_ROW):
        row_items = items[r:r + MAX_TASKS_PER_ROW]
        rows.append(" | ".join(row_items))

    if truncated > 0:
        rows[-1] += f"  +{truncated} more"

    return rows


def compose_fast_frame(
    frame_idx: int,
    render_len: int,
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    frame_task_info: list[tuple[int, str, str]],
    state: np.ndarray,
    action: np.ndarray,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    cam_max_h: int = 300,
    header_h: int = 50,
    info_panel_h: int = 180,
    cam_total_w: int = 1200,
) -> PILImage.Image:
    """Compose a single fast-mode frame using only PIL/numpy (no matplotlib).

    Layout:
      - Header: trajectory info + active task labels with sequential indices
      - Camera row: RGB static + gripper side by side
      - Info panel: state + action numerical values
    """
    # ── Camera row ────────────────────────────────────────────────────────
    images = [rgb_static, rgb_gripper]
    gap = 4
    usable_width = cam_total_w - gap
    per_cam_width = usable_width // 2

    scaled = []
    for img in images:
        h, w = img.shape[:2]
        new_w = per_cam_width
        new_h = int(h * new_w / w)
        pil = PILImage.fromarray(img)
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        scaled.append(pil)

    row_h = max(img.height for img in scaled)
    if row_h > cam_max_h:
        scale = cam_max_h / row_h
        rescaled = []
        for pil in scaled:
            new_w = int(pil.width * scale)
            new_h = int(pil.height * scale)
            pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
            rescaled.append(pil)
        scaled = rescaled
        row_h = cam_max_h

    cam_row = PILImage.new("RGB", (cam_total_w, row_h), color=(0, 0, 0))
    x_off = 0
    for pil in scaled:
        y_off = (row_h - pil.height) // 2
        cam_row.paste(pil, (x_off, y_off))
        x_off += pil.width + gap

    # ── Header: task info with sequential indices ────────────────────────
    task_rows = _layout_task_lines(frame_task_info)
    # Dynamic header height: frame line + task rows
    line_h = 18
    actual_header_h = line_h + len(task_rows) * line_h + 4
    actual_header_h = max(actual_header_h, header_h)  # at least the default

    header = PILImage.new("RGB", (cam_total_w, actual_header_h), color=(20, 20, 20))
    draw_hdr = ImageDraw.Draw(header)

    draw_hdr.text((10, 3), f"Frame {frame_idx}/{render_len}", fill=(200, 200, 200), font=font)
    y = 3 + line_h
    for row in task_rows:
        draw_hdr.text((10, y), row, fill=(255, 255, 255), font=font)
        y += line_h

    # ── Info panel: state + action values ─────────────────────────────────
    info = PILImage.new("RGB", (cam_total_w, info_panel_h), color=(30, 30, 30))
    draw_info = ImageDraw.Draw(info)

    # Action values
    action_vals = action.tolist() if isinstance(action, np.ndarray) else list(action)
    action_lines = _format_value_lines(ACTION_NAMES, action_vals, "Action", max_chars=cam_total_w // 8)
    y = 10
    for line in action_lines:
        draw_info.text((10, y), line, fill=(100, 255, 100), font=font)
        y += 18

    y += 10  # gap between action and state

    # State values
    state_vals = state.tolist() if isinstance(state, np.ndarray) else list(state)
    state_lines = _format_value_lines(STATE_NAMES, state_vals, "State ", max_chars=cam_total_w // 8)
    for line in state_lines:
        draw_info.text((10, y), line, fill=(100, 200, 255), font=font)
        y += 18

    # ── Compose full frame ────────────────────────────────────────────────
    total_h = actual_header_h + row_h + info_panel_h
    full = PILImage.new("RGB", (cam_total_w, total_h), color=(0, 0, 0))
    full.paste(header, (0, 0))
    full.paste(cam_row, (0, actual_header_h))
    full.paste(info, (0, actual_header_h + row_h))

    return full


def render_trajectory_video_fast(
    input_dir: str,
    traj: dict,
    analysis: dict,
    output_path: Path,
    fps_factor: float = 0.2,
    max_frames: int | None = None,
):
    """Render trajectory video using only PIL (no matplotlib).
    Layout: header + camera images + state/action info panel.
    Much faster than matplotlib-based rendering.
    """
    traj_start = traj["start"]
    traj_end = traj["end"]
    traj_len = traj_end - traj_start
    traj_idx = traj["trajectory_index"]

    font = _try_load_font(14)

    # Build per-frame task info from raw annotations (no midpoint split)
    segments = analysis["segments"]
    frame_task_info = build_frame_task_info(segments, traj_start, traj_len)

    # Pre-load all data upfront
    render_len = min(traj_len, max_frames) if max_frames else traj_len
    print(f"  Pre-loading {render_len} frames for trajectory {traj_idx} (fast mode)...")
    states, actions, rgb_statics, rgb_grippers = preload_trajectory_data(
        input_dir, traj_start, render_len
    )

    # Frame dimensions
    cam_total_w = 1200
    header_h = 50
    info_panel_h = 180
    cam_max_h = 300
    output_fps = max(1, int(30 * fps_factor))

    print(f"  Layout: header {header_h}px + cameras {cam_max_h}px + info {info_panel_h}px")
    print(f"  Rendering {render_len} frames at {output_fps}fps (fast mode)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Render all frames as PNGs, then encode with ffmpeg ────────────────
    tmp_dir = tempfile.mkdtemp(prefix="calvin_vis_fast_")
    print(f"  Writing {render_len} raw frames to temp dir...")

    # Determine total frame size from first frame
    first_frame = compose_fast_frame(
        0, render_len, rgb_statics[0], rgb_grippers[0],
        frame_task_info[0] if frame_task_info else [],
        states[0], actions[0], font,
        cam_max_h, header_h, info_panel_h, cam_total_w,
    )
    total_h = first_frame.height
    total_w = first_frame.width

    # Ensure even dimensions for h264
    total_w_even = total_w + (total_w % 2)
    total_h_even = total_h + (total_h % 2)

    for i in range(render_len):
        fti = frame_task_info[min(i, len(frame_task_info) - 1)]

        pil_frame = compose_fast_frame(
            i, render_len, rgb_statics[i], rgb_grippers[i],
            fti, states[i], actions[i], font,
            cam_max_h, header_h, info_panel_h, cam_total_w,
        )

        # Pad to even dimensions if needed
        if total_w_even != total_w or total_h_even != total_h:
            padded = PILImage.new("RGB", (total_w_even, total_h_even), color=(0, 0, 0))
            padded.paste(pil_frame, (0, 0))
            pil_frame = padded

        pil_frame.save(os.path.join(tmp_dir, f"frame_{i:06d}.png"))

        if (i + 1) % 100 == 0 or i == render_len - 1:
            print(f"  Rendered {i+1}/{render_len} frames")

    # ── ffmpeg encode ─────────────────────────────────────────────────────
    nvenc_available = _check_nvenc()
    codec = "h264_nvenc" if nvenc_available else "libx264"
    preset = "p1" if nvenc_available else "fast"
    print(f"  Encoding with ffmpeg ({codec}, preset={preset})...")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(output_fps),
        "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v", codec,
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(output_path),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if nvenc_available:
            print(f"  NVENC failed, falling back to libx264...")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(output_fps),
                "-i", os.path.join(tmp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-crf", "20",
                str(output_path),
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ffmpeg error:\n{result.stderr}")
        else:
            print(f"  ffmpeg error:\n{result.stderr}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Saved video: {output_path}  (fps={output_fps}, {render_len} frames, codec={codec})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Calvin raw dataset annotations and task transitions",
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to Calvin data directory (e.g., /data_16T/.../validation)")
    parser.add_argument("--output_dir", type=str, default="./calvin_annotation_vis",
                        help="Output directory for plots and videos")
    parser.add_argument("--trajectory-index", type=int, nargs="*", default=None,
                        help="Which trajectory(s) to analyze/render (default: all)")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "fast", "analysis"],
                        help="'full' = video (matplotlib) + analysis, 'fast' = PIL-only video + analysis, 'analysis' = text + static plots only")
    parser.add_argument("--fps-factor", type=float, default=0.2,
                        help="Video fps = 30 * fps_factor (default 0.2 = 6fps)")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max frames to render per trajectory (for long trajectories)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_dir}")

    # Load Calvin metadata
    trajectories = load_trajectory_boundaries(str(input_dir))
    annotations = load_lang_annotations(str(input_dir))

    print(f"Trajectories: {len(trajectories)}")
    for t in trajectories:
        print(f"  Trajectory {t['trajectory_index']}: [{t['start']}, {t['end']}) ({t['end']-t['start']} frames)")
    print(f"Annotations: {len(annotations)}")

    # Select trajectories to process
    if args.trajectory_index is not None:
        selected = [t for t in trajectories if t["trajectory_index"] in args.trajectory_index]
    else:
        selected = trajectories

    # Analyze each trajectory
    all_analyses = []
    for traj in selected:
        analysis = analyze_trajectory(annotations, traj)
        all_analyses.append(analysis)
        print_analysis_summary(analysis)

        # Generate static plots
        traj_idx = traj["trajectory_index"]
        plot_overlap_heatmap(analysis, output_dir / f"overlap_heatmap_traj{traj_idx}.png")
        plot_task_sequence_diagram(analysis, output_dir / f"task_sequence_traj{traj_idx}.png")
        plot_annotation_density(analysis, output_dir / f"annotation_density_traj{traj_idx}.png")

        # Render video (full and fast modes)
        if args.mode in ("full", "fast"):
            video_path = output_dir / f"calvin_annotations_traj{traj_idx}.mp4"
            if args.mode == "fast":
                render_trajectory_video_fast(
                    str(input_dir), traj, analysis, video_path,
                    fps_factor=args.fps_factor,
                    max_frames=args.max_frames,
                )
            else:
                render_trajectory_video(
                    str(input_dir), traj, analysis, video_path,
                    fps_factor=args.fps_factor,
                    max_frames=args.max_frames,
                )

    # Overall statistics
    print(f"\n{'='*60}")
    print(f"OVERALL STATISTICS")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Total annotations: {len(annotations)}")
    total_frames = sum(a["traj_len"] for a in all_analyses)
    total_annotated = sum(a["annotated_frames"] for a in all_analyses)
    total_unannotated = sum(a["unannotated_frames"] for a in all_analyses)
    total_multi = sum(a["multi_annotated_frames"] for a in all_analyses)
    print(f"  Total frames: {total_frames}")
    print(f"  Annotated: {total_annotated} ({total_annotated/total_frames*100:.1f}%)")
    print(f"  Unannotated: {total_unannotated} ({total_unannotated/total_frames*100:.1f}%)")
    print(f"  Multi-annotated: {total_multi}")

    # Unique task labels seen
    all_tasks = set()
    for a in all_analyses:
        all_tasks.update(a["task_sequence"])
    print(f"  Unique tasks: {len(all_tasks)} — {sorted(all_tasks)}")

    print(f"\nDone. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()