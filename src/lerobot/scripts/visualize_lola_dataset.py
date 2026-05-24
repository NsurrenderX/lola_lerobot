#!/usr/bin/env python

"""
Visualize LoLADataset episodes as annotated mp4 videos.

Each video frame combines:
  - Camera observation images (side-by-side)
  - Action & state values (text overlay)
  - Task description & episode/frame metadata (header)
  - V2: transition zone indicator, completed tasks display

Output video runs at fps_factor * dataset.fps (default 0.5 = half original fps).

Usage:
  # Basic (V1 dataset)
  python visualize_lola_dataset.py \
      --repo-id lerobot/pusht \
      --num-to-render 3 \
      --output-dir outputs/lola_viz

  # V2 Calvin dataset with transition frames + completed tasks
  python visualize_lola_dataset.py \
      --root /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v4 \
      --episode-indices 0 5 10 \
      --track-completed-tasks \
      --output-dir outputs/lola_viz
"""

import argparse
import json
import random
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lola_dataset import LoLADataset


# ── Helpers ──────────────────────────────────────────────────────────────────


def to_hwc_uint8_numpy(chw_float32: torch.Tensor) -> np.ndarray:
    """(C, H, W) float32 [0,1] → (H, W, C) uint8."""
    return (chw_float32 * 255).type(torch.uint8).permute(1, 2, 0).cpu().numpy()


def flatten_dim_names(names_val, fallback_prefix: str, dim: int) -> list[str]:
    """Flatten meta.names entry (dict-of-lists, list, or None) into [str] of length dim."""
    if names_val is None:
        return [f"{fallback_prefix}{i}" for i in range(dim)]
    if isinstance(names_val, list):
        return names_val if len(names_val) == dim else [f"{fallback_prefix}{i}" for i in range(dim)]
    if isinstance(names_val, dict):
        flat = []
        for sub in names_val.values():
            if isinstance(sub, list):
                flat.extend(sub)
        return flat if len(flat) == dim else [f"{fallback_prefix}{i}" for i in range(dim)]
    return [f"{fallback_prefix}{i}" for i in range(dim)]


def format_value_lines(
    names: list[str],
    values: list[float],
    prefix: str,
    max_chars_per_line: int = 100,
) -> list[str]:
    """Wrap value pairs into multiple text lines."""
    lines: list[str] = []
    cur = f"{prefix}: "
    for name, val in zip(names, values):
        entry = f"{name}={val:+.3f}  "
        if len(cur) + len(entry) > max_chars_per_line:
            lines.append(cur)
            cur = f"    {entry}"
        else:
            cur += entry
    lines.append(cur)
    return lines


def make_invalid_placeholder(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((width // 4, height // 4), "INVALID", fill=(255, 255, 255))
    return img


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


# ── Frame composition ─────────────────────────────────────────────────────────


def compose_frame(
    item: dict,
    dataset: LoLADataset,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    action_names: list[str],
    state_names: list[str],
    episode_metadata: dict | None = None,
    ep_idx: int | None = None,
    header_h: int = 60,
    info_panel_h: int = 200,
) -> Image.Image:
    """Compose a single annotated video frame from a dataset item."""

    # ── Camera images ────────────────────────────────────────────────────────
    cam_keys = dataset.meta.camera_keys
    cam_images: list[Image.Image] = []

    if cam_keys:
        first_cam_shape = dataset.meta.info["features"][cam_keys[0]]["shape"]
        target_h = first_cam_shape[1] if len(first_cam_shape) >= 2 else 480

        for cam_key in cam_keys:
            chw = item.get(cam_key)
            if chw is None or not isinstance(chw, torch.Tensor) or chw.ndim != 3:
                shape = dataset.meta.info["features"].get(cam_key, {}).get("shape", [3, 480, 640])
                w, h = shape[2], shape[1]
                placeholder = make_invalid_placeholder(w, h)
                placeholder = placeholder.resize(
                    (int(w * target_h / h), target_h), Image.LANCZOS
                )
                cam_images.append(placeholder)
                continue

            hwc = to_hwc_uint8_numpy(chw)
            pil = Image.fromarray(hwc)
            if pil.height != target_h:
                new_w = int(pil.width * target_h / pil.height)
                pil = pil.resize((new_w, target_h), Image.LANCZOS)
            cam_images.append(pil)
    else:
        cam_images.append(make_invalid_placeholder(640, 480))

    # ── Camera row ──────────────────────────────────────────────────────────
    cam_total_w = sum(img.width for img in cam_images)
    cam_max_h = max(img.height for img in cam_images)
    cam_row = Image.new("RGB", (cam_total_w, cam_max_h), color=(0, 0, 0))
    x_off = 0
    for img in cam_images:
        cam_row.paste(img, (x_off, 0))
        x_off += img.width

    frame_width = cam_total_w

    # ── Header ──────────────────────────────────────────────────────────────
    header = Image.new("RGB", (frame_width, header_h), color=(0, 0, 0))
    draw_hdr = ImageDraw.Draw(header)

    task_text = str(item.get("task", "N/A"))
    bbox = draw_hdr.textbbox((0, 0), task_text, font=font)
    if bbox[2] - bbox[0] > frame_width - 20:
        task_text = task_text[: max(1, int((frame_width - 50) / 6))] + "..."

    ep_idx_val = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]
    frame_idx = item["frame_index"].item() if isinstance(item["frame_index"], torch.Tensor) else item["frame_index"]
    ts = item["timestamp"].item() if isinstance(item["timestamp"], torch.Tensor) else item["timestamp"]

    draw_hdr.text((10, 3), f"Task: {task_text}", fill=(255, 255, 255), font=font)
    draw_hdr.text(
        (10, 20),
        f"Ep: {ep_idx_val}  Frame: {frame_idx}  Timestamp: {ts:.3f}s",
        fill=(200, 200, 200),
        font=font,
    )

    # V2: hist_len + completed tasks info
    transition_info = ""
    completed_info = ""
    if episode_metadata and ep_idx is not None and str(ep_idx) in episode_metadata:
        ep_meta = episode_metadata[str(ep_idx)]
        hist_len = ep_meta.get("hist_len", 0)
        annotation_len = ep_meta.get("annotation_len", 0)
        completed_tasks = ep_meta.get("completed_tasks", [])

        transition_info = f"HistLen: {hist_len}  Annotation: {annotation_len}  Frame: {frame_idx}/{annotation_len - 1}"

        if completed_tasks:
            # Show up to 5 most recently completed tasks (reverse order)
            shown = list(reversed(completed_tasks[-5:]))
            completed_info = f"Completed ({len(completed_tasks)}): " + ", ".join(shown)
            if len(completed_tasks) > 5:
                completed_info += f" +{len(completed_tasks) - 5} earlier"
        else:
            completed_info = "Completed: (none)"

    if transition_info:
        draw_hdr.text((10, 37), transition_info, fill=(255, 200, 100), font=font)
    if completed_info:
        # Truncate if too wide
        bbox = draw_hdr.textbbox((0, 0), completed_info, font=font)
        if bbox[2] - bbox[0] > frame_width - 20:
            completed_info = completed_info[: max(1, int((frame_width - 50) / 6))] + "..."
        draw_hdr.text((10, 50), completed_info, fill=(150, 200, 255), font=font)

    # ── Info panel (action + state) ──────────────────────────────────────────
    info = Image.new("RGB", (frame_width, info_panel_h), color=(30, 30, 30))
    draw_info = ImageDraw.Draw(info)

    # Action values
    action = item["action"]
    action_vals = action.tolist() if isinstance(action, torch.Tensor) else action
    action_lines = format_value_lines(action_names, action_vals, "Action", max_chars_per_line=frame_width // 8)
    y = 10
    for line in action_lines:
        draw_info.text((10, y), line, fill=(100, 255, 100), font=font)
        y += 18

    y += 10  # gap between action and state

    # State values
    state = item.get("observation.state")
    if state is not None:
        state_vals = state.tolist() if isinstance(state, torch.Tensor) else state
        state_lines = format_value_lines(state_names, state_vals, "State ", max_chars_per_line=frame_width // 8)
        for line in state_lines:
            draw_info.text((10, y), line, fill=(100, 200, 255), font=font)
            y += 18

    # V2: completed_tasks_ann (randomly selected for this sample)
    completed_tasks_ann = item.get("completed_tasks_ann")
    if completed_tasks_ann:
        y += 10
        draw_info.text((10, y), "Completed tasks (ann, recent first):", fill=(200, 200, 100), font=font)
        y += 18
        for i, ann in enumerate(reversed(completed_tasks_ann[-8:])):
            draw_info.text((20, y), f"{i + 1}. {ann}", fill=(220, 200, 130), font=font)
            y += 16
        if len(completed_tasks_ann) > 8:
            draw_info.text((20, y), f"  ... +{len(completed_tasks_ann) - 8} earlier", fill=(180, 180, 100), font=font)
            y += 16

    # ── Compose full frame ──────────────────────────────────────────────────
    total_h = header_h + cam_max_h + info_panel_h
    full = Image.new("RGB", (frame_width, total_h), color=(0, 0, 0))
    full.paste(header, (0, 0))
    full.paste(cam_row, (0, header_h))
    full.paste(info, (0, header_h + cam_max_h))

    return full


# ── Episode rendering ─────────────────────────────────────────────────────────


def render_episode(
    dataset: LoLADataset,
    ep_idx: int,
    output_path: Path,
    fps_factor: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    episode_metadata: dict | None = None,
) -> None:
    """Render one episode as an annotated mp4 video."""
    ep = dataset.meta.episodes[ep_idx]
    from_idx = ep["dataset_from_index"]
    to_idx = ep["dataset_to_index"]

    # Pre-compute dimension names (constant per dataset)
    action_dim = dataset.action_dim
    action_names_raw = dataset.meta.names.get("action", None)
    action_names = flatten_dim_names(action_names_raw, "a", action_dim)

    state_dim = dataset.state_dim if hasattr(dataset, "state_dim") and dataset.state_dim else 1
    state_names_raw = dataset.meta.names.get("observation.state", None)
    state_names = flatten_dim_names(state_names_raw, "s", state_dim)

    # V2: check episode metadata for hist_len and annotation_len
    hist_len = 0
    annotation_len = 0
    if episode_metadata and str(ep_idx) in episode_metadata:
        ep_meta = episode_metadata[str(ep_idx)]
        hist_len = ep_meta.get("hist_len", 0)
        annotation_len = ep_meta.get("annotation_len", 0)

    total_frames = to_idx - from_idx
    print(f"  Rendering episode {ep_idx}: {total_frames} frames "
          f"(hist={hist_len}, annotation={annotation_len}) "
          f"[indices {from_idx}..{to_idx - 1}]")

    # Collect all frames
    frames: list[np.ndarray] = []
    for idx in range(from_idx, to_idx):
        item = dataset[idx]
        pil_frame = compose_frame(
            item, dataset, font, action_names, state_names,
            episode_metadata=episode_metadata, ep_idx=ep_idx,
        )
        frames.append(np.array(pil_frame))

    # Determine output dimensions and fps (h264 requires even dimensions)
    h, w, _ = frames[0].shape
    w = w + (w % 2)
    h = h + (h % 2)
    output_fps = max(1, int(dataset.fps * fps_factor))

    # Write video with PyAV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), "w") as container:
        stream = container.add_stream("h264", rate=output_fps)
        stream.pix_fmt = "yuv420p"
        stream.width = w
        stream.height = h
        stream.codec_context.options = {"crf": "23", "preset": "medium"}

        for frame_np in frames:
            fh, fw, _ = frame_np.shape
            if fw != w or fh != h:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:fh, :fw, :] = frame_np
                frame_np = padded
            video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    print(f"  Saved: {output_path}  (fps={output_fps}, {total_frames} frames, {w}x{h})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Visualize LoLADataset episodes as annotated mp4 videos")
    parser.add_argument("--repo-id", default=None, help="Dataset repository ID")
    parser.add_argument("--root", default=None, help="Local dataset root directory")
    parser.add_argument("--num-to-render", type=int, default=3, help="Number of episodes to randomly render")
    parser.add_argument("--episode-indices", type=int, nargs="*", default=None, help="Specific episode indices (overrides random)")
    parser.add_argument("--output-dir", default="outputs/lola_viz", help="Output directory")
    parser.add_argument("--max-history-length", type=int, default=100, help="LoLADataset max_history_length")
    parser.add_argument("--action-chunk-size", type=int, default=10, help="LoLADataset action_chunk_size")
    parser.add_argument("--video-backend", default="torchcodec", help="Video backend: torchcodec (default), pyav")
    parser.add_argument("--fps-factor", type=float, default=0.5, help="Output fps = dataset.fps * fps_factor")
    parser.add_argument("--tolerance-frame", type=int, default=2,
                        help="Frame-level tolerance for video timestamp matching (default: 2)")
    parser.add_argument("--tolerance-s", type=float, default=None,
                        help="Timestamp tolerance in seconds (default: auto from tolerance_frame/fps)")
    # V2 params
    parser.add_argument("--track-completed-tasks", action="store_true", default=False,
                        help="Enable completed tasks tracking (V2 dataset)")
    parser.add_argument("--completed-tasks-use-ann", action="store_true", default=True,
                        help="Use descriptive 'ann' text for completed tasks")
    parser.add_argument("--no-completed-tasks-use-ann", action="store_true",
                        help="Use concise 'task' label instead of 'ann'")
    parser.add_argument("--transition-mask-rate", type=float, default=0.0,
                        help="Mask rate for transition-dominant hist tokens (0=no mask)")
    parser.add_argument("--hist-action-token-drop-rate", type=float, default=0.0,
                        help="Drop rate for task-dominant hist tokens")
    parser.add_argument("--history-type", type=str, default="action", choices=["action", "state"])
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--norm-mode", type=str, default="default",
                        choices=["default", "robovlm", "zscore"],
                        help="Normalization mode")
    parser.add_argument("--norm-min", type=float, default=-0.65)
    parser.add_argument("--norm-max", type=float, default=0.65)
    args = parser.parse_args()

    # ── Normalization setup ──────────────────────────────────────────────────
    norm_action = False
    if args.norm_mode == "robovlm":
        norm_action = True
    elif args.norm_mode == "zscore":
        norm_action = "zscore"

    completed_tasks_use_ann = not args.no_completed_tasks_use_ann if args.no_completed_tasks_use_ann else args.completed_tasks_use_ann

    # ── Instantiate dataset ──────────────────────────────────────────────────
    ds_kwargs = dict(
        repo_id=args.repo_id,
        root=args.root,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        video_backend=args.video_backend,
        norm_action=norm_action,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
        history_type=args.history_type,
        state_dim=args.state_dim,
        tolerance_frame=args.tolerance_frame,
        track_completed_tasks=args.track_completed_tasks,
        transition_mask_rate=args.transition_mask_rate,
        completed_tasks_use_ann=completed_tasks_use_ann,
        hist_action_token_drop_rate=args.hist_action_token_drop_rate,
    )
    if args.tolerance_s is not None:
        ds_kwargs["tolerance_s"] = args.tolerance_s
    dataset = LoLADataset(**ds_kwargs)

    print(f"Dataset: {args.repo_id or args.root}  fps={dataset.fps}  episodes={len(dataset.meta.episodes)}")

    # ── Load V2 episode metadata ─────────────────────────────────────────────
    episode_metadata = None
    if args.root:
        metadata_path = Path(args.root) / "calvin_episode_metadata.json"
    else:
        # Try to find metadata from dataset root
        metadata_path = Path(dataset.root) / "calvin_episode_metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            episode_metadata = json.load(f)
        print(f"Loaded V2 episode metadata: {len(episode_metadata)} episodes")
    else:
        print("No calvin_episode_metadata.json found (V1 dataset mode)")

    # ── Select episodes ──────────────────────────────────────────────────────
    num_episodes = len(dataset.meta.episodes)
    if args.episode_indices is not None:
        ep_indices = args.episode_indices
    else:
        ep_indices = random.sample(range(num_episodes), min(args.num_to_render, num_episodes))

    print(f"Rendering episodes: {ep_indices}")

    # ── Load font ────────────────────────────────────────────────────────────
    font = _try_load_font(14)

    # ── Render each episode ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if args.repo_id is not None:
        repo_id_sanitized = args.repo_id.replace("/", "_")
    elif args.root:
        repo_id_sanitized = Path(args.root).name
    else:
        repo_id_sanitized = "dataset"

    for ep_idx in ep_indices:
        output_path = output_dir / f"{repo_id_sanitized}_episode_{ep_idx}.mp4"
        render_episode(dataset, ep_idx, output_path, args.fps_factor, font,
                       episode_metadata=episode_metadata)

    print(f"\nDone. Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
