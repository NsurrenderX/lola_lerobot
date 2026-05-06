#!/usr/bin/env python

"""
Visualize LoLADataset episodes as annotated mp4 videos.

Each video frame combines:
  - Camera observation images (side-by-side)
  - Action & state values (text overlay)
  - Task description & episode/frame metadata (header)

Output video runs at fps_factor * dataset.fps (default 0.5 = half original fps).

Usage:
  python visualize_lola_dataset.py \
      --repo-id lerobot/pusht \
      --num-to-render 3 \
      --output-dir outputs/lola_viz

  python visualize_lola_dataset.py \
      --repo-id cup_full_plus \
      --root /path/to/data \
      --episode-indices 0 5 10
"""

import argparse
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
    header_h: int = 40,
    info_panel_h: int = 200,
) -> Image.Image:
    """Compose a single annotated video frame from a dataset item."""

    # ── Camera images ────────────────────────────────────────────────────────
    cam_keys = dataset.meta.camera_keys
    cam_images: list[Image.Image] = []

    if cam_keys:
        # Determine target height from first camera's metadata shape
        first_cam_shape = dataset.meta.info["features"][cam_keys[0]]["shape"]
        # shape is (C, H, W) for video/image features
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
        # No cameras — create a blank placeholder
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
    # Truncate task if it's too wide
    bbox = draw_hdr.textbbox((0, 0), task_text, font=font)
    if bbox[2] - bbox[0] > frame_width - 20:
        task_text = task_text[: max(1, int((frame_width - 50) / 6))] + "..."

    ep_idx = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]
    frame_idx = item["frame_index"].item() if isinstance(item["frame_index"], torch.Tensor) else item["frame_index"]
    ts = item["timestamp"].item() if isinstance(item["timestamp"], torch.Tensor) else item["timestamp"]

    draw_hdr.text((10, 5), f"Task: {task_text}", fill=(255, 255, 255), font=font)
    draw_hdr.text(
        (10, 22),
        f"Ep: {ep_idx}  Frame: {frame_idx}  Timestamp: {ts:.3f}s",
        fill=(200, 200, 200),
        font=font,
    )

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
) -> None:
    """Render one episode as an annotated mp4 video."""
    ep = dataset.meta.episodes[ep_idx]
    from_idx = ep["dataset_from_index"]
    to_idx = ep["dataset_to_index"]

    # Pre-compute dimension names (constant per dataset)
    action_dim = dataset.action_dim
    action_names_raw = dataset.meta.names.get("action", None)
    action_names = flatten_dim_names(action_names_raw, "a", action_dim)

    state_dim = 1
    state_names_raw = None
    state_names = [f"s{i}" for i in range(1)]
    if "observation.state" in dataset.meta.names:
        state_names_raw = dataset.meta.names["observation.state"]
        # Determine state_dim from first item
        sample = dataset[from_idx]
        if "observation.state" in sample and isinstance(sample["observation.state"], torch.Tensor):
            state_dim = sample["observation.state"].shape[0]
        state_names = flatten_dim_names(state_names_raw, "s", state_dim)

    # Collect all frames
    frames: list[np.ndarray] = []
    total_frames = to_idx - from_idx
    print(f"  Rendering episode {ep_idx}: {total_frames} frames (indices {from_idx}..{to_idx - 1})")

    for idx in range(from_idx, to_idx):
        item = dataset[idx]
        pil_frame = compose_frame(item, dataset, font, action_names, state_names)
        frames.append(np.array(pil_frame))  # (H, W, 3) uint8

    # Determine output dimensions and fps (h264 requires even dimensions)
    h, w, _ = frames[0].shape
    w = w + (w % 2)  # round up to nearest even width
    h = h + (h % 2)  # round up to nearest even height
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
            # Pad to even dimensions if needed
            fh, fw, _ = frame_np.shape
            if fw != w or fh != h:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:fh, :fw, :] = frame_np
                frame_np = padded
            video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        # Flush encoder
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
    parser.add_argument("--video-backend", default="pyav", help="Video backend (default: pyav)")
    parser.add_argument("--fps-factor", type=float, default=0.5, help="Output fps = dataset.fps * fps_factor")
    args = parser.parse_args()

    # ── Instantiate dataset (no delta_timestamps → single-frame data) ────────
    
    dataset = LoLADataset(
        repo_id=args.repo_id,
        root=args.root,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        video_backend=args.video_backend,
    )

    print(f"Dataset: {args.repo_id}  fps={dataset.fps}  episodes={len(dataset.meta.episodes)}")

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
    else:
        repo_id_sanitized = "dataset"

    for ep_idx in ep_indices:
        output_path = output_dir / f"{repo_id_sanitized}_episode_{ep_idx}.mp4"
        render_episode(dataset, ep_idx, output_path, args.fps_factor, font)

    print(f"\nDone. Videos saved to {output_dir}")


if __name__ == "__main__":
    main()