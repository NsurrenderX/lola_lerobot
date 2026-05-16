#!/usr/bin/env python

"""
Visualize LoLADataset episode trajectories as 3D EEF trajectory + gripper state videos.

Each video shows:
  - Camera observation images (side-by-side, large, full resolution)
  - 3D EEF trajectory (absolute position from observation.state), animated progressively
  - Trajectory color-coded by gripper state (red=closed, green=open)
  - Gripper state timeline panel (obs gripper continuous + action gripper binary)
  - Roll/Pitch/Yaw orientation timeline panel

When cameras are enabled, camera images are composited at full pixel resolution above
the matplotlib plots. Supports 2-4 cameras with dynamic layout.

Usage:
  python visualize_lola_trajectory_3d.py \
      --root /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v3 \
      --num-to-render 3 \
      --output-dir outputs/lola_trajectory_3d

  python visualize_lola_trajectory_3d.py \
      --root /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v3 \
      --episode-indices 0 5 10 \
      --rotate-camera \
      --no-cameras
"""

import argparse
import random
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames


# ── Colors ────────────────────────────────────────────────────────────────────

COLOR_GRIPPER_OPEN = "#2ca02c"       # green
COLOR_GRIPPER_CLOSED = "#d62728"     # red
COLOR_CURRENT_POS = "#1f77b4"        # blue
COLOR_START_MARKER = "#333333"       # dark gray
COLOR_FUTURE = (0.5, 0.5, 0.5, 0.3) # gray, transparent
COLOR_FRAME_LINE = "#1f77b4"         # blue vertical line

RPY_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # roll=red, pitch=green, yaw=blue
RPY_LABELS = ["Roll", "Pitch", "Yaw"]


# ── Data extraction ───────────────────────────────────────────────────────────


def extract_episode_trajectory(dataset: LeRobotDataset, ep_idx: int) -> dict:
    """Extract observation.state and action for all frames in one episode."""
    ep = dataset.meta.episodes[ep_idx]
    from_idx = ep["dataset_from_index"]
    to_idx = ep["dataset_to_index"]

    states = torch.stack(dataset.hf_dataset["observation.state"][from_idx:to_idx]).numpy()
    actions = torch.stack(dataset.hf_dataset["action"][from_idx:to_idx]).numpy()
    timestamps = torch.stack(dataset.hf_dataset["timestamp"][from_idx:to_idx]).tolist()

    return {
        "states": states,
        "actions": actions,
        "timestamps": timestamps,
        "task": ep["tasks"],
        "num_frames": to_idx - from_idx,
    }


def extract_episode_camera_frames(
    dataset: LeRobotDataset,
    ep_idx: int,
    timestamps: list[float],
) -> dict[str, list[np.ndarray]]:
    """Decode all camera frames for one episode. Returns {cam_key: [H,W,C] uint8 list}."""
    ep = dataset.meta.episodes[ep_idx]
    cam_keys = dataset.meta.camera_keys
    if not cam_keys:
        return {}

    cam_frames: dict[str, list[np.ndarray]] = {}
    for vid_key in cam_keys:
        from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
        shifted_ts = [from_timestamp + ts for ts in timestamps]
        video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)

        frames_tensor = decode_video_frames(
            video_path, shifted_ts, dataset.tolerance_s,
            tolerance_frames=dataset.tolerance_frames,
            backend=dataset.video_backend,
        )
        frames_np = (frames_tensor * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        cam_frames[vid_key] = [frames_np[i] for i in range(frames_np.shape[0])]

    return cam_frames


def classify_gripper_from_action(action_gripper: np.ndarray) -> np.ndarray:
    """Classify gripper from action dim (exactly -1 or +1). Returns bool: True=open."""
    return action_gripper > 0


# ── Camera image compositing ─────────────────────────────────────────────────


def compose_camera_row(
    cam_images: list[np.ndarray],
    target_width: int,
    max_height: int = 500,
) -> np.ndarray:
    """Compose 2-4 camera images into a single row, filling target_width.

    All cameras are scaled to the same height (determined by the tallest camera
    after width-based scaling). If the resulting height exceeds max_height, all
    images are further downscaled. Images are placed side-by-side with small gaps,
    centered in target_width. Supports 2-4 cameras.

    Args:
        cam_images: list of (H, W, 3) uint8 arrays, one per camera
        target_width: total width of the output row in pixels
        max_height: maximum height of the camera row in pixels

    Returns:
        (H_out, target_width, 3) uint8 array
    """
    from PIL import Image as PILImage

    num_cams = len(cam_images)
    gap = 4

    # Divide width equally among cameras
    usable_width = target_width - gap * (num_cams - 1)
    per_cam_width = usable_width // num_cams

    # Scale each camera image to its allocated width, preserving aspect ratio
    scaled: list[np.ndarray] = []
    for img in cam_images:
        h, w = img.shape[:2]
        new_w = per_cam_width
        new_h = int(h * new_w / w)
        pil = PILImage.fromarray(img)
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        scaled.append(np.array(pil))

    # Uniform height = max of all scaled heights
    row_h = max(img.shape[0] for img in scaled)

    # If exceeds max_height, downscale all to fit
    if row_h > max_height:
        scale = max_height / row_h
        rescaled: list[np.ndarray] = []
        for img in scaled:
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            pil = PILImage.fromarray(img)
            pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
            rescaled.append(np.array(pil))
        scaled = rescaled
        row_h = max_height

    # Compose into one row (white background)
    row = np.full((row_h, target_width, 3), 255, dtype=np.uint8)
    x_off = 0
    for img in scaled:
        h, w = img.shape[:2]
        y_off = (row_h - h) // 2
        row[y_off:y_off + h, x_off:x_off + w, :] = img
        x_off += w + gap

    return row


# ── Figure setup (plots only, no camera) ─────────────────────────────────────


def setup_figure(
    ep_idx: int,
    task_name: str,
    xyz: np.ndarray,
    rpy: np.ndarray,
    obs_gripper: np.ndarray,
    action_gripper_open: np.ndarray,
    gripper_threshold: float,
):
    """Create figure with 3D trajectory + side panels (no camera row)."""
    n = len(obs_gripper)

    fig = plt.figure(figsize=(14, 6), dpi=100, facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.5, 1],
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.3,
        left=0.06, right=0.96, top=0.92, bottom=0.08,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_grip = fig.add_subplot(gs[0, 1])
    ax_rpy = fig.add_subplot(gs[1, 1])

    margin = 0.02
    ax3d.set_xlim(xyz[:, 0].min() - margin, xyz[:, 0].max() + margin)
    ax3d.set_ylim(xyz[:, 1].min() - margin, xyz[:, 1].max() + margin)
    ax3d.set_zlim(xyz[:, 2].min() - margin, xyz[:, 2].max() + margin)
    ax3d.set_xlabel("X (m)", fontsize=9)
    ax3d.set_ylabel("Y (m)", fontsize=9)
    ax3d.set_zlabel("Z (m)", fontsize=9)
    ax3d.tick_params(labelsize=7)
    ax3d.view_init(elev=25, azim=45)
    ax3d.set_facecolor("white")
    ax_grip.set_facecolor("white")
    ax_rpy.set_facecolor("white")

    ax_grip.set_xlim(-0.5, n - 0.5)
    ax_grip.set_ylim(-1.5, 1.5)
    ax_grip.set_xlabel("Frame", fontsize=8)
    ax_grip.set_ylabel("Value", fontsize=8)
    ax_grip.tick_params(labelsize=7)

    rpy_min, rpy_max = rpy.min(), rpy.max()
    rpy_margin = max(0.1, (rpy_max - rpy_min) * 0.1)
    ax_rpy.set_xlim(-0.5, n - 0.5)
    ax_rpy.set_ylim(rpy_min - rpy_margin, rpy_max + rpy_margin)
    ax_rpy.set_xlabel("Frame", fontsize=8)
    ax_rpy.set_ylabel("Radians", fontsize=8)
    ax_rpy.tick_params(labelsize=7)

    fig.suptitle(f"Episode {ep_idx}: {task_name}", fontsize=13, fontweight="bold")

    return fig, ax3d, ax_grip, ax_rpy


# ── Per-frame rendering (plots only) ─────────────────────────────────────────


def render_frame(
    fig, ax3d, ax_grip, ax_rpy,
    xyz, rpy, obs_gripper, action_gripper, action_gripper_open,
    i, total_frames, gripper_threshold, rotate_camera,
    xlim3d, ylim3d, zlim3d,
    rpy_ylim,
):
    """Render one animation frame (frame index i) -- matplotlib plots only."""

    n = total_frames
    frame_idx = np.arange(n)

    # ── 3D trajectory ──────────────────────────────────────────────────────
    ax3d.cla()
    ax3d.set_facecolor("white")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("lightgray")
    ax3d.yaxis.pane.set_edgecolor("lightgray")
    ax3d.zaxis.pane.set_edgecolor("lightgray")
    ax3d.set_xlim(*xlim3d)
    ax3d.set_ylim(*ylim3d)
    ax3d.set_zlim(*zlim3d)
    ax3d.set_xlabel("X (m)", fontsize=9)
    ax3d.set_ylabel("Y (m)", fontsize=9)
    ax3d.set_zlabel("Z (m)", fontsize=9)
    ax3d.tick_params(labelsize=7)

    for j in range(min(i, n - 1)):
        color = COLOR_GRIPPER_OPEN if action_gripper_open[j] else COLOR_GRIPPER_CLOSED
        ax3d.plot(
            xyz[j : j + 2, 0], xyz[j : j + 2, 1], xyz[j : j + 2, 2],
            color=color, linewidth=2.5, solid_capstyle="round",
        )

    if i < n - 1:
        ax3d.plot(
            xyz[i:, 0], xyz[i:, 1], xyz[i:, 2],
            color=COLOR_FUTURE, linewidth=1.0, linestyle="--",
        )

    ax3d.scatter(*xyz[0], s=80, c=COLOR_START_MARKER, marker="^", zorder=10)
    ax3d.scatter(*xyz[i], s=140, c=COLOR_CURRENT_POS, marker="o", zorder=11,
                 edgecolors="white", linewidths=1.5)

    if rotate_camera:
        ax3d.view_init(elev=25, azim=45 + i * 1.0)

    ax3d.set_title(f"EEF Trajectory  (frame {i}/{n - 1})", fontsize=10)

    # ── Gripper panel ──────────────────────────────────────────────────────
    ax_grip.cla()
    ax_grip.set_facecolor("white")

    ax_grip.fill_between(
        frame_idx, -1.5, 1.5,
        where=action_gripper_open, color=COLOR_GRIPPER_OPEN, alpha=0.15,
    )
    ax_grip.fill_between(
        frame_idx, -1.5, 1.5,
        where=~action_gripper_open, color=COLOR_GRIPPER_CLOSED, alpha=0.15,
    )

    obs_grip_scaled = (obs_gripper / max(obs_gripper.max(), 0.08)) * 1.0
    ax_grip.plot(frame_idx[: i + 1], obs_grip_scaled[: i + 1], "k-", linewidth=1.5, label="Obs gripper")
    ax_grip.plot(frame_idx[: i + 1], action_gripper[: i + 1], "m-", linewidth=1.2,
                 alpha=0.8, label="Action gripper")

    if i < n - 1:
        ax_grip.plot(frame_idx[i:], obs_grip_scaled[i:], color="gray", linewidth=0.5, alpha=0.3)
        ax_grip.plot(frame_idx[i:], action_gripper[i:], color="gray", linewidth=0.5, alpha=0.3)

    ax_grip.axvline(x=i, color=COLOR_FRAME_LINE, linewidth=1.5, linestyle="--")

    ax_grip.set_xlim(-0.5, n - 0.5)
    ax_grip.set_ylim(-1.5, 1.5)
    ax_grip.set_xlabel("Frame", fontsize=8)
    ax_grip.set_ylabel("Value", fontsize=8)
    ax_grip.tick_params(labelsize=7)
    ax_grip.set_title("Gripper State", fontsize=9)
    ax_grip.legend(loc="upper right", fontsize=7)

    # ── RPY panel ──────────────────────────────────────────────────────────
    ax_rpy.cla()
    ax_rpy.set_facecolor("white")

    for d in range(3):
        ax_rpy.plot(
            frame_idx[: i + 1], rpy[: i + 1, d],
            color=RPY_COLORS[d], linewidth=1.2, label=RPY_LABELS[d],
        )
        if i < n - 1:
            ax_rpy.plot(
                frame_idx[i:], rpy[i:, d],
                color=RPY_COLORS[d], linewidth=0.5, alpha=0.3,
            )

    ax_rpy.axvline(x=i, color="black", linewidth=1, linestyle="--")
    ax_rpy.set_xlim(-0.5, n - 0.5)
    ax_rpy.set_ylim(*rpy_ylim)
    ax_rpy.set_xlabel("Frame", fontsize=8)
    ax_rpy.set_ylabel("Radians", fontsize=8)
    ax_rpy.tick_params(labelsize=7)
    ax_rpy.set_title("Orientation (RPY)", fontsize=9)
    ax_rpy.legend(loc="upper right", fontsize=7)

    fig.canvas.draw_idle()


def extract_plot_frame(fig: plt.Figure) -> np.ndarray:
    """Extract the matplotlib figure as an (H, W, 3) uint8 numpy array."""
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()


# ── Video rendering ──────────────────────────────────────────────────────────


def render_trajectory_video(
    dataset: LeRobotDataset,
    ep_idx: int,
    output_path: Path,
    fps_factor: float,
    gripper_threshold: float,
    rotate_camera: bool,
    with_cameras: bool,
):
    """Render one episode trajectory as an mp4 video."""
    data = extract_episode_trajectory(dataset, ep_idx)
    states = data["states"]
    actions = data["actions"]
    timestamps = data["timestamps"]
    task_name = data["task"]
    n = data["num_frames"]

    xyz = states[:, :3]
    rpy = states[:, 3:6]
    obs_gripper = states[:, 6]
    action_gripper = actions[:, 6]
    action_gripper_open = classify_gripper_from_action(action_gripper)

    # Pre-compute axis limits
    margin = 0.02
    xlim3d = (xyz[:, 0].min() - margin, xyz[:, 0].max() + margin)
    ylim3d = (xyz[:, 1].min() - margin, xyz[:, 1].max() + margin)
    zlim3d = (xyz[:, 2].min() - margin, xyz[:, 2].max() + margin)
    rpy_min, rpy_max = rpy.min(), rpy.max()
    rpy_margin = max(0.1, (rpy_max - rpy_min) * 0.1)
    rpy_ylim = (rpy_min - rpy_margin, rpy_max + rpy_margin)

    # Decode camera frames if needed
    cam_frames = None
    cam_keys = dataset.meta.camera_keys
    if with_cameras and cam_keys:
        print(f"  Decoding camera frames for {len(cam_keys)} cameras...")
        cam_frames = extract_episode_camera_frames(dataset, ep_idx, timestamps)

    # Setup matplotlib figure (plots only)
    fig, ax3d, ax_grip, ax_rpy = setup_figure(
        ep_idx, task_name, xyz, rpy, obs_gripper, action_gripper_open,
        gripper_threshold,
    )

    # Pre-render first plot frame to get plot pixel dimensions
    render_frame(
        fig, ax3d, ax_grip, ax_rpy,
        xyz, rpy, obs_gripper, action_gripper, action_gripper_open,
        0, n, gripper_threshold, rotate_camera,
        xlim3d, ylim3d, zlim3d, rpy_ylim,
    )
    fig.canvas.draw()
    plot_img = extract_plot_frame(fig)
    plot_h, plot_w = plot_img.shape[:2]

    # Compute final frame dimensions
    # Camera row max height: cameras are the main focus, allow up to ~65% of
    # total frame height (which will be cam_row + gap + plot_h)
    # This means cam_max_h ~ 1.86 * plot_h when cameras need the space
    cam_max_h = int(plot_h * 1.0)  # cameras can be as tall as the plots

    if with_cameras and cam_frames is not None:
        cam_target_w = plot_w
        first_cam_imgs = [cam_frames[key][0] for key in cam_keys]
        cam_row_sample = compose_camera_row(first_cam_imgs, cam_target_w, cam_max_h)
        cam_row_h = cam_row_sample.shape[0]
        gap = 4
        total_h = cam_row_h + gap + plot_h
        total_w = plot_w
    else:
        cam_row_h = 0
        gap = 0
        total_h = plot_h
        total_w = plot_w

    print(f"  Rendering episode {ep_idx}: {n} frames, task: {task_name}")
    if with_cameras and cam_frames is not None:
        print(f"  Layout: cameras {cam_row_h}px + plots {plot_h}px = {total_h}px total, {total_w}px wide")

    output_fps = max(1, int(dataset.fps * fps_factor))

    # Stream encode to mp4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), "w") as container:
        stream = container.add_stream("h264", rate=output_fps)
        stream.pix_fmt = "yuv420p"
        # H.264 requires even dimensions
        stream.width = total_w + (total_w % 2)
        stream.height = total_h + (total_h % 2)
        stream.codec_context.options = {"crf": "20", "preset": "medium"}

        for i in range(n):
            # Render matplotlib plots
            render_frame(
                fig, ax3d, ax_grip, ax_rpy,
                xyz, rpy, obs_gripper, action_gripper, action_gripper_open,
                i, n, gripper_threshold, rotate_camera,
                xlim3d, ylim3d, zlim3d, rpy_ylim,
            )
            fig.canvas.draw()
            plot_img = extract_plot_frame(fig)

            # Compose final frame
            if with_cameras and cam_frames is not None:
                cam_imgs = [cam_frames[key][i] for key in cam_keys]
                cam_row = compose_camera_row(cam_imgs, total_w, cam_max_h)

                full = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
                full[:cam_row_h, :, :] = cam_row
                full[cam_row_h + gap:, :, :] = plot_img
            else:
                full = plot_img

            # Pad to even dimensions if needed
            fh, fw = full.shape[:2]
            ew = fw + (fw % 2)
            eh = fh + (fh % 2)
            if ew != fw or eh != fh:
                padded = np.full((eh, ew, 3), 255, dtype=np.uint8)
                padded[:fh, :fw, :] = full
                full = padded

            video_frame = av.VideoFrame.from_ndarray(full, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    plt.close(fig)
    print(f"  Saved: {output_path}  (fps={output_fps}, {n} frames)")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LoLADataset episode trajectories as 3D EEF + gripper videos",
    )
    parser.add_argument("--repo-id", default=None, help="Dataset repository ID")
    parser.add_argument("--root", default=None, help="Local dataset root directory")
    parser.add_argument("--num-to-render", type=int, default=3,
                        help="Number of episodes to randomly render (default 3)")
    parser.add_argument("--episode-indices", type=int, nargs="*", default=None,
                        help="Specific episode indices (overrides random selection)")
    parser.add_argument("--output-dir", default="outputs/lola_trajectory_3d",
                        help="Output directory for mp4 videos")
    parser.add_argument("--fps-factor", type=float, default=0.2,
                        help="Output fps = dataset.fps * fps_factor (default 0.2 = 5x slow-motion)")
    parser.add_argument("--rotate-camera", action="store_true",
                        help="Slowly rotate the 3D camera during animation")
    parser.add_argument("--gripper-threshold", type=float, default=0.04,
                        help="Threshold for classifying gripper from observation.state (default 0.04)")
    parser.add_argument("--no-cameras", action="store_true",
                        help="Disable camera observation images in the video")
    args = parser.parse_args()

    with_cameras = not args.no_cameras

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )

    num_episodes = len(dataset.meta.episodes)
    print(f"Dataset: {args.repo_id}  fps={dataset.fps}  episodes={num_episodes}")
    if with_cameras:
        print(f"Camera keys: {dataset.meta.camera_keys}")

    if args.episode_indices is not None:
        ep_indices = args.episode_indices
    else:
        ep_indices = random.sample(range(num_episodes), min(args.num_to_render, num_episodes))

    print(f"Rendering episodes: {ep_indices}")

    output_dir = Path(args.output_dir)
    for ep_idx in ep_indices:
        repo_id_sanitized = (args.repo_id or "dataset").replace("/", "_")
        output_path = output_dir / f"{repo_id_sanitized}_trajectory_ep{ep_idx}.mp4"
        render_trajectory_video(
            dataset, ep_idx, output_path,
            fps_factor=args.fps_factor,
            gripper_threshold=args.gripper_threshold,
            rotate_camera=args.rotate_camera,
            with_cameras=with_cameras,
        )

    print(f"\nDone. Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
