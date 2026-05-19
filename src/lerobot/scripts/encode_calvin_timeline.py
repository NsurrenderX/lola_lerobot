#!/usr/bin/env python
"""Encode Calvin dataset timeline as one continuous MP4 video.

Reads all episode_*.npz files from a Calvin dataset directory in temporal order,
extracts rgb_static images, and encodes them into a single video using PyAV.

Usage:
    # Encode all frames from validation set
    python encode_calvin_timeline.py \
        --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/validation \
        --output_path /tmp/calvin_val_timeline.mp4

    # Encode first 1000 frames with custom codec params
    python encode_calvin_timeline.py \
        --input_dir /data_16T/deepseek/calvin_abc_d/task_ABC_D/validation \
        --output_path /tmp/calvin_val_1000.mp4 \
        --encode_length 1000 --overwrite --crf 20
"""

import argparse
import logging
import sys
from pathlib import Path

import av
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def discover_npz_files(input_dir: Path) -> list[Path]:
    """Find all episode_*.npz files sorted by 7-digit temporal index."""
    files = sorted(input_dir.glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    return files


def encode_timeline_video(
    npz_files: list[Path],
    output_path: Path,
    fps: int,
    vcodec: str,
    crf: str,
    preset: str,
    overwrite: bool,
) -> None:
    """Encode rgb_static frames from NPZ files into one MP4 video."""
    if output_path.exists() and not overwrite:
        logger.warning(f"Output file already exists: {output_path}. Use --overwrite to replace.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calvin rgb_static is 200x200x3 — even dimensions, h264-compatible
    width, height = 200, 200

    av.logging.set_level(av.logging.ERROR)

    total_frames = len(npz_files)
    logger.info(f"Encoding {total_frames} frames to {output_path} "
                f"(codec={vcodec}, fps={fps}, crf={crf}, preset={preset})")

    with av.open(str(output_path), "w") as container:
        stream = container.add_stream(vcodec, rate=fps)
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        stream.codec_context.options = {"crf": crf, "preset": preset}

        for frame_idx, npz_path in enumerate(npz_files):
            data = np.load(str(npz_path), allow_pickle=True)

            try:
                rgb_static = data["rgb_static"]
            except KeyError:
                raise KeyError(
                    f"NPZ file {npz_path} does not contain 'rgb_static' key. "
                    f"Available keys: {list(data.keys())}"
                )

            if rgb_static.shape != (height, width, 3):
                raise ValueError(
                    f"Unexpected rgb_static shape in {npz_path}: {rgb_static.shape}, "
                    f"expected ({height}, {width}, 3)"
                )

            if rgb_static.dtype != np.uint8:
                rgb_static = rgb_static.astype(np.uint8)
            rgb_static = np.ascontiguousarray(rgb_static)

            video_frame = av.VideoFrame.from_ndarray(rgb_static, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

            if (frame_idx + 1) % 1000 == 0 or frame_idx + 1 == total_frames:
                logger.info(f"  Encoded {frame_idx + 1}/{total_frames} frames")

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

    if not output_path.exists():
        raise OSError(f"Video encoding failed — output file not found: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Done: {output_path} ({file_size_mb:.2f} MB, {total_frames} frames, "
                f"{vcodec}/{fps}fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Encode Calvin dataset timeline as one MP4 video (rgb_static from NPZ)"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to Calvin dataset directory (containing episode_*.npz)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output MP4 video path")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS (default: 30)")
    parser.add_argument("--encode_length", type=int, default=None,
                        help="Limit total frames processed (default: all frames)")
    parser.add_argument("--vcodec", type=str, default="h264",
                        help="Video codec (default: h264)")
    parser.add_argument("--crf", type=str, default="23",
                        help="CRF quality value (default: 23)")
    parser.add_argument("--preset", type=str, default="medium",
                        help="Encoding preset (default: medium)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output file if it exists")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    npz_files = discover_npz_files(input_dir)
    if not npz_files:
        logger.error(f"No episode_*.npz files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} NPZ files in {input_dir}")

    if args.encode_length is not None:
        if args.encode_length <= 0:
            logger.error("encode_length must be a positive integer")
            sys.exit(1)
        if args.encode_length < len(npz_files):
            npz_files = npz_files[:args.encode_length]
            logger.info(f"Limiting to first {args.encode_length} frames")
        else:
            logger.info(f"encode_length={args.encode_length} >= total files "
                        f"({len(npz_files)}), processing all")

    encode_timeline_video(
        npz_files=npz_files,
        output_path=output_path,
        fps=args.fps,
        vcodec=args.vcodec,
        crf=args.crf,
        preset=args.preset,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()