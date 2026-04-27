#!/usr/bin/env python
"""
Scan all video files in a LeRobot dataset and compare metadata num_frames
vs actual decodable frame count (via torchcodec probing).

Phase 1: Scan all videos with seek_mode="approximate" (same as decode path).
Phase 2: For any mismatch found, re-probe with seek_mode="exact" to determine
         whether the delta is caused by approximate seek or is a genuine
         metadata/video discrepancy.

Usage:
    python src/lerobot/scripts/scan_video_frame_count.py \
        --dataset_root /data_6t_1/lerobot-v30/merged_0422_sub1/ \
        --num_workers 8

Typical output:
    [OK]   observation.images.primary/chunk-000/file-000.mp4: meta=300000 approx=300000 delta=0
    [WARN] observation.images.primary/chunk-001/file-293.mp4: meta=304716 approx=304714 delta_approx=2 delta_exact=0  ← seek issue
    [WARN] observation.images.primary/chunk-001/file-291.mp4: meta=331900 approx=331898 delta_approx=2 delta_exact=2  ← genuine mismatch
"""

import argparse
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fsspec

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _probe_last_decodable(decoder, num_frames: int, probe_range: int = 10) -> int:
    """Probe from the end of a video to find the last decodable frame index."""
    actual_last = -1
    probe_start = max(0, num_frames - 1)
    probe_end = max(0, num_frames - 1 - probe_range)
    for idx in range(probe_start, probe_end - 1, -1):
        try:
            decoder.get_frames_at(indices=[idx])
            actual_last = idx
            break
        except RuntimeError:
            continue

    # Fallback: full scan if the last probe_range frames all fail
    if actual_last == -1:
        for idx in range(num_frames - 1, -1, -1):
            try:
                decoder.get_frames_at(indices=[idx])
                actual_last = idx
                break
            except RuntimeError:
                continue

    return actual_last


def probe_single_video(video_path: str) -> dict:
    """Probe one video with both approximate and exact seek modes."""
    from torchcodec.decoders import VideoDecoder

    result = {
        "path": video_path,
        "meta_num_frames": -1,
        "approx_last": -1,
        "exact_last": -1,
        "delta_approx": -1,
        "delta_exact": -1,
        "fps": 0.0,
        "error_approx": None,
        "error_exact": None,
    }

    # --- Phase 1: approximate seek ---
    try:
        fh = fsspec.open(video_path).__enter__()
        dec = VideoDecoder(fh, seek_mode="approximate")
        meta = dec.metadata
        result["meta_num_frames"] = meta.num_frames
        result["fps"] = float(meta.average_fps)
        num_frames = meta.num_frames

        approx_last = _probe_last_decodable(dec, num_frames)
        result["approx_last"] = approx_last
        if approx_last >= 0:
            result["delta_approx"] = (num_frames - 1) - approx_last
        fh.close()
    except Exception as e:
        result["error_approx"] = str(e)
        # Can't continue without metadata
        return result

    # --- Phase 2: exact seek (only if approximate found a mismatch) ---
    if result["delta_approx"] > 0:
        try:
            fh = fsspec.open(video_path).__enter__()
            dec = VideoDecoder(fh, seek_mode="exact")
            num_frames = result["meta_num_frames"]

            exact_last = _probe_last_decodable(dec, num_frames)
            result["exact_last"] = exact_last
            if exact_last >= 0:
                result["delta_exact"] = (num_frames - 1) - exact_last
            fh.close()
        except Exception as e:
            result["error_exact"] = str(e)

    return result


def discover_video_files(dataset_root: str) -> list[str]:
    """Discover all .mp4 video files under dataset_root/videos/."""
    videos_dir = os.path.join(dataset_root, "videos")
    if not os.path.isdir(videos_dir):
        print(f"ERROR: {videos_dir} does not exist")
        sys.exit(1)

    video_files = []
    for root, dirs, files in os.walk(videos_dir):
        for f in files:
            if f.endswith(".mp4"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, videos_dir)
                video_files.append((full_path, rel_path))

    video_files.sort(key=lambda x: x[1])
    return video_files


def main():
    parser = argparse.ArgumentParser(description="Scan video metadata vs actual frame count")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root directory (contains videos/ subdir)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers for probing")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every video result (default: only print mismatches)")
    args = parser.parse_args()

    video_files = discover_video_files(args.dataset_root)
    total = len(video_files)
    print(f"Found {total} video files under {args.dataset_root}/videos/")
    print(f"Probing with {args.num_workers} workers (approx + exact for mismatches)...")
    print()

    start_time = time.time()
    results = []
    ok_count = 0
    mismatch_count = 0
    seek_issue_count = 0
    genuine_mismatch_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(probe_single_video, full_path): rel_path
            for full_path, rel_path in video_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            rel_path = futures[future]
            r = future.result()
            results.append(r)

            if r["error_approx"] is not None:
                error_count += 1
                print(f"[ERR]  {rel_path}: {r['error_approx']}")
            elif r["delta_approx"] == 0:
                ok_count += 1
                if args.verbose:
                    print(f"[OK]   {rel_path}: meta={r['meta_num_frames']} delta=0")
            elif r["delta_approx"] > 0:
                mismatch_count += 1
                da = r["delta_approx"]
                de = r["delta_exact"] if r["delta_exact"] >= 0 else "?"
                de_raw = r["delta_exact"] if r["delta_exact"] >= 0 else -1

                if r["error_exact"]:
                    tag = "exact_failed"
                    print(f"[WARN] {rel_path}: meta={r['meta_num_frames']} "
                          f"approx_last={r['approx_last']} delta_approx={da} "
                          f"exact_probe_failed: {r['error_exact']}")
                elif r["delta_exact"] == 0:
                    tag = "seek_issue"
                    seek_issue_count += 1
                    print(f"[WARN] {rel_path}: meta={r['meta_num_frames']} "
                          f"approx_last={r['approx_last']} delta_approx={da} "
                          f"exact_last={r['exact_last']} delta_exact=0  ← SEEK ISSUE")
                else:
                    tag = "genuine"
                    genuine_mismatch_count += 1
                    print(f"[WARN] {rel_path}: meta={r['meta_num_frames']} "
                          f"approx_last={r['approx_last']} delta_approx={da} "
                          f"exact_last={r['exact_last']} delta_exact={de}  ← GENUINE MISMATCH")
            else:
                # delta < 0 (more actual frames than metadata)
                mismatch_count += 1
                print(f"[???]  {rel_path}: meta={r['meta_num_frames']} "
                      f"approx_last={r['approx_last']} delta_approx={r['delta_approx']}")

            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                speed = i / max(elapsed, 1e-6)
                eta = (total - i) / max(speed, 1e-6)
                print(f"  Progress: {i}/{total} ({i/total*100:.1f}%), "
                      f"{speed:.1f} vid/s, ETA {eta:.0f}s")

    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print("Scan Summary")
    print("=" * 60)
    print(f"Total videos:         {total}")
    print(f"OK (delta=0):         {ok_count}")
    print(f"Mismatch (approx):    {mismatch_count}")
    print(f"  ├─ Seek issue only: {seek_issue_count}  (delta_exact=0, approximate seek can't reach last frames)")
    print(f"  └─ Genuine mismatch:{genuine_mismatch_count}  (delta_exact>0, metadata vs actual truly differ)")
    print(f"Errors:               {error_count}")
    print(f"Time:                 {elapsed:.1f}s ({elapsed/max(total,1):.2f}s/video)")

    if mismatch_count > 0:
        print()
        print("--- Mismatching videos ---")
        print(f"{'Path':<60} {'meta':>8} {'approx_last':>12} {'Δapprox':>8} {'exact_last':>11} {'Δexact':>7} {'Verdict':<18}")
        print("-" * 130)
        for r in results:
            if r["delta_approx"] > 0 and r["error_approx"] is None:
                rel = os.path.relpath(r["path"], os.path.join(args.dataset_root, "videos"))
                da = r["delta_approx"]
                de = r["delta_exact"] if r["delta_exact"] >= 0 else "N/A"
                el = r["exact_last"] if r["exact_last"] >= 0 else "N/A"
                if r["error_exact"]:
                    verdict = "exact_failed"
                elif r["delta_exact"] == 0:
                    verdict = "SEEK ISSUE"
                else:
                    verdict = "GENUINE MISMATCH"
                print(f"  {rel:<58} {r['meta_num_frames']:>8} {r['approx_last']:>12} {da:>8} {str(el):>11} {str(de):>7} {verdict:<18}")

    # Save JSON
    if args.output_json:
        output_data = {
            "dataset_root": args.dataset_root,
            "total_videos": total,
            "ok_count": ok_count,
            "mismatch_count_approx": mismatch_count,
            "seek_issue_count": seek_issue_count,
            "genuine_mismatch_count": genuine_mismatch_count,
            "error_count": error_count,
            "mismatch_details": [
                {
                    "path": os.path.relpath(r["path"], os.path.join(args.dataset_root, "videos")),
                    "meta_num_frames": r["meta_num_frames"],
                    "approx_last": r["approx_last"],
                    "delta_approx": r["delta_approx"],
                    "exact_last": r["exact_last"],
                    "delta_exact": r["delta_exact"],
                    "fps": r["fps"],
                    "error_exact": r["error_exact"],
                    "verdict": (
                        "SEEK ISSUE" if r["delta_exact"] == 0 and not r["error_exact"]
                        else "GENUINE MISMATCH" if r.get("delta_exact", -1) > 0
                        else "exact_failed"
                    ),
                }
                for r in results if r["delta_approx"] > 0 and r["error_approx"] is None
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print()
    if mismatch_count > 0:
        if seek_issue_count > 0:
            print(f"Found {seek_issue_count} videos where approximate seek causes false mismatch "
                  f"(exact seek resolves them).")
        if genuine_mismatch_count > 0:
            print(f"Found {genuine_mismatch_count} videos with genuine metadata/actual frame count mismatch!")
    else:
        print("All videos have consistent metadata vs actual frame counts.")


if __name__ == "__main__":
    main()