#!/usr/bin/env python3
"""Scan dataset and compute tier configuration for LoLA per-tier batching.

Uses calibrated coefficients from Phase 1a (calibrate_memory_cost.py) to
compute per-episode equivalent token cost, determine optimal tier boundaries,
and output a detailed data profile including frame-level distribution, cost
decomposition, estimated sequence lengths, and cost histogram.

The output JSON is used by Phase 2 training pipeline to configure per-tier
DataLoaders and dynamic gradient accumulation.

Usage:
    python src/lerobot/scripts/scan_dataset_tier_config.py \
        --dataset_root /data_6t_1/lerobot-v30/merged_0422_sub1/ \
        --calibration_path /data_6t_1/calibration_coefficients.json \
        --num_tiers 4 --tier_method kmeans \
        --output /data_6t_1/tier_config_merged_0422_sub1.json
"""

import argparse
import json
import math
import os
import time

import numpy as np


def smart_resize_qwen3vl(height, width, factor=32, min_pixels=65536, max_pixels=230400):
    """Exact replica of Qwen3.5-VL smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too large: {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def compute_visual_tokens(original_h, original_w, min_pixels=65536, max_pixels=230400):
    """Compute visual token count for one image at given resolution.

    For Qwen3.5-4B: num_tokens = h_bar * w_bar / 1024.
    """
    h_bar, w_bar = smart_resize_qwen3vl(original_h, original_w, factor=32,
                                          min_pixels=min_pixels, max_pixels=max_pixels)
    return h_bar * w_bar // 1024


def compute_optimal_tier_boundaries(costs, num_tiers=3, method="kmeans"):
    """Determine tier boundaries from cost distribution.

    Methods:
      "kmeans"  — 1D K-means clustering on costs. Boundaries at midpoints.
      "percentile" — Equal-count percentile split.

    Returns boundaries [lower_0, boundary_1, ..., inf].
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans
        costs_1d = costs.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_tiers, random_state=42, n_init=10)
        kmeans.fit(costs_1d)
        centers = sorted(kmeans.cluster_centers_.flatten())
        boundaries = [0.0]
        for i in range(len(centers) - 1):
            boundaries.append(float((centers[i] + centers[i + 1]) / 2))
        boundaries.append(float("inf"))
    elif method == "percentile":
        percentiles = [100 * i / num_tiers for i in range(1, num_tiers)]
        boundaries = [0.0]
        for p in percentiles:
            boundaries.append(float(np.percentile(costs, p)))
        boundaries.append(float("inf"))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'percentile'.")
    return boundaries


def classify_tier(cost, boundaries):
    """Classify an item into a tier based on cost and boundaries."""
    for i in range(len(boundaries) - 1):
        if cost < boundaries[i + 1]:
            return i
    return len(boundaries) - 2


def main():
    parser = argparse.ArgumentParser(description="Scan dataset and compute tier config")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--calibration_path", type=str, required=True,
                        help="Path to calibration JSON from Phase 1a")
    parser.add_argument("--max_image_pixels", type=int, default=230400)
    parser.add_argument("--min_image_pixels", type=int, default=65536)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--max_history_length", type=int, default=1024)
    parser.add_argument("--vlm_text_tokens", type=int, default=30,
                        help="Estimated VLM text tokens per frame (prompt template)")
    parser.add_argument("--num_tiers", type=int, default=3,
                        help="Number of length tiers")
    parser.add_argument("--tier_method", type=str, default="kmeans",
                        choices=["kmeans", "percentile"],
                        help="Method for determining tier boundaries")
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None,
                        help="Path to dataset-to-episodes JSON mapping (for sub-dataset info)")
    parser.add_argument("--scan_num_workers", type=int, default=8,
                        help="Number of workers for video metadata scan")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output tier config JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("LoLA Dataset Tier Configuration Scanner")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Calibration path: {args.calibration_path}")
    print(f"Num tiers: {args.num_tiers}")
    start_time = time.time()

    # ── Step 1: Load calibration coefficients ────────────────────
    print("\n--- Loading calibration coefficients ---")
    with open(args.calibration_path) as f:
        calibration = json.load(f)

    vision_tower_multiplier = calibration["coefficients"]["vision_tower_multiplier"]
    action_token_weight = calibration["coefficients"]["action_token_weight"]
    print(f"  vision_tower_multiplier: {vision_tower_multiplier:.4f}")
    print(f"  action_token_weight: {action_token_weight:.4f}")

    # ── Step 2: Load dataset metadata ────────────────────────────
    print("\n--- Loading dataset metadata ---")
    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.lola_pretrain_streaming_dataset import _load_episodes_polars, _discover_parquet_files

    meta_root = Path(args.dataset_root)
    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.repo_id = ""
    meta.revision = "v30"
    meta.root = meta_root
    meta.writer = None
    meta.latest_episode = None

    import json as _json
    with open(meta_root / "meta" / "info.json") as f:
        meta.info = _json.load(f)

    from lerobot.datasets.lerobot_dataset import load_tasks
    meta.tasks = load_tasks(meta_root)

    meta.episodes = _load_episodes_polars(str(meta_root))

    num_episodes = len(meta.episodes)
    camera_keys = [k for k in meta.info["features"] if meta.info["features"][k].get("dtype") == "video"]
    print(f"  Episodes: {num_episodes}")
    print(f"  Camera keys: {camera_keys}")

    # ── Step 3: Scan video resolutions ────────────────────────────
    print("\n--- Scanning video resolutions ---")
    from lerobot.datasets.video_utils import scan_video_metadata

    video_meta = scan_video_metadata(args.dataset_root, num_workers=args.scan_num_workers)
    video_resolution_map = {}
    for rel_path, vmeta in video_meta.items():
        h, w = vmeta["height"], vmeta["width"]
        visual_tokens = compute_visual_tokens(h, w, args.min_image_pixels, args.max_image_pixels) if h > 0 and w > 0 else 0
        video_resolution_map[rel_path] = {
            "seek_mode": vmeta["seek_mode"],
            "height": h,
            "width": w,
            "visual_tokens": visual_tokens,
        }
    print(f"  Scanned {len(video_resolution_map)} video files")

    # ── Step 4: Load sub-dataset info ────────────────────────────
    print("\n--- Loading sub-dataset info ---")
    sub_dataset_dims = [(args.pred_chunk_size, 0)]
    episode_to_ds_idx = np.full(num_episodes, 0, dtype=np.int16)

    if args.dataset_to_episodes_path and os.path.isfile(args.dataset_to_episodes_path):
        with open(args.dataset_to_episodes_path) as f:
            ds_episodes_map = json.load(f)

        sub_dataset_names = []
        sub_dataset_dims_list = []
        for ds_name, ds_info in ds_episodes_map.items():
            sub_dataset_names.append(ds_name)
            action_dim = ds_info.get("action_dim", 20)
            state_dim = ds_info.get("state_dim", 0)
            sub_dataset_dims_list.append((action_dim, state_dim))

            ep_start = ds_info.get("episode_start", 0)
            ep_end = ds_info.get("episode_end", num_episodes)
            for ep_idx in range(ep_start, ep_end):
                if ep_idx < num_episodes:
                    episode_to_ds_idx[ep_idx] = len(sub_dataset_names) - 1

        sub_dataset_dims = sub_dataset_dims_list
        print(f"  Sub-datasets: {len(sub_dataset_names)}")
        print(f"  Action dims: {[d[0] for d in sub_dataset_dims]}")

    # ── Step 4b: Load episode lengths (frame counts) ──────────────
    print("\n--- Loading episode lengths ---")
    episode_lengths = []
    for ep_idx in range(num_episodes):
        ep_meta = meta.episodes[ep_idx]
        length = ep_meta.get("length", 0)
        if length == 0:
            length = ep_meta.get("dataset_to_index", 0) - ep_meta.get("dataset_from_index", 0)
        episode_lengths.append(length)

    lengths_arr = np.array(episode_lengths)
    total_frames = int(lengths_arr.sum())
    print(f"  Total episodes: {num_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Avg frames/episode: {lengths_arr.mean():.1f}")
    print(f"  Frames/episode range: [{lengths_arr.min()}, {lengths_arr.max()}]")
    print(f"  Frames/episode std: {lengths_arr.std():.1f}")

    # ── Step 5: Compute per-episode cost with decomposition ───────
    print("\n--- Computing per-episode equivalent token cost (with decomposition) ---")
    video_path_template = meta.info.get("video_path",
                                         "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    episode_costs = []
    episode_visual_costs = []
    episode_action_costs = []
    episode_camera_counts = []
    episode_visual_tokens_per_cam = []  # per-camera visual tokens for each episode
    episode_vlm_seq_lengths = []
    episode_dit_seq_lengths = []

    pred_tokens = args.pred_chunk_size // args.action_chunk_size
    hist_tokens = args.max_history_length // args.action_chunk_size

    for ep_idx in range(num_episodes):
        ep_meta = meta.episodes[ep_idx]

        # Visual token cost
        visual_cost = 0.0
        valid_camera_count = 0
        total_visual_tokens = 0
        per_cam_tokens = {}
        for cam_key in camera_keys:
            is_valid = ep_meta.get(f"videos/{cam_key}/is_valid", 1)
            if is_valid == 1:
                valid_camera_count += 1
                chunk_idx = ep_meta.get(f"videos/{cam_key}/chunk_index", 0)
                file_idx = ep_meta.get(f"videos/{cam_key}/file_index", 0)
                video_rel = video_path_template.format(
                    video_key=cam_key, chunk_index=chunk_idx, file_index=file_idx)
                video_rel_key = video_rel[len("videos/"):] if video_rel.startswith("videos/") else video_rel

                n_tokens = 0
                if video_rel_key in video_resolution_map:
                    n_tokens = video_resolution_map[video_rel_key]["visual_tokens"]
                elif video_rel_key in video_meta:
                    h, w = video_meta[video_rel_key]["height"], video_meta[video_rel_key]["width"]
                    if h > 0 and w > 0:
                        n_tokens = compute_visual_tokens(h, w, args.min_image_pixels, args.max_image_pixels)
                    else:
                        shape = meta.info["features"][cam_key]["shape"]
                        n_tokens = compute_visual_tokens(shape[1], shape[2],
                                                         args.min_image_pixels, args.max_image_pixels)
                else:
                    shape = meta.info["features"][cam_key]["shape"]
                    n_tokens = compute_visual_tokens(shape[1], shape[2],
                                                     args.min_image_pixels, args.max_image_pixels)

                visual_cost += n_tokens
                total_visual_tokens += n_tokens
                per_cam_tokens[cam_key] = n_tokens

        visual_cost *= (1 + vision_tower_multiplier)

        # Action token cost
        ds_idx = int(episode_to_ds_idx[ep_idx])
        action_dim = sub_dataset_dims[ds_idx][0]
        action_cost = (pred_tokens + hist_tokens) * action_token_weight * action_dim

        total_cost = visual_cost + action_cost
        vlm_seq_len = args.vlm_text_tokens + total_visual_tokens
        dit_seq_len = pred_tokens * action_dim

        episode_costs.append(total_cost)
        episode_visual_costs.append(visual_cost)
        episode_action_costs.append(action_cost)
        episode_camera_counts.append(valid_camera_count)
        episode_visual_tokens_per_cam.append(per_cam_tokens)
        episode_vlm_seq_lengths.append(vlm_seq_len)
        episode_dit_seq_lengths.append(dit_seq_len)

    costs_arr = np.array(episode_costs)
    print(f"  Cost range: [{costs_arr.min():.0f}, {costs_arr.max():.0f}]")
    print(f"  Cost median: {np.median(costs_arr):.0f}, mean: {costs_arr.mean():.0f}, std: {costs_arr.std():.0f}")
    print(f"  Visual cost median: {np.median(episode_visual_costs):.0f}")
    print(f"  Action cost median: {np.median(episode_action_costs):.0f}")
    print(f"  Avg camera count: {np.mean(episode_camera_counts):.2f}")
    print(f"  VLM seq_len range: [{min(episode_vlm_seq_lengths)}, {max(episode_vlm_seq_lengths)}]")
    print(f"  DiT seq_len range: [{min(episode_dit_seq_lengths)}, {max(episode_dit_seq_lengths)}]")

    # ── Step 6: Determine optimal tier boundaries ─────────────────
    print("\n--- Determining optimal tier boundaries ---")
    tier_boundaries = compute_optimal_tier_boundaries(costs_arr, num_tiers=args.num_tiers, method=args.tier_method)
    print(f"  Method: {args.tier_method}")
    print(f"  Tier boundaries: {tier_boundaries}")

    # ── Step 7: Classify episodes into tiers with frame-level stats ─
    print("\n--- Tier statistics (episode + frame level) ---")
    episode_tiers = [classify_tier(c, tier_boundaries) for c in episode_costs]
    episode_tiers_arr = np.array(episode_tiers)
    tier_counts = np.bincount(episode_tiers, minlength=args.num_tiers)

    tier_stats = {}
    for i in range(args.num_tiers):
        mask = episode_tiers_arr == i
        tier_costs = costs_arr[mask]
        tier_lengths = lengths_arr[mask]
        tier_frames = int(tier_lengths.sum())
        tier_visual = np.array(episode_visual_costs)[mask]
        tier_action = np.array(episode_action_costs)[mask]
        tier_cams = np.array(episode_camera_counts)[mask]
        tier_vlm = np.array(episode_vlm_seq_lengths)[mask]
        tier_dit = np.array(episode_dit_seq_lengths)[mask]

        n = int(tier_counts[i])
        if n == 0:
            tier_stats[str(i)] = {"episode_count": 0, "frame_count": 0}
            continue

        tier_stats[str(i)] = {
            "episode_count": n,
            "episode_ratio": float(n / num_episodes),
            "frame_count": tier_frames,
            "frame_ratio": float(tier_frames / total_frames),
            "avg_cost": float(tier_costs.mean()),
            "std_cost": float(tier_costs.std()),
            "min_cost": float(tier_costs.min()),
            "max_cost": float(tier_costs.max()),
            "median_cost": float(np.median(tier_costs)),
            "p25_cost": float(np.percentile(tier_costs, 25)),
            "p75_cost": float(np.percentile(tier_costs, 75)),
            "p90_cost": float(np.percentile(tier_costs, 90)),
            "avg_frames_per_episode": float(tier_lengths.mean()),
            "std_frames_per_episode": float(tier_lengths.std()),
            "min_frames_per_episode": int(tier_lengths.min()),
            "max_frames_per_episode": int(tier_lengths.max()),
            "avg_visual_cost": float(tier_visual.mean()),
            "avg_action_cost": float(tier_action.mean()),
            "avg_camera_count": float(tier_cams.mean()),
            "min_camera_count": int(tier_cams.min()),
            "max_camera_count": int(tier_cams.max()),
            "avg_vlm_seq_len": float(tier_vlm.mean()),
            "std_vlm_seq_len": float(tier_vlm.std()),
            "min_vlm_seq_len": int(tier_vlm.min()),
            "max_vlm_seq_len": int(tier_vlm.max()),
            "avg_dit_seq_len": float(tier_dit.mean()),
            "std_dit_seq_len": float(tier_dit.std()),
            "min_dit_seq_len": int(tier_dit.min()),
            "max_dit_seq_len": int(tier_dit.max()),
        }

        print(f"  Tier {i}: {n} episodes ({n/num_episodes:.1%}), "
              f"{tier_frames} frames ({tier_frames/total_frames:.1%}), "
              f"avg_cost={tier_costs.mean():.0f}±{tier_costs.std():.0f} "
              f"[{tier_costs.min():.0f}-{tier_costs.max():.0f}], "
              f"avg_len={tier_lengths.mean():.0f}±{tier_lengths.std():.0f}f/ep, "
              f"cam={tier_cams.mean():.1f}[{tier_cams.min()}-{tier_cams.max()}], "
              f"vlm_seq={tier_vlm.mean():.0f}±{tier_vlm.std():.0f} "
              f"[{tier_vlm.min()}-{tier_vlm.max()}], "
              f"dit_seq={tier_dit.mean():.0f}±{tier_dit.std():.0f} "
              f"[{tier_dit.min()}-{tier_dit.max()}]")

    # ── Step 8: Cost histogram ────────────────────────────────────
    print("\n--- Computing cost histogram ---")
    hist_bins = 50
    cost_hist, cost_bin_edges = np.histogram(costs_arr, bins=hist_bins)
    frame_weighted_hist = np.zeros(hist_bins)
    for ep_idx in range(num_episodes):
        bin_idx = np.searchsorted(cost_bin_edges, episode_costs[ep_idx], side='right') - 1
        bin_idx = min(max(bin_idx, 0), hist_bins - 1)
        frame_weighted_hist[bin_idx] += episode_lengths[ep_idx]

    # VLM seq_len histogram
    vlm_arr = np.array(episode_vlm_seq_lengths)
    vlm_hist, vlm_bin_edges = np.histogram(vlm_arr, bins=30)
    vlm_frame_weighted = np.zeros(30)
    for ep_idx in range(num_episodes):
        bin_idx = np.searchsorted(vlm_bin_edges, episode_vlm_seq_lengths[ep_idx], side='right') - 1
        bin_idx = min(max(bin_idx, 0), 29)
        vlm_frame_weighted[bin_idx] += episode_lengths[ep_idx]

    print(f"  Cost histogram: {hist_bins} bins")
    print(f"  VLM seq_len histogram: 30 bins")

    # ── Step 9: Camera distribution summary ───────────────────────
    print("\n--- Camera count distribution ---")
    cam_counts_arr = np.array(episode_camera_counts)
    for cam_count in sorted(set(cam_counts_arr)):
        mask = cam_counts_arr == cam_count
        ep_count = int(mask.sum())
        frame_count = int(lengths_arr[mask].sum())
        avg_cost = float(costs_arr[mask].mean())
        avg_vlm = float(np.array(episode_vlm_seq_lengths)[mask].mean())
        print(f"  {cam_count} cameras: {ep_count} episodes ({ep_count/num_episodes:.1%}), "
              f"{frame_count} frames ({frame_count/total_frames:.1%}), "
              f"avg_cost={avg_cost:.0f}, avg_vlm_seq={avg_vlm:.0f}")

    # ── Step 10: Save to JSON ──────────────────────────────────────
    result = {
        "dataset_root": args.dataset_root,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "scan_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_elapsed_seconds": round(time.time() - start_time, 1),
        "params": {
            "calibration_path": args.calibration_path,
            "tier_method": args.tier_method,
            "num_tiers": args.num_tiers,
            "vision_tower_multiplier": vision_tower_multiplier,
            "action_token_weight": action_token_weight,
            "max_image_pixels": args.max_image_pixels,
            "min_image_pixels": args.min_image_pixels,
            "pred_chunk_size": args.pred_chunk_size,
            "action_chunk_size": args.action_chunk_size,
            "max_history_length": args.max_history_length,
            "vlm_text_tokens": args.vlm_text_tokens,
        },
        "video_resolution_map": video_resolution_map,
        "episode_costs": episode_costs,
        "episode_lengths": episode_lengths,
        "episode_visual_costs": episode_visual_costs,
        "episode_action_costs": episode_action_costs,
        "episode_camera_counts": episode_camera_counts,
        "episode_vlm_seq_lengths": episode_vlm_seq_lengths,
        "episode_dit_seq_lengths": episode_dit_seq_lengths,
        "episode_tiers": episode_tiers,
        "tier_boundaries": tier_boundaries,
        "tier_stats": tier_stats,
        "cost_histogram": {
            "bins": cost_bin_edges.tolist(),
            "episode_counts": cost_hist.tolist(),
            "frame_counts": frame_weighted_hist.tolist(),
        },
        "vlm_seq_len_histogram": {
            "bins": vlm_bin_edges.tolist(),
            "episode_counts": vlm_hist.tolist(),
            "frame_counts": vlm_frame_weighted.tolist(),
        },
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTier config saved to: {args.output}")
    elapsed = time.time() - start_time
    print(f"Total scan time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()