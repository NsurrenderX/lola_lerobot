#!/usr/bin/env python3
"""Recompute stats.json for Calvin lerobot datasets from actual parquet data.

Fixes the stats.json swap bug where v2's stats.json contained v3's statistics
and vice versa. Recomputes per-feature statistics from actual parquet data.

Usage:
    python recompute_calvin_meta.py --dataset_dir /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v2
    python recompute_calvin_meta.py --dataset_dir /data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v3
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def compute_stats(values: np.ndarray) -> dict:
    """Compute per-dimension statistics for a feature."""
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_dims = values.shape[1]
    stats = {
        "min": [float(values[:, d].min()) for d in range(n_dims)],
        "max": [float(values[:, d].max()) for d in range(n_dims)],
        "mean": [float(values[:, d].mean()) for d in range(n_dims)],
        "std": [float(values[:, d].std()) for d in range(n_dims)],
        "count": [int(values.shape[0])],
        "q01": [float(np.percentile(values[:, d], 1)) for d in range(n_dims)],
        "q10": [float(np.percentile(values[:, d], 10)) for d in range(n_dims)],
        "q50": [float(np.percentile(values[:, d], 50)) for d in range(n_dims)],
        "q90": [float(np.percentile(values[:, d], 90)) for d in range(n_dims)],
        "q99": [float(np.percentile(values[:, d], 99)) for d in range(n_dims)],
    }
    return stats


def compute_stats_from_parquet(dataset_dir: str) -> dict:
    """Read parquet data and compute per-feature statistics."""
    dataset_dir = Path(dataset_dir)

    # Find all parquet files (only file*.parquet, not temp files)
    data_dir = dataset_dir / "data"
    parquet_files = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
            for f in sorted(chunk_dir.iterdir()):
                if f.name.startswith("file") and f.suffix == ".parquet":
                    parquet_files.append(f)

    if not parquet_files:
        raise FileNotFoundError(f"No file*.parquet found in {data_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Read and concatenate all data
    all_data = {}
    total_rows = 0
    for pf in parquet_files:
        table = pq.read_table(str(pf))
        logger.info(f"  {pf}: {table.num_rows} rows")
        total_rows += table.num_rows

        for col_name in table.column_names:
            col_data = table.column(col_name)
            if col_name in ("action", "observation.state"):
                values = np.array([row.as_py() for row in col_data], dtype=np.float64)
            elif col_name in ("episode_index", "frame_index", "index", "task_index", "timestamp"):
                values = np.array([row.as_py() if hasattr(row, 'as_py') else float(row) for row in col_data], dtype=np.float64)
            else:
                continue  # Skip video/image columns

            if col_name not in all_data:
                all_data[col_name] = []
            all_data[col_name].append(values)

    logger.info(f"Total rows: {total_rows}")

    # Concatenate
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)

    # Compute stats for each feature
    stats_dict = {}
    for key, values in all_data.items():
        logger.info(f"Computing stats for {key}...")
        stats_dict[key] = compute_stats(values)

    # Add placeholder for video/image features (min=0, max=1, mean/std from info)
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        for feat_name, feat_info in info.get("features", {}).items():
            if feat_info.get("dtype") == "video" and feat_name not in stats_dict:
                stats_dict[feat_name] = {
                    "min": [[[0.0], [0.0], [0.0]]],
                    "max": [[[1.0], [1.0], [1.0]]],
                    "mean": [[[0.65], [0.55], [0.47]]],
                    "std": [[[0.11], [0.11], [0.09]]],
                    "count": [total_rows],
                    "q01": [[[0.05], [0.01], [0.02]]],
                    "q10": [[[0.45], [0.31], [0.22]]],
                    "q50": [[[0.64], [0.47], [0.32]]],
                    "q90": [[[0.92], [0.90], [0.90]]],
                    "q99": [[[0.996], [0.996], [0.996]]],
                }

    return stats_dict


def main():
    parser = argparse.ArgumentParser(description="Recompute stats.json from actual parquet data")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to lerobot dataset directory")
    parser.add_argument("--dry_run", action="store_true", help="Print stats without writing to file")
    args = parser.parse_args()

    stats = compute_stats_from_parquet(args.dataset_dir)

    if args.dry_run:
        print(json.dumps(stats, indent=2))
        return

    # Write to stats.json
    stats_path = Path(args.dataset_dir) / "meta" / "stats.json"
    logger.info(f"Writing stats to {stats_path}")

    # Backup existing stats
    if stats_path.exists():
        backup_path = stats_path.with_suffix(".json.bak")
        logger.info(f"Backing up existing stats to {backup_path}")
        import shutil
        shutil.copy2(str(stats_path), str(backup_path))

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats recomputed and saved successfully")

    # Verify key stats
    if "observation.state" in stats:
        state_grip = stats["observation.state"]["min"][6]
        state_grip_max = stats["observation.state"]["max"][6]
        if state_grip == -1.0 and state_grip_max == 1.0:
            logger.info("  observation.state dim6: binary gripper {-1, 1}")
        else:
            logger.info(f"  observation.state dim6: gripper width range [{state_grip:.4f}, {state_grip_max:.4f}]")

    if "action" in stats:
        action_count = stats["action"]["count"][0]
        logger.info(f"  action count: {action_count}")


if __name__ == "__main__":
    main()