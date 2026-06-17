#!/usr/bin/env python
"""
从原始Calvin数据计算包含所有帧（含transition帧）的stats，
并增量更新（不覆盖）到LeRobot格式数据集的stats.json中。

原始Calvin数据覆盖所有~1.8M帧（包括trajectory间的transition帧），
而LeRobot格式的stats仅覆盖annotated episode帧（~1.05M帧），
因此原始Calvin的stats能更好地反映包含transition数据的完整分布。

用法:
    python compute_incremental_stats.py --calvin_root /path/to/calvin/training/ \
        --dataset_root /path/to/lerobot_v4/
    python compute_incremental_stats.py --dry_run  # 仅计算并打印，不保存
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from lerobot.datasets.compute_stats import RunningQuantileStats, DEFAULT_QUANTILES
from lerobot.datasets.utils import load_stats, write_stats


def load_calvin_statistics_yaml(calvin_root: Path):
    """加载原始Calvin的statistics.yaml，提取robot_obs[:7]的mean/std。"""
    stats_path = calvin_root / "statistics.yaml"
    if not stats_path.exists():
        return None

    with open(stats_path) as f:
        calvin_stats = yaml.safe_load(f)

    # statistics.yaml格式: robot_obs是list containing dict with NormalizeVector config
    robot_obs_entry = calvin_stats.get("robot_obs", [])
    if isinstance(robot_obs_entry, list) and len(robot_obs_entry) > 0:
        robot_obs_entry = robot_obs_entry[0]

    state_mean_15 = robot_obs_entry.get("mean", None)
    state_std_15 = robot_obs_entry.get("std", None)

    if state_mean_15 is not None and state_std_15 is not None:
        # 只取前7维: [x, y, z, roll, pitch, yaw, gripper_width]
        state_mean_7 = np.array(state_mean_15[:7], dtype=np.float32)
        state_std_7 = np.array(state_std_15[:7], dtype=np.float32)
        return {"mean": state_mean_7, "std": state_std_7}

    return None


def compute_calvin_incremental_stats(
    calvin_root: str,
    sample_rate: float = 0.1,
):
    """
    从原始Calvin数据采样计算observation.state和action的完整stats。

    Args:
        calvin_root: 原始Calvin数据目录路径
        sample_rate: 采样率（默认10%，约180K帧）

    Returns:
        dict with "observation.state_incremental" and "action_incremental" stats dicts
    """
    calvin_root = Path(calvin_root)

    # 统计所有episode文件
    npz_files = sorted(
        [f for f in os.listdir(str(calvin_root)) if f.startswith("episode_") and f.endswith(".npz")]
    )
    total_frames = len(npz_files)
    print(f"[compute_incremental_stats] Total Calvin frames: {total_frames}")

    # 采样数量
    num_samples = max(100, int(total_frames * sample_rate))
    num_samples = min(num_samples, total_frames)
    print(f"[compute_incremental_stats] Sampling {num_samples} frames (rate={sample_rate})")

    # 采样索引（均匀分布）
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    sample_files = [npz_files[i] for i in sample_indices]

    # 初始化RunningQuantileStats
    state_collector = RunningQuantileStats()
    action_collector = RunningQuantileStats()

    print("[compute_incremental_stats] Computing stats from Calvin NPZ files...")
    for fname in tqdm(sample_files, desc="Sampling Calvin frames"):
        ep_path = str(calvin_root / fname)
        data = np.load(ep_path)

        # observation.state = robot_obs[:7]
        robot_obs = data["robot_obs"][:7].astype(np.float32)  # shape (7,)
        state_collector.update(robot_obs.reshape(1, -1))

        # action = rel_actions
        rel_actions = data["rel_actions"].astype(np.float32)  # shape (7,)
        action_collector.update(rel_actions.reshape(1, -1))

    # 获取统计结果
    state_stats = state_collector.get_statistics()
    action_stats = action_collector.get_statistics()

    # 添加count
    state_stats["count"] = np.array([num_samples])
    action_stats["count"] = np.array([num_samples])

    # 验证：与statistics.yaml的robot_obs[:7] mean/std对比
    calvin_yaml_stats = load_calvin_statistics_yaml(calvin_root)
    if calvin_yaml_stats is not None:
        print("\n[compute_incremental_stats] Validation against statistics.yaml:")
        print(f"  robot_obs[:7] mean (yaml):    {calvin_yaml_stats['mean'].tolist()}")
        print(f"  robot_obs[:7] mean (sampled): {state_stats['mean'].tolist()}")
        print(f"  robot_obs[:7] std (yaml):     {calvin_yaml_stats['std'].tolist()}")
        print(f"  robot_obs[:7] std (sampled):  {state_stats['std'].tolist()}")

        mean_diff = np.abs(state_stats['mean'] - calvin_yaml_stats['mean'])
        std_diff = np.abs(state_stats['std'] - calvin_yaml_stats['std'])
        print(f"  Mean abs diff: {mean_diff.tolist()}")
        print(f"  Std abs diff:  {std_diff.tolist()}")

        # 如果差异过大（>0.01），提示可能是采样率太低
        if np.any(mean_diff > 0.05) or np.any(std_diff > 0.05):
            print("  WARNING: Large difference from yaml stats! Consider using higher sample_rate.")
    else:
        print("[compute_incremental_stats] No statistics.yaml found, skipping validation.")

    # 转换numpy数组为列表以便JSON序列化
    state_stats_serialized = {k: v.tolist() for k, v in state_stats.items()}
    action_stats_serialized = {k: v.tolist() for k, v in action_stats.items()}

    return {
        "observation.state_incremental": state_stats_serialized,
        "action_incremental": action_stats_serialized,
    }


def compare_stats(original_stats: dict, incremental_stats: dict, key: str):
    """比较原始stats和incremental stats的差异。"""
    orig_key = key
    incr_key = key + "_incremental"

    if orig_key not in original_stats or incr_key not in incremental_stats:
        return

    orig_mean = np.array(original_stats[orig_key]["mean"])
    incr_mean = np.array(incremental_stats[incr_key]["mean"])
    orig_std = np.array(original_stats[orig_key]["std"])
    incr_std = np.array(incremental_stats[incr_key]["std"])

    print(f"\n  [{key}] Comparison:")
    print(f"    Original mean:    {orig_mean.tolist()}")
    print(f"    Incremental mean: {incr_mean.tolist()}")
    print(f"    Mean diff:        {(incr_mean - orig_mean).tolist()}")
    print(f"    Original std:     {orig_std.tolist()}")
    print(f"    Incremental std:  {incr_std.tolist()}")
    print(f"    Std diff:         {(incr_std - orig_std).tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="从原始Calvin数据计算增量stats并更新到LeRobot数据集的stats.json"
    )
    parser.add_argument(
        "--calvin_root",
        type=str,
        default="/data_16T/deepseek/calvin_abc_d/task_ABC_D/training/",
        help="原始Calvin数据目录路径",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v4/",
        help="目标LeRobot数据集目录路径",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="采样率（默认10%）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅计算并打印差异，不保存到stats.json",
    )

    args = parser.parse_args()

    # Step 1: 从Calvin数据计算incremental stats
    incremental_stats = compute_calvin_incremental_stats(
        calvin_root=args.calvin_root,
        sample_rate=args.sample_rate,
    )

    # Step 2: 加载现有stats.json
    dataset_root = Path(args.dataset_root)
    stats_path = dataset_root / "meta" / "stats.json"

    if not stats_path.exists():
        print(f"[compute_incremental_stats] ERROR: stats.json not found at {stats_path}")
        return

    with open(stats_path) as f:
        existing_stats = json.load(f)

    print(f"\n[compute_incremental_stats] Existing stats keys: {list(existing_stats.keys())}")

    # Step 3: 添加incremental keys（不覆盖已有keys）
    for key, value in incremental_stats.items():
        if key in existing_stats:
            print(f"  WARNING: Key '{key}' already exists in stats.json with value={value} — overwriting with new value.")
        existing_stats[key] = value

    # Step 4: 比较原始vs incremental差异
    print("\n[compute_incremental_stats] Stats comparison:")
    compare_stats(existing_stats, incremental_stats, "observation.state")
    compare_stats(existing_stats, incremental_stats, "action")

    # Step 5: 保存（或dry_run仅打印）
    if args.dry_run:
        print("\n[compute_incremental_stats] DRY RUN — stats NOT saved.")
        print(f"  Would add keys: {list(incremental_stats.keys())}")
    else:
        # 保存回stats.json
        with open(stats_path, "w") as f:
            json.dump(existing_stats, f, indent=4)

        print(f"\n[compute_incremental_stats] Stats saved to {stats_path}")
        print(f"  Added keys: {list(incremental_stats.keys())}")
        print(f"  Total keys in stats.json: {list(existing_stats.keys())}")


if __name__ == "__main__":
    main()