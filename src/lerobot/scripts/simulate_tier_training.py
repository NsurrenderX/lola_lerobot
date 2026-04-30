#!/usr/bin/env python
"""
LoLA Tier 训练数据分布模拟脚本

模拟多卡多 worker 的 tier DataLoader 行为，不实际解码视频也不加载模型。
用于验证：
1. 每个 optimizer step 中各 tier 的数据比例是否符合预期
2. 1 个 epoch 内各 worker 是否有数据重叠（同一 frame 被 >1 个 worker yield）
3. 各 worker 的 frame 分配是否均衡

使用方法:
    # 模拟 4 卡 × 2 workers, 3 tiers, effective_batch_size=256
    python src/lerobot/scripts/simulate_tier_training.py \
        --tier_config_path /path/to/tier_config.json \
        --dataset_root /mnt/data/lerobot-dataset \
        --dataset_to_episodes_path /mnt/data/dataset_to_episodes.json \
        --num_gpus 4 --num_workers 2 \
        --effective_batch_size 256 --balance_mode frame_weighted \
        --max_steps 100

    # 仅数值模拟（不需要数据集）
    python src/lerobot/scripts/simulate_tier_training.py \
        --tier_config_path /path/to/tier_config.json \
        --skip_data_loading \
        --effective_batch_size 2048

    # 快速扫描：每 worker 只迭代少量 batch
    python src/lerobot/scripts/simulate_tier_training.py \
        --tier_config_path /path/to/tier_config.json \
        --dataset_root /mnt/data/lerobot-dataset \
        --quick_scan
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.datasets.lola_pretrain_streaming_dataset import LoLAPretrainStreamingDataset
from lerobot.scripts.train_lola_azure_stream import compute_tier_schedule


def _run_single_parallel_id(
    rank: int,
    worker_id: int,
    total_gpus: int,
    num_dataloader_workers: int,
    tier_idx: int,
    dataset: LoLAPretrainStreamingDataset,
    micro_batch: int,
    max_batches: int,
):
    """Run a single parallel_id's data iteration in the current process.

    Patches torch.utils.data.get_worker_info() to return the desired
    (rank, worker_id) so that dataset.__iter__ computes the correct
    parallel_id = rank * num_workers + worker_id for data sharding.

    Returns dict with items, stats, etc.
    """
    parallel_id = rank * num_dataloader_workers + worker_id

    # Set distributed env vars (dataset reads RANK / WORLD_SIZE)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(total_gpus)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    from torch.utils.data import DataLoader
    import torch.utils.data as _data_utils

    def light_collate(batch):
        result = {
            "_tier": [],
            "_memory_cost": [],
            "episode_index": [],
            "frame_index": [],
            "action_dim": [],
        }
        for item in batch:
            result["_tier"].append(item.get("_tier", -1))
            result["_memory_cost"].append(item.get("_memory_cost", 0.0))
            result["episode_index"].append(item.get("episode_index", -1))
            result["frame_index"].append(item.get("frame_index", -1))
            result["action_dim"].append(item.get("action_dim", 0))
        return result

    # Patch get_worker_info to return our mock
    _orig_get_worker_info = _data_utils.get_worker_info

    class _MockWorkerInfo:
        pass

    mock_info = _MockWorkerInfo()
    mock_info.num_workers = num_dataloader_workers
    mock_info.id = worker_id
    mock_info.seed = 42 + parallel_id
    mock_info.dataset = dataset

    _data_utils.get_worker_info = lambda: mock_info

    try:
        loader = DataLoader(
            dataset,
            batch_size=micro_batch,
            num_workers=0,
            collate_fn=light_collate,
        )

        items = []
        tier_counts = defaultdict(int)
        memory_costs = []
        batches_collected = 0

        for batch in loader:
            batches_collected += 1
            for i in range(len(batch["_tier"])):
                ep_idx = batch["episode_index"][i]
                fr_idx = batch["frame_index"][i]
                t = batch["_tier"][i]
                mc = batch["_memory_cost"][i]
                items.append((ep_idx, fr_idx))
                tier_counts[t] += 1
                memory_costs.append(mc)

            if batches_collected >= max_batches:
                break

        return {
            "rank": rank,
            "worker_id": worker_id,
            "parallel_id": parallel_id,
            "tier_idx": tier_idx,
            "total_items": len(items),
            "tier_counts": dict(tier_counts),
            "avg_memory_cost": float(np.mean(memory_costs)) if memory_costs else 0,
            "batches_collected": batches_collected,
            "items": items,
            "error": None,
        }

    except Exception as e:
        import traceback
        return {
            "parallel_id": parallel_id,
            "tier_idx": tier_idx,
            "error": f"{e}\n{traceback.format_exc()}",
        }
    finally:
        # Restore original get_worker_info
        _data_utils.get_worker_info = _orig_get_worker_info


def simulate_tier_data_loading(
    tier_idx: int,
    num_gpus: int,
    num_workers: int,
    dataset_kwargs: dict,
    micro_batch: int,
    max_batches: int = 50,
):
    """Sequentially simulate each (rank, worker_id) combination.

    Creates one dataset instance, then iterates it multiple times with
    different get_worker_info patches to simulate each parallel_id.
    This avoids the heavy multiprocessing overhead of 8+ processes
    all doing full dataset initialization.
    """
    total_parallel = num_gpus * num_workers

    print(f"  Creating dataset for tier {tier_idx}...", flush=True)
    try:
        dataset = LoLAPretrainStreamingDataset(
            **dataset_kwargs,
            yield_tier=tier_idx,
        )
    except Exception as e:
        print(f"  ERROR creating dataset: {e}", flush=True)
        return {}, {}

    all_worker_items = {}
    all_worker_stats = {}

    for rank in range(num_gpus):
        for worker_id in range(num_workers):
            parallel_id = rank * num_workers + worker_id
            print(f"  Simulating rank={rank} worker={worker_id} "
                  f"(parallel_id={parallel_id})...", flush=True)

            result = _run_single_parallel_id(
                rank, worker_id, num_gpus, num_workers,
                tier_idx, dataset, micro_batch, max_batches,
            )

            if result.get("error"):
                print(f"  ERROR: {result['error'][:200]}", flush=True)
                continue

            pid = result["parallel_id"]
            all_worker_items[pid] = result["items"]
            all_worker_stats[pid] = {
                "total_items": result["total_items"],
                "tier_counts": result["tier_counts"],
                "avg_memory_cost": result["avg_memory_cost"],
                "batches_collected": result["batches_collected"],
            }
            print(f"  [Tier {tier_idx}] Rank {result['rank']} Worker {result['worker_id']} "
                  f"(pid={pid}): {result['total_items']} items, "
                  f"{result['batches_collected']} batches", flush=True)

    return all_worker_items, all_worker_stats


def check_worker_overlap(all_worker_items: dict, tier_idx: int):
    """检查同一 tier 内不同 worker 是否有数据重叠。"""
    worker_sets = {}
    for pid, items in all_worker_items.items():
        worker_sets[pid] = set(items)

    pids = sorted(worker_sets.keys())
    overlaps = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            overlap = worker_sets[pids[i]] & worker_sets[pids[j]]
            if overlap:
                overlaps.append((pids[i], pids[j], len(overlap)))

    if overlaps:
        print(f"\n  [Tier {tier_idx}] OVERLAP DETECTED:")
        for pid_i, pid_j, count in overlaps:
            total_i = len(worker_sets[pid_i])
            total_j = len(worker_sets[pid_j])
            print(f"    Worker {pid_i} ({total_i} items) ∩ "
                  f"Worker {pid_j} ({total_j} items): {count} shared frames")
        return False
    else:
        total_items = sum(len(s) for s in worker_sets.values())
        print(f"  [Tier {tier_idx}] No overlap across {len(pids)} workers "
              f"({total_items} total items) ✓")
        return True


def check_worker_balance(all_worker_stats: dict, tier_idx: int):
    """检查各 worker 的数据量是否均衡。"""
    item_counts = [s["total_items"] for s in all_worker_stats.values()]
    if not item_counts:
        print(f"  [Tier {tier_idx}] No data collected")
        return 0

    avg = np.mean(item_counts)
    std = np.std(item_counts)
    min_c = min(item_counts)
    max_c = max(item_counts)
    max_deviation = max(abs(c - avg) / avg * 100 for c in item_counts) if avg > 0 else 0

    status = "⚠ Imbalanced" if max_deviation > 30 else ("~ Acceptable" if max_deviation > 15 else "✓ Balanced")
    print(f"  [Tier {tier_idx}] Worker balance: avg={avg:.0f} std={std:.0f} "
          f"min={min_c} max={max_c} max_dev={max_deviation:.1f}% {status}")

    # Per-worker detail
    for pid in sorted(all_worker_stats.keys()):
        s = all_worker_stats[pid]
        dev = abs(s["total_items"] - avg) / avg * 100 if avg > 0 else 0
        print(f"    Worker {pid}: {s['total_items']} items "
              f"(avg_cost={s['avg_memory_cost']:.0f}, dev={dev:.1f}%)")

    return max_deviation


def check_tier_purity(all_worker_items: dict, tier_idx: int):
    """检查 yield_tier 过滤是否纯净——是否所有 frame 都属于目标 tier。"""
    wrong_tier_count = 0
    total = 0
    for pid, items in all_worker_items.items():
        for ep_idx, fr_idx in items:
            total += 1

    # items 只存了 (ep_idx, frame_idx)，没有 tier 信息
    # 需要重新检查 tier。我们无法在 sub-process 中回传 _tier 字段
    # 因为 light_collate 只提取了 (ep_idx, frame_idx)
    # 改为在 _worker_process 中额外统计 wrong tier
    # 这需要在 _worker_process 中添加逻辑
    print(f"  [Tier {tier_idx}] Tier purity: checking via _tier field in items...")
    # 实际上我们在 light_collate 中没有收集每条 item 的 _tier，
    # 只收集了 batch 级别的 _tier list。这个检查在后续版本添加。
    pass


def simulate_training_steps(
    tier_schedule: dict,
    tier_micro_batches: list,
    accum_order: list,
    balance_weights: list,
    max_steps: int = 100,
):
    """纯数值模拟训练循环的数据分布。不需要实例化 dataset。"""
    print("\n" + "=" * 70)
    print("TRAINING STEP SIMULATION (numerical)")
    print("=" * 70)

    tier_accum_steps = tier_schedule["tier_accum_steps"]
    total_accum = len(accum_order)

    # 每个 optimizer step 的理论比例
    tier_samples = {}
    total_samples = 0
    for t in range(len(tier_micro_batches)):
        s = tier_micro_batches[t] * int(tier_accum_steps[str(t)])
        tier_samples[t] = s
        total_samples += s

    print(f"\nPer optimizer step (total_accum={total_accum} micro-steps):")
    for t in range(len(tier_micro_batches)):
        actual_ratio = tier_samples[t] / total_samples
        target_ratio = balance_weights[t]
        print(f"  Tier {t}: samples={tier_samples[t]} "
              f"({actual_ratio:.3f} vs target {target_ratio:.3f}, "
              f"dev={actual_ratio - target_ratio:+.4f})")
    print(f"  Total effective batch: {total_samples}")

    # 模拟多步累积
    print(f"\nSimulating {max_steps} optimizer steps...")
    cumulative_tier_samples = defaultdict(int)

    for step in range(1, max_steps + 1):
        for micro_idx, (tier_idx, micro_batch) in enumerate(accum_order):
            cumulative_tier_samples[tier_idx] += micro_batch

        if step % max(1, max_steps // 10) == 0 or step <= 5:
            total = sum(cumulative_tier_samples.values())
            ratios = {t: cumulative_tier_samples[t] / total for t in cumulative_tier_samples}
            deviations = {t: ratios[t] - balance_weights[t] for t in ratios}
            max_dev = max(abs(d) for d in deviations.values())
            print(f"  Step {step}: total={total} samples, "
                  f"max_dev={max_dev:.4f}, "
                  f"ratios={{{', '.join(f'T{t}:{ratios[t]:.3f}' for t in sorted(ratios))}}}")

    total = sum(cumulative_tier_samples.values())
    print(f"\nAfter {max_steps} steps:")
    for t in sorted(cumulative_tier_samples.keys()):
        actual_ratio = cumulative_tier_samples[t] / total
        target_ratio = balance_weights[t]
        print(f"  Tier {t}: {cumulative_tier_samples[t]} samples "
              f"({actual_ratio:.3f} vs target {target_ratio:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Simulate LoLA tier training data distribution")

    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None)
    parser.add_argument("--sub_root", type=str, default=None)
    parser.add_argument("--temp_process", action="store_true")
    parser.add_argument("--episode_chunk_size", type=int, default=8)

    # 模拟参数
    parser.add_argument("--num_gpus", type=int, default=4, help="Simulated GPU count")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per GPU")
    parser.add_argument("--max_steps", type=int, default=50, help="Simulated optimizer steps (numerical)")
    parser.add_argument("--max_batches_per_worker", type=int, default=50,
                        help="Max batches per worker to iterate (data loading sim)")
    parser.add_argument("--buffer_size", type=int, default=5000)

    # Tier 参数
    parser.add_argument("--tier_config_path", type=str, required=True)
    parser.add_argument("--effective_batch_size", type=int, default=2048)
    parser.add_argument("--balance_mode", type=str, default="frame_weighted",
                        choices=["frame_weighted", "equal", "episode_weighted"])
    parser.add_argument("--gpu_utilization_target", type=float, default=0.92)
    parser.add_argument("--gpu_memory_budget_gb", type=float, default=None)
    parser.add_argument("--tier_micro_batches_override", type=str, default=None)
    parser.add_argument("--calibration_path", type=str, default=None)

    # 控制参数
    parser.add_argument("--skip_data_loading", action="store_true",
                        help="Skip actual data loading, only do numerical simulation")
    parser.add_argument("--quick_scan", action="store_true",
                        help="Quick scan: only iterate 3 batches per worker")

    args = parser.parse_args()

    if not args.skip_data_loading and args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided "
                         "(not needed with --skip_data_loading).")

    # ── 1. Load tier config and compute schedule ──────────────────────
    print("=" * 70)
    print("TIER SCHEDULE COMPUTATION")
    print("=" * 70)

    with open(args.tier_config_path) as f:
        tier_config = json.load(f)

    tier_stats = tier_config["tier_stats"]
    num_tiers = len(tier_stats)

    calibration_path = args.calibration_path or tier_config.get("params", {}).get("calibration_path")

    schedule = compute_tier_schedule(
        tier_stats=tier_stats,
        effective_batch_size=args.effective_batch_size,
        balance_mode=args.balance_mode,
        gpu_memory_budget_gb=args.gpu_memory_budget_gb,
        tier_micro_batches_override=args.tier_micro_batches_override,
        gpu_utilization_target=args.gpu_utilization_target,
        calibration_path=calibration_path,
    )

    tier_micro_batches = schedule["tier_micro_batches"]
    tier_accum_steps = schedule["tier_accum_steps"]
    accum_order = schedule["accum_order"]
    balance_weights = schedule["balance_weights"]
    actual_effective_batch = schedule["actual_effective_batch"]

    print(f"\nAuto-computed schedule ({args.balance_mode}, target={args.effective_batch_size}):")
    for t in range(num_tiers):
        print(f"  Tier {t}: micro_batch={tier_micro_batches[t]}, "
              f"accum_steps={tier_accum_steps[str(t)]}, "
              f"weight={balance_weights[t]:.3f}, "
              f"samples/step={tier_micro_batches[t] * tier_accum_steps[str(t)]}")
    print(f"  Total effective batch: {actual_effective_batch}")

    # ── 2. Numerical training simulation ──────────────────────────────
    simulate_training_steps(
        tier_schedule=schedule,
        tier_micro_batches=tier_micro_batches,
        accum_order=accum_order,
        balance_weights=balance_weights,
        max_steps=args.max_steps,
    )

    if args.skip_data_loading:
        print("\n--skip_data_loading: Skipping actual data loading.")
        return

    # ── 3. Actual data loading simulation ─────────────────────────────
    print("\n" + "=" * 70)
    print("DATA LOADING SIMULATION (actual dataset iteration)")
    print("=" * 70)
    print(f"Simulating: {args.num_gpus} GPUs × {args.num_workers} workers "
          f"= {args.num_gpus * args.num_workers} parallel workers per tier")

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.lola import LoLAConfig

    print("\nLoading dataset metadata...")
    dataset_metadata = LoLAPretrainStreamingDataset._build_metadata_polars(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=None,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    action_dim = features["action"].shape[0] if "action" in features else 20
    fps = dataset_metadata.fps

    print(f"Dataset: {dataset_metadata.total_episodes} episodes, "
          f"{dataset_metadata.total_frames} frames, fps={fps}")

    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        vlm_path="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/",
        action_dim=action_dim,
        action_chunk_size=10,
        pred_chunk_size=50,
        n_obs_steps=1,
        input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
        output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
        load_full_history=True,
        max_history_length=100,
    )

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    # 每个 tier 分别模拟
    overall_no_overlap = True
    for tier_idx in range(num_tiers):
        print(f"\n{'─' * 60}")
        print(f"Tier {tier_idx}: micro_batch={tier_micro_batches[tier_idx]}, "
              f"accum_steps={tier_accum_steps[str(tier_idx)]}")
        print(f"{'─' * 60}")

        dataset_kwargs = dict(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            sub_root=args.sub_root,
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=args.buffer_size,
            seed=42,
            shuffle=True,
            deferred_video_decode=True,
            async_decode=False,
            decode_device="cpu",
            decode_num_threads=1,
            num_dataloader_workers=args.num_workers,
            dataset_to_episodes_path=args.dataset_to_episodes_path,
            temp_process=args.temp_process,
            episode_chunk_size=args.episode_chunk_size,
            tier_config_path=args.tier_config_path,
        )

        max_batches = args.max_batches_per_worker
        if args.quick_scan:
            max_batches = 3

        t0 = time.time()
        all_worker_items, all_worker_stats = simulate_tier_data_loading(
            tier_idx=tier_idx,
            num_gpus=args.num_gpus,
            num_workers=args.num_workers,
            dataset_kwargs=dataset_kwargs,
            micro_batch=tier_micro_batches[tier_idx],
            max_batches=max_batches,
        )
        elapsed = time.time() - t0
        print(f"  Data loading took {elapsed:.1f}s")

        # 检查 worker 间重叠
        no_overlap = check_worker_overlap(all_worker_items, tier_idx)
        if not no_overlap:
            overall_no_overlap = False

        # 检查 worker 间均衡
        check_worker_balance(all_worker_stats, tier_idx)

    # ── 4. 总结 ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tiers: {num_tiers}")
    print(f"Parallelism: {args.num_gpus} GPUs × {args.num_workers} workers "
          f"= {args.num_gpus * args.num_workers} per tier")
    print(f"Effective batch size: {actual_effective_batch} (target: {args.effective_batch_size})")
    print(f"Balance mode: {args.balance_mode}")
    print(f"Worker overlap: {'NONE ✓' if overall_no_overlap else 'DETECTED ✗'}")
    for t in range(num_tiers):
        print(f"  Tier {t}: micro_batch={tier_micro_batches[t]}, "
              f"accum={tier_accum_steps[str(t)]}, "
              f"weight={balance_weights[t]:.3f}")


if __name__ == "__main__":
    main()
