"""
Micro-Benchmark: Varlen SDPA vs Dense SDPA

Validates whether PyTorch nested tensor (jagged layout) SDPA on cuDNN backend
truly skips padding computation, and whether pack/unpack overhead is acceptable.

This benchmark simulates Flux2 DiT attention patterns:
- Dense SDPA with 2D bool mask reshape to 4D (current baseline)
- Varlen SDPA with nested tensor (strip padding → jagged → SDPA → unpack)

Test scenarios: different padding ratios (0%, 20%, 30%, 50%)
"""

import argparse
import time
import json
import torch
import torch.nn.functional as F


def create_padding_mask(batch_size, seq_len, padding_ratio, device):
    """Create a 2D bool mask where True=valid, False=padding.
    Padding is on the LEFT side (consistent with Qwen3.5 left-padding).
    Each batch item has different valid lengths to simulate real data variance."""
    valid_lengths = []
    base_valid = int(seq_len * (1 - padding_ratio))
    for i in range(batch_size):
        # Vary valid length per item: base ± 15% to simulate real data
        variation = int(base_valid * 0.15 * (i / max(batch_size - 1, 1)))
        valid_len = max(8, base_valid + variation)
        valid_len = min(valid_len, seq_len)
        valid_lengths.append(valid_len)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i, vl in enumerate(valid_lengths):
        mask[i, seq_len - vl:] = True  # left-padding: valid tokens on right
    return mask, valid_lengths


def dense_sdpa_attention(query, key, value, attn_mask_2d):
    """Baseline: Dense SDPA with 2D bool mask reshaped to 4D.
    This is how Flux2 currently handles attention_mask."""
    # attn_mask_2d: [B, S] where True=valid → convert to SDPA convention
    # SDPA bool mask: True = mask OUT (add -inf), False = attend
    sdpa_mask = ~attn_mask_2d  # True=mask_out, False=attend
    # Reshape to [B, 1, 1, S] for SDPA broadcasting
    sdpa_mask_4d = sdpa_mask.unsqueeze(1).unsqueeze(1)

    # Q/K/V layout: [B, S, H, D] → permute to [B, H, S, D] for SDPA
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask_4d)
    return out.permute(0, 2, 1, 3)  # back to [B, S, H, D]


def varlen_sdpa_attention(query, key, value, attn_mask_2d):
    """Varlen: Strip padding → nested tensor → SDPA → unpack back to dense.
    attn_mask_2d: [B, S] where True=valid, False=padding.
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    # Build per-item valid tensors and construct nested tensors
    q_parts = []
    k_parts = []
    v_parts = []
    valid_lengths = []

    for b in range(batch_size):
        valid_indices = attn_mask_2d[b]  # bool [S]
        vl = valid_indices.sum().item()
        valid_lengths.append(vl)
        # Strip padding: take only valid positions
        q_parts.append(query[b, valid_indices].contiguous())  # [vl, H, D]
        k_parts.append(key[b, valid_indices].contiguous())
        v_parts.append(value[b, valid_indices].contiguous())

    # Construct nested tensors with jagged layout
    q_nested = torch.nested.nested_tensor(q_parts, layout=torch.jagged)
    k_nested = torch.nested.nested_tensor(k_parts, layout=torch.jagged)
    v_nested = torch.nested.nested_tensor(v_parts, layout=torch.jagged)

    # SDPA with nested tensors
    out_nested = F.scaled_dot_product_attention(
        q_nested, k_nested, v_nested, enable_gqa=True
    )

    # Unpack back to dense [B, S, H, D] with padding positions = 0
    out_dense = torch.zeros(batch_size, seq_len, num_heads, head_dim,
                             dtype=query.dtype, device=query.device)
    for b in range(batch_size):
        vl = valid_lengths[b]
        valid_indices = attn_mask_2d[b]
        out_dense[b, valid_indices] = out_nested[b]

    return out_dense


def benchmark_fn(fn, query, key, value, mask, num_warmup=5, num_iters=20):
    """Benchmark a function, return mean time in ms."""
    # Warmup
    for _ in range(num_warmup):
        _ = fn(query, key, value, mask)
        torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(query, key, value, mask)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    mean_time = sum(times) / len(times)
    return mean_time, peak_mem


def benchmark_pack_unpack(query, key, value, attn_mask_2d, num_warmup=5, num_iters=50):
    """Benchmark only the pack/unpack overhead (no SDPA computation)."""
    batch_size, seq_len, num_heads, head_dim = query.shape

    # Warmup
    for _ in range(num_warmup):
        q_parts = []
        for b in range(batch_size):
            q_parts.append(query[b, attn_mask_2d[b]].contiguous())
        q_nested = torch.nested.nested_tensor(q_parts, layout=torch.jagged)
        out_dense = torch.zeros(batch_size, seq_len, num_heads, head_dim,
                                 dtype=query.dtype, device=query.device)
        for b in range(batch_size):
            vl = attn_mask_2d[b].sum().item()
            out_dense[b, attn_mask_2d[b]] = q_nested[b]
        torch.cuda.synchronize()

    # Measure pack
    pack_times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q_parts = []
        for b in range(batch_size):
            q_parts.append(query[b, attn_mask_2d[b]].contiguous())
        q_nested = torch.nested.nested_tensor(q_parts, layout=torch.jagged)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pack_times.append((t1 - t0) * 1000)

    # Measure unpack
    unpack_times = []
    for _ in range(num_iters):
        q_parts = []
        for b in range(batch_size):
            q_parts.append(query[b, attn_mask_2d[b]].contiguous())
        q_nested = torch.nested.nested_tensor(q_parts, layout=torch.jagged)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_dense = torch.zeros(batch_size, seq_len, num_heads, head_dim,
                                 dtype=query.dtype, device=query.device)
        for b in range(batch_size):
            out_dense[b, attn_mask_2d[b]] = q_nested[b]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        unpack_times.append((t1 - t0) * 1000)

    pack_mean = sum(pack_times) / len(pack_times)
    unpack_mean = sum(unpack_times) / len(unpack_times)
    return pack_mean, unpack_mean


def run_benchmark(config):
    """Run the full benchmark with given config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA required for this benchmark")
        return

    B = config.batch_size
    H = config.num_heads
    D = config.head_dim
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    print(f"Varlen SDPA Micro-Benchmark")
    print(f"{'='*60}")
    print(f"Config: B={B}, H={H}, D={D}, dtype={dtype}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    results = {}

    for seq_len in config.seq_lengths:
        for padding_ratio in config.padding_ratios:
            print(f"\n--- S={seq_len}, padding={padding_ratio:.0%} ---")

            # Create data
            mask, valid_lengths = create_padding_mask(B, seq_len, padding_ratio, device)
            query = torch.randn(B, seq_len, H, D, dtype=dtype, device=device)
            key = torch.randn(B, seq_len, H, D, dtype=dtype, device=device)
            value = torch.randn(B, seq_len, H, D, dtype=dtype, device=device)

            # Reset memory before each test
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Benchmark dense SDPA
            dense_time, dense_mem = benchmark_fn(dense_sdpa_attention, query, key, value, mask)
            print(f"  Dense SDPA:  {dense_time:.3f} ms, peak_mem={dense_mem:.1f} MB")

            # Reset memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Benchmark varlen SDPA
            try:
                varlen_time, varlen_mem = benchmark_fn(varlen_sdpa_attention, query, key, value, mask)
                print(f"  Varlen SDPA: {varlen_time:.3f} ms, peak_mem={varlen_mem:.1f} MB")
            except Exception as e:
                print(f"  Varlen SDPA: FAILED - {e}")
                varlen_time = None
                varlen_mem = None

            # Pack/unpack overhead
            try:
                pack_ms, unpack_ms = benchmark_pack_unpack(query, key, value, mask)
                total_pack_unpack = pack_ms + unpack_ms
                print(f"  Pack overhead: {pack_ms:.3f} ms, Unpack: {unpack_ms:.3f} ms, Total: {total_pack_unpack:.3f} ms")
            except Exception as e:
                print(f"  Pack/unpack: FAILED - {e}")
                pack_ms = None
                unpack_ms = None
                total_pack_unpack = None

            # Compute speedup
            if varlen_time is not None:
                speedup = dense_time / varlen_time
                overhead_pct = (total_pack_unpack / varlen_time * 100) if total_pack_unpack else float('inf')
                print(f"  Speedup: {speedup:.2f}x, Pack/unpack overhead: {overhead_pct:.1f}%")
            else:
                speedup = None
                overhead_pct = None

            avg_valid = sum(valid_lengths) / len(valid_lengths)
            actual_padding = 1 - avg_valid / seq_len
            print(f"  Actual padding ratio: {actual_padding:.1%} (avg valid={avg_valid:.0f})")

            result_key = f"S{seq_len}_p{padding_ratio:.0%}"
            results[result_key] = {
                "seq_len": seq_len,
                "padding_ratio": padding_ratio,
                "actual_padding_ratio": actual_padding,
                "avg_valid_length": avg_valid,
                "dense_time_ms": dense_time,
                "dense_peak_mem_mb": dense_mem,
                "varlen_time_ms": varlen_time,
                "varlen_peak_mem_mb": varlen_mem,
                "pack_ms": pack_ms,
                "unpack_ms": unpack_ms,
                "total_pack_unpack_ms": total_pack_unpack,
                "speedup": speedup,
                "overhead_pct": overhead_pct,
            }

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<20} {'Dense(ms)':<10} {'Varlen(ms)':<10} {'Speedup':<8} {'Overhead%':<10} {'Mem Diff':<10}")
    print(f"{'-'*60}")
    for key, r in results.items():
        dense_t = r['dense_time_ms']
        varlen_t = r['varlen_time_ms'] if r['varlen_time_ms'] else 'FAIL'
        speedup = f"{r['speedup']:.2f}x" if r['speedup'] else 'N/A'
        overhead = f"{r['overhead_pct']:.1f}%" if r['overhead_pct'] else 'N/A'
        if r['dense_peak_mem_mb'] and r['varlen_peak_mem_mb']:
            mem_diff = f"{r['dense_peak_mem_mb'] - r['varlen_peak_mem_mb']:.1f}MB"
        else:
            mem_diff = 'N/A'
        print(f"{key:<20} {dense_t:<10.3f} {varlen_t:<10} {speedup:<8} {overhead:<10} {mem_diff:<10}")

    # Decision
    print(f"\n{'='*60}")
    print(f"DECISION CRITERIA")
    print(f"{'='*60}")
    passed = False
    for key, r in results.items():
        if r['speedup'] and r['overhead_pct']:
            actual_p = r['actual_padding_ratio']
            if actual_p >= 0.3:
                if r['speedup'] >= 1.3 and r['overhead_pct'] < 10:
                    print(f"  {key}: PASS (≥1.3x speedup, <10% overhead at {actual_p:.0%} padding)")
                    passed = True
                elif r['speedup'] >= 1.2:
                    print(f"  {key}: MARGINAL (1.2-1.3x speedup at {actual_p:.0%} padding)")
                else:
                    print(f"  {key}: FAIL (<1.2x speedup at {actual_p:.0%} padding)")

    if passed:
        print(f"\n  RECOMMENDATION: Proceed with varlen SDPA implementation.")
    else:
        print(f"\n  RECOMMENDATION: Do NOT proceed. Varlen SDPA does not provide sufficient benefit.")

    # Save results
    output_file = config.output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Varlen SDPA Micro-Benchmark")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[440, 880, 1760])
    parser.add_argument("--padding_ratios", type=float, nargs="+", default=[0.0, 0.2, 0.3, 0.5])
    parser.add_argument("--output_file", type=str, default="benchmark_varlen_results.json")
    config = parser.parse_args()
    run_benchmark(config)


if __name__ == "__main__":
    main()