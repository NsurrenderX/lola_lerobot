#!/usr/bin/env python3
"""Calibrate memory cost coefficients for LoLA tier-based batching.

Instantiates the actual model components on a single GPU and empirically measures
how much GPU memory each type of token consumes, deriving:
- vision_tower_multiplier: visual token cost relative to text token
- action_token_weight: action token chunk cost relative to text token (per-chunk ratio)

No pretrained LoLA checkpoint is required — the DiT is instantiated from config
with random weights, since memory cost depends on architecture, not weight values.

These coefficients are used by scan_dataset_tier_config.py to compute
equivalent token cost per episode and determine tier boundaries.

Usage:
    python src/lerobot/scripts/calibrate_memory_cost.py \
        --vlm_path /mnt/pvc/training_data/weights/Qwen3.5-4B \
        --device cuda:0 \
        --output calibration_coefficients.json
"""

import argparse
import gc
import json
import math
import sys
import time

import numpy as np
import torch


def smart_resize_qwen3vl(height, width, factor=32, min_pixels=65536, max_pixels=230400):
    """Exact replica of Qwen3.5-VL smart_resize.

    factor=32 = patch_size(16) * spatial_merge_size(2).
    Returns (h_bar, w_bar) — both divisible by factor, within pixel bounds.
    """
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

    For Qwen3.5-4B: num_tokens = h_bar * w_bar / 1024
    where 1024 = patch_size^2 * spatial_merge_size^2 = 16*16*4.
    """
    h_bar, w_bar = smart_resize_qwen3vl(
        original_h, original_w, factor=32, min_pixels=min_pixels, max_pixels=max_pixels
    )
    num_tokens = h_bar * w_bar // 1024
    return num_tokens


def fit_linear(x_vals, y_vals):
    """Fit y = a*x + b and return (slope, intercept)."""
    coeffs = np.polyfit(x_vals, y_vals, 1)
    return float(coeffs[0]), float(coeffs[1])


def _cleanup_gpu(device):
    """Force-release all stray GPU allocations before a measurement."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def measure_text_baseline(vlm_model, device, text_lengths, batch_size, dtype=torch.bfloat16):
    """Measure peak GPU memory for pure text sequences at varying lengths.

    Runs full forward + backward (gradients enabled) to capture training-time
    memory: activations + gradients. Uses batch_size > 1 to amplify per-token
    cost above fragmentation noise.

    Returns dict: {num_text_tokens: peak_memory_bytes}
    """
    results = {}
    for n_tokens in text_lengths:
        _cleanup_gpu(device)

        input_ids = torch.randint(0, 1000, (batch_size, n_tokens), device=device)
        attention_mask = torch.ones(batch_size, n_tokens, dtype=torch.long, device=device)

        try:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = vlm_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = out.logits.sum()
                loss.backward()
        except Exception as e:
            print(f"  Warning: text forward+backward failed at {n_tokens} tokens: {e}")
            vlm_model.zero_grad(set_to_none=True)
            del input_ids, attention_mask
            continue

        peak_mem = torch.cuda.max_memory_allocated(device)
        results[n_tokens] = peak_mem
        print(f"  Text {n_tokens} tokens (B={batch_size}): {peak_mem / 1e9:.2f} GB")

        # Cleanup after measurement
        del out, loss, input_ids, attention_mask
        vlm_model.zero_grad(set_to_none=True)

    return results


def measure_visual_cost(vlm_model, device, resolutions, min_pixels, max_pixels,
                        batch_size, dtype=torch.bfloat16):
    """Measure peak GPU memory delta for images at various resolutions.

    Uses DIRECT text-only baseline measurement at each resolution's actual
    sequence length (not linear-fit interpolation), to avoid baseline prediction
    errors from non-linear memory scaling.

    For each resolution:
    1. Process image through processor → get actual_seq_len
    2. Measure VLM memory with image (forward+backward)
    3. Measure VLM memory with text-only at same actual_seq_len (forward+backward)
    4. delta = visual_peak - text_peak

    Returns dict: {num_visual_tokens: delta_mem_bytes}
    """
    results = {}
    for h, w in resolutions:
        n_visual_tokens = compute_visual_tokens(h, w, min_pixels, max_pixels)
        _cleanup_gpu(device)

        try:
            from transformers import AutoProcessor
            from PIL import Image

            processor = AutoProcessor.from_pretrained(vlm_model.config._name_or_path
                                                       if hasattr(vlm_model.config, '_name_or_path')
                                                       else "Qwen/Qwen3.5-4B")
            processor.image_processor.max_pixels = max_pixels
            processor.image_processor.min_pixels = min_pixels

            img = Image.new("RGB", (w, h), (255, 255, 255))
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Perform the robot task."},
            ]}]] * batch_size

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Extract actual sequence length from processor output
            input_ids_shape = inputs["input_ids"].shape
            actual_seq_len = input_ids_shape[1] if len(input_ids_shape) == 2 else input_ids_shape[0]
            actual_text_tokens = actual_seq_len - n_visual_tokens

            # ── Step A: Measure visual memory (VLM with image) ──
            with torch.amp.autocast("cuda", dtype=dtype):
                out = vlm_model(**inputs)
                loss = out.logits.sum()
                loss.backward()

            visual_peak = torch.cuda.max_memory_allocated(device)

            del out, loss
            for k in list(inputs.keys()):
                del inputs[k]
            vlm_model.zero_grad(set_to_none=True)
            _cleanup_gpu(device)

            # ── Step B: Measure text-only baseline at same actual_seq_len ──
            text_input_ids = torch.randint(0, 1000, (batch_size, actual_seq_len), device=device)
            text_attention_mask = torch.ones(batch_size, actual_seq_len, dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=dtype):
                text_out = vlm_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
                text_loss = text_out.logits.sum()
                text_loss.backward()

            text_peak = torch.cuda.max_memory_allocated(device)

            delta_mem = visual_peak - text_peak

            results[n_visual_tokens] = delta_mem
            print(f"  Visual {h}x{w} -> {n_visual_tokens} tokens (B={batch_size}): "
                  f"seq_len={actual_seq_len} (text={actual_text_tokens}+visual={n_visual_tokens}), "
                  f"visual_peak={visual_peak / 1e9:.2f} GB, "
                  f"text_peak={text_peak / 1e9:.2f} GB, "
                  f"delta={delta_mem / 1e9:.2f} GB")

            del text_out, text_loss, text_input_ids, text_attention_mask
            vlm_model.zero_grad(set_to_none=True)

        except Exception as e:
            print(f"  Warning: visual measurement failed for {h}x{w}: {e}")

        vlm_model.zero_grad(set_to_none=True)

    return results


def measure_action_cost(dit_model, device, action_token_counts, dit_hidden_size,
                        dit_batch_size, vlm_base_seq_len=30, dtype=torch.bfloat16):
    """Measure peak GPU memory for LoLADiT with varying action token counts.

    Uses a SEPARATE (typically much larger) batch size than VLM measurements,
    because DiT (~600M params) is much smaller — a large batch amplifies the
    per-chunk cost signal above the model-weight baseline noise.

    Returns dict: {total_action_chunk_count: peak_memory_bytes}
    """
    results = {}

    for n_chunks in action_token_counts:
        _cleanup_gpu(device)

        # Split chunks: half target (pred), half history
        n_pred = max(1, n_chunks // 2)
        n_hist = max(1, n_chunks - n_pred)

        vlm_features = torch.randn(dit_batch_size, vlm_base_seq_len, dit_hidden_size,
                                   device=device, dtype=dtype)
        empty_emb = torch.randn(dit_batch_size, dit_hidden_size, device=device, dtype=dtype)
        target_actions = torch.randn(dit_batch_size, n_pred, dit_hidden_size,
                                    device=device, dtype=dtype)
        hist_actions = torch.randn(dit_batch_size, n_hist, dit_hidden_size,
                                   device=device, dtype=dtype)
        timestep = torch.rand(dit_batch_size, device=device)

        try:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = dit_model(
                    target_actions=target_actions,
                    hist_actions=hist_actions,
                    vlm_features=vlm_features,
                    empty_emb=empty_emb,
                    timestep=timestep,
                )
                # DiT output is action predictions — take a scalar loss
                loss = out.sum() if isinstance(out, torch.Tensor) else out.loss
                loss.backward()

            peak_mem = torch.cuda.max_memory_allocated(device)
            results[n_chunks] = peak_mem
            print(f"  Action {n_chunks} chunks (pred={n_pred}, hist={n_hist}, B={dit_batch_size}): "
                  f"total={peak_mem / 1e9:.2f} GB")

            # Cleanup after successful measurement
            del out, loss

        except Exception as e:
            print(f"  Warning: action measurement failed for {n_chunks} chunks: {e}")
            import traceback
            traceback.print_exc()

        # Always cleanup input tensors + gradients
        del vlm_features, empty_emb, target_actions, hist_actions, timestep
        dit_model.zero_grad(set_to_none=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Calibrate LoLA memory cost coefficients")
    parser.add_argument("--vlm_path", type=str, required=True,
                        help="Path to Qwen3.5-4B model weights")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_image_pixels", type=int, default=230400)
    parser.add_argument("--min_image_pixels", type=int, default=65536)
    parser.add_argument("--action_dim", type=int, default=20)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    # DiT architecture args (match LoLAConfig defaults)
    parser.add_argument("--dit_hidden_size", type=int, default=1536,
                        help="DiT hidden dimension (default: 1536)")
    parser.add_argument("--dit_num_heads", type=int, default=12,
                        help="DiT attention heads (default: 12)")
    parser.add_argument("--dit_double_layers", type=int, default=4,
                        help="DiT double-stream transformer blocks (default: 4)")
    parser.add_argument("--dit_single_layers", type=int, default=12,
                        help="DiT single-stream transformer blocks (default: 12)")
    parser.add_argument("--vlm_base_seq_len", type=int, default=30,
                        help="Fixed VLM context length for DiT measurement baseline")
    parser.add_argument("--calib_batch_size", type=int, default=10,
                        help="Batch size for VLM calibration measurements (larger = less noise)")
    parser.add_argument("--dit_calib_batch_size", type=int, default=64,
                        help="Batch size for DiT calibration (higher than VLM because DiT is ~600M params)")
    parser.add_argument("--output", type=str, default="calibration_coefficients.json",
                        help="Path to output JSON file")
    parser.add_argument("--text_lengths", type=str, default="50,100,200,300,500",
                        help="Comma-separated text token counts for baseline")
    parser.add_argument("--resolutions", type=str, default="256x256,360x360,480x640,720x1280",
                        help="Comma-separated resolutions (HxW) for visual measurement")
    parser.add_argument("--action_chunk_counts", type=str, default="5,10,20,50",
                        help="Comma-separated action chunk counts for DiT measurement")
    args = parser.parse_args()

    device = torch.device(args.device)
    text_lengths = [int(x) for x in args.text_lengths.split(",")]
    resolutions = [(int(h), int(w)) for h, w in
                   [r.split("x") for r in args.resolutions.split(",")]]
    action_chunk_counts = [int(x) for x in args.action_chunk_counts.split(",")]
    batch_size = args.calib_batch_size
    dit_batch_size = args.dit_calib_batch_size
    dtype = torch.bfloat16

    print("=" * 60)
    print("LoLA Memory Cost Calibration (training mode: fwd+bwd)")
    print("=" * 60)
    print(f"VLM path: {args.vlm_path}")
    print(f"Device: {args.device}")
    print(f"VLM batch size: {batch_size}, DiT batch size: {dit_batch_size}")
    print(f"Max/min pixels: {args.max_image_pixels}/{args.min_image_pixels}")
    print(f"DiT config: hidden={args.dit_hidden_size}, heads={args.dit_num_heads}, "
          f"double={args.dit_double_layers}, single={args.dit_single_layers}")
    print(f"Text lengths: {text_lengths}")
    print(f"Resolutions: {resolutions}")
    print(f"Action chunk counts: {action_chunk_counts}")

    # ── Step 1: Load VLM model ──────────────────────────────────
    print("\n--- Loading VLM model (train mode for gradient memory) ---")
    from transformers import Qwen3_5ForConditionalGeneration

    vlm_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.vlm_path, torch_dtype=dtype, device_map=device
    )
    # Training mode — gradient memory must be included in measurements
    vlm_model.train()

    vlm_cfg = vlm_model.config
    text_cfg = vlm_cfg.text_config if hasattr(vlm_cfg, 'text_config') else vlm_cfg
    vision_cfg = vlm_cfg.vision_config if hasattr(vlm_cfg, 'vision_config') else None
    model_config = {
        "vlm_hidden_size": text_cfg.hidden_size,
        "vlm_num_layers": text_cfg.num_hidden_layers,
        "vit_hidden_size": vision_cfg.hidden_size if vision_cfg else 1024,
        "vit_num_layers": vision_cfg.depth if vision_cfg else 24,
    }
    print(f"Model config: {model_config}")

    # ── Step 2: Text-only baseline ───────────────────────────────
    print("\n--- Measuring text-only baseline (fwd+bwd, gradients on) ---")
    text_mem = measure_text_baseline(vlm_model, device, text_lengths, batch_size, dtype=dtype)

    if len(text_mem) < 2:
        print("ERROR: Need at least 2 text measurements for linear fit")
        sys.exit(1)

    # Fit against total token count (batch_size × n_tokens_per_seq) so
    # slope gives true per-token cost, independent of batch_size.
    text_x = [batch_size * k for k in text_mem.keys()]
    text_y = list(text_mem.values())
    text_slope, text_intercept = fit_linear(text_x, text_y)
    print(f"\nText baseline: mem = {text_slope:.0f} × total_tokens + {text_intercept:.0f}")
    print(f"  Per-text-token cost (fwd+bwd): {text_slope / 1e6:.2f} MB/token")

    # ── Step 3: Visual token cost ────────────────────────────────
    print("\n--- Measuring visual token cost (fwd+bwd, gradients on) ---")
    visual_mem = measure_visual_cost(
        vlm_model, device, resolutions,
        args.min_image_pixels, args.max_image_pixels,
        batch_size, dtype=dtype,
    )

    if visual_mem:
        visual_x = [batch_size * k for k in visual_mem.keys()]
        visual_y = list(visual_mem.values())
        visual_slope = fit_linear(visual_x, visual_y)[0]
        vision_tower_multiplier = visual_slope / text_slope if text_slope > 0 else 0
        print(f"\nVisual cost: delta_mem = {visual_slope:.0f} × total_visual_tokens + ...")
        print(f"  Per-visual-token delta (fwd+bwd): {visual_slope / 1e6:.2f} MB/token")
        print(f"  Vision tower multiplier: {vision_tower_multiplier:.2f}")
    else:
        print("WARNING: No visual measurements, using analytical estimate")
        vision_tower_multiplier = 4.0
        visual_slope = text_slope * vision_tower_multiplier

    # Save VLM info before freeing model (needed for DiT config + fallback)
    vlm_hidden_size = text_cfg.hidden_size
    vlm_num_layers = text_cfg.num_hidden_layers

    # ── Step 4: Action token cost (DiT from config, no checkpoint) ──
    print("\n--- Measuring action token cost (fwd+bwd, gradients on) ---")
    print("  Instantiating LoLADiT from config (random weights, no checkpoint needed)")

    try:
        from lerobot.policies.lola.configuration_lola import LoLAConfig
        from lerobot.policies.lola.modeling_lola import LoLADiT
    except ImportError as e:
        print(f"ERROR: Cannot import LoLA modules: {e}")
        print("  Make sure lerobot is installed (pip install -e .)")
        print("  Falling back to analytical estimate for action tokens")
        dit_total_layers = args.dit_double_layers + args.dit_single_layers
        action_token_weight = (args.dit_hidden_size / vlm_hidden_size
                               * dit_total_layers / vlm_num_layers)
        action_slope = text_slope * action_token_weight
        action_mem = None
    else:
        # Construct LoLAConfig from args + VLM model info
        dit_config = LoLAConfig()
        dit_config.dit_hidden_size = args.dit_hidden_size
        dit_config.dit_num_heads = args.dit_num_heads
        dit_config.dit_double_layers = args.dit_double_layers
        dit_config.dit_single_layers = args.dit_single_layers
        dit_config.action_dim = args.action_dim
        dit_config.action_chunk_size = args.action_chunk_size
        dit_config.pred_chunk_size = args.pred_chunk_size
        dit_config.vlm_hidden_size = vlm_hidden_size

        print(f"  LoLAConfig: dit_hidden_size={dit_config.dit_hidden_size}, "
              f"dit_num_heads={dit_config.dit_num_heads}, "
              f"dit_double_layers={dit_config.dit_double_layers}, "
              f"dit_single_layers={dit_config.dit_single_layers}, "
              f"action_dim={dit_config.action_dim}")

        # Free VLM memory before loading DiT
        del vlm_model
        _cleanup_gpu(device)

        # Instantiate DiT from config (random weights, train mode)
        dit_model = LoLADiT(dit_config).to(device).to(dtype)
        dit_model.train()

        dit_param_count = sum(p.numel() for p in dit_model.parameters())
        dit_param_mem = sum(p.numel() * p.element_size() for p in dit_model.parameters())
        print(f"  DiT parameters: {dit_param_count:,} ({dit_param_mem / 1e9:.2f} GB)")

        action_mem = measure_action_cost(
            dit_model, device, action_chunk_counts, args.dit_hidden_size,
            dit_batch_size, vlm_base_seq_len=args.vlm_base_seq_len, dtype=dtype,
        )

        if action_mem and len(action_mem) >= 2:
            # Fit against total chunks across the batch for true per-chunk slope
            action_x = [dit_batch_size * k for k in action_mem.keys()]
            action_y = list(action_mem.values())
            action_slope, action_intercept = fit_linear(action_x, action_y)
            # action_token_weight = per-chunk ratio relative to per-text-token cost
            action_token_weight = action_slope / text_slope if text_slope > 0 else 0
            print(f"\n  Action cost: mem = {action_slope:.0f} × total_chunks + {action_intercept:.0f}")
            print(f"  Per-action-chunk cost (fwd+bwd): {action_slope / 1e6:.2f} MB/chunk")
            print(f"  Action token weight (per-chunk ratio): {action_token_weight:.4f}")
        else:
            print("WARNING: Insufficient action measurements, using analytical estimate")
            # Analytical: DiT per-position cost vs VLM per-position cost
            # ≈ (dit_hidden/vlm_hidden × dit_total_layers/vlm_layers)
            dit_total_layers = args.dit_double_layers + args.dit_single_layers
            analytical_ratio = (args.dit_hidden_size / vlm_hidden_size
                                * dit_total_layers / vlm_num_layers)
            action_token_weight = analytical_ratio
            action_slope = text_slope * action_token_weight
            action_mem = None

    # ── Step 5: Output ───────────────────────────────────────────
    result = {
        "calibration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vlm_path": args.vlm_path,
        "vlm_calib_batch_size": batch_size,
        "dit_calib_batch_size": dit_batch_size,
        "mode": "fwd+bwd (training gradients enabled)",
        "model_config": model_config,
        "dit_config": {
            "dit_hidden_size": args.dit_hidden_size,
            "dit_num_heads": args.dit_num_heads,
            "dit_double_layers": args.dit_double_layers,
            "dit_single_layers": args.dit_single_layers,
        },
        "measurements": {
            "text_token_slope_bytes": text_slope,
            "visual_token_slope_bytes": visual_slope if visual_mem else None,
            "action_token_slope_bytes": action_slope if action_mem else None,
        },
        "text_mem_detail": {str(k): v for k, v in text_mem.items()},
        "visual_mem_detail": {str(k): v for k, v in visual_mem.items()} if visual_mem else None,
        "action_mem_detail": {str(k): v for k, v in action_mem.items()} if action_mem else None,
        "coefficients": {
            "vision_tower_multiplier": round(vision_tower_multiplier, 4),
            "action_token_weight": round(action_token_weight, 4),
            "text_token_base_cost_bytes": round(text_slope, 2),
        },
        "analytical_reference": {
            "vision_tower_multiplier_range": "3-5",
            "action_token_weight_range": "0.1-0.5",
            "note": "action_token_weight is the per-chunk ratio (DiT per-chunk / VLM per-text-token)",
        },
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibration results saved to: {args.output}")
    print(f"  vision_tower_multiplier = {vision_tower_multiplier:.4f}")
    print(f"  action_token_weight = {action_token_weight:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()