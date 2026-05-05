#!/usr/bin/env python
"""Convert lerobot RoboVLM checkpoint back to the original DeepSpeed/Lightning format.

The lerobot training saves checkpoints with:
  - key "model_state_dict" containing weights WITHOUT the "model." prefix
  - e.g., "act_head.rnn.weight_ih_l0"

The original PyTorch Lightning + DeepSpeed format uses:
  - key "state_dict" containing weights WITH the "model." prefix
  - e.g., "model.act_head.rnn.weight_ih_l0"

This script reverses the conversion done by convert_robovlm_checkpoint.py,
restoring the "model." prefix so the checkpoint can be loaded by the original
RoboVLMs codebase.

Usage:
    python convert_robovlm_checkpoint_reverse.py \
        --input lerobot_checkpoint.pt \
        --output original_robovlm_checkpoint.pt

    # Optional: discard optimizer/scheduler state (original format doesn't include them)
    python convert_robovlm_checkpoint_reverse.py \
        --input lerobot_checkpoint.pt \
        --output original_robovlm_checkpoint.pt \
        --strip_optimizer
"""

import argparse
import torch


def convert_checkpoint_reverse(input_path: str, output_path: str, strip_optimizer: bool = False):
    """Load lerobot checkpoint, add 'model.' prefix back, save in original format."""
    ckpt = torch.load(input_path, map_location="cpu")

    # The lerobot format uses "model_state_dict" (no "model." prefix).
    # It may also come from convert_robovlm_checkpoint.py output, which also
    # uses "model_state_dict".
    if "model_state_dict" in ckpt:
        model_sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        # If somehow loaded an original-format checkpoint, keys already have "model." prefix
        model_sd = ckpt["state_dict"]
        # Check if keys already have "model." prefix
        already_prefixed = any(k.startswith("model.") for k in model_sd)
        if already_prefixed:
            print("Checkpoint already in original format (keys have 'model.' prefix).")
            print("Saving as-is (no changes needed).")
            torch.save(ckpt, output_path)
            print(f"Saved to {output_path}")
            return
    else:
        raise ValueError(
            f"Checkpoint does not contain 'model_state_dict' or 'state_dict' key. "
            f"Available keys: {list(ckpt.keys())}"
        )

    print(f"Input state_dict: {len(model_sd)} keys")

    # Add 'model.' prefix to all keys that don't already have it
    converted_sd = {}
    for key, value in model_sd.items():
        if key.startswith("model."):
            new_key = key
        else:
            new_key = "model." + key
        converted_sd[new_key] = value

    print(f"Converted state_dict: {len(converted_sd)} keys")

    # Show key mapping summary
    prefixes = {}
    for k in converted_sd:
        # Strip "model." for grouping
        bare = k[len("model."):] if k.startswith("model.") else k
        prefix = bare.split(".")[0]
        prefixes.setdefault(prefix, []).append(k)
    for p, ks in sorted(prefixes.items()):
        print(f"  model.{p}: {len(ks)} params")

    # Build output in original Lightning/DeepSpeed format
    output = {
        "state_dict": converted_sd,
        "global_step": ckpt.get("global_step", 0),
        "epoch": ckpt.get("epoch", 0) or ckpt.get("original_epoch", 0),
        "lr_schedulers": [],
    }

    if not strip_optimizer:
        # Carry over optimizer/scheduler state if present (original format
        # typically doesn't include these, but keeping them doesn't hurt)
        if "optimizer_state_dict" in ckpt:
            output["optimizer_state_dict"] = ckpt["optimizer_state_dict"]
        if "scheduler_state_dict" in ckpt:
            output["lr_schedulers"] = [ckpt["scheduler_state_dict"]]

    # Preserve source info if this was a forward-converted checkpoint
    if "source" in ckpt:
        output["source"] = ckpt["source"]

    torch.save(output, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert lerobot RoboVLM checkpoint back to original format")
    parser.add_argument("--input", type=str, required=True, help="Lerobot checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output original-format checkpoint path")
    parser.add_argument("--strip_optimizer", action="store_true",
                        help="Discard optimizer/scheduler state (original format doesn't include them)")
    args = parser.parse_args()
    convert_checkpoint_reverse(args.input, args.output, args.strip_optimizer)


if __name__ == "__main__":
    main()