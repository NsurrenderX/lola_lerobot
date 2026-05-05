#!/usr/bin/env python
"""Convert lerobot CronusVLA checkpoint back to original CronusVLA nested format.

lerobot flat format:
    {
        "model_state_dict": {
            "vlm.vision_backbone.xxx",
            "vlm.projector.xxx",
            "vlm.llm_backbone.xxx",
            "action_model.xxx",
        },
        "source": ...,
    }

Original CronusVLA nested format:
    {
        "model": {
            "vision_backbone": {state_dict},
            "projector": {state_dict},
            "llm_backbone": {state_dict},
            "action_model": {state_dict},
        }
    }

This script reverses the conversion done by convert_cronusvla_checkpoint.py,
grouping flat prefixed keys back into the nested CronusVLA structure.

Usage:
    python convert_cronusvla_checkpoint_reverse.py \
        --input_path /path/to/lerobot_cronusvla.pt \
        --output_path /path/to/cronusvla_checkpoint.pt
"""

import argparse
import torch


# VLM sub-components that had the "vlm." prefix in lerobot format
VLM_SUBCOMPONENTS = ["vision_backbone", "projector", "llm_backbone"]

# Components that kept their own top-level prefix (no "vlm." wrapper)
TOPLEVEL_COMPONENTS = ["action_model"]


def convert_checkpoint_reverse(input_path: str, output_path: str):
    """Load lerobot flat checkpoint, group keys into nested CronusVLA structure, save."""
    ckpt = torch.load(input_path, map_location="cpu")

    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint does not contain 'model_state_dict' key. "
            f"Available keys: {list(ckpt.keys())}"
        )

    flat_sd = ckpt["model_state_dict"]
    print(f"Input flat state_dict: {len(flat_sd)} keys")

    # Group flat keys by prefix into nested structure
    nested_model = {}

    for key, value in flat_sd.items():
        # Check for vlm.<subcomponent>.xxx pattern
        if key.startswith("vlm."):
            # Strip "vlm." prefix, then split on first dot to get sub-component name
            remainder = key[len("vlm."):]
            sub_name = remainder.split(".")[0]
            sub_key = remainder[len(sub_name) + 1:]  # everything after sub_name.

            if sub_name not in nested_model:
                nested_model[sub_name] = {}
            nested_model[sub_name][sub_key] = value

        elif key.startswith("action_model."):
            # action_model.xxx -> action_model group, strip prefix
            sub_name = "action_model"
            sub_key = key[len("action_model."):]

            if sub_name not in nested_model:
                nested_model[sub_name] = {}
            nested_model[sub_name][sub_key] = value

        else:
            # Unknown prefix: try to group by first dot-separated component
            sub_name = key.split(".")[0]
            sub_key = key[len(sub_name) + 1:]

            print(f"Warning: unknown prefix '{sub_name}' for key '{key}', "
                  f"grouping into '{sub_name}'")
            if sub_name not in nested_model:
                nested_model[sub_name] = {}
            nested_model[sub_name][sub_key] = value

    print(f"Nested sub-groups: {list(nested_model.keys())}")
    for sub_name, sub_sd in sorted(nested_model.items()):
        print(f"  {sub_name}: {len(sub_sd)} params")

    # Build output in original CronusVLA format
    output = {
        "model": nested_model,
    }

    # Preserve source info if available
    if "source" in ckpt:
        output["source"] = ckpt["source"]

    # Preserve norm_stats if they were included during forward conversion
    if "norm_stats" in ckpt:
        output["norm_stats"] = ckpt["norm_stats"]

    torch.save(output, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert lerobot CronusVLA checkpoint back to original nested format"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to lerobot-format checkpoint (.pt file)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output CronusVLA-format checkpoint (.pt file)")
    args = parser.parse_args()
    convert_checkpoint_reverse(args.input_path, args.output_path)


if __name__ == "__main__":
    main()