#!/usr/bin/env python
"""Convert original CronusVLA checkpoint to lerobot-compatible flat format.

Original CronusVLA checkpoint structure:
    {
        "model": {
            "vision_backbone": {state_dict},   # TIMM keys like dino_featurizer.blocks.X...
            "projector": {state_dict},          # e.g., 0.weight, 1.weight, etc.
            "llm_backbone": {state_dict},       # HF LlamaForCausalLM keys
            "action_model": {state_dict},       # DiT keys like net.x_embedder.linear.weight
        }
    }

lerobot flat format:
    {
        "model_state_dict": {
            "vlm.vision_backbone.xxx",  # from vision_backbone keys
            "vlm.projector.xxx",        # from projector keys
            "vlm.llm_backbone.xxx",     # from llm_backbone keys (HF format, no mapping)
            "action_model.xxx",         # from action_model keys
        },
        "source": <input_path>,
    }

CronusVLA also stores norm_stats as dataset_statistics.json alongside checkpoints.
If --norm_stats_path is provided, the converter will load and include them.

Usage:
    python convert_cronusvla_checkpoint.py \
        --input_path /path/to/cronusvla_checkpoint.pt \
        --output_path /path/to/lerobot_cronusvla.pt

    # With norm stats:
    python convert_cronusvla_checkpoint.py \
        --input_path /path/to/cronusvla_checkpoint.pt \
        --output_path /path/to/lerobot_cronusvla.pt \
        --norm_stats_path /path/to/dataset_statistics.json
"""

import argparse
import json
import torch


# VLM sub-components that get the "vlm." prefix
VLM_SUBCOMPONENTS = ["vision_backbone", "projector", "llm_backbone"]

# Components that keep their own top-level prefix (no "vlm." wrapper)
TOPLEVEL_COMPONENTS = ["action_model"]


def convert_checkpoint(input_path: str, output_path: str, norm_stats_path: str | None = None):
    """Load CronusVLA checkpoint, flatten nested structure with prefixes, save."""
    ckpt = torch.load(input_path, map_location="cpu")

    # The original CronusVLA checkpoint stores weights under "model"
    if "model" not in ckpt:
        raise ValueError(
            f"Checkpoint does not contain 'model' key. "
            f"Available keys: {list(ckpt.keys())}"
        )

    nested_model = ckpt["model"]
    print(f"Original 'model' sub-groups: {list(nested_model.keys())}")

    # Flatten nested structure into a single state_dict with prefixed keys
    flattened_sd = {}

    for sub_name, sub_sd in nested_model.items():
        if sub_name in VLM_SUBCOMPONENTS:
            # vision_backbone, projector, llm_backbone -> vlm.<sub_name>.<key>
            prefix = f"vlm.{sub_name}"
        elif sub_name in TOPLEVEL_COMPONENTS:
            # action_model -> action_model.<key> (stays top-level)
            prefix = sub_name
        else:
            # Unknown sub-component: preserve under vlm.<sub_name> by default
            print(f"Warning: unknown sub-component '{sub_name}', mapping to 'vlm.{sub_name}'")
            prefix = f"vlm.{sub_name}"

        if not isinstance(sub_sd, dict):
            print(f"Warning: '{sub_name}' is not a dict (type={type(sub_sd).__name__}), skipping")
            continue

        for key, value in sub_sd.items():
            new_key = f"{prefix}.{key}"
            flattened_sd[new_key] = value

        print(f"  {sub_name} -> {prefix}: {len(sub_sd)} params")

    print(f"Total flattened keys: {len(flattened_sd)}")

    # Show key grouping summary
    prefixes = {}
    for k in flattened_sd:
        top = k.split(".")[0]
        prefixes.setdefault(top, []).append(k)
    for p, ks in sorted(prefixes.items()):
        print(f"  {p}: {len(ks)} params")

    # Build output
    output = {
        "model_state_dict": flattened_sd,
        "source": input_path,
    }

    # Handle norm_stats if provided
    if norm_stats_path is not None:
        try:
            with open(norm_stats_path, "r") as f:
                norm_stats = json.load(f)
            output["norm_stats"] = norm_stats
            print(f"Loaded norm_stats from {norm_stats_path}")
        except Exception as e:
            print(f"Warning: failed to load norm_stats from {norm_stats_path}: {e}")
            print("Norm stats will not be included in output checkpoint.")
    else:
        print("No --norm_stats_path provided. CronusVLA norm_stats (dataset_statistics.json) "
              "are typically stored alongside checkpoints. They can be converted separately "
              "or included by re-running with --norm_stats_path.")

    torch.save(output, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CronusVLA checkpoint to lerobot flat format")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to original CronusVLA checkpoint (.pt file)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output lerobot-format checkpoint (.pt file)")
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Optional path to dataset_statistics.json for norm stats")
    args = parser.parse_args()
    convert_checkpoint(args.input_path, args.output_path, args.norm_stats_path)


if __name__ == "__main__":
    main()