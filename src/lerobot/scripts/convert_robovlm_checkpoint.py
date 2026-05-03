#!/usr/bin/env python
"""Convert original RoboVLM DeepSpeed checkpoint to lerobot-compatible format.

Original checkpoint (from PyTorch Lightning + DeepSpeed) has state_dict keys
prefixed with 'model.' (e.g., 'model.act_head.rnn.weight_ih_l0'), while the
migrated RoboVLMModel uses keys without this prefix (e.g., 'act_head.rnn.weight_ih_l0').
Additionally, pretrained checkpoints may lack state embedding weights
(embed_arm_state, embed_gripper_state, embed_state) since use_state=False
was the default during pretraining. These will remain randomly initialized.

Usage:
    python convert_robovlm_checkpoint.py \
        --input /data_16T/deepseek/kosmos2/kosmos_ph_oxe-pretrain.pt \
        --output converted_robovlm_pretrain.pt
"""

import argparse
import torch


def convert_checkpoint(input_path: str, output_path: str):
    """Load original checkpoint, strip 'model.' prefix, save converted."""
    ckpt = torch.load(input_path, map_location="cpu")

    if "state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint does not contain 'state_dict' key. "
            f"Available keys: {list(ckpt.keys())}"
        )

    orig_sd = ckpt["state_dict"]
    print(f"Original state_dict: {len(orig_sd)} keys")

    # Strip 'model.' prefix from all keys
    converted_sd = {}
    for key, value in orig_sd.items():
        if key.startswith("model."):
            new_key = key[len("model."):]
        else:
            new_key = key
        converted_sd[new_key] = value

    print(f"Converted state_dict: {len(converted_sd)} keys")

    # Show key mapping summary
    prefixes = {}
    for k in converted_sd:
        prefix = k.split(".")[0]
        prefixes.setdefault(prefix, []).append(k)
    for p, ks in sorted(prefixes.items()):
        print(f"  {p}: {len(ks)} params")

    # Save as a simple dict with just model weights
    output = {
        "model_state_dict": converted_sd,
        "source": input_path,
        "original_global_step": ckpt.get("global_step", 0),
        "original_epoch": ckpt.get("epoch", 0),
    }

    torch.save(output, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert RoboVLM checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Original checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output converted checkpoint path")
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()