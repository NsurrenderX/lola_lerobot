"""
CronusVLADataset: A dataset wrapper that produces data in the format expected by
CronusVLAModel.forward(), matching the original CronusVLA RLDSBatchTransform +
PaddedCollatorForActionPrediction pipeline.

Key differences from raw LeRobotDataset:
- Quantile normalization (BOUNDS_Q99) using q01/q99 from LeRobot V3 stats.json
- VLM prompt format: "What action should the robot take to {instruction}?"
- Tokenizer-specific special token appending (Llama2 or Qwen2.5)
- Continuous actions (no gripper binarization)
- Future/past action window format: [future_action_window_size+1, action_dim]
- Multi-image support (wrist camera) with view_sequence_len > 1
- Collation with text tokenization + padding + interleaving for multi-view
"""

import logging
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast, Qwen2Tokenizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def normalize_action_quantile(
    action: torch.Tensor,
    q01: torch.Tensor,
    q99: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize action using quantile bounds (BOUNDS_Q99).

    Maps [q01, q99] -> [-1, 1] via: 2 * (action - q01) / (q99 - q01) - 1

    Dimensions where mask=False are left unchanged (not normalized).
    """
    if mask is None:
        mask = torch.ones_like(q01, dtype=torch.bool)

    normalized = 2.0 * (action - q01) / (q99 - q01) - 1.0
    # Only apply normalization where mask is True
    action = torch.where(mask.unsqueeze(0).unsqueeze(0), normalized, action)
    return action


class CronusVLADataset(Dataset):
    """Wraps LeRobotDataset to produce data matching CronusVLA's expected format.

    For each sample, loads a window of observations and actions, normalizes actions
    using quantile bounds (q01/q99), tokenizes text with VLM prompt template,
    applies VLM image transforms, and creates action masks.
    """

    def __init__(
        self,
        repo_id: str,
        config,
        root: str | None = None,
        tokenizer=None,
        image_transform=None,
    ):
        self.config = config
        self.future_action_window_size = config.future_action_window_size
        self.past_action_window_size = config.past_action_window_size
        self.action_dim = config.action_dim
        self.view_sequence_len = config.view_sequence_len
        self.use_wrist_image = config.use_wrist_image
        self.max_text_len = config.max_text_len

        # Load dataset metadata to get fps, camera keys, and statistics
        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        fps = self.meta.fps

        # Extract quantile normalization stats (q01/q99) from LeRobot V3 stats.json
        stats = self.meta.stats
        if stats is not None and "action" in stats:
            action_stats = stats["action"]
            if "q01" in action_stats and "q99" in action_stats:
                self.action_q01 = torch.tensor(action_stats["q01"], dtype=torch.float32)
                self.action_q99 = torch.tensor(action_stats["q99"], dtype=torch.float32)
                # Mask: dimensions to normalize (all True for 7-dim actions)
                self.action_mask = torch.ones_like(self.action_q01, dtype=torch.bool)
            else:
                # Fallback: use min/max as approximation for q01/q99
                logger.warning("q01/q99 not found in dataset stats, using min/max as approximation")
                self.action_q01 = torch.tensor(action_stats.get("min", [0] * self.action_dim), dtype=torch.float32)
                self.action_q99 = torch.tensor(action_stats.get("max", [1] * self.action_dim), dtype=torch.float32)
                self.action_mask = torch.ones_like(self.action_q01, dtype=torch.bool)
        else:
            raise ValueError(f"Dataset {repo_id} does not have action statistics with q01/q99")

        # Compute delta_timestamps:
        # - Observations (images): past_action_window_size + 1 frames
        #   For past=0: just [0/fps] (current frame only)
        #   For past=1: [-1/fps, 0/fps]
        obs_delta_indices = list(range(-self.past_action_window_size, 1))
        obs_delta_ts = [i / fps for i in obs_delta_indices]

        # - Actions: past_action_window_size + future_action_window_size + 1 frames
        #   from -past/fps to future/fps
        action_delta_indices = list(range(-self.past_action_window_size, self.future_action_window_size + 1))
        action_delta_ts = [i / fps for i in action_delta_indices]

        delta_timestamps = {}
        for cam_key in self.meta.camera_keys:
            delta_timestamps[cam_key] = obs_delta_ts
        delta_timestamps["observation.state"] = obs_delta_ts
        delta_timestamps["action"] = action_delta_ts

        self.lerobot_ds = LeRobotDataset(
            repo_id, root=root, delta_timestamps=delta_timestamps, tolerance_frames=2,
        )

        # Identify primary camera key
        self.camera_key = self.meta.camera_keys[0]
        self.wrist_camera_key = None
        if self.use_wrist_image and len(self.meta.camera_keys) > 1:
            self.wrist_camera_key = self.meta.camera_keys[1]

        # VLM image transform and tokenizer (from PrismaticVLM)
        self.image_transform = image_transform
        self.tokenizer = tokenizer

        # Detect tokenizer type for special token handling
        self._is_llama_tokenizer = isinstance(tokenizer, LlamaTokenizerFast)
        self._is_qwen_tokenizer = isinstance(tokenizer, Qwen2Tokenizer)

    def __len__(self):
        return len(self.lerobot_ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.lerobot_ds[idx]

        # --- Images ---
        # Apply VLM image transform (handles DinoSigLIP dual-encoder transforms)
        if self.image_transform is not None:
            images = item[self.camera_key]  # [T, C, H, W] or [C, H, W]
            if images.ndim == 4:
                # Multi-frame: transform each frame
                pixel_values = [self.image_transform(images[t]) for t in range(images.shape[0])]
            elif images.ndim == 3:
                # Single frame
                pixel_values = self.image_transform(images)

            # Handle wrist camera images
            if self.use_wrist_image and self.wrist_camera_key is not None:
                wrist_images = item[self.wrist_camera_key]
                if wrist_images.ndim == 4:
                    wrist_pixel_values = [self.image_transform(wrist_images[t]) for t in range(wrist_images.shape[0])]
                elif wrist_images.ndim == 3:
                    wrist_pixel_values = self.image_transform(wrist_images)

                # Combine primary + wrist images per timestep
                if isinstance(pixel_values, list):
                    pixel_values = [[pv, wpv] for pv, wpv in zip(pixel_values, wrist_pixel_values)]
                else:
                    pixel_values = [pixel_values, wrist_pixel_values]
        else:
            # Fallback: use raw images
            pixel_values = item[self.camera_key]
            if self.use_wrist_image and self.wrist_camera_key is not None:
                pixel_values = {
                    "primary": item[self.camera_key],
                    "wrist": item[self.wrist_camera_key],
                }

        # --- Actions ---
        # [past + future + 1, action_dim] — take first action_dim dims
        actions = item["action"][:, :self.action_dim].clone()

        # Trim to expected length
        expected_len = self.past_action_window_size + self.future_action_window_size + 1
        if actions.shape[0] > expected_len:
            actions = actions[:expected_len]

        # Quantile normalization: [q01, q99] -> [-1, 1]
        actions = normalize_action_quantile(
            actions, self.action_q01, self.action_q99, self.action_mask,
        )

        # Extract future action window
        actions_future = actions[-(self.future_action_window_size + 1):]

        # --- Action masks ---
        action_is_pad = item.get("action_is_pad", None)
        if action_is_pad is not None:
            # Get mask for the future action window portion
            pad_len = action_is_pad.shape[0]
            if pad_len > expected_len:
                action_is_pad = action_is_pad[:expected_len]
            elif pad_len < expected_len:
                action_is_pad = torch.cat([action_is_pad, torch.ones(expected_len - pad_len, dtype=torch.bool)])

            # Extract future portion mask — True means valid (not padded)
            action_masks = ~action_is_pad[-(self.future_action_window_size + 1):]
        else:
            # All valid
            action_masks = torch.ones(self.future_action_window_size + 1, dtype=torch.bool)

        # --- Text tokenization ---
        task = item.get("task", "")
        if isinstance(task, list):
            task = task[0] if len(task) > 0 else ""

        # Build VLA prompt
        prompt_text = f"What action should the robot take to {task.strip().lower()}?"

        if self.tokenizer is not None:
            input_ids = self.tokenizer(prompt_text, add_special_tokens=True).input_ids
            labels = list(input_ids)

            # Add tokenizer-specific special tokens
            if self._is_llama_tokenizer:
                # Add empty token (29871) and cognition/EOS token (2)
                input_ids = input_ids + [29871, 2]
                labels = labels + [29871, 2]
            elif self._is_qwen_tokenizer:
                # Add empty token (220) and cognition token (151645)
                input_ids = input_ids + [220, 151645]
                labels = labels + [220, 151645]

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            raise ValueError("Tokenizer is required for CronusVLADataset")

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "actions": actions_future,           # [future_action_window_size+1, action_dim]
            "action_masks": action_masks,         # [future_action_window_size+1]
            "dataset_name": task,
        }

    def collater(self, samples: list[dict]) -> dict[str, Any]:
        """Collate function: pad input_ids/labels, handle multi-view images,
        stack actions/masks. Matches PaddedCollatorForActionPrediction pattern.
        """
        # --- Pad input_ids and labels ---
        input_ids = [s["input_ids"] for s in samples]
        labels = [s["labels"] for s in samples]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate if necessary
        input_ids = input_ids[:, :self.max_text_len]
        labels = labels[:, :self.max_text_len]

        # Attention mask: non-pad tokens are True
        attention_mask = input_ids.ne(pad_token_id)

        # --- Handle pixel_values ---
        pixel_values_list = [s["pixel_values"] for s in samples]

        # Check if multi-image (list of lists for wrist camera support)
        if isinstance(pixel_values_list[0], list) and isinstance(pixel_values_list[0][0], list):
            # Multi-image per timestep: [[primary, wrist], [primary, wrist], ...]
            obs_window_size = len(pixel_values_list[0])
            # Stack into dict format per camera key
            pixel_values = {
                k: torch.stack([
                    pixel_values_list[idx][ix][k_idx]
                    for idx in range(len(pixel_values_list))
                    for ix in range(len(pixel_values_list[idx]))
                ])
                for k_idx, k in enumerate(pixel_values_list[0][0][0].keys() if isinstance(pixel_values_list[0][0][0], dict) else ["dino", "siglip"])
            }
            # Repeat input_ids/labels/attention_mask by obs_window_size for multi-view
            input_ids = input_ids.repeat_interleave(obs_window_size, dim=0)
            labels = labels.repeat_interleave(obs_window_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(obs_window_size, dim=0)
        elif isinstance(pixel_values_list[0], list):
            # Single image per timestep, but multiple timesteps
            obs_window_size = len(pixel_values_list[0])
            if isinstance(pixel_values_list[0][0], dict):
                # DinoSigLIP: dict with "dino" and "siglip" keys
                pixel_values = {
                    k: torch.stack([
                        pixel_values_list[idx][ix][k]
                        for idx in range(len(pixel_values_list))
                        for ix in range(len(pixel_values_list[idx]))
                    ])
                    for k in pixel_values_list[0][0].keys()
                }
                input_ids = input_ids.repeat_interleave(obs_window_size, dim=0)
                labels = labels.repeat_interleave(obs_window_size, dim=0)
                attention_mask = attention_mask.repeat_interleave(obs_window_size, dim=0)
            else:
                # Plain tensor
                pixel_values = torch.stack([
                    pixel_values_list[idx][ix]
                    for idx in range(len(pixel_values_list))
                    for ix in range(len(pixel_values_list[idx]))
                ])
                input_ids = input_ids.repeat_interleave(obs_window_size, dim=0)
                labels = labels.repeat_interleave(obs_window_size, dim=0)
                attention_mask = attention_mask.repeat_interleave(obs_window_size, dim=0)
        elif isinstance(pixel_values_list[0], dict):
            # DinoSigLIP: dict with "dino" and "siglip" keys (single image)
            pixel_values = {
                k: torch.stack([pixel_values_list[idx][k] for idx in range(len(pixel_values_list))])
                for k in pixel_values_list[0].keys()
            }
        else:
            # Plain tensor (single image, single view)
            pixel_values = torch.stack(pixel_values_list)

        # --- Stack actions and action_masks ---
        actions = torch.stack([s["actions"] for s in samples])
        action_masks = torch.stack([s["action_masks"] for s in samples])

        # --- Dataset names ---
        dataset_names = [s["dataset_name"] for s in samples]

        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
            "action_masks": action_masks,
            "dataset_names": dataset_names,
        }

        return batch