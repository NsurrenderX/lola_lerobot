"""
RoboVLMDataset: A dataset wrapper that produces data in the same format as the
original RoboVLMs DiskCalvinDataset.collater().

Key differences from raw LeRobotDataset:
- Loads window_size + fwd_pred_next_n - 1 action frames (for chunking)
- Optionally applies normalize_action (clip [norm_min, norm_max] -> [-1, 1], preserve gripper)
  - When skip_action_normalize=True (default), assumes data is already in the correct range
- Binarizes gripper in actions
- Trims last timestep from actions before unfold (matching original collater's [:-1])
- Creates action_chunck [B, window_size, fwd_pred_next_n, 7] via .unfold()
- Creates chunk_mask from action_is_pad
- Tokenizes text with Kosmos template
- Applies CLIP image transforms
- Supports multiple camera keys (rgb_static, rgb_gripper, primary, secondary)
"""

import logging
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

logger = logging.getLogger(__name__)

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_action(action: torch.Tensor, action_min: float, action_max: float) -> torch.Tensor:
    """Clip action to [action_min, action_max] and linearly map to [-1, 1].

    The last dimension (gripper) is preserved as-is, matching the original
    `normalize_action(action, maintain_last=True)` from RoboVLMs.
    """
    last_val = action[..., -1].clone()
    action = action.clamp(min=action_min, max=action_max)
    action = 2.0 * (action - action_min) / (action_max - action_min) - 1.0
    action[..., -1] = last_val
    return action


def normalize_action_zscore(
    action: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    gripper_dim_indices: tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Z-score normalize arm dimensions and binarize gripper dimensions.

    Arm dims: (x - mean) / (std + eps)
    Gripper dims: (x > 0).float() — maps {-1, 1} -> {0, 1} for BCE loss

    Args:
        action: [..., action_dim] tensor
        action_mean: [action_dim] per-dimension mean
        action_std: [action_dim] per-dimension std
        gripper_dim_indices: absolute indices of gripper dims (e.g., (9, 19))
        eps: small value to prevent division by zero
    """
    result = action.clone()
    action_dim = action.shape[-1]

    if gripper_dim_indices is not None and len(gripper_dim_indices) > 0:
        arm_mask = torch.ones(action_dim, dtype=torch.bool)
        arm_mask[list(gripper_dim_indices)] = False
        gripper_mask = ~arm_mask

        # Z-score arm dims
        result[..., arm_mask] = (
            (action[..., arm_mask] - action_mean[arm_mask])
            / (action_std[arm_mask] + eps)
        )

        # Binarize gripper dims: {-1, 1} -> {0, 1}
        result[..., gripper_mask] = (action[..., gripper_mask] > 0).float()
    else:
        # No gripper dims specified: z-score all dims
        result = (action - action_mean) / (action_std + eps)

    return result


def unnoramalize_action(action: torch.Tensor, action_min: float, action_max: float) -> torch.Tensor:
    """Map [-1, 1] back to [action_min, action_max].

    The last dimension (gripper) is preserved as-is, matching the original
    `unnoramalize_action` from RoboVLMs data_utils.py.
    """
    last_val = action[..., -1].clone()
    res = 0.5 * (action + 1) * (action_max - action_min) + action_min
    res[..., -1] = last_val
    return res


class RoboVLMDataset(Dataset):
    """Wraps LeRobotDataset to produce data matching the original RoboVLM collater format.

    For each sample, loads a window of observations and actions, normalizes actions,
    creates per-timestep action chunks via .unfold(), binarizes gripper, tokenizes
    text, and applies CLIP image transforms.
    """

    def __init__(
        self,
        repo_id: str,
        config,
        root: str | None = None,
        tokenizer=None,
    ):
        self.config = config
        self.window_size = config.window_size
        self.fwd_pred_next_n = config.fwd_pred_next_n
        self.state_dim = config.state_dim
        self.norm_action = config.norm_action and not config.skip_action_normalize
        self.norm_min = config.norm_min
        self.norm_max = config.norm_max
        self.image_size = config.image_size
        self.image_mean = config.image_mean
        self.image_std = config.image_std

        # Load dataset metadata to get fps and camera keys
        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        fps = self.meta.fps

        # Compute delta_timestamps:
        # - Observations (images): window_size frames ending at current time
        #   [-7/fps, -6/fps, ..., 0/fps] for window_size=8
        # - Actions: need window_size + fwd_pred_next_n frames
        #   because original collater does [:, :-1] trimming before unfold,
        #   so we need T = window_size + fwd_pred_next_n to get window_size chunks after [:-1]
        #   After [:-1]: T-1 = window_size + fwd_pred_next_n - 1
        #   .unfold produces (T-1) - fwd_pred_next_n + 1 = window_size chunks
        # When dataset_tolerance_frames > 0, extend action deltas by tol on each side:
        #   This loads tol extra past + tol extra future frames, so each chunk
        #   contains [tol past, fwd center, tol future] = fwd+2*tol frames.
        obs_delta_indices = list(range(-self.window_size + 1, 1))
        obs_delta_ts = [i / fps for i in obs_delta_indices]

        tol = self.config.dataset_tolerance_frames
        action_delta_indices = list(range(-self.window_size + 1 - tol, self.fwd_pred_next_n + 1 + tol))
        action_delta_ts = [i / fps for i in action_delta_indices]

        delta_timestamps = {}
        for cam_key in self.meta.camera_keys:
            delta_timestamps[cam_key] = obs_delta_ts
        delta_timestamps["observation.state"] = obs_delta_ts
        delta_timestamps["action"] = action_delta_ts

        # Use tolerance_frames=2 to avoid strict tolerance_s assertion failures
        # on videos with slightly imprecise timestamps. tolerance_frames adapts
        # to each video's actual fps, so it's more robust than computing
        # tolerance_s from the dataset's nominal fps.
        self.lerobot_ds = LeRobotDataset(
            repo_id, root=root, delta_timestamps=delta_timestamps, tolerance_frames=2,
        )

        # Identify primary camera key (first one)
        self.camera_key = self.meta.camera_keys[0]

        # CLIP image transform
        self.clip_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(self.image_mean, self.image_std),
        ])

        # Tensor-based CLIP transform for already-tensor images
        self.tensor_resize = T.Resize(
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.BICUBIC,
        )
        self.tensor_normalize = T.Normalize(self.image_mean, self.image_std)

        # Tokenizer
        self.tokenizer = tokenizer
        self.max_text_len = config.max_text_len

    def __len__(self):
        return len(self.lerobot_ds)

    def _transform_image_tensor(self, img: torch.Tensor) -> torch.Tensor:
        """Apply CLIP transforms to a [C, H, W] tensor."""
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = self.tensor_resize(img)
        img = self.tensor_normalize(img)
        return img

    def _transform_images(self, images: torch.Tensor) -> torch.Tensor:
        """Transform a batch of images [T, C, H, W] or [B, T, C, H, W]."""
        if images.ndim == 4:
            return torch.stack([self._transform_image_tensor(images[t]) for t in range(images.shape[0])])
        elif images.ndim == 5:
            return torch.stack([
                torch.stack([self._transform_image_tensor(images[b, t]) for t in range(images.shape[1])])
                for b in range(images.shape[0])
            ])
        return images

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.lerobot_ds[idx]

        # --- Images ---
        # [window_size, C, H, W] — apply CLIP transforms
        images = item[self.camera_key]  # [T, C, H, W]
        if isinstance(images, torch.Tensor):
            images = self._transform_images(images)
        rgb = images  # [window_size, C, H, W]

        # --- Hand camera (rgb_gripper) ---
        hand_rgb = None
        if self.config.use_hand_rgb and len(self.meta.camera_keys) > 1:
            # Find the gripper/hand camera key
            hand_key = None
            for key in self.meta.camera_keys:
                if key != self.camera_key:
                    hand_key = key
                    break
            if hand_key is not None:
                hand_images = item[hand_key]  # [T, C, H, W]
                if isinstance(hand_images, torch.Tensor):
                    hand_images = self._transform_images(hand_images)
                hand_rgb = hand_images

        # --- Actions ---
        # [window_size + fwd_pred_next_n + 2*tol, action_dim]
        tol = self.config.dataset_tolerance_frames
        actions = item["action"][:, :self.config.action_dim].clone()

        # Trim to window_size + fwd_pred_next_n + 2*tol if we got more
        needed_len = self.window_size + self.fwd_pred_next_n + 2 * tol
        if actions.shape[0] > needed_len:
            actions = actions[:needed_len]

        # Original collater does [:, :-1] trimming before unfold — remove last timestep
        if actions.shape[0] > 1:
            actions = actions[:-1]

        # Normalize arm actions to [-1, 1] (preserve gripper)
        # Only apply when data is in the original physical range (e.g. [-0.65, 0.65])
        # Skip when data is already in [-1, 1] (lerobot pre-normalized) or in original range
        if self.norm_action:
            actions = normalize_action(actions, self.norm_min, self.norm_max)

        # Binarize gripper in actions: ((gripper + 1) // 2).float()
        # This maps {-1, 1} -> {0, 1} using integer division
        actions[..., -1] = ((actions[..., -1] + 1) // 2).float()

        # Create action chunks via .unfold()
        # When dataset_tolerance_frames > 0, chunk width = fwd_pred_next_n + 2*tol
        # Each chunk contains [tol past, fwd center, tol future] frames.
        # The center portion (indices [tol : fwd+tol]) aligns with the model prediction.
        chunk_width = self.fwd_pred_next_n + 2 * tol
        T_len = actions.shape[0]
        if T_len >= chunk_width:
            action_chunck = actions.unfold(0, chunk_width, 1).permute(0, 2, 1)
            # action_chunck: [num_chunks, chunk_width, 7]
        else:
            # Fallback: pad if not enough frames
            padded = torch.zeros(needed_len, actions.shape[-1], dtype=actions.dtype)
            padded[:T_len] = actions
            action_chunck = padded.unfold(0, chunk_width, 1).permute(0, 2, 1)

        # Trim to window_size chunks
        action_chunck = action_chunck[:self.window_size]

        # --- Chunk mask from action_is_pad ---
        # action_is_pad: [T] bool
        action_is_pad = item.get("action_is_pad", None)
        if action_is_pad is not None:
            # Pad to needed_len if shorter
            if action_is_pad.shape[0] < needed_len:
                pad_size = needed_len - action_is_pad.shape[0]
                action_is_pad = torch.cat([action_is_pad, torch.ones(pad_size, dtype=torch.bool)])
            action_is_pad = action_is_pad[:needed_len]

            # Same [: -1] trimming as actions
            if action_is_pad.shape[0] > 1:
                action_is_pad = action_is_pad[:-1]

            # unfold same as actions
            if action_is_pad.shape[0] >= chunk_width:
                chunk_mask = action_is_pad.unfold(0, chunk_width, 1)
                # chunk_mask: [num_chunks, chunk_width] — True where padded
            else:
                chunk_mask = torch.ones(self.window_size, chunk_width, dtype=torch.bool)
            chunk_mask = chunk_mask[:self.window_size]
            # Invert: original chunck_mask is True for VALID, not padded
            chunk_mask = ~chunk_mask
        else:
            chunk_mask = torch.ones(self.window_size, chunk_width, dtype=torch.bool)

        # --- Text tokenization ---
        task = item.get("task", "")
        if isinstance(task, list):
            task = task[0] if len(task) > 0 else ""

        result = {
            "rgb": rgb,                                    # [window_size, C, H, W]
            "action_chunck": action_chunck,                # [window_size, fwd_pred_next_n, 7]
            "chunck_mask": chunk_mask,                     # [window_size, fwd_pred_next_n]
            "task": task,                                  # str
        }

        # Add hand camera if available
        if hand_rgb is not None:
            result["hand_rgb"] = hand_rgb                  # [window_size, C, H, W]

        return result

    def collater(self, samples: list[dict]) -> dict[str, Any]:
        """Collate function that tokenizes text and stacks all tensors.

        Text tokenization is done here (at collation time) so that padding
        can be applied to the longest sequence in the batch, matching the
        original collater behavior.
        """
        batch = {}

        # Stack tensor fields
        batch["rgb"] = torch.stack([s["rgb"] for s in samples])                   # [B, ws, C, H, W]
        batch["action_chunck"] = torch.stack([s["action_chunck"] for s in samples])  # [B, ws, fwd, 7]
        batch["chunck_mask"] = torch.stack([s["chunck_mask"] for s in samples])   # [B, ws, fwd]

        # Hand camera (rgb_gripper)
        if "hand_rgb" in samples[0]:
            batch["hand_rgb"] = torch.stack([s["hand_rgb"] for s in samples])     # [B, ws, C, H, W]

        # Tokenize text
        tasks = [s["task"] for s in samples]
        template = "<grounding>An image of a robot {}"
        texts = [template.format(t.strip()) for t in tasks]

        if self.tokenizer is not None:
            self.tokenizer.padding_side = "right"
            encoded = self.tokenizer(
                texts,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=self.max_text_len,
            )
            batch["language"] = encoded["input_ids"]        # [B, max_text_len]
            batch["text_mask"] = encoded["attention_mask"]  # [B, max_text_len]
        else:
            # Fallback: should not happen in practice
            raise ValueError("Tokenizer is required for RoboVLMDataset.collater")

        batch["task"] = tasks

        return batch
