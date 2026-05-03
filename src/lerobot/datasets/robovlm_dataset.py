"""
RoboVLMDataset: A dataset wrapper that produces data in the same format as the
original RoboVLMs DiskCalvinDataset.collater(), matching the
finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10 config.

Key differences from raw LeRobotDataset:
- Loads window_size + fwd_pred_next_n - 1 action frames (for chunking)
- Applies normalize_action (clip [norm_min, norm_max] -> [-1, 1], preserve gripper)
- Binarizes gripper in both action and state
- Creates action_chunck [B, window_size, fwd_pred_next_n, 7] via .unfold()
- Creates chunk_mask from action_is_pad
- Tokenizes text with Kosmos template
- Applies CLIP image transforms
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
        self.norm_action = config.norm_action
        self.norm_min = config.norm_min
        self.norm_max = config.norm_max
        self.image_size = config.image_size
        self.image_mean = config.image_mean
        self.image_std = config.image_std

        # Load dataset metadata to get fps and camera keys
        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        fps = self.meta.fps

        # Compute delta_timestamps:
        # - Observations (images, state): window_size frames ending at current time
        #   [-7/fps, -6/fps, ..., 0/fps] for window_size=8
        # - Actions: need window_size + fwd_pred_next_n - 1 frames
        #   because .unfold(1, fwd_pred_next_n, 1) requires T >= fwd_pred_next_n
        #   and produces T - fwd_pred_next_n + 1 chunks = window_size
        #   So T = window_size + fwd_pred_next_n - 1
        #   Starting from -(window_size-1)/fps to (fwd_pred_next_n-1)/fps
        obs_delta_indices = list(range(-self.window_size + 1, 1))
        obs_delta_ts = [i / fps for i in obs_delta_indices]

        action_delta_indices = list(range(-self.window_size + 1, self.fwd_pred_next_n))
        action_delta_ts = [i / fps for i in action_delta_indices]

        delta_timestamps = {}
        for cam_key in self.meta.camera_keys:
            delta_timestamps[cam_key] = obs_delta_ts
        delta_timestamps["observation.state"] = obs_delta_ts
        delta_timestamps["action"] = action_delta_ts

        self.lerobot_ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)

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

        # --- State ---
        # [window_size, state_dim] — take first state_dim=7 dims, binarize gripper
        state = item["observation.state"][:, :self.state_dim]  # [T, state_dim]
        # Binarize gripper (last dim): threshold at 0 — values >0 -> 1, <=0 -> 0
        # Raw gripper from dataset is typically {-1, 1}, so this maps to {0, 1}
        state[..., -1] = (state[..., -1] > 0).float()

        # --- Actions ---
        # [window_size + fwd_pred_next_n - 1, action_dim]
        actions = item["action"][:, :self.config.action_dim].clone()

        # Trim to window_size + fwd_pred_next_n - 1 if we got more
        needed_len = self.window_size + self.fwd_pred_next_n - 1
        if actions.shape[0] > needed_len:
            actions = actions[:needed_len]

        # Normalize arm actions to [-1, 1] (preserve gripper)
        if self.norm_action:
            actions = normalize_action(actions, self.norm_min, self.norm_max)

        # Binarize gripper in actions: ((gripper + 1) // 2).float()
        # This maps {-1, 1} -> {0, 1} using integer division
        actions[..., -1] = ((actions[..., -1] + 1) // 2).float()

        # Create action chunks via .unfold()
        # actions: [T, 7] where T = window_size + fwd_pred_next_n - 1
        # .unfold(0, fwd_pred_next_n, 1) -> [T - fwd + 1, 7, fwd] = [window_size, 7, fwd]
        # .permute(0, 2, 1) -> [window_size, fwd, 7]
        T_len = actions.shape[0]
        if T_len >= self.fwd_pred_next_n:
            action_chunck = actions.unfold(0, self.fwd_pred_next_n, 1).permute(0, 2, 1)
            # action_chunck: [window_size, fwd_pred_next_n, 7]
        else:
            # Fallback: pad if not enough frames
            padded = torch.zeros(needed_len, actions.shape[-1], dtype=actions.dtype)
            padded[:T_len] = actions
            action_chunck = padded.unfold(0, self.fwd_pred_next_n, 1).permute(0, 2, 1)

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
            # unfold same as actions
            if action_is_pad.shape[0] >= self.fwd_pred_next_n:
                chunk_mask = action_is_pad.unfold(0, self.fwd_pred_next_n, 1)
                # chunk_mask: [window_size, fwd_pred_next_n] — True where padded
            else:
                chunk_mask = torch.ones(self.window_size, self.fwd_pred_next_n, dtype=torch.bool)
            chunk_mask = chunk_mask[:self.window_size]
            # Invert: original chunck_mask is True for VALID, not padded
            chunk_mask = ~chunk_mask
        else:
            chunk_mask = torch.ones(self.window_size, self.fwd_pred_next_n, dtype=torch.bool)

        # --- Text tokenization ---
        task = item.get("task", "")
        if isinstance(task, list):
            task = task[0] if len(task) > 0 else ""

        return {
            "rgb": rgb,                                    # [window_size, C, H, W]
            "rel_state": state,                            # [window_size, state_dim]
            "action_chunck": action_chunck,                # [window_size, fwd_pred_next_n, 7]
            "chunck_mask": chunk_mask,                     # [window_size, fwd_pred_next_n]
            "task": task,                                  # str
        }

    def collater(self, samples: list[dict]) -> dict[str, Any]:
        """Collate function that tokenizes text and stacks all tensors.

        Text tokenization is done here (at collation time) so that padding
        can be applied to the longest sequence in the batch, matching the
        original collater behavior.
        """
        batch = {}

        # Stack tensor fields
        batch["rgb"] = torch.stack([s["rgb"] for s in samples])                   # [B, ws, C, H, W]
        batch["rel_state"] = torch.stack([s["rel_state"] for s in samples])       # [B, ws, state_dim]
        batch["action_chunck"] = torch.stack([s["action_chunck"] for s in samples])  # [B, ws, fwd, 7]
        batch["chunck_mask"] = torch.stack([s["chunck_mask"] for s in samples])   # [B, ws, fwd]

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
