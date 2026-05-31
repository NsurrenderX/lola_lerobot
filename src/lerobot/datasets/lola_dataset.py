#!/usr/bin/env python

# Copyright 2025 LoLA Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoLA专用数据集，支持加载从episode开始到当前帧的完整历史action。

与标准LeRobotDataset的区别：
- 标准LeRobotDataset只加载固定长度的历史帧（n_obs_steps帧）
- LoLADataset加载从episode开始到当前帧的所有action历史
- 支持左侧padding以处理变长历史序列
- V2: 支持completed_tasks（已完成任务序列）和transition-aware token mask
"""

import json
import os
import random
import torch
import torch.nn.functional as F
from typing import Callable

import os
import torch
from typing import Callable

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames, scan_video_seek_modes


class LoLADataset(LeRobotDataset):
    """
    支持加载完整历史action的LoLA专用数据集。

    在标准LeRobotDataset基础上，额外提供：
    - hist_actions_full: 从episode开始到当前帧的所有action
    - hist_actions_mask: 标识哪些是真实action vs padding

    使用方法：
        dataset = LoLADataset(
            repo_id="lerobot/pusht",
            max_history_length=100,
            action_chunk_size=10,  # 历史长度会被补齐到action_chunk_size的整数倍
            delta_timestamps={...},
        )

        item = dataset[0]
        # item["hist_actions_full"]: [padded_length, action_dim]
        # item["hist_actions_mask"]: [padded_length] (1=真实, 0=padding)
        # 其中 padded_length 是 action_chunk_size 的整数倍
    """

    def __init__(
        self,
        repo_id: str,
        max_history_length: int = 100,
        action_chunk_size: int = 10,
        history_padding_side: str = "left",
        root: str | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        tolerance_frame: int = 1,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        norm_action: bool | str = False,
        norm_min: float = -0.65,
        norm_max: float = 0.65,
        gripper_dim_indices_abs: tuple[int, ...] | None = None,
        history_type: str = "action",
        state_dim: int | None = None,
        # V2: completed tasks + transition masking
        track_completed_tasks: bool = True,
        transition_mask_rate: float = 0.0,
        completed_tasks_use_ann: bool = True,
        hist_action_token_drop_rate: float = 0.0,
        max_transition_len: int = 64,
        completed_tasks_history_len: int = 5,
    ):
        """
        Args:
            repo_id: 数据集仓库ID
            max_history_length: 历史action最大长度，超过则截断，不足则padding
            action_chunk_size: action块大小，历史长度会被补齐到该值的整数倍
            history_padding_side: padding方向，"left"或"right"
            root: 本地数据集根目录
            episodes: 指定加载的episode列表
            image_transforms: 图像变换
            delta_timestamps: 时间戳偏移配置
            tolerance_s: 时间戳容差
            revision: 版本
            force_cache_sync: 是否强制同步缓存
            download_videos: 是否下载视频
            video_backend: 视频后端
        """
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            tolerance_frames=tolerance_frame,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        # self.tolerance_frame = tolerance_frame
        self.max_history_length = max_history_length
        self.action_chunk_size = action_chunk_size
        self.history_padding_side = history_padding_side
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.gripper_dim_indices_abs = gripper_dim_indices_abs
        self.history_type = history_type

        # V2: completed tasks + transition masking
        self.track_completed_tasks = track_completed_tasks
        self.transition_mask_rate = transition_mask_rate  # 0=no mask, 1=full mask
        self.completed_tasks_use_ann = completed_tasks_use_ann
        self.hist_action_token_drop_rate = hist_action_token_drop_rate
        self.completed_tasks_history_len = completed_tasks_history_len

        # Z-score normalization stats (computed from dataset metadata)
        self._action_mean = None
        self._action_std = None
        if self.norm_action == "zscore":
            if "action" in self.meta.stats:
                import numpy as np
                _mean = self.meta.stats["action"]["mean"]
                _std = self.meta.stats["action"]["std"]
                self._action_mean = torch.tensor(_mean, dtype=torch.float32) if isinstance(_mean, np.ndarray) else _mean.float()
                self._action_std = torch.tensor(_std, dtype=torch.float32) if isinstance(_std, np.ndarray) else _std.float()
            else:
                raise ValueError("z-score normalization requires 'action' stats in dataset metadata")
            if self.gripper_dim_indices_abs is None:
                raise ValueError("z-score normalization requires gripper_dim_indices_abs to separate arm/gripper dims")

        # State dimension and normalization stats
        if state_dim is not None:
            self.state_dim = state_dim
        elif "observation.state" in self.features:
            self.state_dim = self.features["observation.state"]["shape"][0]
        else:
            self.state_dim = self.action_dim  # fallback

        self._state_mean = None
        self._state_std = None
        if self.norm_action == "zscore" and self.history_type == "state":
            if "observation.state" in self.meta.stats:
                import numpy as np
                _s_mean = self.meta.stats["observation.state"]["mean"]
                _s_std = self.meta.stats["observation.state"]["std"]
                self._state_mean = torch.tensor(_s_mean, dtype=torch.float32) if isinstance(_s_mean, np.ndarray) else _s_mean.float()
                self._state_std = torch.tensor(_s_std, dtype=torch.float32) if isinstance(_s_std, np.ndarray) else _s_std.float()
            else:
                raise ValueError("z-score normalization with history_type='state' requires 'observation.state' stats in dataset metadata")

        # 获取action维度
        if "action" in self.features:
            self.action_dim = self.features["action"]["shape"][0]
        else:
            self.action_dim = 1  # fallback

        # ── Load episode metadata (V2: completed_tasks, transition_len) ────
        self.episode_metadata = {}
        metadata_path = self.root / "calvin_episode_metadata.json"
        if metadata_path.exists():
            with open(str(metadata_path)) as f:
                raw_meta = json.load(f)
            self.episode_metadata = {int(k): v for k, v in raw_meta.items()}
            print(f"[LoLADataset] Loaded episode metadata: {len(self.episode_metadata)} episodes")
        else:
            print("[LoLADataset] No calvin_episode_metadata.json found — V1 mode (no completed_tasks)")

        # ── Load pre-computed hist_action/hist_state from npz (V2) ──────
        self._hist_action_all = None
        self._hist_state_all = None
        self._hist_len_all = None
        npz_path = self.root / "calvin_episode_metadata.npz"
        if npz_path.exists():
            import numpy as np
            npz = np.load(str(npz_path))
            self._hist_action_all = torch.from_numpy(npz["hist_action"])  # [n_ep, max_t, action_dim]
            self._hist_state_all = torch.from_numpy(npz["hist_state"])   # [n_ep, max_t, state_dim]
            self._hist_len_all = torch.from_numpy(npz["hist_len"])       # [n_ep]
            npz_max_t = self._hist_action_all.shape[1]
            if npz_max_t != max_transition_len:
                print(f"[LoLADataset] WARNING: npz max_t={npz_max_t} != max_transition_len={max_transition_len}")
            print(f"[LoLADataset] Loaded hist metadata: {self._hist_action_all.shape}")
        else:
            print("[LoLADataset] No calvin_episode_metadata.npz found — no pre-computed history")

        # ── Seek-mode mapping (scan videos at init) ────────────────
        self._video_seek_modes: dict[str, str] = {}
        if os.path.isdir(os.path.join(str(self.root), "videos")):
            self._video_seek_modes = scan_video_seek_modes(str(self.root), num_workers=8)
            exact_count = sum(1 for v in self._video_seek_modes.values() if v == "exact")
            print(f"[LoLADataset] seek-mode scan: {len(self._video_seek_modes)} videos, "
                  f"{exact_count} require exact mode")

        print(f"[LoLADataset] max_history_length: {max_history_length}")
        print(f"[LoLADataset] action_chunk_size: {action_chunk_size}")
        print(f"[LoLADataset] history_padding_side: {history_padding_side}")
        print(f"[LoLADataset] action_dim: {self.action_dim}")

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Override parent _query_videos to pass seek_mode from init-time scan."""
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)

            # Look up seek_mode from init-time scan mapping
            video_rel = str(self.meta.get_video_file_path(ep_idx, vid_key))
            if video_rel.startswith("videos/"):
                video_rel = video_rel[len("videos/"):]
            seek_mode = self._video_seek_modes.get(video_rel, "approximate")

            frames = decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, tolerance_frames=self.tolerance_frames, backend=self.video_backend, seek_mode=seek_mode)
            
            item[vid_key] = frames.squeeze(0)

        return item

    def __getitem__(self, idx) -> dict:
        """
        获取数据项，包含完整历史action。

        V2 enhancements:
        - completed_tasks / completed_tasks_ann: 任务历史序列（来自episode metadata）
        - hist_actions支持transition zone扩展（向前包含transition帧）
        - chunk-level mask: transition-dominant token用transition_mask_rate,
          task-dominant token用hist_action_token_drop_rate

        Returns:
            dict with additional keys:
            - hist_actions_full: [padded_length, action_dim] 历史action（含padding）
            - hist_actions_mask: [padded_length] 标识真实action (1) vs padding (0)
            - hist_actions_length: 标量，真实action数量
            - completed_tasks: list[str] 已完成的任务标签序列
            - completed_tasks_ann: list[str] 随机选择的annotation文本序列
        """
        # 调用父类方法获取基础数据
        item = super().__getitem__(idx)

        # 获取episode信息
        ep_idx = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # ── V2: Compute completed_tasks + completed_tasks_ann ───────────
        if self.track_completed_tasks and ep_idx in self.episode_metadata:
            meta = self.episode_metadata[ep_idx]
            completed_tasks = meta["completed_tasks"]  # list[str]
            completed_tasks_ann_choices = meta.get("completed_tasks_ann_choices", {})

            # Randomly select one 'ann' per completed task (training diversity)
            # Only keep the most recent completed_tasks_history_len tasks
            max_keep = self.completed_tasks_history_len
            if max_keep == 0:
                completed_tasks = []
            elif len(completed_tasks) > max_keep:
                completed_tasks = completed_tasks[-max_keep:]

            completed_tasks_ann = []
            for task in completed_tasks:
                choices = completed_tasks_ann_choices.get(task, [task])
                if self.completed_tasks_use_ann:
                    selected_ann = random.choice(choices)
                else:
                    selected_ann = task  # concise mode: just use task label
                completed_tasks_ann.append(selected_ann)

            item["completed_tasks"] = completed_tasks
            item["completed_tasks_ann"] = completed_tasks_ann
        elif self.track_completed_tasks:
            # V1 dataset without metadata — no completed tasks
            item["completed_tasks"] = []
            item["completed_tasks_ann"] = []

        # ── V2: Build history from two sources, SEPARATE padding/mask ────
        # Source 1: Transition history (pre-annotation) from npz
        # Source 2: Task history (current episode frames from parquet)
        # Each source is padded/masked independently to chunk_size multiples,
        # then concatenated as [transition_padded | task_padded].
        # This ensures chunk boundaries align so previous_task_end token
        # can be inserted cleanly between transition chunks and task chunks.

        chunk_size = self.action_chunk_size
        max_total_len = self.max_history_length

        # ── Step 1: Extract transition and task data ──
        transition_data = None
        transition_data_len = 0
        if self._hist_action_all is not None and ep_idx < len(self._hist_action_all):
            pre_len = int(self._hist_len_all[ep_idx])
            if self.history_type == "state":
                transition_data = self._hist_state_all[ep_idx]
            else:
                transition_data = self._hist_action_all[ep_idx]
            transition_data_len = pre_len
            # Extract real (non-padded) portion
            offset = transition_data.shape[0] - transition_data_len
            transition_data = transition_data[offset:]

        task_data = None
        task_frame_count = idx - ep_start + 1
        if task_frame_count > 0:
            task_indices = list(range(ep_start, idx + 1))
            if self.history_type == "state":
                task_data_dict = self._query_hf_dataset({"observation.state": task_indices})
                task_data = task_data_dict["observation.state"]
            else:
                task_data_dict = self._query_hf_dataset({"action": task_indices})
                task_data = task_data_dict["action"]

        # ── Step 2: Truncate (prioritize truncating transition, keep task intact) ──
        n_transition = transition_data_len if transition_data is not None else 0

        if transition_data is not None and n_transition > 0:
            total_raw_len = n_transition + (task_data.shape[0] if task_data is not None else 0)
            if total_raw_len > max_total_len:
                # Truncate transition from left, keep task intact
                task_len = task_data.shape[0] if task_data is not None else 0
                max_transition_allowed = max_total_len - task_len
                if max_transition_allowed <= 0:
                    # Task alone exceeds max_history_length, drop all transition
                    transition_data = None
                    n_transition = 0
                elif n_transition > max_transition_allowed:
                    truncated = n_transition - max_transition_allowed
                    transition_data = transition_data[truncated:]
                    n_transition = max_transition_allowed
            # If task still exceeds remaining space
            if task_data is not None and task_data.shape[0] > max_total_len - n_transition:
                task_data = task_data[-(max_total_len - n_transition):]
        elif task_data is not None and task_data.shape[0] > max_total_len:
            task_data = task_data[-max_total_len:]

        # ── Step 3: Separate padding and mask ──
        def _pad_to_chunk_size(data, mask, chunk_size, max_len):
            """Left-pad data and mask to chunk_size multiple, truncate if exceeding max_len."""
            if data.shape[0] == 0:
                return data, mask, 0
            padded_length = ((data.shape[0] + chunk_size - 1) // chunk_size) * chunk_size
            if padded_length > max_len:
                padded_length = (max_len // chunk_size) * chunk_size
                data = data[-padded_length:]
                mask = mask[-padded_length:]
            pad_len = padded_length - data.shape[0]
            if pad_len > 0:
                dim = data.shape[1] if data.dim() > 1 else (self.state_dim if self.history_type == "state" else self.action_dim)
                if data.dim() > 1:
                    padding_data = torch.zeros(pad_len, dim, dtype=data.dtype)
                else:
                    padding_data = torch.zeros(pad_len, dtype=data.dtype)
                padding_mask = torch.zeros(pad_len, dtype=mask.dtype)
                data = torch.cat([padding_data, data], dim=0)
                mask = torch.cat([padding_mask, mask], dim=0)
            return data, mask, padded_length

        # ── Transition part ──
        if transition_data is not None and n_transition > 0:
            t_mask = torch.ones(n_transition, dtype=torch.bool)
            # Pure transition chunks: apply transition_mask_rate per chunk
            if self.transition_mask_rate > 0 and n_transition >= chunk_size:
                num_t_chunks = n_transition // chunk_size
                for ci in range(num_t_chunks):
                    if random.random() < self.transition_mask_rate:
                        t_mask[ci * chunk_size : (ci + 1) * chunk_size] = False
                # Ensure at least one valid chunk
                if int(t_mask.sum().item()) == 0 and n_transition >= chunk_size:
                    t_mask[:chunk_size] = True
            t_max = max_total_len  # transition can use up to full max_history_length
            t_data, t_mask, t_padded_len = _pad_to_chunk_size(transition_data, t_mask, chunk_size, t_max)
            n_transition_chunks = t_padded_len // chunk_size
        else:
            dim = self.state_dim if self.history_type == "state" else self.action_dim
            t_data = torch.zeros(0, dim, dtype=torch.float32)
            t_mask = torch.zeros(0, dtype=torch.bool)
            n_transition_chunks = 0

        # ── Task part ──
        if task_data is not None and task_data.shape[0] > 0:
            task_len = task_data.shape[0]
            k_mask = torch.ones(task_len, dtype=torch.bool)
            # Pure task chunks: apply hist_action_token_drop_rate per chunk
            if self.hist_action_token_drop_rate > 0 and task_len >= chunk_size:
                num_k_chunks = task_len // chunk_size
                for ci in range(num_k_chunks):
                    if random.random() < self.hist_action_token_drop_rate:
                        k_mask[ci * chunk_size : (ci + 1) * chunk_size] = False
                # Ensure at least one valid chunk
                if int(k_mask.sum().item()) == 0 and task_len >= chunk_size:
                    k_mask[:chunk_size] = True
            k_max = max_total_len - t_data.shape[0]  # remaining space after transition
            k_data, k_mask, k_padded_len = _pad_to_chunk_size(task_data, k_mask, chunk_size, k_max)
        else:
            dim = self.state_dim if self.history_type == "state" else self.action_dim
            k_data = torch.zeros(0, dim, dtype=torch.float32)
            k_mask = torch.zeros(0, dtype=torch.bool)

        # ── Step 4: Merge ──
        # Concatenate as [transition_padded | task_padded]
        if t_data.shape[0] > 0 and k_data.shape[0] > 0:
            hist_data = torch.cat([t_data, k_data], dim=0)
            hist_mask = torch.cat([t_mask, k_mask], dim=0)
        elif t_data.shape[0] > 0:
            hist_data = t_data
            hist_mask = t_mask
        elif k_data.shape[0] > 0:
            hist_data = k_data
            hist_mask = k_mask
        else:
            dim = self.state_dim if self.history_type == "state" else self.action_dim
            hist_data = torch.zeros(chunk_size, dim, dtype=torch.float32)
            hist_mask = torch.zeros(chunk_size, dtype=torch.bool)

        # Zero out masked entries in data
        hist_data = hist_data * hist_mask.unsqueeze(-1)
        actual_history_length = int(hist_mask.sum().item())

        hist_key_prefix = "hist_states" if self.history_type == "state" else "hist_actions"
        item[f"{hist_key_prefix}_full"] = hist_data
        item[f"{hist_key_prefix}_mask"] = hist_mask
        item[f"{hist_key_prefix}_length"] = torch.tensor(actual_history_length, dtype=torch.long)
        item["n_transition"] = torch.tensor(n_transition, dtype=torch.long)
        item["n_transition_chunks"] = torch.tensor(n_transition_chunks, dtype=torch.long)

        # Normalization
        if self.norm_action in (True, "minmax", "robovlm"):
            from lerobot.datasets.robovlm_dataset import normalize_action
            if "action" in item:
                item["action"] = normalize_action(item["action"], self.norm_min, self.norm_max)
            if "hist_actions_full" in item:
                item["hist_actions_full"] = normalize_action(item["hist_actions_full"], self.norm_min, self.norm_max)
            if "hist_states_full" in item:
                item["hist_states_full"] = normalize_action(item["hist_states_full"], self.norm_min, self.norm_max)
        elif self.norm_action == "zscore":
            from lerobot.datasets.robovlm_dataset import normalize_action_zscore
            if "action" in item:
                item["action"] = normalize_action_zscore(
                    item["action"], self._action_mean, self._action_std,
                    self.gripper_dim_indices_abs,
                )
            if "hist_actions_full" in item:
                item["hist_actions_full"] = normalize_action_zscore(
                    item["hist_actions_full"], self._action_mean, self._action_std,
                    self.gripper_dim_indices_abs,
                )
            if "hist_states_full" in item and self._state_mean is not None:
                item["hist_states_full"] = (
                    (item["hist_states_full"] - self._state_mean) / (self._state_std + 1e-8)
                )

        return item


class LoLADatasetMetadata(LeRobotDatasetMetadata):
    """LoLA数据集元数据，继承自LeRobotDatasetMetadata"""
    pass