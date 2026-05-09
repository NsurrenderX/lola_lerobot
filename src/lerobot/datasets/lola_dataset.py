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
"""

import os
import torch
import torch.nn.functional as F
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

        # 获取action维度
        if "action" in self.features:
            self.action_dim = self.features["action"]["shape"][0]
        else:
            self.action_dim = 1  # fallback

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

        Returns:
            dict with additional keys:
            - hist_actions_full: [padded_length, action_dim] 历史action（含padding）
            - hist_actions_mask: [padded_length] 标识真实action (1) vs padding (0)
            - hist_actions_length: 标量，真实action数量

        Note:
            padded_length 会被补齐到 action_chunk_size 的整数倍，
            便于模型将每 action_chunk_size 个 action 合并为 1 个 Token。
        """
        # 调用父类方法获取基础数据
        item = super().__getitem__(idx)

        # 获取episode信息
        ep_idx = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # 计算当前帧在episode内的位置
        current_frame_in_ep = idx - ep_start

        # 计算历史长度（从episode开始到当前帧，包含当前帧）
        history_length = current_frame_in_ep + 1

        # 确定实际加载的历史范围
        if history_length > self.max_history_length:
            # 历史超过最大长度，只取最近的max_history_length帧
            start_idx = idx - self.max_history_length + 1
            actual_history_length = self.max_history_length
        else:
            # 历史不足最大长度，从episode开始加载
            start_idx = ep_start
            actual_history_length = history_length

        # 加载历史action
        history_indices = list(range(start_idx, idx + 1))
        hist_actions_dict = self._query_hf_dataset({"action": history_indices})
        hist_actions = hist_actions_dict["action"]  # [actual_length, action_dim]

        # 创建mask：标识真实action
        hist_actions_mask = torch.ones(actual_history_length, dtype=torch.bool)

        # 计算补齐后的长度（action_chunk_size的整数倍）
        # 向上取整到最近的 action_chunk_size 倍数
        padded_length = ((actual_history_length + self.action_chunk_size - 1) // self.action_chunk_size) * self.action_chunk_size

        # 确保不超过 max_history_length（向上取整后可能超过）
        if padded_length > self.max_history_length:
            # 如果补齐后超过最大长度，向下取整到最近的 action_chunk_size 倍数
            padded_length = (self.max_history_length // self.action_chunk_size) * self.action_chunk_size
            # 如果实际历史长度超过了调整后的 padded_length，需要截断
            if actual_history_length > padded_length:
                # 从开头截断（保留最近的动作）
                truncate_length = actual_history_length - padded_length
                hist_actions = hist_actions[truncate_length:]
                hist_actions_mask = hist_actions_mask[truncate_length:]
                actual_history_length = padded_length

        # Padding到 action_chunk_size 的整数倍
        if actual_history_length < padded_length:
            pad_length = padded_length - actual_history_length

            # 创建padding张量
            padding_actions = torch.zeros(pad_length, self.action_dim, dtype=hist_actions.dtype)
            padding_mask = torch.zeros(pad_length, dtype=torch.bool)

            if self.history_padding_side == "left":
                # 左侧padding：padding在前面
                hist_actions = torch.cat([padding_actions, hist_actions], dim=0)
                hist_actions_mask = torch.cat([padding_mask, hist_actions_mask], dim=0)
            else:
                # 右侧padding：padding在后面
                hist_actions = torch.cat([hist_actions, padding_actions], dim=0)
                hist_actions_mask = torch.cat([hist_actions_mask, padding_mask], dim=0)

        # 添加到item
        item["hist_actions_full"] = hist_actions  # [padded_length, action_dim]
        item["hist_actions_mask"] = hist_actions_mask  # [padded_length]
        item["hist_actions_length"] = torch.tensor(actual_history_length, dtype=torch.long)

        # Action normalization
        if self.norm_action in (True, "minmax", "robovlm"):
            # RoboVLM-style: min-max → [-1, 1], preserve gripper as-is
            from lerobot.datasets.robovlm_dataset import normalize_action
            if "action" in item:
                item["action"] = normalize_action(item["action"], self.norm_min, self.norm_max)
            if "hist_actions_full" in item:
                item["hist_actions_full"] = normalize_action(item["hist_actions_full"], self.norm_min, self.norm_max)
        elif self.norm_action == "zscore":
            # Z-score arm dims + binarize gripper dims for BCE
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

        return item


class LoLADatasetMetadata(LeRobotDatasetMetadata):
    """LoLA数据集元数据，继承自LeRobotDatasetMetadata"""
    pass