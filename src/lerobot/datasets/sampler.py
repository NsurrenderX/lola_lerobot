#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Iterator
from itertools import islice

import torch
from torch.utils.data import BatchSampler


class SkipBatchSampler:
    """Skip batch indices before dataset reads, retaining the full epoch length."""

    def __init__(self, batch_sampler: BatchSampler):
        self.batch_sampler = batch_sampler
        self.start_batch = 0

    @property
    def full_length(self) -> int:
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int, start_batch: int = 0):
        if not 0 <= start_batch <= self.full_length:
            raise ValueError(f"Invalid batch offset {start_batch} for {self.full_length} batches")
        self.start_batch = start_batch
        if hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        return islice(self.batch_sampler, self.start_batch, None)

    def __len__(self) -> int:
        return self.full_length - self.start_batch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
