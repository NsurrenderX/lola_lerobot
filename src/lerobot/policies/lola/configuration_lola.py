#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("lola")
@dataclass
class LoLAConfig(PreTrainedConfig):
    # ==========================
    # 1. VLM Settings
    # ==========================
    vlm_model_name: str = "Qwen/Qwen3.5-4B" # Qwen3.5-4B
    vlm_path: str | None = None
    vlm_extract_layers: tuple = (8, 16, 24)
    vlm_hidden_size: int = 2560                # Qwen3.5 提供的 hidden_size
    empty_token_id: int = 248044               # 使用 Qwen3.5 的 eos_token 作为空 Token

    # ==========================
    # 2. Action Encoding Settings
    # ==========================
    n_obs_steps: int = 1
    action_dim: int = 20                       # 3D trans + 6D rot + 1D eef (双臂)
    action_chunk_size: int = 10                # 每 10 个动作合并为 1 个 Token
    pred_chunk_size: int = 50
    
    # ==========================
    # 3. DiT Architecture 
    # ==========================
    dit_hidden_size: int = 1024                # DiT 核心维度
    dit_num_heads: int = 8                     # 1024 / 8 = 128 (Head Dim)
    dit_double_layers: int = 4                 # 双流层数
    dit_single_layers: int = 12                # 单流层数
    dit_arm_ffn_mult: float = 4.0              # Arm 专家 FFN 扩展比 (6D rotation + 3D translation)
    dit_grip_ffn_mult: float = 2.0             # Gripper 专家 FFN 扩展比 (binary open/close)
    dit_ctx_ffn_mult: float = 4.0              # Context 共享 FFN 扩展比 (cross-modal interaction)
    
    # ==========================
    # 4. Flow Matching Settings
    # ==========================
    num_inference_steps: int = 10              # 推理去噪步数
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    action_loss_weight: float = 1.0              # 动作空间重构损失权重
    gripper_loss_weight: float = 1.0              # BCE loss weight for gripper dimension
    gripper_dim_indices: tuple = (-1,)            # Gripper dim indices (supports negative, e.g. (-1,) for single-arm, (-1,-11) for dual-arm)
    
    # Training settings
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)
    
    # VLM training settings (for two-stage training)
    train_vlm: bool = False  # Whether to train VLM (default: False, VLM is frozen)
    vlm_lr: float = 1e-6  # Learning rate for VLM when train_vlm=True (lower than DiT lr)
    
    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    # History action configuration
    load_full_history: bool = False  # Whether to load full episode history actions
    max_history_length: int = 100  # Maximum history length for padding/truncation
    history_padding_side: str = "left"  # Padding side: "left" or "right"

    # Image resolution limits for VLM visual token balancing
    max_image_pixels: int = 230400  # Enforces max visual tokens per image (230400 → max_h≈360p for 720p → 220 tokens)
    min_image_pixels: int = 65536   # Enforces min visual tokens per image (65536 → min 64 tokens, 256x256)

    default_image_resolution: tuple[int, int] = (256, 256)  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Resolve gripper_dim_indices to absolute indices
        resolved = []
        for idx in self.gripper_dim_indices:
            if idx < 0:
                resolved.append(self.action_dim + idx)
            else:
                resolved.append(idx)
        self.gripper_dim_indices_abs = tuple(sorted(resolved))
        self.arm_dim = self.action_dim - len(self.gripper_dim_indices_abs)
        self.gripper_dim = len(self.gripper_dim_indices_abs)

        # Validate configuration
        if self.vlm_model_name not in ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B"]:
            raise ValueError(f"Invalid vlm_model_name: {self.vlm_model_name}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.default_image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        """Return delta indices for observation (history states).
        
        For LoLA, we use n_obs_steps history observations.
        """
        return list(range(-self.n_obs_steps + 1, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.pred_chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
