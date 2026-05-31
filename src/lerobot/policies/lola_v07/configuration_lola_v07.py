from dataclasses import dataclass

from lerobot.policies.lola.configuration_lola import LoLAConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("lola_v07")
@dataclass
class LoLAV07Config(LoLAConfig):
    # Bottleneck dimensions
    action_bottleneck_dim: int = 256       # Arm latent dimension for flow matching
    grip_bottleneck_dim: int = 128         # Grip latent dimension for flow matching
    state_bottleneck_dim: int = 256        # StateEncoder unified mode arm bottleneck
    state_grip_bottleneck_dim: int = 128   # StateEncoder unified mode grip bottleneck

    # Training strategy
    encoder_lr_mult: float = 1.5           # Encoder LR multiplier relative to base LR
    warmup_pct: float = 0.1                # Warm-up fraction (10% of total steps)
    warmup_t_trunc_low: float = 0.1        # t-truncation lower bound during warmup
    warmup_t_trunc_high: float = 0.9       # t-truncation upper bound during warmup

    # Override defaults from LoLAConfig
    action_loss_weight: float = 10.0       # v06 was 1.0

    def __post_init__(self):
        super().__post_init__()
        # Validate bottleneck dimensions
        if self.action_bottleneck_dim >= self.dit_hidden_size:
            raise ValueError(
                f"action_bottleneck_dim ({self.action_bottleneck_dim}) must be < "
                f"dit_hidden_size ({self.dit_hidden_size})"
            )
        if self.grip_bottleneck_dim >= self.dit_hidden_size:
            raise ValueError(
                f"grip_bottleneck_dim ({self.grip_bottleneck_dim}) must be < "
                f"dit_hidden_size ({self.dit_hidden_size})"
            )
        if self.state_bottleneck_dim >= self.dit_hidden_size:
            raise ValueError(
                f"state_bottleneck_dim ({self.state_bottleneck_dim}) must be < "
                f"dit_hidden_size ({self.dit_hidden_size})"
            )
        if self.state_grip_bottleneck_dim >= self.dit_hidden_size:
            raise ValueError(
                f"state_grip_bottleneck_dim ({self.state_grip_bottleneck_dim}) must be < "
                f"dit_hidden_size ({self.dit_hidden_size})"
            )
        # VLM dynamic unfreezing validation
        if self.vlm_lr_mult <= 0:
            raise ValueError(f"vlm_lr_mult ({self.vlm_lr_mult}) must be > 0")
        if not self.train_vlm and self.vlm_unfreeze_v_loss_threshold > 0:
            import warnings
            warnings.warn(
                f"vlm_unfreeze_v_loss_threshold={self.vlm_unfreeze_v_loss_threshold} is set but "
                f"train_vlm=False. The threshold logic will be ignored — VLM will remain frozen. "
                f"Set --train_vlm to enable dynamic VLM unfreezing.",
                UserWarning,
                stacklevel=2,
            )
