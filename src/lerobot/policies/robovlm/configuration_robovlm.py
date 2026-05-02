from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("robovlm")
@dataclass
class RoboVLMConfig(PreTrainedConfig):
    # ==========================
    # 1. VLM Settings
    # ==========================
    vlm_pretrained_path: str = ".vlms/kosmos-2-patch14-224"
    vlm_model_type: str = "AutoModelForVision2Seq"

    # ==========================
    # 2. Image Settings
    # ==========================
    image_size: int = 224
    image_mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple = (0.26862954, 0.26130258, 0.27577711)

    # ==========================
    # 3. VLM Hidden Size
    # ==========================
    hidden_size: int = 1024

    # ==========================
    # 4. Observation / Action
    # ==========================
    window_size: int = 8
    fwd_pred_next_n: int = 10
    action_dim: int = 7
    history_type: str = "post"

    # ==========================
    # 5. State Settings
    # ==========================
    use_state: bool = True
    state_dim: int = 7

    # ==========================
    # 6. LSTM Head Settings
    # ==========================
    lstm_hidden_size: int = 1024
    lstm_num_layers: int = 4
    lstm_dropout_p: float = 0.0
    lstm_down_sample: str = "none"
    lstm_latent: int = 1

    # ==========================
    # 7. Loss Settings
    # ==========================
    arm_gripper_loss_ratio: float = 0.01

    # ==========================
    # 8. Training Setup
    # ==========================
    freeze_backbone: bool = False
    train_vision: bool = True
    train_text_embedding: bool = True
    gradient_checkpointing: bool = False
    dtype: str = "bfloat16"
    device: str | None = None

    # ==========================
    # 9. Normalization
    # ==========================
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ==========================
    # 10. Optimizer Settings
    # ==========================
    optimizer_lr: float = 2e-5
    optimizer_weight_decay: float = 0.0
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0

    # ==========================
    # 11. Scheduler Settings
    # ==========================
    scheduler_warmup_steps: int = 250
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2e-6

    def __post_init__(self):
        super().__post_init__()

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.state_dim,),
            )

        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

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
        return list(range(-self.window_size + 1, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.fwd_pred_next_n))

    @property
    def reward_delta_indices(self) -> None:
        return None