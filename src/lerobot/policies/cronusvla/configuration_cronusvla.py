from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("cronusvla")
@dataclass
class CronusVLAConfig(PreTrainedConfig):
    # ==========================
    # 1. VLM Settings (loaded via prismatic package from CronusVLA repo)
    # ==========================
    vlm_base: str = "prism-dinosiglip-224px+7b"
    vlm_arch_specifier: str = "fused-gelu-mlp"  # "gelu-mlp", "fused-gelu-mlp", or "linear"
    view_sequence_len: int = 1  # 1=primary only, 2=primary+wrist
    use_wrist_image: bool = False
    local_vlm_path: str | None = None  # Path to local .pt checkpoint; avoids HF Hub download
    hf_token: str | None = None  # HF API token for gated models (e.g., Llama-2-7b-hf)

    # ==========================
    # 4. Image Settings
    # ==========================
    image_size: int = 224

    # ==========================
    # 5. VLM Hidden Size (must match LLM lm_head.in_features)
    # ==========================
    hidden_size: int = 4096  # 4096 for Llama-2-7B, 896 for Qwen2.5-0.5B

    # ==========================
    # 6. DiT Diffusion Decoder Settings
    # ==========================
    action_model_type: str = "DiT-B"  # "DiT-S", "DiT-B", "DiT-L"
    diffusion_steps: int = 100
    noise_schedule: str = "squaredcos_cap_v2"
    class_dropout_prob: float = 0.1
    extend_num: int = 6

    # ==========================
    # 7. Action / Temporal Settings
    # ==========================
    action_dim: int = 7
    future_action_window_size: int = 15
    past_action_window_size: int = 0

    # ==========================
    # 8. Training Settings
    # ==========================
    repeated_diffusion_steps: int = 4
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = False
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    device: str | None = None

    # ==========================
    # 9. Inference Settings
    # ==========================
    cfg_scale: float = 1.5
    use_ddim: bool = False
    num_ddim_steps: int = 5

    # ==========================
    # 10. Normalization
    # ==========================
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    norm_type: str = "BOUNDS_Q99"

    # ==========================
    # 11. Text Tokenizer
    # ==========================
    max_text_len: int = 256

    # ==========================
    # 12. Optimizer Settings
    # ==========================
    optimizer_lr: float = 2e-5
    optimizer_weight_decay: float = 0.0
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0

    # ==========================
    # 13. Scheduler Settings
    # ==========================
    scheduler_type: str = "constant"
    scheduler_warmup_steps: int = 250

    def __post_init__(self):
        super().__post_init__()
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
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
        # Scheduler is constructed directly in train_cronusvla.py
        return None

    @property
    def observation_delta_indices(self) -> list:
        return list(range(-self.past_action_window_size, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.future_action_window_size + 1))

    @property
    def reward_delta_indices(self) -> None:
        return None