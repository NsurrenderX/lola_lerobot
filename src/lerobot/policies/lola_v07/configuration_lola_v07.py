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

    # DiT gradient checkpointing (独立于 VLM 的 gradient_checkpointing 开关)。
    # DiT 是 256D bottleneck 的小模型, 激活 ~1GB, GC 的完整重算不划算 — 默认关;
    # VLM (激活大头) 的 GC 仍由 gradient_checkpointing && train_vlm 控制。
    # 显存吃紧时可用 --dit_gradient_checkpointing 重新打开。
    dit_gradient_checkpointing: bool = False

    # Override defaults from LoLAConfig
    action_loss_weight: float = 10.0       # v06 was 1.0

    # ==========================
    # EMA (2026-08-12)
    # ==========================
    # 全模型 EMA (含 VLM): ZeRO-3 分片本地维护, 每 rank 只存自己 shard 的副本。
    # 冻结期的 VLM 权重不变, EMA 恒等于基座; 动态解冻时以当前权重初始化 VLM 分片的
    # EMA 并继续跟踪 (分片布局在 engine 重建前后不变, 见 zero3-dynamic-unfreeze 调研)。
    # 0 = 关闭; 建议 0.999 (~1000 步有效窗口)。
    ema_decay: float = 0.0

    # ==========================
    # 图像增强 (2026-08-12, LoLADataset train-only)
    # ==========================
    # brightness/contrast/saturation 小幅 jitter (不碰 hue, 保住颜色-指令绑定) +
    # mild affine (平移/缩放, reflection 填充保内容完整)。样本内所有相机/所有帧共享
    # 同一组随机参数 (上下文语义一致), 样本间独立采样。全 0/默认 = 关闭。
    image_aug_brightness: float = 0.0        # jitter 幅度, 0.2 → U(0.8, 1.2)
    image_aug_contrast: float = 0.0
    image_aug_saturation: float = 0.0
    image_aug_translate: float = 0.0         # 相对图幅的平移幅度, 如 0.1 = ±10%
    image_aug_scale_min: float = 1.0         # 缩放下界, 如 0.9
    image_aug_scale_max: float = 1.0         # 缩放上届, 如 1.1

    # ==========================
    # Visual token drop (2026-08-12, bridge 输入侧, training-only)
    # ==========================
    # 以概率 p 将 visual token 的 VLM hidden 特征置零 (置零而非 attention 删除,
    # 序列长度/位置结构不变, DiT 侧无联动)。eval 由 self.training 门控自动关闭。
    visual_token_drop_rate: float = 0.0

    # ==========================
    # Chunk 帧观测 (2026-08-12)
    # ==========================
    # 观测 = [上一个 action chunk 起始帧, 当前帧], 提供 chunk 尺度的动作-场景反馈。
    # 推理侧由 policy 缓存上一 chunk 起始帧 (episode 首 chunk 自复制, 与训练侧
    # episode 边界 clamp 语义对齐)。开启后 observation_delta_indices = [-chunk, 0]。
    obs_prev_chunk_frame: bool = False

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
        # 2026-08-12 新增字段校验
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay ({self.ema_decay}) must be in [0, 1)")
        if not (0.0 <= self.visual_token_drop_rate < 1.0):
            raise ValueError(f"visual_token_drop_rate ({self.visual_token_drop_rate}) must be in [0, 1)")
        if self.image_aug_scale_min > self.image_aug_scale_max:
            raise ValueError(
                f"image_aug_scale_min ({self.image_aug_scale_min}) > "
                f"image_aug_scale_max ({self.image_aug_scale_max})"
            )
        for name in ("image_aug_brightness", "image_aug_contrast", "image_aug_saturation", "image_aug_translate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def observation_delta_indices(self) -> list:
        """obs_prev_chunk_frame 开启时观测为 [上一 chunk 起始帧, 当前帧]。"""
        if self.obs_prev_chunk_frame:
            return [-self.action_chunk_size, 0]
        return super().observation_delta_indices
