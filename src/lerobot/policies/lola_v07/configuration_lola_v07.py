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

    # ==========================
    # 双 Segment Summary (2026-08-23)
    # ==========================
    # transition / 当前 task 各自独立保留最近 N 帧, 各压缩为 1 个语义 summary token。
    # 与 chunks 模式的区别: history stream 长度恒为 5
    # (hist_start | transition_summary | PTE | task_summary | hist_end)。
    #
    # 字段分为两类 (见方案 §6.4):
    #   - 架构 fingerprint: 改变参数拓扑, 决定 checkpoint 能否加载
    #   - 训练语义: 不改参数 shape 但改行为, 仍参与 resume 匹配
    # 所有 flag 型字段都是"参数恒建、flag 只门控 forward", 因此开关切换不换架构。
    history_architecture_version: int = 0   # 0 = chunks, 1 = segment_summary v1
    history_tokenization_mode: str = "chunks"  # "chunks" | "segment_summary"

    history_summary_num_heads: int = 8
    max_transition_summary_frames: int = 32
    max_task_summary_frames: int = 32

    # content dropout: 只把 states 替换为 null, 不动 mask/length/present/PTE
    transition_summary_drop_rate: float = 0.7
    task_summary_drop_rate: float = 0.7

    # chain-position sampling: 联动截断 completed_tasks 文本与 transition 几何,
    # 使训练分布接近 CALVIN 五任务链的 0..4 前置已完成任务分布
    history_chain_position_mode: str = "none"  # "none" | "uniform_0_4"
    history_chain_max_position: int = 4

    history_summary_last_chunk_residual: bool = True
    history_summary_last_gate_init: float = -4.0   # sigmoid ≈ 0.018

    transition_summary_length_encoding: bool = False
    task_summary_length_encoding: bool = True
    history_summary_total_length_cap: int = 64

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
        self._validate_history_tokenization()

    def _validate_history_tokenization(self):
        """双 Segment Summary 的模式校验 (方案 §6.3)。

        整除校验不可省略: LolaV07StateEncoder._pad_and_chunk() 在 seq 不整除
        chunk_size 时会在【最新帧右侧】补零, 静默污染 last-valid chunk。
        """
        valid_modes = ("chunks", "segment_summary")
        if self.history_tokenization_mode not in valid_modes:
            raise ValueError(
                f"history_tokenization_mode must be one of {valid_modes}, "
                f"got {self.history_tokenization_mode!r}"
            )

        # 模式 <-> version 双向绑定
        expected_version = 1 if self.history_tokenization_mode == "segment_summary" else 0
        if self.history_architecture_version != expected_version:
            raise ValueError(
                f"history_architecture_version must be {expected_version} when "
                f"history_tokenization_mode={self.history_tokenization_mode!r}, "
                f"got {self.history_architecture_version}"
            )

        for name in ("transition_summary_drop_rate", "task_summary_drop_rate"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")

        valid_chain_modes = ("none", "uniform_0_4")
        if self.history_chain_position_mode not in valid_chain_modes:
            raise ValueError(
                f"history_chain_position_mode must be one of {valid_chain_modes}, "
                f"got {self.history_chain_position_mode!r}"
            )

        if self.history_tokenization_mode != "segment_summary":
            return

        required = {
            "history_type": "state",
            "state_encoder_mode": "unified",
            "load_full_history": True,
            "use_special_tokens": True,
            "use_previous_task_end": True,
            "hist_action_token_drop_rate": 0.0,
            "transition_mask_rate": 0.0,
            # CALVIN 五任务链前置已完成任务最多 4 条; 5 条这个 prompt 格式在评测中
            # 不存在。模式级不变量 — 否则 R1a/R1b 对照会被 5 vs 4 污染 (方案 §6.3)。
            "completed_tasks_history_len": 4,
        }
        for name, expected in required.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(
                    f"segment_summary mode requires {name}={expected!r}, got {actual!r}"
                )

        for name in ("max_transition_summary_frames", "max_task_summary_frames"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0 in segment_summary mode, got {value}")
            if value % self.action_chunk_size != 0:
                raise ValueError(
                    f"{name} ({value}) must be a multiple of action_chunk_size "
                    f"({self.action_chunk_size}) — otherwise the state encoder pads zeros "
                    f"to the RIGHT of the newest frame and silently corrupts the last chunk"
                )

        if self.history_summary_num_heads <= 0:
            raise ValueError(
                f"history_summary_num_heads must be > 0, got {self.history_summary_num_heads}"
            )
        if self.dit_hidden_size % self.history_summary_num_heads != 0:
            raise ValueError(
                f"dit_hidden_size ({self.dit_hidden_size}) must be divisible by "
                f"history_summary_num_heads ({self.history_summary_num_heads})"
            )
        if self.history_summary_total_length_cap <= 0:
            raise ValueError(
                f"history_summary_total_length_cap must be > 0, "
                f"got {self.history_summary_total_length_cap}"
            )
        if self.history_chain_position_mode == "uniform_0_4" and self.history_chain_max_position != 4:
            raise ValueError(
                f"history_chain_max_position must be 4 when "
                f"history_chain_position_mode='uniform_0_4', got {self.history_chain_max_position}"
            )

    @property
    def is_segment_summary(self) -> bool:
        return self.history_tokenization_mode == "segment_summary"

    @property
    def summary_padded_frames(self) -> int:
        """两段在模型入口对齐到的公共帧预算 P = max(T, K) (方案 §10.3)。

        不等 budget 时较短的一段左 pad 到 P (内容 null / mask False), 再沿 batch
        合并成一次 state encoder 前向 — 保住 ZeRO-3 的单次模块调用约束。
        """
        return max(self.max_transition_summary_frames, self.max_task_summary_frames)

    @property
    def summary_num_chunks(self) -> int:
        return self.summary_padded_frames // self.action_chunk_size

    @property
    def observation_delta_indices(self) -> list:
        """obs_prev_chunk_frame 开启时观测为 [上一 chunk 起始帧, 当前帧]。"""
        if self.obs_prev_chunk_frame:
            return [-self.action_chunk_size, 0]
        return super().observation_delta_indices
