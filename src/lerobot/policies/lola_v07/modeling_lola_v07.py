"""LoLA v07: Latent Flow Matching with Bottleneck Architecture.

Core changes from v06:
- ActionEncoder/StateEncoder have explicit Bottleneck (256D arm / 128D grip)
- Flow Matching operates in Bottleneck latent space, not 1024D DiT space
- Noise generated in latent space, decoded via ActionEncoder decoder (shared)
- v-loss computed in latent space via dit_to_latent projections (pseudoinverse-init)
- DiT continues operating in 1024D, unmodified
- FP32 isolation for encoders, BF16 for DiT
"""

import math
import logging
from collections import deque
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config
from lerobot.policies.lola.modeling_lola import (
    LolaVLMContextBridge,
    build_vlm_bridge,
    LoLADiT,
    LoLAPolicy,
)
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action Encoder with Bottleneck
# ---------------------------------------------------------------------------

class LolaV07ActionEncoder(nn.Module):
    """Dual-Token Action Chunking with explicit Bottleneck.

    Architecture: enc1 -> enc2 (bottleneck) -> dec (pure linear)
    - encode(): returns heterogeneous latents (arm [B,N,256], grip [B,N,128])
    - decode(): reconstructs 1024D tokens with optional modality embeddings
    - forward(): encode + decode, optionally returning latents
    """

    def __init__(self, config: LoLAV07Config):
        super().__init__()
        self.chunk_size = config.action_chunk_size
        self.arm_dim = config.arm_dim
        self.gripper_dim = config.gripper_dim
        self.config = config
        hidden = config.dit_hidden_size
        arm_btl = config.action_bottleneck_dim
        grip_btl = config.grip_bottleneck_dim

        # Arm: chunk -> hidden -> bottleneck -> hidden (decoder)
        self.arm_enc1 = nn.Sequential(
            nn.Linear(self.chunk_size * self.arm_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-6),
            nn.SiLU(),
        )
        self.arm_enc2 = nn.Sequential(
            nn.Linear(hidden, arm_btl),
            nn.LayerNorm(arm_btl, eps=1e-6),
            nn.SiLU(),
        )
        self.arm_dec = nn.Linear(arm_btl, hidden)

        # Gripper: chunk -> hidden -> bottleneck -> hidden (decoder)
        self.grip_enc1 = nn.Sequential(
            nn.Linear(self.chunk_size * self.gripper_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-6),
            nn.SiLU(),
        )
        self.grip_enc2 = nn.Sequential(
            nn.Linear(hidden, grip_btl),
            nn.LayerNorm(grip_btl, eps=1e-6),
            nn.SiLU(),
        )
        self.grip_dec = nn.Linear(grip_btl, hidden)

        # Orthogonal init + norm conservation for decoders
        nn.init.orthogonal_(self.arm_dec.weight)
        self.arm_dec.weight.data *= math.sqrt(arm_btl / hidden)
        nn.init.zeros_(self.arm_dec.bias)

        nn.init.orthogonal_(self.grip_dec.weight)
        self.grip_dec.weight.data *= math.sqrt(grip_btl / hidden)
        nn.init.zeros_(self.grip_dec.bias)

        # Modality embeddings (applied in decode, not encode)
        self.arm_modality_emb = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.gripper_modality_emb = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

    def _pad_and_chunk(self, actions: torch.Tensor, dim_size: int) -> torch.Tensor:
        b, seq_len, d = actions.shape
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            actions = F.pad(actions, (0, 0, 0, pad_len))
            seq_len += pad_len
        return actions.view(b, seq_len // self.chunk_size, self.chunk_size * dim_size)

    def encode(self, arm_chunked: torch.Tensor, grip_chunked: torch.Tensor):
        """Return heterogeneous latents.

        Args:
            arm_chunked: [B, num_chunks, chunk_size * arm_dim]
            grip_chunked: [B, num_chunks, chunk_size * grip_dim]

        Returns:
            arm_latent: [B, num_chunks, arm_btl]
            grip_latent: [B, num_chunks, grip_btl]
        """
        arm_latent = self.arm_enc2(self.arm_enc1(arm_chunked))
        grip_latent = self.grip_enc2(self.grip_enc1(grip_chunked))
        return arm_latent, grip_latent

    def decode(self, arm_latent: torch.Tensor, grip_latent: torch.Tensor,
               add_modality_emb: bool = True) -> torch.Tensor:
        """Reconstruct 1024D tokens from heterogeneous latents.

        Args:
            arm_latent: [B, num_chunks, arm_btl]
            grip_latent: [B, num_chunks, grip_btl]
            add_modality_emb: if True, add modality embeddings (for final output);
                              if False, skip (DiT adds its own modality_emb internally)

        Returns:
            tokens: [B, 2*num_chunks, dit_hidden_size]
        """
        arm_tokens = self.arm_dec(arm_latent)
        grip_tokens = self.grip_dec(grip_latent)
        if add_modality_emb:
            arm_tokens = arm_tokens + self.arm_modality_emb
            grip_tokens = grip_tokens + self.gripper_modality_emb
        return torch.cat([arm_tokens, grip_tokens], dim=1)

    def forward(self, actions: torch.Tensor, return_latent: bool = False):
        gripper_indices = list(self.config.gripper_dim_indices_abs)
        all_indices = list(range(actions.shape[-1]))
        arm_indices = [i for i in all_indices if i not in gripper_indices]

        arm_actions = actions[..., arm_indices]
        gripper_actions = actions[..., gripper_indices]

        arm_chunked = self._pad_and_chunk(arm_actions, self.arm_dim)
        grip_chunked = self._pad_and_chunk(gripper_actions, self.gripper_dim)

        arm_latent, grip_latent = self.encode(arm_chunked, grip_chunked)
        tokens = self.decode(arm_latent, grip_latent)

        if return_latent:
            return tokens, arm_latent, grip_latent
        return tokens


# ---------------------------------------------------------------------------
# State Encoder with Bottleneck
# ---------------------------------------------------------------------------

class LolaV07StateEncoder(nn.Module):
    """State History Encoder with Bottleneck, compatible with 5-stream DiT.

    Supports two modes:
    - "unified": overcomplete extraction -> split -> per-branch bottleneck
    - "separated": fully independent arm/grip pipelines with bottleneck
    """

    def __init__(self, config: LoLAV07Config):
        super().__init__()
        self.chunk_size = config.action_chunk_size
        self.state_dim = config.state_dim
        self.config = config
        self.mode = config.state_encoder_mode
        hidden = config.dit_hidden_size
        arm_btl = config.state_bottleneck_dim
        grip_btl = config.state_grip_bottleneck_dim

        if self.mode == "unified":
            # Stage 1: overcomplete extraction (no bottleneck)
            self.state_proj = nn.Sequential(
                nn.Linear(self.chunk_size * self.state_dim, 2 * hidden),
                nn.LayerNorm(2 * hidden, eps=1e-6),
                nn.SiLU(),
                nn.Linear(2 * hidden, 2 * hidden),
                nn.LayerNorm(2 * hidden, eps=1e-6),
            )
            # Stage 2: per-branch bottleneck after split
            self.arm_bottleneck = nn.Sequential(
                nn.Linear(hidden, arm_btl),
                nn.LayerNorm(arm_btl, eps=1e-6),
                nn.SiLU(),
                nn.Linear(arm_btl, hidden),  # pure linear decoder
            )
            self.grip_bottleneck = nn.Sequential(
                nn.Linear(hidden, grip_btl),
                nn.LayerNorm(grip_btl, eps=1e-6),
                nn.SiLU(),
                nn.Linear(grip_btl, hidden),  # pure linear decoder
            )
            # Orthogonal init for bottleneck decoders
            nn.init.orthogonal_(self.arm_bottleneck[-1].weight)
            self.arm_bottleneck[-1].weight.data *= math.sqrt(arm_btl / hidden)
            nn.init.zeros_(self.arm_bottleneck[-1].bias)

            nn.init.orthogonal_(self.grip_bottleneck[-1].weight)
            self.grip_bottleneck[-1].weight.data *= math.sqrt(grip_btl / hidden)
            nn.init.zeros_(self.grip_bottleneck[-1].bias)

        else:  # "separated"
            num_gripper = len(config.gripper_dim_indices_abs)
            self.state_gripper_indices = tuple(range(self.state_dim - num_gripper, self.state_dim))
            self.state_arm_dim = self.state_dim - num_gripper
            self.state_gripper_dim = num_gripper

            # Arm: chunk -> hidden -> bottleneck -> hidden (decoder)
            self.arm_state_enc1 = nn.Sequential(
                nn.Linear(self.chunk_size * self.state_arm_dim, hidden),
                nn.LayerNorm(hidden, eps=1e-6),
                nn.SiLU(),
            )
            self.arm_state_enc2 = nn.Sequential(
                nn.Linear(hidden, arm_btl),
                nn.LayerNorm(arm_btl, eps=1e-6),
                nn.SiLU(),
            )
            self.arm_state_dec = nn.Linear(arm_btl, hidden)

            # Grip: chunk -> hidden -> bottleneck -> hidden (decoder)
            self.grip_state_enc1 = nn.Sequential(
                nn.Linear(self.chunk_size * self.state_gripper_dim, hidden),
                nn.LayerNorm(hidden, eps=1e-6),
                nn.SiLU(),
            )
            self.grip_state_enc2 = nn.Sequential(
                nn.Linear(hidden, grip_btl),
                nn.LayerNorm(grip_btl, eps=1e-6),
                nn.SiLU(),
            )
            self.grip_state_dec = nn.Linear(grip_btl, hidden)

            # Orthogonal init for decoders
            nn.init.orthogonal_(self.arm_state_dec.weight)
            self.arm_state_dec.weight.data *= math.sqrt(arm_btl / hidden)
            nn.init.zeros_(self.arm_state_dec.bias)

            nn.init.orthogonal_(self.grip_state_dec.weight)
            self.grip_state_dec.weight.data *= math.sqrt(grip_btl / hidden)
            nn.init.zeros_(self.grip_state_dec.bias)

        self.arm_ctx_state_emb = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.grip_ctx_state_emb = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

    def _pad_and_chunk(self, states: torch.Tensor, dim_size: int) -> torch.Tensor:
        b, seq_len, d = states.shape
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            states = F.pad(states, (0, 0, 0, pad_len))
            seq_len += pad_len
        return states.view(b, seq_len // self.chunk_size, self.chunk_size * dim_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.mode == "unified":
            state_chunked = self._pad_and_chunk(states, self.state_dim)
            projected = self.state_proj(state_chunked)  # [B, num_chunks, 2 * hidden]
            hidden = self.config.dit_hidden_size
            arm_part = projected[..., :hidden]
            grip_part = projected[..., hidden:]
            arm_tokens = self.arm_bottleneck(arm_part) + self.arm_ctx_state_emb
            grip_tokens = self.grip_bottleneck(grip_part) + self.grip_ctx_state_emb
        else:  # "separated"
            gripper_indices = list(self.state_gripper_indices)
            all_indices = list(range(states.shape[-1]))
            arm_indices = [i for i in all_indices if i not in gripper_indices]

            arm_states = states[..., arm_indices]
            grip_states = states[..., gripper_indices]

            arm_chunked = self._pad_and_chunk(arm_states, self.state_arm_dim)
            grip_chunked = self._pad_and_chunk(grip_states, self.state_gripper_dim)

            arm_latent = self.arm_state_enc2(self.arm_state_enc1(arm_chunked))
            arm_tokens = self.arm_state_dec(arm_latent) + self.arm_ctx_state_emb

            grip_latent = self.grip_state_enc2(self.grip_state_enc1(grip_chunked))
            grip_tokens = self.grip_state_dec(grip_latent) + self.grip_ctx_state_emb

        return torch.cat([arm_tokens, grip_tokens], dim=1)  # [B, 2*num_chunks, hidden]


# ---------------------------------------------------------------------------
# Core Model: LoLAV07Pytorch
# ---------------------------------------------------------------------------

class LoLAV07Pytorch(nn.Module):
    """LoLA v07 core model with Latent Flow Matching.

    Key difference from v06: Flow Matching operates in Bottleneck latent space
    (arm 256D / grip 128D), not in 1024D DiT hidden space. The DiT continues
    operating in 1024D for 5-stream attention compatibility.

    Latent <-> DiT bridging:
    - Latent -> DiT: reuses ActionEncoder.arm_dec / grip_dec (shared params, no modality_emb)
    - DiT -> Latent: arm_dit_to_latent / grip_dit_to_latent (separate params, pseudoinverse init)
    """

    def __init__(self, config: LoLAV07Config):
        super().__init__()
        self.config = config
        hidden = config.dit_hidden_size
        arm_btl = config.action_bottleneck_dim
        grip_btl = config.grip_bottleneck_dim

        # VLM Bridge and DiT (reused from lola)
        self.vlm_bridge = build_vlm_bridge(config)

        # Encoders with Bottleneck (FP32)
        self.action_encoder = LolaV07ActionEncoder(config).float()
        self.state_encoder = LolaV07StateEncoder(config).float() if config.history_type == "state" else None

        # DiT (BF16, unmodified)
        self.dit = LoLADiT(config)

        # dit_to_latent projections (trainable, pseudoinverse-initialized)
        def _init_dit_to_latent(dec_linear, name):
            W = dec_linear.weight.detach().float()
            with torch.no_grad():
                S = torch.linalg.svdvals(W)
                cond = S[0] / S[-1]
                if cond > 1e4:
                    logger.warning(
                        f"{name} condition number {cond:.2e}, pseudoinverse may be unstable"
                    )
            return torch.linalg.pinv(W, rcond=1e-3)

        arm_pinv = _init_dit_to_latent(self.action_encoder.arm_dec, "arm_dec")
        self.arm_dit_to_latent = nn.Linear(hidden, arm_btl)
        self.arm_dit_to_latent.weight.data.copy_(arm_pinv)
        self.arm_dit_to_latent.bias.data.zero_()

        grip_pinv = _init_dit_to_latent(self.action_encoder.grip_dec, "grip_dec")
        self.grip_dit_to_latent = nn.Linear(hidden, grip_btl)
        self.grip_dit_to_latent.weight.data.copy_(grip_pinv)
        self.grip_dit_to_latent.bias.data.zero_()

        # Gradient checkpointing
        self.gradient_checkpointing_enabled = False
        self._checkpoint_fn = torch.utils.checkpoint.checkpoint
        self._checkpoint_fn_kwargs = {"use_reentrant": False, "preserve_rng_state": False}

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        logger.info("Enabled gradient checkpointing for LoLAV07Pytorch model")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        logger.info("Disabled gradient checkpointing for LoLAV07Pytorch model")

    def set_deepspeed_checkpointing(self):
        import deepspeed
        self._checkpoint_fn = deepspeed.checkpointing.non_reentrant_checkpoint
        self._checkpoint_fn_kwargs = {}
        self.dit._checkpoint_fn = self._checkpoint_fn
        self.dit._checkpoint_fn_kwargs = self._checkpoint_fn_kwargs

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return self._checkpoint_fn(func, *args, **kwargs, **self._checkpoint_fn_kwargs)
        return func(*args, **kwargs)

    def _run_vlm_bridge(self, hidden_states_all_layers, vlm_attention_mask=None):
        """统一的桥接器调用入口: transformer 模式透传 attention_mask, legacy 模式忽略."""
        if isinstance(self.vlm_bridge, LolaVLMContextBridge):
            return self.vlm_bridge(hidden_states_all_layers, attention_mask=vlm_attention_mask)
        return self.vlm_bridge(hidden_states_all_layers)

    def forward(self, hidden_states_all_layers, input_ids, hist_actions, target_actions,
                hist_actions_mask=None, vlm_attention_mask=None, time=None, noise=None, state_emb=None,
                n_transition_chunks=0, n_transition_chunks_batch=None, return_per_sample=False):
        """Training forward with Latent Flow Matching.

        v-loss is computed in Bottleneck latent space (256D/128D), not 1024D.

        Args:
            n_transition_chunks: number of transition chunks in history (for previous_task_end placement)
            n_transition_chunks_batch: per-sample [B] tensor for attention mask construction
            return_per_sample: 额外返回未聚合的逐样本 loss 张量 ([B], keys 以 "_per_sample" 结尾),
                供 t-网格探针在 CRN 固定噪声下记录逐样本逐 t 曲线
        """
        b = target_actions.shape[0]
        device = target_actions.device

        # 1. VLM features (BF16, unchanged)
        vlm_features, empty_emb = self._run_vlm_bridge(hidden_states_all_layers, vlm_attention_mask)
        target_dtype = vlm_features.dtype

        # 2. History encoding (force FP32 to survive DeepSpeed BF16 autocast)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
            if self.state_encoder is not None:
                hist_chunks = self.state_encoder(hist_actions)
            else:
                hist_chunks = self.action_encoder(hist_actions)
        hist_chunks = hist_chunks.to(target_dtype)

        # 3. Target action encoding to latent (force FP32, output to BF16)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
            _, target_arm_latent, target_grip_latent = self.action_encoder(
                target_actions, return_latent=True
            )
        target_arm_latent = target_arm_latent.to(target_dtype)
        target_grip_latent = target_grip_latent.to(target_dtype)

        # 4. Hist actions mask processing (same as v06)
        hist_chunks_mask = None
        if hist_actions_mask is not None:
            chunk_size = self.config.action_chunk_size
            seq_len = hist_actions_mask.shape[1]
            remainder = seq_len % chunk_size
            if remainder != 0:
                pad_len = chunk_size - remainder
                hist_actions_mask = F.pad(hist_actions_mask, (0, pad_len), value=0)
                seq_len = hist_actions_mask.shape[1]
            num_chunks = seq_len // chunk_size
            hist_actions_mask_reshaped = hist_actions_mask.view(b, num_chunks, chunk_size)
            hist_chunks_mask = hist_actions_mask_reshaped.any(dim=2).float()
            hist_chunks_mask = torch.cat([hist_chunks_mask, hist_chunks_mask], dim=1)

        # 5. Flow Matching in latent space (BF16)
        if noise is None:
            noise_arm_latent = torch.randn_like(target_arm_latent)
            noise_grip_latent = torch.randn_like(target_grip_latent)
        else:
            noise_arm_latent, noise_grip_latent = noise
            noise_arm_latent = noise_arm_latent.to(target_dtype)
            noise_grip_latent = noise_grip_latent.to(target_dtype)

        if time is None:
            dist = torch.distributions.Beta(
                self.config.time_sampling_beta_alpha,
                self.config.time_sampling_beta_beta,
            )
            time = dist.sample((b,)).to(device)

        time = time.to(target_dtype)
        t_expand = time[:, None, None]

        z_t_arm = (1 - t_expand) * target_arm_latent + t_expand * noise_arm_latent
        z_t_grip = (1 - t_expand) * target_grip_latent + t_expand * noise_grip_latent

        u_t_arm = noise_arm_latent - target_arm_latent
        u_t_grip = noise_grip_latent - target_grip_latent

        # 6. Latent -> DiT (FP32 for decoder precision, output to BF16)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
            z_t_arm_dit = self.action_encoder.arm_dec(z_t_arm)
            z_t_grip_dit = self.action_encoder.grip_dec(z_t_grip)
        z_t_dit = torch.cat([z_t_arm_dit.to(target_dtype), z_t_grip_dit.to(target_dtype)], dim=1)

        # 7. DiT forward with optional special tokens
        use_st = self.config.use_special_tokens

        # ── Split hist_chunks into arm/grip halves ──
        num_hist_chunks = hist_chunks.shape[1] // 2
        arm_hist = hist_chunks[:, :num_hist_chunks, :]
        grip_hist = hist_chunks[:, num_hist_chunks:, :]

        # ── Special token concatenation (all done here, DiT receives pre-assembled sequences) ──
        # Also build corresponding attention masks for each stream

        # VLM attention mask preprocessing
        if vlm_attention_mask is not None:
            vlm_mask = vlm_attention_mask[:, :-1].bool()  # exclude empty_token
        else:
            vlm_mask = torch.ones(b, vlm_features.shape[1], dtype=torch.bool, device=device)

        # Hist mask preprocessing (chunk-level, per-sample)
        arm_hist_mask_base = None
        grip_hist_mask_base = None
        if hist_chunks_mask is not None:
            hist_mask_bool = hist_chunks_mask.bool()
            arm_hist_mask_base = hist_mask_bool[:, :num_hist_chunks]
            grip_hist_mask_base = hist_mask_bool[:, num_hist_chunks:]
        else:
            arm_hist_mask_base = torch.ones(b, num_hist_chunks, dtype=torch.bool, device=device)
            grip_hist_mask_base = torch.ones(b, num_hist_chunks, dtype=torch.bool, device=device)

        # Build the final sequences and masks
        if use_st:
            # ── ctx_vlm stream: vlm_start + vlm_features + vlm_end ──
            vlm_stream = torch.cat([
                self.dit.vlm_start_emb.expand(b, -1, -1),
                vlm_features,
                self.dit.vlm_end_emb.expand(b, -1, -1),
            ], dim=1)
            vlm_stream_mask = torch.cat([
                torch.ones(b, 1, dtype=torch.bool, device=device),  # vlm_start
                vlm_mask,
                torch.ones(b, 1, dtype=torch.bool, device=device),  # vlm_end
            ], dim=1)

            # ── ctx_arm/ctx_grip streams: hist_start + [transition_chunks + previous_task_end + task_chunks] + hist_end ──
            max_n_tc = n_transition_chunks  # already batch-max from Policy layer
            n_tc_batch = n_transition_chunks_batch  # [B] tensor for per-sample mask

            # 方案B 旋钮: 禁用 transition/task 拆分与 previous_task_end,
            # 历史走连续序列 (hist_start + all chunks + hist_end), 边界语义交给文本
            if not getattr(self.config, "use_previous_task_end", True):
                max_n_tc = 0

            # 提升 expand 到单次调用: 同一 Parameter 在计算图中只被消费一次
            # (ZeRO-3 resume 后, 双重 AccumulateGrad 会命中 grad 分区视图的 shape 失配)
            hist_start = self.dit.hist_start_emb.expand(b, -1, -1)
            prev_task_end = self.dit.previous_task_end_emb.expand(b, -1, -1)
            hist_end = self.dit.hist_end_emb.expand(b, -1, -1)

            if max_n_tc > 0:
                # Split into transition and task chunks
                arm_transition = arm_hist[:, :max_n_tc, :]
                arm_task = arm_hist[:, max_n_tc:, :]
                grip_transition = grip_hist[:, :max_n_tc, :]
                grip_task = grip_hist[:, max_n_tc:, :]

                # Per-sample transition validity 不再使用独立的 tc_valid 层:
                # 历史数据在 transition 槽位内是右对齐的 (dataset + collate 均左 pad),
                # 而 tc_valid = arange < n_tc 标记的是左端 (pad 区), 方向相反 —
                # 非 batch-max 样本的 transition 上下文会被整体误杀。
                # 真实 chunk mask 随数据走、无方向问题, 且已编码 pad 与
                # transition_mask_rate 的 drop, 直接作为 transition 段 mask。

                # previous_task_end validity: per-sample, only if n_tc > 0 for that sample
                pte_valid = (n_tc_batch > 0)  # [B]

                # Arm stream assembly
                arm_stream = torch.cat([
                    hist_start,
                    arm_transition,
                    prev_task_end,
                    arm_task,
                    hist_end,
                ], dim=1)
                # transition 段 mask 直接用真实 chunk mask (随数据右对齐, 含 pad/drop 信息)
                arm_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_start
                    arm_hist_mask_base[:, :max_n_tc],                   # transition chunks (per-sample)
                    pte_valid.unsqueeze(1),                            # previous_task_end
                    arm_hist_mask_base[:, max_n_tc:],                 # task chunks
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_end
                ], dim=1)

                # Grip stream assembly (symmetric)
                grip_stream = torch.cat([
                    hist_start,
                    grip_transition,
                    prev_task_end,
                    grip_task,
                    hist_end,
                ], dim=1)
                grip_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_start
                    grip_hist_mask_base[:, :max_n_tc],                  # transition chunks (per-sample)
                    pte_valid.unsqueeze(1),                            # previous_task_end
                    grip_hist_mask_base[:, max_n_tc:],                 # task chunks
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_end
                ], dim=1)
            else:
                # No transition data: hist_start + all chunks + hist_end (no previous_task_end)
                arm_stream = torch.cat([
                    hist_start,
                    arm_hist,
                    hist_end,
                ], dim=1)
                arm_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_start
                    arm_hist_mask_base,                                  # all chunks
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_end
                ], dim=1)

                grip_stream = torch.cat([
                    hist_start,
                    grip_hist,
                    hist_end,
                ], dim=1)
                grip_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_start
                    grip_hist_mask_base,                                  # all chunks
                    torch.ones(b, 1, dtype=torch.bool, device=device),  # hist_end
                ], dim=1)

            # Reassemble hist_actions for DiT: [arm_stream | grip_stream] in the standard order
            # DiT expects: hist_actions = [arm_half, grip_half] concatenated along dim=1
            hist_actions_for_dit = torch.cat([arm_stream, grip_stream], dim=1)
            # Build full mask in DiT's expected order: [vlm, grip_hist, arm_hist, grip_target, arm_target]
            # (matching LoLADiT.forward's attention mask concatenation order)
            grip_target_mask = torch.ones(b, z_t_dit.shape[1] // 2, dtype=torch.bool, device=device)
            arm_target_mask = torch.ones(b, z_t_dit.shape[1] // 2, dtype=torch.bool, device=device)
            full_mask = torch.cat([vlm_stream_mask, grip_stream_mask, arm_stream_mask,
                                   grip_target_mask, arm_target_mask], dim=1)

            pred_z0_dit = self.dit(
                z_t_dit, hist_actions_for_dit, vlm_stream, empty_emb, time,
                state_emb=state_emb,
                hist_actions_mask=None,  # mask is passed via joint_attention_kwargs
                vlm_attention_mask=None,
                return_chunks=True,
                use_gradient_checkpointing=self.gradient_checkpointing_enabled and self.training,
                joint_attention_kwargs={'attention_mask': full_mask},
            )
        else:
            # No special tokens: use standard DiT forward (unchanged)
            pred_z0_dit = self.dit(
                z_t_dit, hist_chunks, vlm_features, empty_emb, time,
                state_emb=state_emb,
                hist_actions_mask=hist_chunks_mask,
                vlm_attention_mask=vlm_attention_mask,
                return_chunks=True,
                use_gradient_checkpointing=self.gradient_checkpointing_enabled and self.training,
            )

        # 8. DiT -> Latent (FP32 for dit_to_latent precision, output to BF16)
        num_chunks = pred_z0_dit.shape[1] // 2
        pred_z0_arm_dit = pred_z0_dit[:, :num_chunks, :]
        pred_z0_grip_dit = pred_z0_dit[:, num_chunks:, :]

        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
            pred_z0_arm_latent = self.arm_dit_to_latent(pred_z0_arm_dit)
            pred_z0_grip_latent = self.grip_dit_to_latent(pred_z0_grip_dit)
        pred_z0_arm_latent = pred_z0_arm_latent.to(target_dtype)
        pred_z0_grip_latent = pred_z0_grip_latent.to(target_dtype)

        # 9. v-loss in latent space (BF16)
        t_expand_clamped = t_expand.clamp(min=1e-5)
        v_pred_arm = (z_t_arm - pred_z0_arm_latent) / t_expand_clamped
        v_pred_grip = (z_t_grip - pred_z0_grip_latent) / t_expand_clamped

        v_loss_arm = F.mse_loss(v_pred_arm, u_t_arm, reduction="none")
        v_loss_grip = F.mse_loss(v_pred_grip, u_t_grip, reduction="none")
        v_loss = (v_loss_arm.mean() + v_loss_grip.mean()) / 2.0

        # 10. Decode latent predictions to 1024D tokens for action-space losses
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
            pred_z0_tokens = self.action_encoder.decode(pred_z0_arm_latent, pred_z0_grip_latent)
        pred_z0_tokens = pred_z0_tokens.to(target_dtype)

        # 11. Action-space losses (unchanged from v06)
        num_target_chunks = pred_z0_tokens.shape[1] // 2
        pred_x0_arm = pred_z0_tokens[:, :num_target_chunks, :]
        pred_x0_grip = pred_z0_tokens[:, num_target_chunks:, :]

        pred_arm = self.dit.arm_out_proj(pred_x0_arm)
        pred_arm = pred_arm.view(b, num_target_chunks * self.config.action_chunk_size, self.config.arm_dim)

        pred_gripper_logits = self.dit.gripper_out_proj(pred_x0_grip)
        pred_gripper_logits = pred_gripper_logits.view(
            b, num_target_chunks * self.config.action_chunk_size, self.config.gripper_dim
        )

        # Match target lengths
        min_len = min(target_actions.shape[1], pred_arm.shape[1])
        arm_indices = [i for i in range(self.config.action_dim) if i not in list(self.config.gripper_dim_indices_abs)]
        gripper_indices = list(self.config.gripper_dim_indices_abs)

        target_arm = target_actions[:, :min_len, arm_indices].to(target_dtype)
        target_gripper = target_actions[:, :min_len, gripper_indices].to(target_dtype)
        pred_arm_matched = pred_arm[:, :min_len, :]
        pred_gripper_logits_matched = pred_gripper_logits[:, :min_len, :]

        arm_loss = F.huber_loss(pred_arm_matched, target_arm, reduction="none")
        arm_loss_mean = arm_loss.mean()
        arm_loss_per_dim = arm_loss.mean(dim=(0, 1))

        target_gripper_01 = (target_gripper > 0).to(target_dtype)
        gripper_loss = F.binary_cross_entropy_with_logits(pred_gripper_logits_matched, target_gripper_01)

        # 12. Total loss
        action_loss_weight = getattr(self.config, 'action_loss_weight', 10.0)
        gripper_loss_weight = getattr(self.config, 'gripper_loss_weight', 1.0)
        total_loss = v_loss + action_loss_weight * arm_loss_mean + gripper_loss_weight * gripper_loss
        total_loss = total_loss.to(target_dtype)

        result = {
            "total_loss": total_loss,
            "v_loss": v_loss.float(),
            "arm_loss": arm_loss_mean.float(),
            "gripper_loss": gripper_loss.float(),
            "arm_loss_per_dim": arm_loss_per_dim.float(),
        }

        # 13. Per-sample losses (t-网格探针用: CRN 固定噪声下逐样本逐 t 记录)
        # 与 step 9/11/12 相同的公式, 仅去掉 batch 维度的聚合
        if return_per_sample:
            v_loss_ps = (v_loss_arm.mean(dim=(1, 2)) + v_loss_grip.mean(dim=(1, 2))) / 2.0
            arm_huber_ps = arm_loss.mean(dim=(1, 2))
            arm_mse_ps = F.mse_loss(pred_arm_matched, target_arm, reduction="none").mean(dim=(1, 2))
            total_ps = v_loss_ps + action_loss_weight * arm_huber_ps
            result["v_loss_per_sample"] = v_loss_ps.float()
            result["arm_huber_per_sample"] = arm_huber_ps.float()
            result["arm_mse_per_sample"] = arm_mse_ps.float()
            if pred_gripper_logits_matched.shape[-1] > 0:
                grip_bce_ps = F.binary_cross_entropy_with_logits(
                    pred_gripper_logits_matched, target_gripper_01, reduction="none",
                ).mean(dim=(1, 2))
                grip_acc_ps = (
                    (pred_gripper_logits_matched > 0) == (target_gripper_01 > 0.5)
                ).float().mean(dim=(1, 2))
                total_ps = total_ps + gripper_loss_weight * grip_bce_ps
            else:
                grip_bce_ps = torch.zeros(b, device=device)
                grip_acc_ps = torch.zeros(b, device=device)
            result["gripper_bce_per_sample"] = grip_bce_ps.float()
            result["gripper_acc_per_sample"] = grip_acc_ps.float()
            result["total_loss_per_sample"] = total_ps.float()

        return result

    @torch.no_grad()
    def sample_actions(self, hidden_states_all_layers, hist_actions, hist_actions_mask=None,
                       state_emb=None,
                       n_transition_chunks=None):
        """Inference: Euler integration in latent space.

        Args:
            n_transition_chunks: number of transition chunks (None = auto-detect 0)
        """
        b = hist_actions.shape[0]
        device = hist_actions.device

        vlm_features, empty_emb = self._run_vlm_bridge(hidden_states_all_layers)
        target_dtype = vlm_features.dtype

        # History encoding (force FP32 to survive DeepSpeed BF16 autocast)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
            if self.state_encoder is not None:
                hist_chunks = self.state_encoder(hist_actions)
            else:
                hist_chunks = self.action_encoder(hist_actions)
        hist_chunks = hist_chunks.to(target_dtype)

        # Hist mask processing
        hist_chunks_mask = None
        if hist_actions_mask is not None:
            chunk_size = self.config.action_chunk_size
            seq_len = hist_actions_mask.shape[1]
            remainder = seq_len % chunk_size
            if remainder != 0:
                pad_len = chunk_size - remainder
                hist_actions_mask = F.pad(hist_actions_mask, (0, pad_len), value=0)
                seq_len = hist_actions_mask.shape[1]
            num_chunks = seq_len // chunk_size
            hist_actions_mask_reshaped = hist_actions_mask.view(b, num_chunks, chunk_size)
            hist_chunks_mask = hist_actions_mask_reshaped.any(dim=2).float()
            hist_chunks_mask = torch.cat([hist_chunks_mask, hist_chunks_mask], dim=1)

        # Handle n_transition_chunks default
        if n_transition_chunks is None:
            n_transition_chunks = 0
        # 方案B 旋钮: 推理同样禁用 transition/task 拆分
        if not getattr(self.config, "use_previous_task_end", True):
            n_transition_chunks = 0

        # ── Special token handling for inference ──
        use_st = self.config.use_special_tokens

        if use_st:
            num_hist_chunks = hist_chunks.shape[1] // 2
            arm_hist = hist_chunks[:, :num_hist_chunks, :]
            grip_hist = hist_chunks[:, num_hist_chunks:, :]

            # VLM mask
            vlm_mask = torch.ones(b, vlm_features.shape[1], dtype=torch.bool, device=device)

            # Hist mask (chunk-level)
            if hist_chunks_mask is not None:
                hist_mask_bool = hist_chunks_mask.bool()
                arm_hist_mask_base = hist_mask_bool[:, :num_hist_chunks]
                grip_hist_mask_base = hist_mask_bool[:, num_hist_chunks:]
            else:
                arm_hist_mask_base = torch.ones(b, num_hist_chunks, dtype=torch.bool, device=device)
                grip_hist_mask_base = torch.ones(b, num_hist_chunks, dtype=torch.bool, device=device)

            # VLM stream: vlm_start + vlm_features + vlm_end
            vlm_stream = torch.cat([
                self.dit.vlm_start_emb.expand(b, -1, -1),
                vlm_features,
                self.dit.vlm_end_emb.expand(b, -1, -1),
            ], dim=1)
            vlm_stream_mask = torch.cat([
                torch.ones(b, 1, dtype=torch.bool, device=device),
                vlm_mask,
                torch.ones(b, 1, dtype=torch.bool, device=device),
            ], dim=1)

            # Hist streams with special tokens
            if n_transition_chunks > 0:
                arm_stream = torch.cat([
                    self.dit.hist_start_emb.expand(b, -1, -1),
                    arm_hist[:, :n_transition_chunks, :],
                    self.dit.previous_task_end_emb.expand(b, -1, -1),
                    arm_hist[:, n_transition_chunks:, :],
                    self.dit.hist_end_emb.expand(b, -1, -1),
                ], dim=1)
                arm_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    arm_hist_mask_base[:, :n_transition_chunks],
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    arm_hist_mask_base[:, n_transition_chunks:],
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                ], dim=1)
                grip_stream = torch.cat([
                    self.dit.hist_start_emb.expand(b, -1, -1),
                    grip_hist[:, :n_transition_chunks, :],
                    self.dit.previous_task_end_emb.expand(b, -1, -1),
                    grip_hist[:, n_transition_chunks:, :],
                    self.dit.hist_end_emb.expand(b, -1, -1),
                ], dim=1)
                grip_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    grip_hist_mask_base[:, :n_transition_chunks],
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    grip_hist_mask_base[:, n_transition_chunks:],
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                ], dim=1)
            else:
                arm_stream = torch.cat([
                    self.dit.hist_start_emb.expand(b, -1, -1),
                    arm_hist,
                    self.dit.hist_end_emb.expand(b, -1, -1),
                ], dim=1)
                arm_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    arm_hist_mask_base,
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                ], dim=1)
                grip_stream = torch.cat([
                    self.dit.hist_start_emb.expand(b, -1, -1),
                    grip_hist,
                    self.dit.hist_end_emb.expand(b, -1, -1),
                ], dim=1)
                grip_stream_mask = torch.cat([
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    grip_hist_mask_base,
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                ], dim=1)

            hist_actions_for_dit = torch.cat([arm_stream, grip_stream], dim=1)
        else:
            vlm_stream = vlm_features
            hist_actions_for_dit = hist_chunks
            hist_chunks_mask_for_dit = hist_chunks_mask
            vlm_attention_mask_for_dit = None  # will use default VLM mask in DiT
        predict_chunks_len = self.config.pred_chunk_size // self.config.action_chunk_size
        arm_btl = self.config.action_bottleneck_dim
        grip_btl = self.config.grip_bottleneck_dim

        z_t_arm = torch.randn(b, predict_chunks_len, arm_btl, device=device, dtype=torch.float32)
        z_t_grip = torch.randn(b, predict_chunks_len, grip_btl, device=device, dtype=torch.float32)

        dt = -1.0 / self.config.num_inference_steps
        time = torch.tensor(1.0, device=device, dtype=torch.float32)

        while time >= -dt / 2:
            expanded_time = time.expand(b)

            # 2. Latent -> DiT (FP32 for decoder precision, output to BF16)
            with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
                z_t_arm_dit = self.action_encoder.arm_dec(z_t_arm)
                z_t_grip_dit = self.action_encoder.grip_dec(z_t_grip)
            z_t_dit = torch.cat([z_t_arm_dit.to(target_dtype), z_t_grip_dit.to(target_dtype)], dim=1)

            # 3. DiT forward
            if use_st:
                # Build full mask for special token mode
                grip_target_mask = torch.ones(b, z_t_dit.shape[1] // 2, dtype=torch.bool, device=device)
                arm_target_mask = torch.ones(b, z_t_dit.shape[1] // 2, dtype=torch.bool, device=device)
                full_mask = torch.cat([vlm_stream_mask, grip_stream_mask, arm_stream_mask,
                                       grip_target_mask, arm_target_mask], dim=1)
                pred_z0_dit = self.dit(
                    target_actions=z_t_dit,
                    hist_actions=hist_actions_for_dit,
                    vlm_features=vlm_stream,
                    empty_emb=empty_emb,
                    timestep=expanded_time,
                    state_emb=state_emb,
                    hist_actions_mask=None,
                    vlm_attention_mask=None,
                    return_chunks=True,
                    use_gradient_checkpointing=False,
                    joint_attention_kwargs={'attention_mask': full_mask},
                )
            else:
                pred_z0_dit = self.dit(
                    target_actions=z_t_dit,
                    hist_actions=hist_actions_for_dit,
                    vlm_features=vlm_stream,
                    empty_emb=empty_emb,
                    timestep=expanded_time,
                    state_emb=state_emb,
                    hist_actions_mask=hist_chunks_mask_for_dit,
                    vlm_attention_mask=vlm_attention_mask_for_dit,
                    return_chunks=True,
                    use_gradient_checkpointing=False,
                )

            # 4. DiT -> Latent (FP32 for dit_to_latent precision)
            num = pred_z0_dit.shape[1] // 2
            with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
                pred_z0_arm_latent = self.arm_dit_to_latent(pred_z0_dit[:, :num, :])
                pred_z0_grip_latent = self.grip_dit_to_latent(pred_z0_dit[:, num:, :])

            # 5. Euler step in latent space
            t_expand = time.clamp(min=1e-5)
            v_pred_arm = (z_t_arm - pred_z0_arm_latent) / t_expand
            v_pred_grip = (z_t_grip - pred_z0_grip_latent) / t_expand
            z_t_arm = z_t_arm + dt * v_pred_arm
            z_t_grip = z_t_grip + dt * v_pred_grip

            time = time + dt

        # 6. Final decode: latent -> 1024D tokens -> action space
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=True):
            pred_z0_tokens = self.action_encoder.decode(z_t_arm, z_t_grip)
        pred_z0_tokens = pred_z0_tokens.to(target_dtype)

        num_chunks = pred_z0_tokens.shape[1] // 2
        pred_x0_arm = pred_z0_tokens[:, :num_chunks, :]
        pred_x0_grip = pred_z0_tokens[:, num_chunks:, :]

        pred_arm = self.dit.arm_out_proj(pred_x0_arm).view(b, -1, self.config.arm_dim)
        pred_gripper_logits = self.dit.gripper_out_proj(pred_x0_grip).view(b, -1, self.config.gripper_dim)
        pred_gripper_probs = torch.sigmoid(pred_gripper_logits)
        pred_gripper_binary = (pred_gripper_probs > self.config.gripper_threshold).float()
        pred_gripper = (pred_gripper_binary - 0.5) * 2.0

        # Reassemble into original action_dim ordering
        actions = torch.zeros(
            b, pred_arm.shape[1], self.config.action_dim,
            device=device, dtype=pred_arm.dtype,
        )
        arm_indices = [i for i in range(self.config.action_dim) if i not in list(self.config.gripper_dim_indices_abs)]
        actions[:, :, arm_indices] = pred_arm
        actions[:, :, list(self.config.gripper_dim_indices_abs)] = pred_gripper.to(actions.dtype)
        return actions


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class LoLAV07Policy(PreTrainedPolicy):
    """LoLA v07 Policy with Latent Flow Matching."""
    config_class = LoLAV07Config
    name = "lola_v07"

    def __init__(self, config: LoLAV07Config):
        super().__init__(config)
        self.config = config

        if isinstance(config.dtype, str):
            self._dtype = getattr(torch, config.dtype)
        else:
            self._dtype = config.dtype

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core model
        self.model = LoLAV07Pytorch(config)

        # VLM loading — backbone resolved via registry (config.vlm_backbone)
        from lerobot.policies.lola.vlm_backbone import get_vlm_backbone
        vlm_spec = get_vlm_backbone(self.config.vlm_backbone)
        if self.config.vlm_path is not None:
            self.vlm = vlm_spec.load_model(
                self.config.vlm_path,
                dtype=self._dtype,
                local_files_only=True,
            )
        else:
            self.vlm = vlm_spec.load_model(
                self.config.vlm_model_name,
                dtype=self._dtype,
            )

        # Remove unused VLM layers
        last_extract_layer = max(self.config.vlm_extract_layers)
        lang_model = self.vlm.language_model
        for i in range(len(lang_model.layers) - 1, last_extract_layer - 1, -1):
            del lang_model.layers[i]
        lang_model.norm = nn.Identity()

        self.model.to(self._dtype)

        # Re-isolate encoders to FP32 after dtype conversion
        self.model.action_encoder = self.model.action_encoder.float()
        if self.model.state_encoder is not None:
            self.model.state_encoder = self.model.state_encoder.float()

        # Gradient checkpointing
        # VLM 是激活大头, GC 必须保留; DiT (256D bottleneck, 激活 ~1GB) 默认不做 GC,
        # 避免 backward 里一次完整重算 — 由 config.dit_gradient_checkpointing 单独控制。
        if config.gradient_checkpointing:
            if getattr(config, "dit_gradient_checkpointing", False):
                self.model.gradient_checkpointing_enable()
            if config.train_vlm:
                self.vlm.gradient_checkpointing_enable()

        # Action queue
        self._action_queue = deque(maxlen=self.config.action_chunk_size * 5)
        # obs_prev_chunk_frame 推理缓存 (见 inject_prev_chunk_frame)
        self._prev_chunk_frames = None

        # VLM forward mode
        if config.gradient_checkpointing and config.train_vlm:
            self._vlm_forward_mode = "output_hidden_states"
        elif config.compile_model:
            import warnings
            warnings.warn(
                "torch.compile + FSDP is incompatible with hook-based hidden state capture. "
                "Switching VLM forward mode to 'output_hidden_states'.",
                UserWarning,
                stacklevel=2,
            )
            self._vlm_forward_mode = "output_hidden_states"
        else:
            self._vlm_forward_mode = "hook"

        # Hook infrastructure
        self._captured_hidden_states: Dict[int, torch.Tensor] = {}
        self._hook_handles: List = []
        self._in_vlm_forward: bool = False
        if self._vlm_forward_mode == "hook":
            self._register_vlm_hooks()

    # ---- VLM hook infrastructure (same as LoLAPolicy) ----

    def _register_vlm_hooks(self):
        for extract_layer_idx in self.config.vlm_extract_layers:
            decoder_layer_idx = extract_layer_idx - 1
            decoder_layer = self.vlm.language_model.layers[decoder_layer_idx]

            def make_hook(eidx):
                def hook_fn(module, input, output):
                    if not self._in_vlm_forward:
                        return
                    self._captured_hidden_states[eidx] = output
                return hook_fn

            handle = decoder_layer.register_forward_hook(make_hook(extract_layer_idx))
            self._hook_handles.append(handle)

    def _remove_vlm_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._captured_hidden_states = {}

    def _move_to_device(self, device: torch.device):
        self._device = device
        self.model = self.model.to(device)
        self.vlm = self.vlm.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.action_chunk_size * 5)
        # obs_prev_chunk_frame 推理缓存: {cam_key: 上一 chunk 起始帧 (C,H,W)}
        self._prev_chunk_frames = None

    def inject_prev_chunk_frame(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """obs_prev_chunk_frame 推理适配 (2026-08-12)。

        在图像 processor (LolaImageProcessor) 之前对原始观测调用:
        chunk 边界 (action queue 为空, 本步将预测新 chunk) 时, 把每个相机的当前帧
        与缓存的上一 chunk 起始帧堆叠为 (2, C, H, W), 与训练侧
        observation_delta_indices=[-action_chunk_size, 0] 严格对齐; episode 首
        chunk 用当前帧自复制, 对应训练侧 episode 边界 clamp。
        非边界步原样返回 (该步动作从 queue 弹出, batch 不被消费)。
        仅处理 (C,H,W) tensor 形式的相机值 (CALVIN eval 路径)。
        """
        if not self.config.obs_prev_chunk_frame or len(self._action_queue) > 0:
            return observation
        obs = dict(observation)
        new_cache = {}
        for key, val in observation.items():
            if not (key.startswith("observation.images.")
                    and isinstance(val, torch.Tensor) and val.ndim == 3):
                continue
            prev = self._prev_chunk_frames.get(key) if self._prev_chunk_frames else None
            obs[key] = torch.stack([prev if prev is not None else val, val], dim=0)
            new_cache[key] = val
        if new_cache:
            self._prev_chunk_frames = new_cache
        return obs

    # ---- Data preparation (same as LoLAPolicy) ----

    def prepare_hist_actions(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.history_type == "state":
            if "hist_states_full" in batch:
                hist_input = batch["hist_states_full"]
                if hist_input.ndim == 2:
                    hist_input = hist_input.unsqueeze(0)
                hist_mask = batch.get("hist_states_mask", None)
                if hist_mask is not None:
                    hist_mask = hist_mask.float()
                else:
                    hist_mask = torch.ones(
                        hist_input.shape[0], hist_input.shape[1],
                        dtype=torch.float32, device=hist_input.device,
                    )
                return hist_input, hist_mask
            elif "observation.state" in batch:
                hist_input = batch["observation.state"]
                if hist_input.ndim == 2:
                    hist_input = hist_input.unsqueeze(1)
                return hist_input, None
            else:
                b = batch.get("action", batch.get("input_ids", torch.zeros(1))).shape[0]
                hist_input = torch.zeros(
                    b, self.config.action_chunk_size, self.config.state_dim,
                    device=self.device, dtype=self.dtype,
                )
                return hist_input, None
        else:
            if "hist_actions_full" in batch:
                hist_input = batch["hist_actions_full"]
                if hist_input.ndim == 2:
                    hist_input = hist_input.unsqueeze(0)
                hist_mask = batch.get("hist_actions_mask", None)
                if hist_mask is not None:
                    hist_mask = hist_mask.float()
                else:
                    hist_mask = torch.ones(
                        hist_input.shape[0], hist_input.shape[1],
                        dtype=torch.float32, device=hist_input.device,
                    )
                return hist_input, hist_mask
            elif "hist_actions" in batch:
                hist_input = batch["hist_actions"]
                if hist_input.ndim == 2:
                    hist_input = hist_input.unsqueeze(1)
                return hist_input, None
            elif "observation.state" in batch:
                hist_input = batch["observation.state"]
                if hist_input.ndim == 2:
                    hist_input = hist_input.unsqueeze(1)
                return hist_input, None
            else:
                b = batch.get("action", batch.get("input_ids", torch.zeros(1))).shape[0]
                hist_input = torch.zeros(
                    b, self.config.action_chunk_size, self.config.action_dim,
                    device=self.device, dtype=self.dtype,
                )
                return hist_input, None

    def prepare_target_actions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        actions = batch["action"]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        return actions

    def prepare_vlm_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
        elif "observation.language.tokens" in batch:
            input_ids = batch["observation.language.tokens"]
        elif "observation.language_tokens" in batch:
            input_ids = batch["observation.language_tokens"]
        else:
            b = batch["action"].shape[0]
            input_ids = torch.full(
                (b, 1), self.config.empty_token_id,
                dtype=torch.long, device=self.device,
            )

        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        # mm_token_type_ids is required by Cosmos3-Nano (Qwen3-VL) M-RoPE when
        # image_grid_thw is passed; Qwen3.5's processor never emits it.
        mm_token_type_ids = batch.get("mm_token_type_ids", None)
        attention_mask = (
            batch.get("attention_mask", None)
            or batch.get("observation.language.attention_mask", None)
        )

        if "hidden_states_all_layers" in batch:
            raw = batch["hidden_states_all_layers"]
            if isinstance(raw, dict):
                hidden_states_all_layers = raw
            else:
                hidden_states_all_layers = {i: raw[i] for i in self.config.vlm_extract_layers}
        else:
            forward_kwargs = {
                "input_ids": input_ids,
                "return_dict": True,
            }
            if self._vlm_forward_mode == "output_hidden_states":
                forward_kwargs["output_hidden_states"] = True
            else:
                forward_kwargs["output_hidden_states"] = False

            if pixel_values is not None:
                forward_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                forward_kwargs["image_grid_thw"] = image_grid_thw
            if mm_token_type_ids is not None:
                forward_kwargs["mm_token_type_ids"] = mm_token_type_ids
            if attention_mask is not None:
                forward_kwargs["attention_mask"] = attention_mask

            # 运行时冻结判定: delayed-unfreeze 期间 config.train_vlm=True 但 VLM
            # 参数 requires_grad=False。冻结的 VLM 必须在 no_grad 下运行 — 否则
            # ZeRO-3 的 PreBackwardFunctionForModule 会把 VLM 激活接进 autograd 图
            # (rg=True 链路到 loss), 叠加训练循环 model.train() 递归打开的 HF
            # 逐层 checkpoint, 每层保存 ZeRO-3 gather 的权重 (RMSNorm [4096] /
            # qk-norm [128]), backward recompute 时参数已 release → CheckpointError
            # ("Recomputed values ... different metadata", [4096]→[0])。
            vlm_needs_grad = self.config.train_vlm and any(
                p.requires_grad for p in self.vlm.parameters()
            )
            if self._vlm_forward_mode == "hook":
                self._captured_hidden_states = {}
                self._in_vlm_forward = True
                try:
                    if not vlm_needs_grad:
                        with torch.no_grad():
                            self.vlm(**forward_kwargs)
                    else:
                        self.vlm(**forward_kwargs)
                finally:
                    self._in_vlm_forward = False
                hidden_states_all_layers = self._captured_hidden_states
            else:
                if not vlm_needs_grad:
                    with torch.no_grad():
                        vlm_output = self.vlm(**forward_kwargs)
                else:
                    vlm_output = self.vlm(**forward_kwargs)
                hidden_states_all_layers = {
                    i: vlm_output.hidden_states[i] for i in self.config.vlm_extract_layers
                }

        return hidden_states_all_layers, input_ids

    # ---- Training & Inference ----

    def _drop_visual_tokens(self, hidden_states_all_layers, input_ids, batch):
        """按 visual token 位置随机置零各层 hidden 特征 (见 forward 中调用点注释)。"""
        rate = self.config.visual_token_drop_rate
        image_token_id = getattr(self.vlm.config, "image_token_id", None)
        if image_token_id is not None:
            vis_mask = input_ids == image_token_id
        else:
            # 退路: Cosmos3/Qwen3-VL processor 的 mm_token_type_ids (1=image)
            mm_types = batch.get("mm_token_type_ids", None)
            if mm_types is None:
                if not getattr(self, "_visual_drop_warned", False):
                    self._visual_drop_warned = True
                    print("[visual_token_drop] ⚠️ 无法定位 visual token "
                          "(无 image_token_id / mm_token_type_ids), 跳过 drop")
                return hidden_states_all_layers
            vis_mask = mm_types == 1
        keep = (torch.rand(vis_mask.shape, device=vis_mask.device) >= rate) | ~vis_mask
        keep = keep.to(next(iter(hidden_states_all_layers.values())).dtype).unsqueeze(-1)
        return {k: v * keep for k, v in hidden_states_all_layers.items()}

    def forward(self, batch: Dict[str, torch.Tensor], compute_per_dim: bool = False, time=None,
                noise=None, return_per_sample: bool = False):
        """Training forward. NOTE: hist_actions/target_actions dtype is managed by LoLAV07Pytorch.forward().

        noise: 可选 (noise_arm_latent, noise_grip_latent) 元组, 外部注入固定噪声 (CRN 探针用)。
        return_per_sample: loss_dict 额外携带逐样本 loss 张量 (keys 以 "_per_sample" 结尾)。
        """
        hist_actions, hist_actions_mask = self.prepare_hist_actions(batch)
        target_actions = self.prepare_target_actions(batch)
        hidden_states_all_layers, input_ids = self.prepare_vlm_inputs(batch)

        vlm_attention_mask = (
            batch.get("attention_mask", None)
            or batch.get("observation.language.attention_mask", None)
        )

        # Extract n_transition_chunks for special token placement
        n_transition_chunks_batch = batch.get("n_transition_chunks", None)
        max_n_tc = 0
        if n_transition_chunks_batch is not None and isinstance(n_transition_chunks_batch, torch.Tensor):
            max_n_tc = int(n_transition_chunks_batch.max())
            n_transition_chunks_batch = n_transition_chunks_batch.to(self.device)
        else:
            n_transition_chunks_batch = torch.tensor([0], dtype=torch.long, device=self.device)

        # State conditioning: extract observation.state when enabled
        state_emb = None
        if self.config.use_state_condition:
            state_tensor = batch.get("observation.state", None)
            if state_tensor is not None:
                if state_tensor.ndim == 3:
                    # [B, n_obs_steps, max_state_dim] -> [B, state_dim]
                    state_emb = state_tensor[:, -1, :self.config.state_dim]
                else:
                    # [B, max_state_dim] -> [B, state_dim]
                    state_emb = state_tensor[:, :self.config.state_dim]
                state_emb = state_emb.to(self.dtype)

        # v07: do NOT convert hist_actions/target_actions to self.dtype here;
        # LoLAV07Pytorch.forward() handles FP32 isolation for encoders internally.
        if hist_actions_mask is not None:
            hist_actions_mask = hist_actions_mask.to(self.dtype)
        hidden_states_all_layers = {k: v.to(self.dtype) for k, v in hidden_states_all_layers.items()}

        # Visual token drop (2026-08-12): bridge 输入侧以概率 p 将 visual token 的
        # 特征置零 (置零而非 attention 删除 — 序列长度/位置结构不变, DiT 无联动)。
        # training-only; eval 由 self.training 门控自动关闭。
        if self.training and self.config.visual_token_drop_rate > 0:
            hidden_states_all_layers = self._drop_visual_tokens(
                hidden_states_all_layers, input_ids, batch
            )

        losses = self.model(
            hidden_states_all_layers=hidden_states_all_layers,
            input_ids=input_ids,
            hist_actions=hist_actions,
            target_actions=target_actions,
            hist_actions_mask=hist_actions_mask,
            vlm_attention_mask=vlm_attention_mask,
            time=time,
            noise=noise,
            state_emb=state_emb,
            n_transition_chunks=max_n_tc,
            n_transition_chunks_batch=n_transition_chunks_batch,
            return_per_sample=return_per_sample,
        )

        loss = losses["total_loss"]
        loss_dict = {
            "loss": loss.item(),
            "v_loss": losses["v_loss"].item(),
            "arm_loss": losses["arm_loss"].item(),
            "gripper_loss": losses["gripper_loss"].item(),
        }
        if compute_per_dim:
            loss_dict["arm_loss_per_dim"] = losses["arm_loss_per_dim"].detach()
        if return_per_sample:
            for key, value in losses.items():
                if key.endswith("_per_sample"):
                    loss_dict[key] = value.detach()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self.model.eval()

        hist_actions, hist_actions_mask = self.prepare_hist_actions(batch)
        # v07: dtype conversion happens inside model
        if hist_actions_mask is not None:
            hist_actions_mask = hist_actions_mask.to(self.dtype)
        hidden_states_all_layers, input_ids = self.prepare_vlm_inputs(batch)

        # Extract n_transition_chunks for special token placement (CALVIN inference)
        n_transition_chunks = batch.get("n_transition_chunks", None)
        if n_transition_chunks is not None and isinstance(n_transition_chunks, torch.Tensor):
            n_transition_chunks = int(n_transition_chunks.max())  # batch max
        else:
            n_transition_chunks = None  # default: auto-detect 0

        # State conditioning: extract observation.state when enabled
        state_emb = None
        if self.config.use_state_condition:
            state_tensor = batch.get("observation.state", None)
            if state_tensor is not None:
                if state_tensor.ndim == 3:
                    state_emb = state_tensor[:, -1, :self.config.state_dim]
                else:
                    state_emb = state_tensor[:, :self.config.state_dim]
                state_emb = state_emb.to(self.dtype)

        actions = self.model.sample_actions(
            hidden_states_all_layers=hidden_states_all_layers,
            hist_actions=hist_actions,
            hist_actions_mask=hist_actions_mask,
            state_emb=state_emb,
            n_transition_chunks=n_transition_chunks,
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            for i in range(actions.shape[1]):
                self._action_queue.append(actions[:, i, :])
        return self._action_queue.popleft()
