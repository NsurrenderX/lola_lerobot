# Copyright 2025 Lola Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

from diffusers.models.transformers.transformer_flux2 import (
    Flux2Modulation,
    Flux2FeedForward,
    )
from diffusers.models.embeddings import apply_rotary_emb


from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.lola.configuration_lola import LoLAConfig

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def create_sinusoidal_pos_embedding(time: torch.Tensor, dimension: int, min_period: float = 4e-3, max_period: float = 1.0) -> torch.Tensor:
    """生成 Timestep 的正弦位置编码 (参考 OpenPI/LeRobot)"""
    assert dimension % 2 == 0 # 确保维度为偶数
    half_dim = dimension // 2
    fraction = torch.linspace(0.0, 1.0, half_dim, dtype=torch.float32, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

# ----------------------------------------------------------------------
# 1. Sub-Modules
# ----------------------------------------------------------------------
class LolaActionEncoder(nn.Module):
    """Dual-Token Action Chunking: 将手臂和夹爪完全解耦，分别映射为独立的 Token，
    并注入模态编码以区分身份，最后在 Sequence 维度拼接。
    输出长度为原来的 2 倍: [B, 2 * num_chunks, dit_hidden_size]
    """
    def __init__(self, config: LoLAConfig):
        super().__init__()
        self.chunk_size = config.action_chunk_size
        self.arm_dim = config.arm_dim
        self.gripper_dim = config.gripper_dim
        self.config = config

        self.arm_proj = nn.Sequential(
            nn.Linear(self.chunk_size * self.arm_dim, config.dit_hidden_size),
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        )

        self.gripper_proj = nn.Sequential(
            nn.Linear(self.chunk_size * self.gripper_dim, config.dit_hidden_size),
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        )

        self.arm_modality_emb = nn.Parameter(torch.randn(1, 1, config.dit_hidden_size) * 0.02)
        self.gripper_modality_emb = nn.Parameter(torch.randn(1, 1, config.dit_hidden_size) * 0.02)

    def _pad_and_chunk(self, actions: torch.Tensor, dim_size: int) -> torch.Tensor:
        b, seq_len, d = actions.shape
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            actions = F.pad(actions, (0, 0, 0, pad_len))
            seq_len += pad_len
        return actions.view(b, seq_len // self.chunk_size, self.chunk_size * dim_size)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        gripper_indices = list(self.config.gripper_dim_indices_abs)
        all_indices = list(range(actions.shape[-1]))
        arm_indices = [i for i in all_indices if i not in gripper_indices]

        arm_actions = actions[..., arm_indices]
        gripper_actions = actions[..., gripper_indices]

        arm_chunked = self._pad_and_chunk(arm_actions, self.arm_dim)
        gripper_chunked = self._pad_and_chunk(gripper_actions, self.gripper_dim)

        arm_tokens = self.arm_proj(arm_chunked) + self.arm_modality_emb
        gripper_tokens = self.gripper_proj(gripper_chunked) + self.gripper_modality_emb

        return torch.cat([arm_tokens, gripper_tokens], dim=1)

class LolaVLMFeatureExtractor(nn.Module):
    """提取 Qwen3.5 特征层与全局空 Token"""
    def __init__(self, config: LoLAConfig):
        super().__init__()
        # vlm_extract_layers 指定的是 transformer 层编号 (如 8, 16, 24)
        # hidden_states 元组结构: [embedding层, transformer层1, transformer层2, ...]
        # 所以实际的索引不需要额外处理 (embedding 层是索引 0)
        self.extract_layers = [layer_idx for layer_idx in config.vlm_extract_layers]
        
        # Qwen3.5 的 3层 Hidden 拼接
        concat_dim = config.vlm_hidden_size * len(self.extract_layers)
        
        # 两个独立的投影网络映射到 1536 维度
        self.feature_proj =nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.LayerNorm(concat_dim, eps=1e-6),
            nn.SiLU(),
            nn.Linear(concat_dim, config.dit_hidden_size),
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        ) 
        self.feature_shortcut = nn.Linear(concat_dim, config.dit_hidden_size)
        self.feature_out_proj = nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        
        self.empty_token_proj = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.LayerNorm(concat_dim, eps=1e-6),
            nn.SiLU(),
            nn.Linear(concat_dim, config.dit_hidden_size),
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        )
        self.empty_token_shortcut = nn.Linear(concat_dim, config.dit_hidden_size)
        self.empty_token_out_proj = nn.Linear(config.dit_hidden_size, config.dit_hidden_size)

    def forward(self, hidden_states_all_layers: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states_all_layers: Dict[int, Tensor] 仅包含所需层的 hidden_states
                通过 forward hook 捕获，key 为层编号 (如 8, 16, 24)

        Returns:
            vlm_emb: [B, Seq_Len-1, dit_hidden_size] - VLM 特征（不包含最后一个 token）
            empty_emb: [B, dit_hidden_size] - 空 token 特征（最后一个 token）
        """
        # 提取指定的 transformer 层
        selected_hiddens = [hidden_states_all_layers[i] for i in self.extract_layers]
        stacked_features = torch.cat(selected_hiddens, dim=-1) # [B, SeqLen, 7680]
        
        # 假设空 Token 已经在 input_ids 的最末尾
        vlm_features = stacked_features[:, :-1, :]   # [B, Seq_Len-1, 7680]
        empty_token = stacked_features[:, -1, :]     # [B, 7680]
        
        vlm_fused = self.feature_proj(vlm_features) + self.feature_shortcut(vlm_features)
        vlm_emb = self.feature_out_proj(vlm_fused)    # [B, Seq_Len-1, 1536]

        empty_fused = self.empty_token_proj(empty_token) + self.empty_token_shortcut(empty_token)
        empty_emb = self.empty_token_out_proj(empty_fused) # [B, 1536]
        
        return vlm_emb, empty_emb

class LolaConditionEmbedder(nn.Module):
    """融合 Timestep 与空 Token 生成 Modulation 条件"""
    def __init__(self, config: LoLAConfig):
        super().__init__()
        self.min_period = config.min_period
        self.max_period = config.max_period
        self.time_mlp = nn.Sequential(
            nn.Linear(256, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        )

    def forward(self, timestep: torch.Tensor, empty_emb: torch.Tensor) -> torch.Tensor:
        time_emb = create_sinusoidal_pos_embedding(timestep, 256, self.min_period, self.max_period).to(empty_emb.dtype)
        t_feat = self.time_mlp(time_emb)
        c_feat = self.cond_mlp(empty_emb)
        return t_feat + c_feat # [B, 1536]

# ----------------------------------------------------------------------
# 2. Dual-Expert Transformer Blocks
# ----------------------------------------------------------------------
class LoLA5StreamDoubleBlock(nn.Module):
    """5-Stream Double Block: ctx_vlm / ctx_arm / ctx_grip / target_arm / target_grip.
    Context sub-streams share a common FFN for cross-modal feature interaction,
    but have independent QKV projections per sub-stream.
    Target streams have fully independent QKV + FFN (heterogeneous experts).
    All 5 streams participate in joint attention.
    """
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 arm_mlp_ratio=4.0, grip_mlp_ratio=2.0, ctx_mlp_ratio=4.0,
                 eps=1e-6, bias=False):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        # --- Context sub-streams: independent QKV, shared FFN ---
        for prefix in ['ctx_vlm', 'ctx_arm', 'ctx_grip']:
            setattr(self, f'{prefix}_norm1', nn.LayerNorm(dim, elementwise_affine=False, eps=eps))
            setattr(self, f'{prefix}_to_q', nn.Linear(dim, self.inner_dim, bias=bias))
            setattr(self, f'{prefix}_to_k', nn.Linear(dim, self.inner_dim, bias=bias))
            setattr(self, f'{prefix}_to_v', nn.Linear(dim, self.inner_dim, bias=bias))
            setattr(self, f'{prefix}_norm_q', nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True))
            setattr(self, f'{prefix}_norm_k', nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True))
            setattr(self, f'{prefix}_to_out', nn.Linear(self.inner_dim, dim, bias=bias))
            setattr(self, f'{prefix}_norm2', nn.LayerNorm(dim, elementwise_affine=False, eps=eps))
            setattr(self, f'{prefix}_modulation', Flux2Modulation(dim, mod_param_sets=2, bias=bias))

        self.ctx_shared_ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=ctx_mlp_ratio, bias=bias)

        # --- Target arm expert ---
        self.arm_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.arm_to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.arm_to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.arm_to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.arm_norm_q = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.arm_norm_k = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.arm_to_out = nn.Linear(self.inner_dim, dim, bias=bias)
        self.arm_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.arm_ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=arm_mlp_ratio, bias=bias)
        self.arm_modulation = Flux2Modulation(dim, mod_param_sets=2, bias=bias)

        # --- Target gripper expert ---
        self.grip_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.grip_to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.grip_to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.grip_to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.grip_norm_q = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.grip_norm_k = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.grip_to_out = nn.Linear(self.inner_dim, dim, bias=bias)
        self.grip_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.grip_ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=grip_mlp_ratio, bias=bias)
        self.grip_modulation = Flux2Modulation(dim, mod_param_sets=2, bias=bias)

    def _ctx_qkv(self, prefix, hidden, mod):
        """Compute QKV for a context sub-stream."""
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(mod, 2)
        normed = getattr(self, f'{prefix}_norm1')(hidden)
        normed = (1 + scale_msa) * normed + shift_msa
        q = getattr(self, f'{prefix}_to_q')(normed)
        k = getattr(self, f'{prefix}_to_k')(normed)
        v = getattr(self, f'{prefix}_to_v')(normed)
        q = getattr(self, f'{prefix}_norm_q')(q.unflatten(-1, (self.num_heads, self.head_dim)))
        k = getattr(self, f'{prefix}_norm_k')(k.unflatten(-1, (self.num_heads, self.head_dim)))
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        return q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def forward(self, ctx_vlm_hidden, ctx_arm_hidden, ctx_grip_hidden,
                arm_hidden, grip_hidden,
                temb_mod_ctx_vlm, temb_mod_ctx_arm, temb_mod_ctx_grip,
                temb_mod_arm, temb_mod_grip,
                image_rotary_emb, joint_attention_kwargs=None):
        joint_attention_kwargs = joint_attention_kwargs or {}
        ctx_vlm_len = ctx_vlm_hidden.shape[1]
        ctx_arm_len = ctx_arm_hidden.shape[1]
        ctx_grip_len = ctx_grip_hidden.shape[1]
        arm_len = arm_hidden.shape[1]
        grip_len = grip_hidden.shape[1]

        # 1. Per-stream modulation + QKV for context sub-streams
        vlm_q, vlm_k, vlm_v, vlm_gate_msa, vlm_shift_mlp, vlm_scale_mlp, vlm_gate_mlp = \
            self._ctx_qkv('ctx_vlm', ctx_vlm_hidden, temb_mod_ctx_vlm)
        arm_ctx_q, arm_ctx_k, arm_ctx_v, arm_ctx_gate_msa, arm_ctx_shift_mlp, arm_ctx_scale_mlp, arm_ctx_gate_mlp = \
            self._ctx_qkv('ctx_arm', ctx_arm_hidden, temb_mod_ctx_arm)
        grip_ctx_q, grip_ctx_k, grip_ctx_v, grip_ctx_gate_msa, grip_ctx_shift_mlp, grip_ctx_scale_mlp, grip_ctx_gate_mlp = \
            self._ctx_qkv('ctx_grip', ctx_grip_hidden, temb_mod_ctx_grip)

        # 2. Per-stream modulation + QKV for target streams
        (arm_shift_msa, arm_scale_msa, arm_gate_msa), (arm_shift_mlp, arm_scale_mlp, arm_gate_mlp) = \
            Flux2Modulation.split(temb_mod_arm, 2)
        arm_norm = (1 + arm_scale_msa) * self.arm_norm1(arm_hidden) + arm_shift_msa
        arm_q = self.arm_norm_q(self.arm_to_q(arm_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        arm_k = self.arm_norm_k(self.arm_to_k(arm_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        arm_v = self.arm_to_v(arm_norm).unflatten(-1, (self.num_heads, self.head_dim))

        (grip_shift_msa, grip_scale_msa, grip_gate_msa), (grip_shift_mlp, grip_scale_mlp, grip_gate_mlp) = \
            Flux2Modulation.split(temb_mod_grip, 2)
        grip_norm = (1 + grip_scale_msa) * self.grip_norm1(grip_hidden) + grip_shift_msa
        grip_q = self.grip_norm_q(self.grip_to_q(grip_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        grip_k = self.grip_norm_k(self.grip_to_k(grip_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        grip_v = self.grip_to_v(grip_norm).unflatten(-1, (self.num_heads, self.head_dim))

        # 3. Concatenate Q/K/V: [ctx_vlm, ctx_grip, ctx_arm, target_grip, target_arm]
        query = torch.cat([vlm_q, grip_ctx_q, arm_ctx_q, grip_q, arm_q], dim=1)
        key = torch.cat([vlm_k, grip_ctx_k, arm_ctx_k, grip_k, arm_k], dim=1)
        value = torch.cat([vlm_v, grip_ctx_v, arm_ctx_v, grip_v, arm_v], dim=1)

        # 4. RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # 5. Joint attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_mask = joint_attention_kwargs.get('attention_mask', None)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        attn_output = attn_output.transpose(1, 2).flatten(2, 3).to(query.dtype)

        # 6. Split attention output back into 5 streams
        pos = 0
        vlm_attn = attn_output[:, pos:pos + ctx_vlm_len, :]; pos += ctx_vlm_len
        grip_ctx_attn = attn_output[:, pos:pos + ctx_grip_len, :]; pos += ctx_grip_len
        arm_ctx_attn = attn_output[:, pos:pos + ctx_arm_len, :]; pos += ctx_arm_len
        grip_attn = attn_output[:, pos:pos + grip_len, :]; pos += grip_len
        arm_attn = attn_output[:, pos:pos + arm_len, :]; pos += arm_len

        # 7. Output projections + residual + gate (attention sub-block)
        vlm_attn = self.ctx_vlm_to_out(vlm_attn)
        arm_ctx_attn = self.ctx_arm_to_out(arm_ctx_attn)
        grip_ctx_attn = self.ctx_grip_to_out(grip_ctx_attn)
        arm_attn = self.arm_to_out(arm_attn)
        grip_attn = self.grip_to_out(grip_attn)

        ctx_vlm_hidden = ctx_vlm_hidden + vlm_gate_msa * vlm_attn
        ctx_arm_hidden = ctx_arm_hidden + arm_ctx_gate_msa * arm_ctx_attn
        ctx_grip_hidden = ctx_grip_hidden + grip_ctx_gate_msa * grip_ctx_attn
        arm_hidden = arm_hidden + arm_gate_msa * arm_attn
        grip_hidden = grip_hidden + grip_gate_msa * grip_attn

        # 8. FFN sub-block
        # Context sub-streams: per-stream norm + mod, then shared FFN
        vlm_norm2 = self.ctx_vlm_norm2(ctx_vlm_hidden) * (1 + vlm_scale_mlp) + vlm_shift_mlp
        arm_ctx_norm2 = self.ctx_arm_norm2(ctx_arm_hidden) * (1 + arm_ctx_scale_mlp) + arm_ctx_shift_mlp
        grip_ctx_norm2 = self.ctx_grip_norm2(ctx_grip_hidden) * (1 + grip_ctx_scale_mlp) + grip_ctx_shift_mlp

        ctx_merged_norm = torch.cat([vlm_norm2, grip_ctx_norm2, arm_ctx_norm2], dim=1)
        ctx_ffn_out = self.ctx_shared_ff(ctx_merged_norm)
        vlm_ffn = ctx_ffn_out[:, :ctx_vlm_len, :]
        grip_ctx_ffn = ctx_ffn_out[:, ctx_vlm_len:ctx_vlm_len + ctx_grip_len, :]
        arm_ctx_ffn = ctx_ffn_out[:, ctx_vlm_len + ctx_grip_len:, :]

        ctx_vlm_hidden = ctx_vlm_hidden + vlm_gate_mlp * vlm_ffn
        ctx_arm_hidden = ctx_arm_hidden + arm_ctx_gate_mlp * arm_ctx_ffn
        ctx_grip_hidden = ctx_grip_hidden + grip_ctx_gate_mlp * grip_ctx_ffn

        # Target streams: independent FFN
        arm_norm2 = self.arm_norm2(arm_hidden) * (1 + arm_scale_mlp) + arm_shift_mlp
        arm_hidden = arm_hidden + arm_gate_mlp * self.arm_ff(arm_norm2)
        grip_norm2 = self.grip_norm2(grip_hidden) * (1 + grip_scale_mlp) + grip_shift_mlp
        grip_hidden = grip_hidden + grip_gate_mlp * self.grip_ff(grip_norm2)

        return ctx_vlm_hidden, ctx_arm_hidden, ctx_grip_hidden, arm_hidden, grip_hidden


class LoLA5StreamSingleBlock(nn.Module):
    """5-Stream Single Block: parallel pattern (attn + FFN combined with single gate).
    Same 5-stream structure as DoubleBlock but with parallel ViT-22B style computation.
    Context sub-streams share FFN, target streams have independent FFN.
    """
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 arm_mlp_ratio=4.0, grip_mlp_ratio=2.0, ctx_mlp_ratio=4.0,
                 eps=1e-6, bias=False):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        # --- Context sub-streams: independent QKV, shared FFN ---
        for prefix in ['ctx_vlm', 'ctx_arm', 'ctx_grip']:
            setattr(self, f'{prefix}_norm', nn.LayerNorm(dim, elementwise_affine=False, eps=eps))
            setattr(self, f'{prefix}_to_q', nn.Linear(dim, self.inner_dim, bias=bias))
            setattr(self, f'{prefix}_to_k', nn.Linear(dim, self.inner_dim, bias=bias))
            setattr(self, f'{prefix}_to_v', nn.Linear(dim, self.inner_dim, bias=bias))
            setattr(self, f'{prefix}_norm_q', nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True))
            setattr(self, f'{prefix}_norm_k', nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True))
            setattr(self, f'{prefix}_modulation', Flux2Modulation(dim, mod_param_sets=1, bias=bias))

        self.ctx_shared_ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=ctx_mlp_ratio, bias=bias)

        # --- Target arm expert ---
        self.arm_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.arm_to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.arm_to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.arm_to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.arm_norm_q = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.arm_norm_k = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.arm_ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=arm_mlp_ratio, bias=bias)
        self.arm_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=bias)

        # --- Target gripper expert ---
        self.grip_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.grip_to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.grip_to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.grip_to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.grip_norm_q = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.grip_norm_k = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.grip_ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=grip_mlp_ratio, bias=bias)
        self.grip_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=bias)

    def forward(self, ctx_vlm_hidden, ctx_arm_hidden, ctx_grip_hidden,
                arm_hidden, grip_hidden,
                temb_mod_ctx_vlm, temb_mod_ctx_arm, temb_mod_ctx_grip,
                temb_mod_arm, temb_mod_grip,
                image_rotary_emb, joint_attention_kwargs=None):
        joint_attention_kwargs = joint_attention_kwargs or {}
        ctx_vlm_len = ctx_vlm_hidden.shape[1]
        ctx_arm_len = ctx_arm_hidden.shape[1]
        ctx_grip_len = ctx_grip_hidden.shape[1]
        arm_len = arm_hidden.shape[1]
        grip_len = grip_hidden.shape[1]

        # 1. Per-stream modulation + norm + QKV for context sub-streams
        ctx_norms = []
        for prefix, hidden, temb_mod in [
            ('ctx_vlm', ctx_vlm_hidden, temb_mod_ctx_vlm),
            ('ctx_grip', ctx_grip_hidden, temb_mod_ctx_grip),
            ('ctx_arm', ctx_arm_hidden, temb_mod_ctx_arm),
        ]:
            (shift, scale, gate) = Flux2Modulation.split(temb_mod, 1)[0]
            norm = (1 + scale) * getattr(self, f'{prefix}_norm')(hidden) + shift
            ctx_norms.append((norm, gate))

        # 2. Per-stream modulation + norm + QKV for target streams
        (arm_shift, arm_scale, arm_gate) = Flux2Modulation.split(temb_mod_arm, 1)[0]
        arm_norm = (1 + arm_scale) * self.arm_norm(arm_hidden) + arm_shift

        (grip_shift, grip_scale, grip_gate) = Flux2Modulation.split(temb_mod_grip, 1)[0]
        grip_norm = (1 + grip_scale) * self.grip_norm(grip_hidden) + grip_shift

        # 3. QKV projections for all 5 streams
        qkv_list = []
        for prefix, (norm, _) in zip(['ctx_vlm', 'ctx_grip', 'ctx_arm'], ctx_norms):
            q = getattr(self, f'{prefix}_to_q')(norm)
            k = getattr(self, f'{prefix}_to_k')(norm)
            v = getattr(self, f'{prefix}_to_v')(norm)
            q = getattr(self, f'{prefix}_norm_q')(q.unflatten(-1, (self.num_heads, self.head_dim)))
            k = getattr(self, f'{prefix}_norm_k')(k.unflatten(-1, (self.num_heads, self.head_dim)))
            v = v.unflatten(-1, (self.num_heads, self.head_dim))
            qkv_list.append((q, k, v))

        arm_q = self.arm_norm_q(self.arm_to_q(arm_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        arm_k = self.arm_norm_k(self.arm_to_k(arm_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        arm_v = self.arm_to_v(arm_norm).unflatten(-1, (self.num_heads, self.head_dim))
        grip_q = self.grip_norm_q(self.grip_to_q(grip_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        grip_k = self.grip_norm_k(self.grip_to_k(grip_norm).unflatten(-1, (self.num_heads, self.head_dim)))
        grip_v = self.grip_to_v(grip_norm).unflatten(-1, (self.num_heads, self.head_dim))

        # 4. Concatenate Q/K/V: [ctx_vlm, ctx_grip, ctx_arm, target_grip, target_arm]
        all_q = [qkv_list[0][0], qkv_list[1][0], qkv_list[2][0], grip_q, arm_q]
        all_k = [qkv_list[0][1], qkv_list[1][1], qkv_list[2][1], grip_k, arm_k]
        all_v = [qkv_list[0][2], qkv_list[1][2], qkv_list[2][2], grip_v, arm_v]
        query = torch.cat(all_q, dim=1)
        key = torch.cat(all_k, dim=1)
        value = torch.cat(all_v, dim=1)

        # 5. RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # 6. Joint attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_mask = joint_attention_kwargs.get('attention_mask', None)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        attn_output = attn_output.transpose(1, 2).flatten(2, 3).to(query.dtype)

        # 7. Split attention output
        pos = 0
        vlm_attn = attn_output[:, pos:pos + ctx_vlm_len, :]; pos += ctx_vlm_len
        grip_ctx_attn = attn_output[:, pos:pos + ctx_grip_len, :]; pos += ctx_grip_len
        arm_ctx_attn = attn_output[:, pos:pos + ctx_arm_len, :]; pos += ctx_arm_len
        grip_attn = attn_output[:, pos:pos + grip_len, :]; pos += grip_len
        arm_attn = attn_output[:, pos:pos + arm_len, :]; pos += arm_len

        # 8. Parallel pattern: attn + FFN combined with single gate
        # Context: shared FFN on merged norms
        ctx_merged_norm = torch.cat([ctx_norms[0][0], ctx_norms[1][0], ctx_norms[2][0]], dim=1)
        ctx_ffn_out = self.ctx_shared_ff(ctx_merged_norm)
        vlm_ffn = ctx_ffn_out[:, :ctx_vlm_len, :]
        grip_ctx_ffn = ctx_ffn_out[:, ctx_vlm_len:ctx_vlm_len + ctx_grip_len, :]
        arm_ctx_ffn = ctx_ffn_out[:, ctx_vlm_len + ctx_grip_len:, :]

        vlm_combined = vlm_attn + vlm_ffn
        grip_ctx_combined = grip_ctx_attn + grip_ctx_ffn
        arm_ctx_combined = arm_ctx_attn + arm_ctx_ffn

        # Target: independent FFN
        arm_combined = arm_attn + self.arm_ff(arm_norm)
        grip_combined = grip_attn + self.grip_ff(grip_norm)

        # 9. Residual + gate per stream
        ctx_vlm_hidden = ctx_vlm_hidden + ctx_norms[0][1] * vlm_combined
        ctx_grip_hidden = ctx_grip_hidden + ctx_norms[1][1] * grip_ctx_combined
        ctx_arm_hidden = ctx_arm_hidden + ctx_norms[2][1] * arm_ctx_combined
        arm_hidden = arm_hidden + arm_gate * arm_combined
        grip_hidden = grip_hidden + grip_gate * grip_combined

        return ctx_vlm_hidden, ctx_arm_hidden, ctx_grip_hidden, arm_hidden, grip_hidden

# ----------------------------------------------------------------------
# 2. Core LoLA DiT Modeling (5-Stream Symmetric Architecture)
# ----------------------------------------------------------------------
class LoLADiT(nn.Module):
    def __init__(self, config: LoLAConfig):
        super().__init__()
        self.config = config
        self.cond_embedder = LolaConditionEmbedder(config)

        # 5-Stream Modulation: ctx_vlm, ctx_arm, ctx_grip, target_arm, target_grip
        dim = config.dit_hidden_size
        # Double blocks: 2-set modulation
        self.ctx_vlm_double_modulation = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.ctx_arm_double_modulation = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.ctx_grip_double_modulation = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.arm_double_modulation = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.grip_double_modulation = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        # Single blocks: 1-set modulation
        self.ctx_vlm_single_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=False)
        self.ctx_arm_single_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=False)
        self.ctx_grip_single_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=False)
        self.arm_single_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=False)
        self.grip_single_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=False)

        # 5 Modality Embeddings
        self.vlm_modality_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.arm_ctx_modality_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.grip_ctx_modality_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.arm_target_modality_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.grip_target_modality_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # 5-Stream Transformer Blocks
        attention_head_dim = config.dit_hidden_size // config.dit_num_heads

        self.double_blocks = nn.ModuleList([
            LoLA5StreamDoubleBlock(
                dim=config.dit_hidden_size,
                num_attention_heads=config.dit_num_heads,
                attention_head_dim=attention_head_dim,
                arm_mlp_ratio=config.dit_arm_ffn_mult,
                grip_mlp_ratio=config.dit_grip_ffn_mult,
                ctx_mlp_ratio=config.dit_ctx_ffn_mult,
            )
            for _ in range(config.dit_double_layers)
        ])

        self.single_blocks = nn.ModuleList([
            LoLA5StreamSingleBlock(
                dim=config.dit_hidden_size,
                num_attention_heads=config.dit_num_heads,
                attention_head_dim=attention_head_dim,
                arm_mlp_ratio=config.dit_arm_ffn_mult,
                grip_mlp_ratio=config.dit_grip_ffn_mult,
                ctx_mlp_ratio=config.dit_ctx_ffn_mult,
            )
            for _ in range(config.dit_single_layers)
        ])

        # Dual output heads: arm regression + gripper classification
        self.arm_out_proj = nn.Sequential(
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.arm_dim * config.action_chunk_size)
        )
        self.gripper_out_proj = nn.Sequential(
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.gripper_dim * config.action_chunk_size)
        )

        # Checkpointing function abstraction
        self._checkpoint_fn = torch.utils.checkpoint.checkpoint
        self._checkpoint_fn_kwargs = {"use_reentrant": False, "preserve_rng_state": False}

    def _prepare_rope_emb(self, vlm_len, arm_hist_len, grip_hist_len,
                          arm_target_len, grip_target_len, device, dtype):
        """Multi-axis RoPE for 5-stream architecture.
        Sequence order: [ctx_vlm(T=0), ctx_grip(T=1), ctx_arm(T=1), target_grip(T=2), target_arm(T=2)]
        """
        head_dim = self.config.dit_hidden_size // self.config.dit_num_heads
        rope_dim = head_dim
        axes_dims = (rope_dim // 4, rope_dim // 4, rope_dim // 4, rope_dim // 4)
        assert sum(axes_dims) == rope_dim

        def make_coords(length, t_axis):
            coords = torch.zeros((length, 4), dtype=torch.long, device=device)
            coords[:, 0] = t_axis
            coords[:, 3] = torch.arange(length, device=device)
            return coords

        # 5 streams with T-axis differentiation
        vlm_coords = make_coords(vlm_len, 0)
        grip_hist_coords = make_coords(grip_hist_len, 1)
        arm_hist_coords = make_coords(arm_hist_len, 1)
        grip_target_coords = make_coords(grip_target_len, 2)
        arm_target_coords = make_coords(arm_target_len, 2)

        all_coords = torch.cat([vlm_coords, grip_hist_coords, arm_hist_coords,
                                grip_target_coords, arm_target_coords], dim=0)

        def compute_multiaxis_rope(coords):
            freqs_list = []
            for i, dim in enumerate(axes_dims):
                pos = coords[:, i].float()
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
                freqs = torch.outer(pos, inv_freq)
                freqs = freqs.repeat_interleave(2, dim=-1)
                freqs_list.append(freqs)
            emb = torch.cat(freqs_list, dim=-1)
            return (emb.cos().to(dtype), emb.sin().to(dtype))

        return compute_multiaxis_rope(all_coords)

    def forward(self, target_actions, hist_actions, vlm_features, empty_emb, timestep,
                hist_actions_mask=None, vlm_attention_mask=None, return_chunks: bool = False,
                use_gradient_checkpointing: bool = False):
        b = target_actions.shape[0]

        # Split target_actions into arm (first half) and gripper (second half)
        num_target_chunks = target_actions.shape[1] // 2
        arm_target = target_actions[:, :num_target_chunks, :]
        grip_target = target_actions[:, num_target_chunks:, :]

        # Split hist_actions into arm and gripper halves
        num_hist_chunks = hist_actions.shape[1] // 2
        arm_hist = hist_actions[:, :num_hist_chunks, :]
        grip_hist = hist_actions[:, num_hist_chunks:, :]

        # Add modality embeddings
        vlm_features = vlm_features + self.vlm_modality_emb
        arm_hist = arm_hist + self.arm_ctx_modality_emb
        grip_hist = grip_hist + self.grip_ctx_modality_emb
        arm_target = arm_target + self.arm_target_modality_emb
        grip_target = grip_target + self.grip_target_modality_emb

        temb = self.cond_embedder(timestep, empty_emb)

        # Per-stream modulation for double-stream blocks
        temb_mod_ctx_vlm_d = self.ctx_vlm_double_modulation(temb)
        temb_mod_ctx_arm_d = self.ctx_arm_double_modulation(temb)
        temb_mod_ctx_grip_d = self.ctx_grip_double_modulation(temb)
        temb_mod_arm_d = self.arm_double_modulation(temb)
        temb_mod_grip_d = self.grip_double_modulation(temb)

        # Per-stream modulation for single-stream blocks
        temb_mod_ctx_vlm_s = self.ctx_vlm_single_modulation(temb)
        temb_mod_ctx_arm_s = self.ctx_arm_single_modulation(temb)
        temb_mod_ctx_grip_s = self.ctx_grip_single_modulation(temb)
        temb_mod_arm_s = self.arm_single_modulation(temb)
        temb_mod_grip_s = self.grip_single_modulation(temb)

        # Build attention mask
        # Sequence order: [ctx_vlm, ctx_grip, ctx_arm, target_grip, target_arm]
        joint_attention_kwargs = {}
        if hist_actions_mask is not None or vlm_attention_mask is not None:
            vlm_len = vlm_features.shape[1]
            arm_hist_len = arm_hist.shape[1]
            grip_hist_len = grip_hist.shape[1]
            arm_target_len = arm_target.shape[1]
            grip_target_len = grip_target.shape[1]

            if vlm_attention_mask is not None:
                vlm_mask = vlm_attention_mask[:, :-1].bool()
            else:
                vlm_mask = torch.ones(b, vlm_len, dtype=torch.bool, device=target_actions.device)

            if hist_actions_mask is not None:
                # Split hist mask into arm/grip halves
                hist_mask_bool = hist_actions_mask.bool()
                arm_hist_mask = hist_mask_bool[:, :num_hist_chunks]
                grip_hist_mask = hist_mask_bool[:, num_hist_chunks:]
            else:
                arm_hist_mask = torch.ones(b, arm_hist_len, dtype=torch.bool, device=target_actions.device)
                grip_hist_mask = torch.ones(b, grip_hist_len, dtype=torch.bool, device=target_actions.device)

            # Randomly drop valid history action tokens during training to prevent overfitting
            if self.training and self.config.hist_action_token_drop_rate > 0.0:
                drop_rate = self.config.hist_action_token_drop_rate
                arm_keep = torch.rand(b, arm_hist_len, device=target_actions.device) >= drop_rate
                grip_keep = torch.rand(b, grip_hist_len, device=target_actions.device) >= drop_rate
                arm_hist_mask = arm_hist_mask & arm_keep
                grip_hist_mask = grip_hist_mask & grip_keep

            grip_target_mask = torch.ones(b, grip_target_len, dtype=torch.bool, device=target_actions.device)
            arm_target_mask = torch.ones(b, arm_target_len, dtype=torch.bool, device=target_actions.device)

            full_mask = torch.cat([vlm_mask, grip_hist_mask, arm_hist_mask,
                                   grip_target_mask, arm_target_mask], dim=1)
            # SDPA boolean mask convention: True = attend, False = ignore
            # full_mask: True = valid token, so pass directly (no inversion)
            joint_attention_kwargs['attention_mask'] = full_mask

        # RoPE
        all_rope = self._prepare_rope_emb(
            vlm_len=vlm_features.shape[1],
            arm_hist_len=arm_hist.shape[1],
            grip_hist_len=grip_hist.shape[1],
            arm_target_len=arm_target.shape[1],
            grip_target_len=grip_target.shape[1],
            device=target_actions.device,
            dtype=target_actions.dtype,
        )

        # Double-stream blocks
        for block in self.double_blocks:
            vlm_features, arm_hist, grip_hist, arm_target, grip_target = block(
                ctx_vlm_hidden=vlm_features,
                ctx_arm_hidden=arm_hist,
                ctx_grip_hidden=grip_hist,
                arm_hidden=arm_target,
                grip_hidden=grip_target,
                temb_mod_ctx_vlm=temb_mod_ctx_vlm_d,
                temb_mod_ctx_arm=temb_mod_ctx_arm_d,
                temb_mod_ctx_grip=temb_mod_ctx_grip_d,
                temb_mod_arm=temb_mod_arm_d,
                temb_mod_grip=temb_mod_grip_d,
                image_rotary_emb=all_rope,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Single-stream blocks
        for block in self.single_blocks:
            vlm_features, arm_hist, grip_hist, arm_target, grip_target = block(
                ctx_vlm_hidden=vlm_features,
                ctx_arm_hidden=arm_hist,
                ctx_grip_hidden=grip_hist,
                arm_hidden=arm_target,
                grip_hidden=grip_target,
                temb_mod_ctx_vlm=temb_mod_ctx_vlm_s,
                temb_mod_ctx_arm=temb_mod_ctx_arm_s,
                temb_mod_ctx_grip=temb_mod_ctx_grip_s,
                temb_mod_arm=temb_mod_arm_s,
                temb_mod_grip=temb_mod_grip_s,
                image_rotary_emb=all_rope,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Output: only target streams
        if return_chunks:
            return torch.cat([arm_target, grip_target], dim=1)  # [arm, grip] format
        else:
            arm_actions = self.arm_out_proj(arm_target)
            grip_logits = self.gripper_out_proj(grip_target)
            arm_actions = arm_actions.view(b, num_target_chunks * self.config.action_chunk_size, self.config.arm_dim)
            grip_logits = grip_logits.view(b, num_target_chunks * self.config.action_chunk_size, self.config.gripper_dim)

            out_actions = torch.zeros(b, num_target_chunks * self.config.action_chunk_size, self.config.action_dim,
                                      device=target_actions.device, dtype=target_actions.dtype)
            arm_indices = [i for i in range(self.config.action_dim) if i not in list(self.config.gripper_dim_indices_abs)]
            out_actions[:, :, arm_indices] = arm_actions
            out_actions[:, :, list(self.config.gripper_dim_indices_abs)] = grip_logits
            return out_actions

# ----------------------------------------------------------------------
# 3. Main Model & Policy Wrapping
# ----------------------------------------------------------------------
class LoLAPytorch(nn.Module):
    """结合了特征提取、编码和 DiT 的核心包装类"""
    def __init__(self, config: LoLAConfig):
        super().__init__()
        self.config = config
        self.vlm_bridge = LolaVLMFeatureExtractor(config)
        self.action_encoder = LolaActionEncoder(config)
        self.dit = LoLADiT(config)
        self.gradient_checkpointing_enabled = False
        self._checkpoint_fn = torch.utils.checkpoint.checkpoint
        self._checkpoint_fn_kwargs = {"use_reentrant": False, "preserve_rng_state": False}

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        logger.info("Enabled gradient checkpointing for LoLAPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        logger.info("Disabled gradient checkpointing for LoLAPytorch model")

    def set_deepspeed_checkpointing(self):
        """Switch to DeepSpeed activation checkpointing. Called by trainer during DeepSpeed setup."""
        import deepspeed
        self._checkpoint_fn = deepspeed.checkpointing.non_reentrant_checkpoint
        self._checkpoint_fn_kwargs = {}
        self.dit._checkpoint_fn = self._checkpoint_fn
        self.dit._checkpoint_fn_kwargs = self._checkpoint_fn_kwargs

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return self._checkpoint_fn(func, *args, **kwargs, **self._checkpoint_fn_kwargs)
        return func(*args, **kwargs)

    def forward(self, hidden_states_all_layers, input_ids, hist_actions, target_actions,
                hist_actions_mask=None, vlm_attention_mask=None, time=None, noise=None):
        """
        训练时的前向传播，实现 x-pred + v-loss 的 Flow Matching

        前向流程:
        - 网络输出: 模型参数化为直接预测干净的动作本体 (x-pred)
        - 损失监督: 采用预测流速与真实流速的均方差计算损失 (v-loss)
        """
        b = target_actions.shape[0]
        device = target_actions.device

        # 1. 获取基础特征
        vlm_features, empty_emb = self.vlm_bridge(hidden_states_all_layers)
        hist_chunks = self.action_encoder(hist_actions)
        target_chunks = self.action_encoder(target_actions)

        # 处理 hist_actions_mask（如果提供）
        hist_chunks_mask = None
        if hist_actions_mask is not None:
            # hist_actions_mask: [B, hist_seq_len]
            # 需要将 mask 转换为 chunk 级别
            # 每个 chunk 包含 action_chunk_size 个 action
            # chunk mask = 该 chunk 内是否有任何有效 action
            chunk_size = self.config.action_chunk_size
            seq_len = hist_actions_mask.shape[1]

            # Pad seq_len 到 chunk_size 的整数倍
            remainder = seq_len % chunk_size
            if remainder != 0:
                pad_len = chunk_size - remainder
                hist_actions_mask = F.pad(hist_actions_mask, (0, pad_len), value=0)
                seq_len = hist_actions_mask.shape[1]

            # Reshape 并聚合：[B, num_chunks, chunk_size] -> [B, num_chunks]
            num_chunks = seq_len // chunk_size
            hist_actions_mask_reshaped = hist_actions_mask.view(b, num_chunks, chunk_size)
            # 一个 chunk 有效如果其中任何一个 action 有效
            hist_chunks_mask = hist_actions_mask_reshaped.any(dim=2).float()  # [B, num_chunks]
            # Dual-Token: double the mask for arm+gripper token pairs
            hist_chunks_mask = torch.cat([hist_chunks_mask, hist_chunks_mask], dim=1)  # [B, 2*num_chunks]

        # 确保 dtype 一致性 (DeepSpeed BF16 训练时需要)
        target_dtype = vlm_features.dtype
        hist_chunks = hist_chunks.to(target_dtype)
        target_chunks = target_chunks.to(target_dtype)

        # 2. Flow Matching: 加噪
        # x_t = t * noise + (1 - t) * x_0
        if noise is None:
            noise = torch.randn_like(target_chunks)
        if time is None:
            dist = torch.distributions.Beta(self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta)
            time = dist.sample((b,)).to(device)

        # 确保 time 和 noise 的 dtype 与 target_chunks 一致 (DeepSpeed BF16 训练时需要)
        time = time.to(target_dtype)
        noise = noise.to(target_dtype)

        t_expand = time[:, None, None]
        x_t = t_expand * noise + (1 - t_expand) * target_chunks

        # Ground Truth Flow: v = noise - x_0
        u_t = noise - target_chunks

        # 3. DiT 前向传播，预测干净的 x_0 (x-pred)
        # 注意：DiT 在 chunk 空间操作，输入输出都是 [B, num_chunks, dit_hidden_size]
        pred_x0_chunks = self.dit(
            x_t, hist_chunks, vlm_features, empty_emb, time,
            hist_actions_mask=hist_chunks_mask,
            vlm_attention_mask=vlm_attention_mask,
            return_chunks=True,
            use_gradient_checkpointing=self.gradient_checkpointing_enabled and self.training,
        )
        
        # 4. v-loss: unchanged, operates on all doubled tokens in hidden space
        t_expand_clamped = t_expand.clamp(min=1e-5)
        v_pred = (x_t - pred_x0_chunks) / t_expand_clamped
        v_loss = F.mse_loss(v_pred, u_t, reduction="none")
        v_loss_mean = v_loss.mean()

        # 5. Split chunk features into arm and gripper
        num_chunks = pred_x0_chunks.shape[1] // 2
        pred_x0_arm = pred_x0_chunks[:, :num_chunks, :]
        pred_x0_gripper = pred_x0_chunks[:, num_chunks:, :]

        # 6. Arm: decode via arm_out_proj, compute Huber loss
        pred_arm = self.dit.arm_out_proj(pred_x0_arm)
        pred_arm = pred_arm.view(b, num_chunks * self.config.action_chunk_size, self.config.arm_dim)

        # 7. Gripper: decode via gripper_out_proj, compute BCE loss
        pred_gripper_logits = self.dit.gripper_out_proj(pred_x0_gripper)
        pred_gripper_logits = pred_gripper_logits.view(b, num_chunks * self.config.action_chunk_size, self.config.gripper_dim)

        # Match target lengths
        min_len = min(target_actions.shape[1], pred_arm.shape[1])
        arm_indices = [i for i in range(self.config.action_dim) if i not in list(self.config.gripper_dim_indices_abs)]
        gripper_indices = list(self.config.gripper_dim_indices_abs)

        target_arm = target_actions[:, :min_len, arm_indices]
        target_gripper = target_actions[:, :min_len, gripper_indices]
        pred_arm_matched = pred_arm[:, :min_len, :]
        pred_gripper_logits_matched = pred_gripper_logits[:, :min_len, :]

        # Arm: Huber loss (Smooth L1, robust to outliers)
        arm_loss = F.huber_loss(pred_arm_matched, target_arm, reduction="none")
        arm_loss_mean = arm_loss.mean()
        arm_loss_per_dim = arm_loss.mean(dim=(0, 1))

        # Gripper: BCE loss (binary classification, {-1,1} -> {0,1})
        target_gripper_01 = (target_gripper > 0).float()
        gripper_loss = F.binary_cross_entropy_with_logits(pred_gripper_logits_matched, target_gripper_01)

        # 8. Combine losses
        action_loss_weight = getattr(self.config, 'action_loss_weight', 1.0)
        gripper_loss_weight = getattr(self.config, 'gripper_loss_weight', 1.0)
        total_loss = v_loss_mean + action_loss_weight * arm_loss_mean + gripper_loss_weight * gripper_loss

        return {
            "total_loss": total_loss,
            "v_loss": v_loss_mean,
            "arm_loss": arm_loss_mean,
            "gripper_loss": gripper_loss,
            "arm_loss_per_dim": arm_loss_per_dim,
        }

    @torch.no_grad()
    def sample_actions(self, hidden_states_all_layers, hist_actions, hist_actions_mask=None):
        """推理阶段：欧拉积分去噪 (Dual-Token 版本)"""
        b = hist_actions.shape[0]
        device = hist_actions.device

        vlm_features, empty_emb = self.vlm_bridge(hidden_states_all_layers)
        hist_chunks = self.action_encoder(hist_actions)

        # Process hist_actions_mask and double it for dual-token
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
            # Double mask for arm+gripper token pairs
            hist_chunks_mask = torch.cat([hist_chunks_mask, hist_chunks_mask], dim=1)

        # Noise shape doubled for arm+gripper tokens
        predict_chunks_len = self.config.pred_chunk_size // self.config.action_chunk_size
        noise_shape = (b, predict_chunks_len * 2, self.config.dit_hidden_size)
        x_t = torch.randn(noise_shape, device=device, dtype=empty_emb.dtype)

        dt = -1.0 / self.config.num_inference_steps
        time = torch.tensor(1.0, device=device, dtype=torch.float32)

        while time >= -dt / 2:
            expanded_time = time.expand(b)
            pred_x0_chunks = self.dit(
                target_actions=x_t,
                hist_actions=hist_chunks,
                vlm_features=vlm_features,
                empty_emb=empty_emb,
                timestep=expanded_time,
                hist_actions_mask=hist_chunks_mask,
                return_chunks=True,
                use_gradient_checkpointing=False,
            )

            t_expand = time.clamp(min=1e-5)
            v_pred = (x_t - pred_x0_chunks) / t_expand
            x_t = x_t + dt * v_pred
            time = time + dt

        # Dual-head decode + gripper discretization
        num_chunks = pred_x0_chunks.shape[1] // 2
        pred_x0_arm = pred_x0_chunks[:, :num_chunks, :]
        pred_x0_gripper = pred_x0_chunks[:, num_chunks:, :]

        # Arm: continuous regression
        pred_arm = self.dit.arm_out_proj(pred_x0_arm).view(b, -1, self.config.arm_dim)

        # Gripper: sigmoid threshold -> {-1, 1}
        pred_gripper_logits = self.dit.gripper_out_proj(pred_x0_gripper).view(b, -1, self.config.gripper_dim)
        pred_gripper_probs = torch.sigmoid(pred_gripper_logits)
        pred_gripper_binary = (pred_gripper_probs > self.config.gripper_threshold).float()
        pred_gripper = (pred_gripper_binary - 0.5) * 2.0

        # Reassemble into original action_dim ordering
        actions = torch.zeros(b, pred_arm.shape[1], self.config.action_dim, device=device, dtype=pred_arm.dtype)
        arm_indices = [i for i in range(self.config.action_dim) if i not in list(self.config.gripper_dim_indices_abs)]
        actions[:, :, arm_indices] = pred_arm
        actions[:, :, list(self.config.gripper_dim_indices_abs)] = pred_gripper.to(actions.dtype)
        return actions

class LoLAPolicy(PreTrainedPolicy):
    """适配 LeRobot 的 Policy API"""
    config_class = LoLAConfig
    name = "lola"

    def __init__(self, config: LoLAConfig):
        super().__init__(config)
        self.config = config
        
        # 设置 dtype - 将字符串转换为 torch.dtype
        if isinstance(config.dtype, str):
            self._dtype = getattr(torch, config.dtype)
        else:
            self._dtype = config.dtype
        
        # 设置 device - 对于分布式训练，延迟设备分配让框架管理
        # 先设置为 meta device，后续由 Lightning 策略处理
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心模型 (DiT, Action Encoder, VLM Bridge)
        self.model = LoLAPytorch(config)
        
        # VLM 加载策略：
        # 对于分布式训练（DeepSpeed/FSDP），VLM 需要特殊处理
        # 方案：先加载到 CPU，然后让分布式策略管理设备分配
        if self.config.vlm_path is not None:
            self.vlm = Qwen3_5Model.from_pretrained(
                self.config.vlm_path,
                torch_dtype=self._dtype,
                device_map=None,  # 不自动分配，让分布式策略管理
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation="sdpa",
            )
        else:
            self.vlm = Qwen3_5Model.from_pretrained(
                self.config.vlm_model_name,
                torch_dtype=self._dtype,
                device_map=None,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )

        # Remove unused VLM parameters to fix DDP find_unused_parameters=False error.
        # LoLA only extracts hidden_states from layers 8/16/24, so layers 24-31,
        # final norm, and lm_head are dead branches with zero gradients.
        # Qwen3_5Model (vs Qwen3_5ForConditionalGeneration) already eliminates lm_head.
        last_extract_layer = max(self.config.vlm_extract_layers)
        lang_model = self.vlm.language_model
        for i in range(len(lang_model.layers) - 1, last_extract_layer - 1, -1):
            del lang_model.layers[i]
        lang_model.norm = nn.Identity()

        self.model.to(self._dtype)

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if config.train_vlm:
                self.vlm.gradient_checkpointing_enable()

        # 初始化动作队列
        self._action_queue = deque(maxlen=self.config.action_chunk_size * 5)

        # VLM forward mode selection:
        # - "hook": uses forward hooks to capture only 3 needed layers (memory-efficient,
        #   ~30MB). Works with FSDP but NOT with torch.compile (hooks are Python-level
        #   side effects that Dynamo cannot propagate across graph breaks).
        # - "output_hidden_states": uses output_hidden_states=True to get all 33 layers
        #   from VLM output, then selects 3 (~330MB peak). Compatible with both FSDP
        #   and torch.compile because hidden_states are tensor-level model returns.
        # - "split": manually iterates decoder layers, NOT compatible with FSDP
        #   (bypasses _pre_forward_unshard, causing size-0 parameter errors).
        if config.compile_model:
            import warnings
            warnings.warn(
                "torch.compile + FSDP is incompatible with hook-based hidden state capture "
                "(hooks are Python side effects that Dynamo cannot propagate). "
                "Switching VLM forward mode from 'hook' to 'output_hidden_states'. "
                "This materializes all 33 VLM layers (~330MB peak) instead of 3 (~30MB), "
                "but the impact is negligible relative to B200's 171GB reserved memory.",
                UserWarning,
                stacklevel=2,
            )
            self._vlm_forward_mode = "output_hidden_states"
        else:
            self._vlm_forward_mode = "hook"

        # Hook infrastructure: only actively used for hook mode
        if self._vlm_forward_mode == "hook":
            self._captured_hidden_states: Dict[int, torch.Tensor] = {}
            self._hook_handles: List = []
            self._in_vlm_forward: bool = False
            self._register_vlm_hooks()
        else:
            self._captured_hidden_states: Dict[int, torch.Tensor] = {}
            self._hook_handles: List = []
            self._in_vlm_forward: bool = False

    def _register_vlm_hooks(self):
        """在 VLM 的指定 decoder 层上注册 forward hook，仅捕获所需的 hidden states。

        替代 output_hidden_states=True（会物化所有 33 层），仅在需要的 3 层上
        注册 hook，减少 30 个无用张量的分配和内存压力。

        层索引映射: hidden_states[8] = layers[7] 的输出 (因为 hidden_states[0] 是 embedding)
        """
        for extract_layer_idx in self.config.vlm_extract_layers:
            decoder_layer_idx = extract_layer_idx - 1
            decoder_layer = self.vlm.language_model.layers[decoder_layer_idx]

            def make_hook(eidx):
                def hook_fn(module, input, output):
                    if not self._in_vlm_forward:
                        return
                    # Qwen3_5DecoderLayer.forward() 返回单个 tensor (非 tuple)
                    self._captured_hidden_states[eidx] = output
                return hook_fn

            handle = decoder_layer.register_forward_hook(make_hook(extract_layer_idx))
            self._hook_handles.append(handle)

    def _remove_vlm_hooks(self):
        """移除所有 VLM forward hooks并清理状态。"""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._captured_hidden_states = {}

    def _move_to_device(self, device: torch.device):
        """将模型移动到指定设备（供 Lightning 策略调用）"""
        self._device = device
        self.model = self.model.to(device)
        self.vlm = self.vlm.to(device)
        return self
    
    @property
    def device(self) -> torch.device:
        """返回模型所在设备"""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """返回模型数据类型"""
        return self._dtype

    def get_optim_params(self) -> dict:
        """返回所有可训练参数，包括 VLM 和 DiT 模型"""
        # 返回所有参数（包括 VLM 和 model）
        return self.parameters()
        
    def reset(self):
        """每当环境重置时清空动作队列"""
        self._action_queue = deque(maxlen=self.config.action_chunk_size * 5) # 假设缓存最多 50 步

    # =========================================================
    # 数据准备环节 (Prepare & Preprocess)
    # 对齐 LeRobot (例如 pi0) 的批处理结构
    # =========================================================
    def prepare_hist_actions(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 batch 中提取历史动作和对应的 mask。

        支持多种数据源（按优先级）：
        1. hist_actions_full: LoLADataset 提供的完整历史action（含padding）
        2. hist_actions: 直接提供的历史动作
        3. observation.state: 观测状态作为历史动作
        4. 全零占位：fallback

        Returns:
            hist_actions: [B, SeqLen, ActionDim] 历史动作
            hist_actions_mask: [B, SeqLen] 历史动作mask (1=有效, 0=padding), None表示全部有效
        """
        # 优先使用 LoLADataset 提供的完整历史action
        if "hist_actions_full" in batch:
            hist_actions = batch["hist_actions_full"]
            # 确保是3D张量 [B, SeqLen, ActionDim]
            if hist_actions.ndim == 2:
                hist_actions = hist_actions.unsqueeze(0)

            # 提取对应的 mask
            hist_actions_mask = batch.get("hist_actions_mask", None)
            if hist_actions_mask is not None:
                # 转换为 float (1.0=有效, 0.0=padding)
                hist_actions_mask = hist_actions_mask.float()
            else:
                # 如果没有 mask，创建全 1 的 mask
                hist_actions_mask = torch.ones(
                    hist_actions.shape[0], hist_actions.shape[1],
                    dtype=torch.float32, device=hist_actions.device
                )

            return hist_actions, hist_actions_mask

        elif "hist_actions" in batch:
            # 支持直接提供的历史动作
            hist_actions = batch["hist_actions"]
            if hist_actions.ndim == 2:
                hist_actions = hist_actions.unsqueeze(1)
            # 没有 mask，返回 None 表示全部有效
            return hist_actions, None

        elif "observation.state" in batch:
            hist_actions = batch["observation.state"]
            # 保证满足 [B, SeqLen, ActionDim]
            if hist_actions.ndim == 2:
                hist_actions = hist_actions.unsqueeze(1)
            # 没有 mask，返回 None 表示全部有效
            return hist_actions, None

        else:
            # Fallback：如果没有历史动作，使用当前 batch 中对应设备的全零张量占位
            # 优先使用 "action" 键，如果没有则尝试其他键获取 batch size
            if "action" in batch:
                b = batch["action"].shape[0]
            elif "target_actions" in batch:
                b = batch["target_actions"].shape[0]
            elif "input_ids" in batch:
                b = batch["input_ids"].shape[0]
            else:
                raise KeyError("Cannot determine batch size: no 'action', 'target_actions', or 'input_ids' in batch")
            hist_actions = torch.zeros((b, self.config.action_chunk_size, self.config.action_dim), device=self.device, dtype=self.dtype)
            return hist_actions, None

    def prepare_target_actions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从 batch 中提取目标动作"""
        # LeRobot 默认动作目标 key 为 "action"
        actions = batch["action"]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        return actions

    def prepare_vlm_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        处理和准备 VLM (Qwen3.5) 的输入特征。

        通过 forward hook 仅捕获指定层 (8, 16, 24) 的 hidden_states，
        替代 output_hidden_states=True（会物化所有 33 层但只使用 3 层）。

        Args:
            batch: 包含输入数据的字典，可能包含以下键:
                - "input_ids": 文本 token IDs [B, seq_len]
                - "observation.language_tokens": 语言指令 tokens (备选)
                - "pixel_values": 图像像素值 (用于视觉输入)
                - "image_grid_thw": 图像网格信息 (Qwen3.5 视觉模型需要)
                - "attention_mask": 注意力掩码

        Returns:
            hidden_states_all_layers: Dict[int, Tensor] 仅包含所需层的 hidden_states
            input_ids: 输入 token IDs
        """
        # 1. 提取 input_ids
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
        elif "observation.language.tokens" in batch:
            input_ids = batch["observation.language.tokens"]
        elif "observation.language_tokens" in batch:
            input_ids = batch["observation.language_tokens"]
        else:
            # 如果没有文本输入，使用 empty_token_id 作为占位
            b = batch["action"].shape[0]
            input_ids = torch.full((b, 1), self.config.empty_token_id, dtype=torch.long, device=self.device)

        # 2. 提取视觉输入（如果有）
        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        attention_mask = batch.get("attention_mask", None) or batch.get("observation.language.attention_mask", None)

        # 3. 调用 Qwen3.5 获取 hidden_states
        # 如果 batch 中已经提供了预计算的 hidden_states，直接使用
        if "hidden_states_all_layers" in batch:
            raw = batch["hidden_states_all_layers"]
            if isinstance(raw, dict):
                hidden_states_all_layers = raw
            else:
                # Legacy tuple format: 仅提取所需层，转为 dict
                hidden_states_all_layers = {i: raw[i] for i in self.config.vlm_extract_layers}
        else:
            # 端到端调用 Qwen3.5 模型
            if self._vlm_forward_mode == "split":
                # Split forward: not compatible with FSDP (bypasses _pre_forward_unshard)
                hidden_states_all_layers = self._vlm_split_forward(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )
            elif self._vlm_forward_mode == "output_hidden_states":
                # output_hidden_states mode: compatible with torch.compile + FSDP.
                # Uses VLM's native output_hidden_states=True to get all 33 layers as
                # tensor-level model output, then selects only the 3 needed layers.
                # Dynamo can properly track tensor returns (not Python side effects).
                forward_kwargs = {
                    "input_ids": input_ids,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                if pixel_values is not None:
                    forward_kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    forward_kwargs["image_grid_thw"] = image_grid_thw
                if attention_mask is not None:
                    forward_kwargs["attention_mask"] = attention_mask

                if not self.config.train_vlm:
                    with torch.no_grad():
                        vlm_output = self.vlm(**forward_kwargs)
                else:
                    vlm_output = self.vlm(**forward_kwargs)
                hidden_states_all_layers = {
                    i: vlm_output.hidden_states[i] for i in self.config.vlm_extract_layers
                }
            else:
                # Hook mode: memory-efficient (only 3 layers), but NOT compatible with
                # torch.compile (Python side effects). Works with FSDP alone.
                forward_kwargs = {
                    "input_ids": input_ids,
                    "output_hidden_states": False,  # 不物化所有 33 层，改用 hook 捕获
                    "return_dict": True,
                }

                # 添加视觉输入（如果有）
                if pixel_values is not None:
                    forward_kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    forward_kwargs["image_grid_thw"] = image_grid_thw
                if attention_mask is not None:
                    forward_kwargs["attention_mask"] = attention_mask

                # 激活 hook 捕获
                self._captured_hidden_states = {}
                self._in_vlm_forward = True
                try:
                    if not self.config.train_vlm:
                        with torch.no_grad():
                            self.vlm(**forward_kwargs)
                    else:
                        self.vlm(**forward_kwargs)
                finally:
                    self._in_vlm_forward = False

                hidden_states_all_layers = self._captured_hidden_states

        return hidden_states_all_layers, input_ids

    def _vlm_split_forward(self, input_ids, pixel_values=None, image_grid_thw=None, attention_mask=None):
        """Split VLM forward into segments at extract layers, avoiding hooks.

        Instead of registering forward hooks on decoder layers (which causes graph breaks
        in torch.compile), we manually run the VLM in segments:
        1. Vision encoder + embedding → inputs_embeds (same as Qwen3_5Model.forward)
        2. Layers [0, extract[0]) → capture hidden at extract[0]
        3. Layers [extract[0], extract[1]) → capture hidden at extract[1]
        ...
        N. Layers [extract[-1], num_layers) + final norm

        This makes the entire VLM forward visible to torch.compile as a single graph.
        """
        vlm_model = self.vlm  # Qwen3_5Model
        lang_model = vlm_model.language_model  # Qwen3_5TextModel

        # Step 1: Embed inputs (same as Qwen3_5Model.forward)
        inputs_embeds = vlm_model.get_input_embeddings()(input_ids)

        # Inject visual embeddings — inlined from Qwen3_5Model.forward to avoid
        # get_image_features() which uses .tolist() / torch.split and causes
        # graph breaks in torch.compile. Also enables compiling vision blocks.
        if pixel_values is not None:
            visual = vlm_model.visual
            pixel_values = pixel_values.to(dtype=visual.dtype)

            # Run vision encoder forward (Qwen3_5VisionModel.forward inlined)
            hidden_states = visual.patch_embed(pixel_values)
            pos_embeds = visual.fast_pos_embed_interpolate(image_grid_thw)
            hidden_states = hidden_states + pos_embeds
            rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)

            seq_len_v, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len_v, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len_v, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings_v = (emb.cos(), emb.sin())

            cu_seqlens = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            for blk in visual.blocks:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings_v,
                )

            # Merge and scatter image embeddings into text
            # visual.merger output: (total_visual_tokens, out_hidden_size)
            image_embeds = visual.merger(hidden_states)

            # Per-image split + cat (equivalent to get_image_features logic,
            # but using torch.split with a tensor instead of .tolist())
            merge_size = visual.spatial_merge_size ** 2
            split_sizes = image_grid_thw.prod(-1) // merge_size
            image_embeds_list = torch.split(image_embeds, split_sizes.tolist())
            image_embeds_cat = torch.cat(image_embeds_list, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            # Scatter into text embeddings (same as Qwen3_5Model.forward)
            image_mask, _ = vlm_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_list
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        # Step 2: Compute position_ids using Qwen3_5Model's method (handles 3D mrope for vision)
        position_ids = vlm_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        # Step 3: Compute position embeddings and causal mask (same as Qwen3_5TextModel.forward)
        batch_size, seq_len = inputs_embeds.shape[:2]
        cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)

        # Handle position_ids dimensions
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        position_embeddings = lang_model.rotary_emb(inputs_embeds, position_ids)

        # text_position_ids for causal mask
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
        else:
            text_position_ids = position_ids[0]

        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=lang_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        # Qwen3.5 uses alternating linear_attention / full_attention layers.
        # Linear attention (DeltaNet) layers need a separate mask.
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None

        # Step 4: Run decoder layers in segments, capturing at extract points
        hidden_states = inputs_embeds
        extract_layers = sorted(self.config.vlm_extract_layers)
        num_layers = len(lang_model.layers)
        # Segment boundaries: [0, extract[0], extract[1], ..., extract[-1], num_layers]
        boundaries = [0] + [e for e in extract_layers] + [num_layers]

        captured = {}
        for seg_idx in range(len(boundaries) - 1):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]

            for layer_idx in range(start, end):
                layer = lang_model.layers[layer_idx]
                layer_mask = linear_attn_mask if layer.layer_type == "linear_attention" else causal_mask
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                )

            # After running layers [start, end), the output is from layer end-1
            # which corresponds to hidden_states[end] (0-indexed: embedding=0, layer0=1, ...)
            if end in extract_layers:
                captured[end] = hidden_states

        return captured

    # =========================================================
    # 训练和推理接口
    # =========================================================
    def forward(self, batch: Dict[str, torch.Tensor], compute_per_dim: bool = False) -> Tuple[torch.Tensor, dict]:
        """训练过程的前向传播，计算 Flow Matching Loss"""
        hist_actions, hist_actions_mask = self.prepare_hist_actions(batch)
        target_actions = self.prepare_target_actions(batch)
        hidden_states_all_layers, input_ids = self.prepare_vlm_inputs(batch)

        # Extract VLM attention_mask for DiT (to fix vlm_mask bug)
        vlm_attention_mask = batch.get("attention_mask", None) or batch.get("observation.language.attention_mask", None)

        # 转换为正确的精度
        hist_actions = hist_actions.to(self.dtype)
        target_actions = target_actions.to(self.dtype)
        if hist_actions_mask is not None:
            hist_actions_mask = hist_actions_mask.to(self.dtype)
        # 将 hidden_states_all_layers 也转换为正确的 dtype (解决 BF16 训练时的 dtype 不匹配问题)
        hidden_states_all_layers = {k: v.to(self.dtype) for k, v in hidden_states_all_layers.items()}

        losses = self.model(
            hidden_states_all_layers=hidden_states_all_layers,
            input_ids=input_ids,
            hist_actions=hist_actions,
            target_actions=target_actions,
            hist_actions_mask=hist_actions_mask,
            vlm_attention_mask=vlm_attention_mask,
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
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """推理阶段：预测一整段 Action Chunk"""
        self.model.eval()

        hist_actions, hist_actions_mask = self.prepare_hist_actions(batch)
        hist_actions = hist_actions.to(self.dtype)
        if hist_actions_mask is not None:
            hist_actions_mask = hist_actions_mask.to(self.dtype)
        hidden_states_all_layers, input_ids = self.prepare_vlm_inputs(batch)

        actions = self.model.sample_actions(
            hidden_states_all_layers=hidden_states_all_layers,
            hist_actions=hist_actions,
            hist_actions_mask=hist_actions_mask,
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """推理阶段：根据环境观测选取单步 Action (使用 action queue 缓存机制)"""
        if len(self._action_queue) == 0:
            # 如果动作缓存为空，则进行一次大规模的 chunk 预测
            actions = self.predict_action_chunk(batch) # [B, Seq_Len, Action_Dim]
            
            # 将预测出的一串动作按照批次放入队列
            # LeRobot 在评估时通常 batch_size = 1，所以可以直接按顺序塞入
            for i in range(actions.shape[1]):
                self._action_queue.append(actions[:, i, :])
                
        return self._action_queue.popleft()