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
    Flux2TransformerBlock,
    Flux2SingleTransformerBlock,
    Flux2Modulation,
    )


from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.lola.configuration_lola import LoLAConfig

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

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
    """Action Chunking: 将 Chunk Size x Action Dim 的物理动作压缩为 Action Expert Dim Token"""
    def __init__(self, config: LoLAConfig):
        super().__init__()
        self.chunk_size = config.action_chunk_size
        self.action_dim = config.action_dim
        
        # 使用 MLP 并加入 LayerNorm，稳定不同物理量纲带来的方差偏移
        self.proj = nn.Sequential(
            nn.Linear(self.chunk_size * self.action_dim, config.dit_hidden_size),
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)
        )
        
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        b, seq_len, d = actions.shape
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            actions = F.pad(actions, (0, 0, 0, pad_len))
            seq_len += pad_len
            
        chunked = actions.view(b, seq_len // self.chunk_size, self.chunk_size * d)
        return self.proj(chunked)

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
# 2. Core LoLA DiT Modeling (800M Parameter Scale)
# ----------------------------------------------------------------------
class LoLADiT(nn.Module):
    def __init__(self, config: LoLAConfig):
        super().__init__()
        self.config = config
        self.cond_embedder = LolaConditionEmbedder(config)
        
        # FLUX.2 Modulation 层 
        self.double_stream_modulation_img = Flux2Modulation(config.dit_hidden_size, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(config.dit_hidden_size, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(config.dit_hidden_size, mod_param_sets=1, bias=False)

        # FLUX.2 Transformer Blocks 实例化
        attention_head_dim = config.dit_hidden_size // config.dit_num_heads
        
        self.double_blocks = nn.ModuleList([
            Flux2TransformerBlock(
                dim=config.dit_hidden_size,
                num_attention_heads=config.dit_num_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(config.dit_double_layers)
        ])
        
        self.single_blocks = nn.ModuleList([
            Flux2SingleTransformerBlock(
                dim=config.dit_hidden_size,
                num_attention_heads=config.dit_num_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(config.dit_single_layers)
        ])
        
        self.action_out_proj = nn.Sequential(
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.action_dim * config.action_chunk_size)
        )

    def _prepare_rope_emb(self, vlm_len: int, hist_len: int, target_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成多轴旋转位置编码 (Multi-axis RoPE)。
        构建 4D 坐标系: (T, H, W, L)，支持长上下文与 2D 视觉 Patch 的扩展。
        
        注意: apply_rotary_emb 在 diffusers 中的实现:
        - 输入 x 形状: [B, H, S, D]，D 是完整的 head_dim
        - cos/sin 形状: [S, D]，也必须是完整的 head_dim
        - 内部会将 x reshape 为 [B, H, S, D//2, 2]，然后 flatten 回 [B, H, S, D]
        - 最终乘法在 flatten 后的完整 D 维度上进行
        """
        head_dim = self.config.dit_hidden_size // self.config.dit_num_heads
        # apply_rotary_emb 期望 cos/sin 的维度是完整的 head_dim
        rope_dim = head_dim
        # 将 rope_dim 分配给 4 个轴 (T, H, W, L)
        axes_dims = (rope_dim // 4, rope_dim // 4, rope_dim // 4, rope_dim // 4)
        assert sum(axes_dims) == rope_dim, f"Axes dims sum must match rope_dim {rope_dim}"
        
        # 1. 显式构建 4D 坐标
        # VLM Features (T=0, 未来需区分 2D 图像时可激活 H, W)
        vlm_coords = torch.zeros((vlm_len, 4), dtype=torch.long, device=device)
        vlm_coords[:, 0] = 0
        vlm_coords[:, 3] = torch.arange(vlm_len, device=device)
        
        # History Actions (T=1, 仅限 1D，H=W=0)
        hist_coords = torch.zeros((hist_len, 4), dtype=torch.long, device=device)
        hist_coords[:, 0] = 1
        hist_coords[:, 3] = torch.arange(hist_len, device=device)
        
        # Target Actions (T=2, 仅限 1D，H=W=0)
        target_coords = torch.zeros((target_len, 4), dtype=torch.long, device=device)
        target_coords[:, 0] = 2
        target_coords[:, 3] = torch.arange(target_len, device=device)
        
        context_coords = torch.cat([vlm_coords, hist_coords], dim=0)
        all_coords = torch.cat([context_coords, target_coords], dim=0)
        
        def compute_multiaxis_rope(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """根据 4D 坐标独立计算各个轴的旋转频率，并拼接到 Head 维度"""
            freqs_list = []
            for i, dim in enumerate(axes_dims):
                pos = coords[:, i].float()
                
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
                freqs = torch.outer(pos, inv_freq) # [Seq_Len, dim // 2]
                freqs = freqs.repeat_interleave(2, dim=-1) # [Seq_Len, dim]
                freqs_list.append(freqs)
            
            # 将所有轴拼接到一起: [Seq_Len, rope_dim]
            emb = torch.cat(freqs_list, dim=-1)
            
            # apply_rotary_emb 期望的格式: (cos, sin)，维度为 [S, D]
            # 注意：不要在这里 unsqueeze，apply_rotary_emb 会内部处理
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
            return (cos, sin)

        context_rope = compute_multiaxis_rope(context_coords)
        target_rope = compute_multiaxis_rope(target_coords)
        all_rope = compute_multiaxis_rope(all_coords)
        
        return context_rope, target_rope, all_rope

    def forward(self, target_actions, hist_actions, vlm_features, empty_emb, timestep,
                hist_actions_mask=None, vlm_attention_mask=None, return_chunks: bool = False,
                use_gradient_checkpointing: bool = False):
        """
        Args:
            target_actions: [B, num_chunks, dit_hidden_size] - 加噪后的目标动作chunks
            hist_actions: [B, hist_chunks, dit_hidden_size] - 历史动作chunks
            vlm_features: [B, vlm_seq_len, dit_hidden_size] - VLM特征
            empty_emb: [B, dit_hidden_size] - 空token嵌入
            timestep: [B] - 时间步
            hist_actions_mask: [B, hist_seq_len] - 历史动作的mask，1表示有效，0表示padding
            return_chunks: 如果为True，返回chunk特征而非解码后的动作
        """
        temb = self.cond_embedder(timestep, empty_emb)

        temb_mod_img = self.double_stream_modulation_img(temb)
        temb_mod_txt = self.double_stream_modulation_txt(temb)
        temb_mod_single = self.single_stream_modulation(temb)

        context_features = torch.cat([vlm_features, hist_actions], dim=1)

        # 构建 attention_mask（如果提供了 mask 信息）
        # 注意：Flux2Attention 内部会拼接 encoder_hidden_states 和 hidden_states
        # 所以 mask 需要覆盖完整序列
        joint_attention_kwargs = {}
        if hist_actions_mask is not None or vlm_attention_mask is not None:
            vlm_len = vlm_features.shape[1]
            hist_len = hist_actions.shape[1]
            target_len = target_actions.shape[1]

            # vlm_attention_mask: [B, vlm_seq_len+1] (original VLM mask including empty token)
            # vlm_features excludes the empty token ([:, :-1, :]), so mask must also exclude it
            if vlm_attention_mask is not None:
                vlm_mask = vlm_attention_mask[:, :-1].bool()  # [B, vlm_len]
            else:
                vlm_mask = torch.ones(
                    vlm_features.shape[0], vlm_len,
                    dtype=torch.bool,
                    device=target_actions.device
                )

            if hist_actions_mask is not None:
                hist_mask_bool = hist_actions_mask.bool()
            else:
                hist_mask_bool = torch.ones(
                    hist_actions.shape[0], hist_len,
                    dtype=torch.bool,
                    device=target_actions.device
                )

            target_mask = torch.ones(
                target_actions.shape[0], target_len,
                dtype=torch.bool,
                device=target_actions.device
            )

            # 拼接: [B, full_seq_len]
            # True 表示该位置有效，False 表示 padding
            full_mask = torch.cat([vlm_mask, hist_mask_bool, target_mask], dim=1)

            # scaled_dot_product_attention 的 boolean mask:
            # True 表示该位置应该被 MASK OUT（不参与 attention）
            # False 表示该位置有效（参与 attention）
            attn_mask = ~full_mask

            joint_attention_kwargs['attention_mask'] = attn_mask

        # 获取多模态 RoPE 特征 - 修复：使用 .shape[1] 获取序列长度
        context_rope, target_rope, all_rope = self._prepare_rope_emb(
            vlm_len=vlm_features.shape[1],
            hist_len=hist_actions.shape[1],
            target_len=target_actions.shape[1],
            device=target_actions.device,
            dtype=target_actions.dtype
        )
        
        for block in self.double_blocks:
            if use_gradient_checkpointing:
                context_features, target_actions = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states=target_actions,
                    encoder_hidden_states=context_features,
                    temb_mod_img=temb_mod_img,
                    temb_mod_txt=temb_mod_txt,
                    image_rotary_emb=all_rope,
                    joint_attention_kwargs=joint_attention_kwargs,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                context_features, target_actions = block(
                    hidden_states=target_actions,
                    encoder_hidden_states=context_features,
                    temb_mod_img=temb_mod_img,
                    temb_mod_txt=temb_mod_txt,
                    image_rotary_emb=all_rope,  # 使用 all_rope 因为 attention 中会拼接 encoder 和 target
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        for block in self.single_blocks:
            if use_gradient_checkpointing:
                context_features, target_actions = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states=target_actions,
                    encoder_hidden_states=context_features,
                    temb_mod=temb_mod_single,
                    image_rotary_emb=all_rope,
                    joint_attention_kwargs=joint_attention_kwargs,
                    split_hidden_states=True,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                context_features, target_actions = block(
                    hidden_states=target_actions,
                    encoder_hidden_states=context_features,
                    temb_mod=temb_mod_single,
                    image_rotary_emb=all_rope,
                    joint_attention_kwargs=joint_attention_kwargs,
                    split_hidden_states=True
                )
            
        # 根据 return_chunks 决定返回 chunk 特征还是解码后的动作
        if return_chunks:
            # 返回 chunk 特征，用于训练时的 v-loss 计算
            return target_actions  # [B, num_chunks, dit_hidden_size]
        else:
            # 解码为具体动作
            out_actions = self.action_out_proj(target_actions)
            b, num_chunks, _ = out_actions.shape
            out_actions = out_actions.view(b, num_chunks * self.config.action_chunk_size, self.config.action_dim)
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

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        logger.info("Enabled gradient checkpointing for LoLAPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        logger.info("Disabled gradient checkpointing for LoLAPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
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
        
        # 4. 从预测的 x_0 推导预测的流速 v
        # Flow Matching 公式: x_t = t * noise + (1 - t) * x_0
        # 因此: v = noise - x_0
        # 或者等价地: v = (x_t - x_0) / t  (当 x_t 已知时)
        # 
        # 使用 (x_t - pred_x0) / t 形式，隐式引入 1/t^2 的 SNR 权重
        # 这使得模型在 t->0 时受到更严厉的惩罚，提升高精度抓取成功率
        t_expand_clamped = t_expand.clamp(min=1e-5)  # 避免 t 接近 0 时除零崩溃
        v_pred = (x_t - pred_x0_chunks) / t_expand_clamped
        
        # 5. 使用 v-loss 进行监督 (chunk 空间)
        v_loss = F.mse_loss(v_pred, u_t, reduction="none")
        
        # 6. 【关键修复】动作空间重构损失 - 确保 action_out_proj 被训练
        # 将预测的 chunk 特征解码到动作空间
        pred_actions = self.dit.action_out_proj(pred_x0_chunks)
        num_chunks = pred_x0_chunks.shape[1]
        pred_actions = pred_actions.view(b, num_chunks * self.config.action_chunk_size, self.config.action_dim)
        
        # 处理 target_actions 以匹配输出长度
        # target_actions: [B, action_seq_len, action_dim]
        target_action_len = target_actions.shape[1]
        pred_action_len = pred_actions.shape[1]
        
        if target_action_len >= pred_action_len:
            # 截取目标动作的前 pred_action_len 步
            target_actions_matched = target_actions[:, :pred_action_len, :]
        else:
            # 如果目标动作较短，对预测动作进行截取
            pred_actions = pred_actions[:, :target_action_len, :]
            target_actions_matched = target_actions
        
        # 计算动作空间的重构损失
        action_loss = F.mse_loss(pred_actions, target_actions_matched, reduction="none")
        
        # 7. 组合损失
        # action_loss_weight 控制 action_out_proj 训练的强度
        # 建议从配置中读取，默认为 1.0
        action_loss_weight = getattr(self.config, 'action_loss_weight', 1.0)
        total_loss = v_loss.mean() + action_loss_weight * action_loss.mean()
        
        return total_loss

    @torch.no_grad()
    def sample_actions(self, hidden_states_all_layers, hist_actions, hist_actions_mask=None):
        """
        推理阶段：欧拉积分去噪

        根据README，模型采用x-pred预测，在推理阶段可以完美消除欧拉积分在最后微小时间步的截断误差。
        Flow Matching ODE: dx/dt = v(x, t) = x_0 - noise
        欧拉积分: x_{t+dt} = x_t + dt * v(x_t, t)
        """
        b = hist_actions.shape[0]
        device = hist_actions.device

        # 修复：vlm_bridge.forward 只接受一个参数
        vlm_features, empty_emb = self.vlm_bridge(hidden_states_all_layers)
        hist_chunks = self.action_encoder(hist_actions)

        # 处理 hist_actions_mask（如果提供）
        hist_chunks_mask = None
        if hist_actions_mask is not None:
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
            hist_chunks_mask = hist_actions_mask_reshaped.any(dim=2).float()

        # 纯噪声起点 (此处假设预测长度与 target 相同，例如 5 个 Chunk 即 50 steps)
        # 具体需要预测多少长度可以在外部通过参数传入，此处固定为预测配置中约定的块数
        predict_chunks_len = self.config.pred_chunk_size // self.config.action_chunk_size
        noise_shape = (b, predict_chunks_len, self.config.dit_hidden_size)
        x_t = torch.randn(noise_shape, device=device, dtype=empty_emb.dtype)

        dt = -1.0 / self.config.num_inference_steps
        time = torch.tensor(1.0, device=device, dtype=torch.float32)

        # Euler 步进
        # 在 x-pred 设置下，我们需要先预测 x_0，然后计算 v = x_0 - x_t (等价于 noise - x_0 的反向)
        # 但更高效的做法是：直接使用预测的 x_0 来计算下一步
        while time >= -dt / 2:
            expanded_time = time.expand(b)

            # DiT 预测干净的 x_0 (x-pred)，返回 chunk 特征
            pred_x0_chunks = self.dit(
                target_actions=x_t,
                hist_actions=hist_chunks,
                vlm_features=vlm_features,
                empty_emb=empty_emb,
                timestep=expanded_time,
                hist_actions_mask=hist_chunks_mask,
                return_chunks=True,
                use_gradient_checkpointing=False,  # No checkpointing during inference
            )

            # 计算流速: v = noise - x_0，但噪声未知
            # 使用 Flow Matching 的另一种形式: v = (x_t - x_0) / t 当 t > 0
            # 但在推理时我们不需要显式计算 v，可以直接用 x-pred 的性质

            # 标准欧拉积分: x_{t+dt} = x_t + dt * v
            # 其中 v = x_0 - x_t (从 x_t = t*noise + (1-t)*x_0 推导)
            # 因此: x_{t+dt} = x_t + dt * (pred_x0 - x_t)
            # 这等价于向预测的干净样本移动
            t_expand = time.clamp(min=1e-5)
            # v = (pred_x0 - x_t) / t 不正确
            # 正确的推导: 从 x_t = t*noise + (1-t)*x_0
            # dx/dt = noise - x_0 = v
            # 所以 v_pred = (x_t - pred_x0) / t (当 t -> 0 时 pred_x0 -> x_0)
            v_pred = (x_t - pred_x0_chunks) / t_expand
            
            x_t = x_t + dt * v_pred
            time = time + dt
            
        # 返回解码到具体维度的物理 Action [B, Predict_Steps, Action_Dim]
        return self.dit.action_out_proj(pred_x0_chunks).view(b, -1, self.config.action_dim)

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
            self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.config.vlm_path,
                torch_dtype=self._dtype,
                device_map=None,  # 不自动分配，让分布式策略管理
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        else:
            self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.config.vlm_model_name,
                torch_dtype=self._dtype,
                device_map=None,
                low_cpu_mem_usage=True,
            )
        
        self.model.to(self._dtype)

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if config.train_vlm:
                self.vlm.gradient_checkpointing_enable()

        # 初始化动作队列
        self._action_queue = deque(maxlen=self.config.action_chunk_size * 5)

        # VLM forward hooks: 仅捕获需要的层 (8, 16, 24)，替代 output_hidden_states=True
        self._captured_hidden_states: Dict[int, torch.Tensor] = {}
        self._hook_handles: List = []
        self._in_vlm_forward: bool = False
        self._register_vlm_hooks()

    def _register_vlm_hooks(self):
        """在 VLM 的指定 decoder 层上注册 forward hook，仅捕获所需的 hidden states。

        替代 output_hidden_states=True（会物化所有 33 层），仅在需要的 3 层上
        注册 hook，减少 30 个无用张量的分配和内存压力。

        层索引映射: hidden_states[8] = layers[7] 的输出 (因为 hidden_states[0] 是 embedding)
        """
        for extract_layer_idx in self.config.vlm_extract_layers:
            decoder_layer_idx = extract_layer_idx - 1
            decoder_layer = self.vlm.model.language_model.layers[decoder_layer_idx]

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
            # 端到端调用 Qwen3.5 模型，仅通过 hook 捕获所需层
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

    # =========================================================
    # 训练和推理接口
    # =========================================================
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
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

        loss = losses.mean()
        loss_dict = {
            "loss": loss.item(),
        }
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