# **LoLA 架构重构指南：双流 Token 解耦与联合注意力机制**

## **核心设计动机 (Why we do this)**

当前模型将连续的 6 维手臂轨迹和离散的 1 维夹爪状态通过同一个 nn.Linear 压缩到了同一个 1536 维隐空间中。这导致在 DiT 的 MLP 层中发生了严重的**梯度污染**（夹爪二值化的阶跃梯度冲毁了平滑的手臂轨迹梯度）。  
**解决方案 (The "Dual-Token" Approach)：**

1. **输入解耦**：将 7 维切分为 6+1，分别用两个独立的 Linear 映射，加上独立的模态编码（Modality Embedding）以区分身份，然后在**序列维度**（Sequence Length）拼接。  
2. **时序耦合 (共享 RoPE)**：在拼接后的长序列中，让对应的手臂 Token 和夹爪 Token 共享完全相同的旋转位置编码（RoPE ID），使得 Self-Attention 认为它们发生在同一物理瞬间，实现完美参照。  
3. **输出解耦**：分离 MSE 损失（手臂）和 BCE 损失（夹爪），进行精准的梯度惩罚。

## **实施步骤 (修改 modeling_lola.py)**

### **步骤 1: 重构 Action Encoder**

找到 LolaActionEncoder 类，用以下代码完全替换它。这一步引入了模态特征，并把输入序列长度翻倍。  
class LolaActionEncoder(nn.Module):  
    """  
    Dual-Token Action Chunking: 将手臂和夹爪完全解耦，分别映射为独立的 Token，  
    并注入模态编码以区分身份，最后在 Sequence 维度拼接。  
    """  
    def __init__(self, config: LoLAConfig):  
        super().__init__()  
        self.chunk_size = config.action_chunk_size  
        self.action_dim = config.action_dim  # 默认应该是 7  
        self.arm_dim = self.action_dim - 1   # 6维  
        self.gripper_dim = 1                 # 1维  
          
        # 独立的手臂投影网络  
        self.arm_proj = nn.Sequential(  
            nn.Linear(self.chunk_size * self.arm_dim, config.dit_hidden_size),  
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),  
            nn.SiLU(),  
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)  
        )  
          
        # 独立的夹爪投影网络  
        self.gripper_proj = nn.Sequential(  
            nn.Linear(self.chunk_size * self.gripper_dim, config.dit_hidden_size),  
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),  
            nn.SiLU(),  
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size)  
        )

        # 模态编码 (Modality Embeddings)，区分相同时序下的 Arm 和 Gripper  
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
        # 分离手臂与夹爪动作  
        arm_actions = actions[..., :-1]  
        gripper_actions = actions[..., -1:]  
          
        # 分别进行 Chunking  
        arm_chunked = self._pad_and_chunk(arm_actions, self.arm_dim)  
        gripper_chunked = self._pad_and_chunk(gripper_actions, self.gripper_dim)  
          
        # 分别投影并加上各自的模态思想钢印  
        arm_tokens = self.arm_proj(arm_chunked) + self.arm_modality_emb  
        gripper_tokens = self.gripper_proj(gripper_chunked) + self.gripper_modality_emb  
          
        # 在序列维度 (dim=1) 拼接  
        # 此时输出长度变为原来的 2 倍: [B, 2 * num_chunks, 1536]  
        return torch.cat([arm_tokens, gripper_tokens], dim=1)

### **步骤 2: 分离 DiT 的输出头**

找到 LoLADiT.__init__ 方法的末尾，删除原来的 self.action_out_proj，替换为下面两个独立的解码头：  
        # ========== 新增：双输出解码头 ==========  
        # 1. 手臂回归头 (输出 6 维连续量)  
        self.arm_out_proj = nn.Sequential(  
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),  
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),  
            nn.SiLU(),  
            nn.Linear(config.dit_hidden_size, (config.action_dim - 1) * config.action_chunk_size)  
        )  
          
        # 2. 夹爪分类头 (输出 1 维 Logits)  
        self.gripper_out_proj = nn.Sequential(  
            nn.LayerNorm(config.dit_hidden_size, eps=1e-6),  
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),  
            nn.SiLU(),  
            nn.Linear(config.dit_hidden_size, 1 * config.action_chunk_size)  
        )

### **步骤 3: 改造 RoPE（核心：强制时间戳对齐）**

找到 LoLADiT._prepare_rope_emb 方法，在构建 hist_coords 和 target_coords 时，适配翻倍的序列长度，并让前后半段**共享 Pos ID**。  
    def _prepare_rope_emb(self, vlm_len: int, hist_len: int, target_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        # ... (前面的代码保持不变，直到构建 4D 坐标) ...  
          
        # 1. 显式构建 4D 坐标  
        vlm_coords = torch.zeros((vlm_len, 4), dtype=torch.long, device=device)  
        vlm_coords[:, 0] = 0  
        vlm_coords[:, 3] = torch.arange(vlm_len, device=device)  
          
        # History Actions (T=1)  
        # 注意：hist_len 现在是翻倍的，前半段是 arm，后半段是 gripper  
        hist_num_chunks = hist_len // 2  
        hist_coords = torch.zeros((hist_len, 4), dtype=torch.long, device=device)  
        hist_coords[:, 0] = 1  
        # 关键修改：让前后两半的 chunk_id 一模一样，共享 RoPE 旋转角度  
        hist_coords[:, 3] = torch.cat([torch.arange(hist_num_chunks, device=device),  
                                       torch.arange(hist_num_chunks, device=device)], dim=0)  
          
        # Target Actions (T=2)  
        target_num_chunks = target_len // 2  
        target_coords = torch.zeros((target_len, 4), dtype=torch.long, device=device)  
        target_coords[:, 0] = 2  
        # 关键修改：同样让 Target 的前后两半共享相同的 Pos ID  
        target_coords[:, 3] = torch.cat([torch.arange(target_num_chunks, device=device),  
                                         torch.arange(target_num_chunks, device=device)], dim=0)  
          
        # ... (后续 compute_multiaxis_rope 代码保持不变) ...

### **步骤 4: 重写 Loss 计算（分离 Huber Loss 与 BCE Loss）**

找到 LoLAPytorch.forward。在把数据传入 self.dit 之前，我们需要修复一下历史动作的 Mask；在计算 Loss 时，我们需要分别结算。  
**在准备输入前 (修复 Mask 拼接)：**  
        # (在准备好 vlm_features, hist_chunks, target_chunks 后，进入 self.dit 之前)  
        # 修复 hist_actions_mask 以匹配翻倍的 hist_chunks 长度  
        if hist_actions_mask is not None:  
            hist_mask_bool = hist_actions_mask.bool()  
            # 复制一份拼接到后面  
            hist_actions_mask = torch.cat([hist_mask_bool, hist_mask_bool], dim=1)

**修改 Loss 结算：**  
        # 4. 流匹配损失：在全体 1536 维隐空间计算总 v-loss (确保隐空间不崩塌)  
        t_expand_clamped = t_expand.clamp(min=1e-5)  
        v_pred = (x_t - pred_x0_chunks) / t_expand_clamped  
        v_loss = F.mse_loss(v_pred, u_t, reduction="none")  
        v_loss_mean = v_loss.mean()  
          
        # 5. 将联合隐空间 Token 拆分回 Arm 和 Gripper  
        num_chunks = pred_x0_chunks.shape[1] // 2  
        pred_x0_arm = pred_x0_chunks[:, :num_chunks, :]  
        pred_x0_gripper = pred_x0_chunks[:, num_chunks:, :]  
          
        # 6. 独立解码  
        pred_arm = self.dit.arm_out_proj(pred_x0_arm)                    # [B, num_chunks, chunk_size * 6]  
        pred_gripper_logits = self.dit.gripper_out_proj(pred_x0_gripper) # [B, num_chunks, chunk_size * 1]  
          
        pred_arm = pred_arm.view(b, -1, self.config.action_dim - 1)  
        pred_gripper_logits = pred_gripper_logits.view(b, -1)

        # 获取对齐长度的真实标签  
        target_arm = target_actions[:, :pred_arm.shape[1], :-1]  
        target_gripper = target_actions[:, :pred_gripper_logits.shape[1], -1]  
          
        # 7. 分别计算损失  
        # 连续动作使用 Huber Loss (Smooth L1)，有效抑制多模态下的模糊化和异常尖峰  
        action_loss = F.huber_loss(pred_arm, target_arm, reduction="none")  
        action_loss_mean = action_loss.mean()  
          
        # 夹爪使用二元交叉熵 (BCE)。先将真实标签 {-1, 1} 转为 {0, 1} 格式。  
        target_gripper_01 = (target_gripper > 0).float()  
        gripper_loss = F.binary_cross_entropy_with_logits(pred_gripper_logits, target_gripper_01)  
          
        # 组合损失 (确保 config 里有对应的 weight 属性，或者写死为 1.0)  
        action_loss_weight = getattr(self.config, 'action_loss_weight', 1.0)  
        gripper_loss_weight = getattr(self.config, 'gripper_loss_weight', 1.0)  
          
        total_loss = v_loss_mean + action_loss_weight * action_loss_mean + gripper_loss_weight * gripper_loss

        return {  
            "total_loss": total_loss,  
            "v_loss": v_loss_mean,  
            "action_loss": action_loss_mean,  
            "gripper_loss": gripper_loss, # 加入日志监控，看着它暴降  
        }

### **步骤 5: 修改推理代码以处理二值化输出**

找到 LoLAPytorch.sample_actions。因为我们的解码机制变了，我们需要把它拼装回 [-1, 1] 的物理空间。  
    @torch.no_grad()  
    def sample_actions(self, hidden_states_all_layers, hist_actions, hist_actions_mask=None):  
        # ... (前置处理不变)  
          
        # 1. 修复推理时的 history mask  
        if hist_actions_mask is not None:  
            hist_mask_bool = hist_actions_mask.bool()  
            hist_actions_mask = torch.cat([hist_mask_bool, hist_mask_bool], dim=1)

        # 2. 噪音长度现在是原来的两倍 (前半段手臂，后半段夹爪)  
        predict_chunks_len = self.config.pred_chunk_size // self.config.action_chunk_size  
        noise_shape = (b, predict_chunks_len * 2, self.config.dit_hidden_size)  
        x_t = torch.randn(noise_shape, device=device, dtype=empty_emb.dtype)

        dt = -1.0 / self.config.num_inference_steps  
        time = torch.tensor(1.0, device=device, dtype=torch.float32)

        # ... (Euler 步进代码 while time >= -dt / 2: ... 完全保持不变) ...  
              
        # 3. 去噪结束后，分离解码并拼合物理 Action  
        num_chunks = pred_x0_chunks.shape[1] // 2  
        pred_x0_arm = pred_x0_chunks[:, :num_chunks, :]  
        pred_x0_gripper = pred_x0_chunks[:, num_chunks:, :]  
          
        # 手臂直接解码  
        pred_arm = self.dit.arm_out_proj(pred_x0_arm).view(b, -1, self.config.action_dim - 1)  
          
        # 夹爪先解码为 Logits，过 Sigmoid 算出概率，再强行截断为 -1 或 1  
        pred_gripper_logits = self.dit.gripper_out_proj(pred_x0_gripper).view(b, -1, 1)  
        pred_gripper = (torch.sigmoid(pred_gripper_logits) > 0.5).float() # [0, 1]  
        pred_gripper = (pred_gripper - 0.5) * 2.0                         # [-1, 1]  
          
        # 返回拼接后的完整 7 维动作，交给环境执行  
        return torch.cat([pred_arm, pred_gripper], dim=-1)

**💡 预期效果：**  
修改完成并重新启动训练后，应该能立即观察到 gripper_loss 的平滑下降，并在验证集上看到夹爪维度的 **Precision / Recall 双双突破 90%**，同时连续维度的 L1 Error 也会因摆脱梯度干扰而显著降低。