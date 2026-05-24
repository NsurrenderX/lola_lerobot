以下是下一版LoLA的**可执行修改路线图**，按**依赖顺序**与**风险等级**分为四个阶段。每个阶段包含**修改内容**、**理论依据**、**代码要点**与**验收标准**。

---

# LoLA 模型修改总方案

## 核心原则（贯穿全部阶段）

1. **Flow Matching 必须在 Bottleneck 子空间中进行**：Noise 与 Target 共享 Decoder，禁止在 1024D 空间用各向同性高斯噪声。
2. **v-loss 是配角，rollout 成功率是主角**：接受 v-loss "良性恶化"（0.1–0.3），不以压低 v-loss 为目标。
3. **BF16 下绝不使用 Zero Init**：避免深层 Bottleneck 死锁。
4. **Encoder 与 DiT 解耦**：Encoder 负责"压缩到紧凑流形"，DiT 负责"在紧凑流形上做 Flow Matching"。

---

## Phase 1：核心架构重构（解决维度错配）

**目标**：让 Flow Matching 从 1024D 冗余空间降维到 256D/128D 子空间，消除 768 维零压制。

### 1.1 ActionEncoder：引入显式 Bottleneck + 编解码接口

**修改内容**：
- 将 `arm_proj` 拆分为 `arm_enc1` → `arm_enc2` (Bottleneck) → `arm_dec`。
- `gripper_proj` 同理，但 Bottleneck 更窄（128D）。
- 暴露 `encode()` / `decode()` 方法，供外部复用 Decoder 生成噪声。

**代码要点**：

```python
class LolaActionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.dit_hidden_size
        btl = getattr(config, 'action_bottleneck_dim', 256)
        grip_btl = max(btl // 2, 64)

        # Arm: chunk -> hidden -> 256 -> hidden
        self.arm_enc1 = nn.Sequential(
            nn.Linear(self.chunk_size * self.arm_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-6), nn.SiLU(),
        )
        self.arm_enc2 = nn.Sequential(
            nn.Linear(hidden, btl),
            nn.LayerNorm(btl, eps=1e-6), nn.SiLU(),
        )
        self.arm_dec = nn.Sequential(
            nn.Linear(btl, hidden),
            nn.LayerNorm(hidden, eps=1e-6),
        )
        # 正交初始化 + 范数守恒（防止 BF16 噪声放大）
        nn.init.orthogonal_(self.arm_dec[0].weight)
        self.arm_dec[0].weight.data *= math.sqrt(btl / hidden)
        nn.init.zeros_(self.arm_dec[0].bias)

        # Gripper: chunk -> hidden -> 128 -> hidden
        self.grip_enc1 = nn.Sequential(...)
        self.grip_enc2 = nn.Sequential(...)
        self.grip_dec = nn.Sequential(...)
        nn.init.orthogonal_(self.grip_dec[0].weight)
        self.grip_dec[0].weight.data *= math.sqrt(grip_btl / hidden)
        nn.init.zeros_(self.grip_dec[0].bias)

    def encode(self, arm_chunked, grip_chunked):
        arm_latent = self.arm_enc2(self.arm_enc1(arm_chunked))   # [B,N,256]
        grip_latent = self.grip_enc2(self.grip_enc1(grip_chunked)) # [B,N,128]
        return arm_latent, grip_latent

    def decode(self, arm_latent, grip_latent):
        arm_tokens = self.arm_dec(arm_latent) + self.arm_modality_emb
        grip_tokens = self.grip_dec(grip_latent) + self.gripper_modality_emb
        return torch.cat([arm_tokens, grip_tokens], dim=1)

    def forward(self, actions, return_latent=False):
        # ... chunking + split ...
        arm_latent, grip_latent = self.encode(arm_chunked, grip_chunked)
        tokens = self.decode(arm_latent, grip_latent)
        if return_latent:
            return tokens, arm_latent, grip_latent
        return tokens
```

**验收标准**：`decode()` 被成功复用于 noise 生成（见 1.2）。

---

### 1.2 Flow Matching：Noise 在 Latent 空间生成

**修改内容**：在 `LoLAPytorch.forward` 中，noise 不再在 1024D token 空间生成，而是在 `arm_latent` / `grip_latent` 同维度生成，再通过 `action_encoder.decode()` 映射。

**代码要点**：

```python
# LoLAPytorch.forward
target_actions_f32 = target_actions.float()
target_tokens, arm_latent, grip_latent = self.action_encoder(target_actions_f32, return_latent=True)

# Noise 在 latent 空间生成，复用 decoder（关键！）
noise_arm_latent = torch.randn_like(arm_latent)   # [B,N,256]
noise_grip_latent = torch.randn_like(grip_latent) # [B,N,128]
noise_tokens = self.action_encoder.decode(noise_arm_latent, noise_grip_latent)

# 回到 BF16 给 DiT
target_tokens = target_tokens.to(self.dtype)
noise_tokens = noise_tokens.to(self.dtype)
noise = noise_tokens  # 复用变量名

# Flow Matching（保持原有逻辑）
t = dist.sample((b,)).to(device).to(self.dtype)
x_t = (1 - t) * target_tokens + t * noise_tokens
u_t = noise_tokens - target_tokens  # 或 noise_latent - target_latent 视参数化而定
```

**理论依据**：确保 $x_t$ 全程严格位于 Decoder 的列空间 $\mathcal{C}(W_{\text{dec}})$ 中，消除 768 维正交补空间的零压制任务。

**验收标准**：训练初期 v-loss 不再呈现"先暴跌后反弹"的虚假收敛，而是平稳下降。

---

### 1.3 StateEncoder (Unified 模式)：Split 后各自 Bottleneck

**修改内容**：`state_proj` 先统一提取到 `2*hidden`（无 Bottleneck），split 为 arm/grip 后，各自独立 Bottleneck。

**代码要点**：

```python
class LolaStateEncoder(nn.Module):
    def __init__(self, config):
        ...
        # Stage 1: 统一过完备提取（无 bottleneck）
        self.state_proj = nn.Sequential(
            nn.Linear(self.chunk_size * self.state_dim, 2 * hidden),
            nn.LayerNorm(2 * hidden, eps=1e-6), nn.SiLU(),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.LayerNorm(2 * hidden, eps=1e-6),
        )
        # Stage 2: Split 后各自独立压缩
        self.arm_bottleneck = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.LayerNorm(bottleneck256, eps=1e-6), nn.SiLU(),
            nn.Linear(bottleneck, hidden), nn.LayerNorm(hidden, eps=1e-6),
        )
        self.grip_bottleneck = nn.Sequential(
            nn.Linear(hidden, bottleneck_gripper), nn.LayerNorm(bottleneck_gripper, eps=1e-6), nn.SiLU(),
            nn.Linear(bottleneck_gripper, hidden), nn.LayerNorm(hidden, eps=1e-6),
        )

    def forward(self, states):
        projected = self.state_proj(self._pad_and_chunk(states, self.state_dim))
        arm_part = projected[..., :hidden]
        grip_part = projected[..., hidden:]
        arm_tokens = self.arm_bottleneck(arm_part) + self.arm_ctx_state_emb
        grip_tokens = self.grip_bottleneck(grip_part) + self.grip_ctx_state_emb
        return torch.cat([arm_tokens, grip_tokens], dim=1)
```

**验收标准**：StateEncoder 输出与 ActionEncoder 输出统计特性一致（均值/方差同级）。

以下是针对两个遗漏点的补充方案。这是**显式 Latent Flow Matching** 的完整实现：DiT 仍在 1024D 运行以兼容 5-Stream Attention，但 Flow Matching 的加噪、去噪与 v-loss 计算全部发生在 Bottleneck 维度（Arm 256D / Grip 128D）。

---

## Phase 1 补充 A：DiT 的 Latent Flow Matching 改造

### 架构改动概览

```
ActionEncoder.encode()          StateEncoder()
       │                              │
       ▼                              ▼
[arm_latent 256D]              [hist_tokens 1024D]
[grip_latent 128D]                    │
       │                              │
       ├─── Flow Matching ────────────┤
       │    (在 latent 空间生成噪声)   │
       ▼                              ▼
[arm_latent_to_dit]            (直接输入 DiT)
[grip_latent_to_dit]                 │
       │                             │
       └──► [DiT 5-Stream @ 1024D] ◄─┘
                 │
       [arm_dit_to_latent 256D]
       [grip_dit_to_latent 128D]
                 │
                 ▼
         v-loss @ latent
                 │
       ActionEncoder.decode()
                 │
       arm_out_proj / gripper_out_proj
```

### 1.1 LoLAPytorch：新增异构 Adapter

在 `__init__` 中增加四个轻量 Adapter（无 shortcut，正常初始化）：

```python
class LoLAPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden = config.dit_hidden_size
        arm_btl = getattr(config, 'action_bottleneck_dim', 256)
        grip_btl = max(arm_btl // 2, 64)  # 128

        # ActionEncoder 与 StateEncoder（FP32，见前文）
        self.action_encoder = LolaActionEncoder(config).float()
        self.state_encoder = LolaStateEncoder(config).float()

        # DiT 核心（保持 1024D，与 VLM/history 兼容）
        self.dit = LoLADiT(config)

        # --- Latent <-> DiT 异构 Adapter（纯线性，安全稳定） ---
        # Arm: 256 <-> 1024
        self.arm_latent_to_dit = nn.Linear(arm_btl, hidden)
        self.arm_dit_to_latent = nn.Linear(hidden, arm_btl)

        # Grip: 128 <-> 1024
        self.grip_latent_to_dit = nn.Linear(grip_btl, hidden)
        self.grip_dit_to_latent = nn.Linear(hidden, grip_btl)

        # 初始化建议：使用正交初始化
        nn.init.orthogonal_(self.arm_latent_to_dit.weight)
        nn.init.orthogonal_(self.arm_dit_to_latent.weight)
        nn.init.orthogonal_(self.grip_latent_to_dit.weight)
        nn.init.orthogonal_(self.grip_dit_to_latent.weight)
```

### 1.2 ActionEncoder 接口调整

`encode()` 与 `decode()` 必须显式分离，且支持异构 latent：

```python
class LolaActionEncoder(nn.Module):
    def __init__(self, config):
        # ... 同前文，arm_btl=256, grip_btl=128 ...

    def encode(self, arm_chunked, grip_chunked):
        """返回异构 latent"""
        arm_latent = self.arm_enc2(self.arm_enc1(arm_chunked))   # [B, N, 256]
        grip_latent = self.grip_enc2(self.grip_enc1(grip_chunked)) # [B, N, 128]
        return arm_latent, grip_latent

    def decode(self, arm_latent, grip_latent):
        """从异构 latent 重建 1024D token"""
        arm_tokens = self.arm_dec(arm_latent) + self.arm_modality_emb
        grip_tokens = self.grip_dec(grip_latent) + self.gripper_modality_emb
        return torch.cat([arm_tokens, grip_tokens], dim=1)  # [B, 2N, 1024]

    def forward(self, actions, return_latent=False):
        # ... chunking ...
        arm_latent, grip_latent = self.encode(arm_chunked, grip_chunked)
        tokens = self.decode(arm_latent, grip_latent)
        if return_latent:
            return tokens, arm_latent, grip_latent
        return tokens
```

### 1.3 LoLAPytorch.forward：完整的 Latent FM 流程

```python
def forward(self, hidden_states_all_layers, input_ids, hist_actions, target_actions,
            hist_actions_mask=None, vlm_attention_mask=None, time=None, noise=None):
    b = target_actions.shape[0]
    device = target_actions.device

    # 1. VLM 特征（保持 BF16）
    vlm_features, empty_emb = self.vlm_bridge(hidden_states_all_layers)

    # 2. 历史动作/状态（FP32 下编码，输出 1024D，直接给 DiT）
    hist_actions_f32 = hist_actions.float()
    if self.state_encoder is not None:
        hist_chunks = self.state_encoder(hist_actions_f32)
    else:
        hist_chunks = self.action_encoder(hist_actions_f32)
    hist_chunks = hist_chunks.to(self.dtype)

    # 3. 目标动作编码到异构 Latent（FP32）
    target_actions_f32 = target_actions.float()
    # 需要复用 chunking 逻辑，或者让 action_encoder 暴露 chunking 方法
    # 这里假设 action_encoder.forward 内部处理 chunking
    _, target_arm_latent, target_grip_latent = self.action_encoder(target_actions_f32, return_latent=True)

    # 4. Flow Matching：在异构 Latent 空间生成噪声
    noise_arm_latent = torch.randn_like(target_arm_latent) if noise is None else noise[0]
    noise_grip_latent = torch.randn_like(target_grip_latent) if noise is None else noise[1]

    if time is None:
        dist = torch.distributions.Beta(self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta)
        time = dist.sample((b,)).to(device)

    # 确保 FP32 计算
    time_f32 = time.float()
    target_arm_latent = target_arm_latent.float()
    target_grip_latent = target_grip_latent.float()
    noise_arm_latent = noise_arm_latent.float()
    noise_grip_latent = noise_grip_latent.float()

    t_expand = time_f32[:, None, None]
    z_t_arm = (1 - t_expand) * target_arm_latent + t_expand * noise_arm_latent
    z_t_grip = (1 - t_expand) * target_grip_latent + t_expand * noise_grip_latent

    # 5. 升维到 DiT（异构 Adapter）
    z_t_arm_dit = self.arm_latent_to_dit(z_t_arm.to(self.dtype))
    z_t_grip_dit = self.grip_latent_to_dit(z_t_grip.to(self.dtype))
    z_t_dit = torch.cat([z_t_arm_dit, z_t_grip_dit], dim=1)  # [B, 2N, 1024]

    # 6. DiT 前向（1024D，与 VLM/history 交互）
    pred_z0_dit = self.dit(
        target_actions=z_t_dit,
        hist_actions=hist_chunks,
        vlm_features=vlm_features,
        empty_emb=empty_emb,
        timestep=time,
        hist_actions_mask=hist_chunks_mask,
        vlm_attention_mask=vlm_attention_mask,
        return_chunks=True,
        use_gradient_checkpointing=self.gradient_checkpointing_enabled and self.training,
    )  # [B, 2N, 1024]

    # 7. 降维回异构 Latent
    num_chunks = pred_z0_dit.shape[1] // 2
    pred_z0_arm_dit = pred_z0_dit[:, :num_chunks, :]
    pred_z0_grip_dit = pred_z0_dit[:, num_chunks:, :]

    pred_z0_arm_latent = self.arm_dit_to_latent(pred_z0_arm_dit).float()
    pred_z0_grip_latent = self.grip_dit_to_latent(pred_z0_grip_dit).float()

    # 8. v-loss：在异构 Latent 空间计算（关键！）
    t_expand_clamped = t_expand.clamp(min=1e-5)
    v_pred_arm = (z_t_arm - pred_z0_arm_latent) / t_expand_clamped
    v_pred_grip = (z_t_grip - pred_z0_grip_latent) / t_expand_clamped

    v_loss_arm = F.mse_loss(v_pred_arm, noise_arm_latent - target_arm_latent, reduction="none")
    v_loss_grip = F.mse_loss(v_pred_grip, noise_grip_latent - target_grip_latent, reduction="none")
    v_loss = (v_loss_arm.mean() + v_loss_grip.mean()) / 2.0

    # 9. 解码到 1024D token，进入原有输出头
    pred_z0_tokens = self.action_encoder.decode(pred_z0_arm_latent, pred_z0_grip_latent)
    pred_z0_tokens = pred_z0_tokens.to(self.dtype)

    # 10. 后续与原有代码完全一致（arm_out_proj / gripper_out_proj / Huber / BCE）
    num_target_chunks = pred_z0_tokens.shape[1] // 2
    pred_x0_arm = pred_z0_tokens[:, :num_target_chunks, :]
    pred_x0_grip = pred_z0_tokens[:, num_target_chunks:, :]

    pred_arm = self.dit.arm_out_proj(pred_x0_arm)
    pred_grip_logits = self.dit.gripper_out_proj(pred_x0_grip)
    # ... reshape, match target length, Huber, BCE ...
```

### 1.4 sample_actions 的同步修改

推理阶段的 Euler 积分必须在 **Latent 空间**进行，而非 1024D：

```python
@torch.no_grad()
def sample_actions(self, hidden_states_all_layers, hist_actions, hist_actions_mask=None):
    b = hist_actions.shape[0]
    device = hist_actions.device

    vlm_features, empty_emb = self.vlm_bridge(hidden_states_all_layers)
    hist_chunks = ...  # 同前，1024D

    # 1. Noise 在异构 Latent 空间初始化
    predict_chunks_len = self.config.pred_chunk_size // self.config.action_chunk_size
    z_t_arm = torch.randn(b, predict_chunks_len, 256, device=device, dtype=torch.float32)
    z_t_grip = torch.randn(b, predict_chunks_len, 128, device=device, dtype=torch.float32)

    dt = -1.0 / self.config.num_inference_steps
    time = torch.tensor(1.0, device=device, dtype=torch.float32)

    while time >= -dt / 2:
        expanded_time = time.expand(b)

        # 升维到 DiT
        z_t_arm_dit = self.arm_latent_to_dit(z_t_arm.to(self.dtype))
        z_t_grip_dit = self.grip_latent_to_dit(z_t_grip.to(self.dtype))
        z_t_dit = torch.cat([z_t_arm_dit, z_t_grip_dit], dim=1)

        pred_z0_dit = self.dit(
            target_actions=z_t_dit,
            hist_actions=hist_chunks,
            vlm_features=vlm_features,
            empty_emb=empty_emb,
            timestep=expanded_time,
            return_chunks=True,
            use_gradient_checkpointing=False,
        )

        # 降维回 Latent
        num = pred_z0_dit.shape[1] // 2
        pred_z0_arm_latent = self.arm_dit_to_latent(pred_z0_dit[:, :num, :]).float()
        pred_z0_grip_latent = self.grip_dit_to_latent(pred_z0_dit[:, num:, :]).float()

        # Euler 积分（在 Latent 空间！）
        t_expand = time.clamp(min=1e-5)
        v_pred_arm = (z_t_arm - pred_z0_arm_latent) / t_expand
        v_pred_grip = (z_t_grip - pred_z0_grip_latent) / t_expand
        z_t_arm = z_t_arm + dt * v_pred_arm
        z_t_grip = z_t_grip + dt * v_pred_grip
        time = time + dt

    # 2. 最终解码到动作
    pred_z0_tokens = self.action_encoder.decode(z_t_arm, z_t_grip)
    # ... split + arm_out_proj + gripper_out_proj（同前）...
```

---

## Phase 1 补充 B：StateEncoder Split 模式改造

Split 模式的 StateEncoder 与 ActionEncoder 完全对称：各自独立 MLP + Bottleneck + Decoder。

```python
class LolaStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size = config.action_chunk_size
        self.state_dim = config.state_dim
        self.config = config
        self.mode = config.state_encoder_mode
        hidden = config.dit_hidden_size
        arm_btl = getattr(config, 'state_bottleneck_dim', 256)  # 可与 action 共用配置
        grip_btl = max(arm_btl // 2, 64)

        if self.mode == "unified":
            # ... 前文已给（统一提取后 split + 各自 bottleneck）...

        else:  # "separated"
            num_gripper = len(config.gripper_dim_indices_abs)
            self.state_gripper_indices = tuple(range(self.state_dim - num_gripper, self.state_dim))
            self.state_arm_dim = self.state_dim - num_gripper
            self.state_gripper_dim = num_gripper

            # --- Arm State: chunk -> hidden -> 256 -> hidden ---
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
            self.arm_state_dec = nn.Sequential(
                nn.Linear(arm_btl, hidden),
                nn.LayerNorm(hidden, eps=1e-6),
            )
            # 正交初始化 + 范数守恒
            nn.init.orthogonal_(self.arm_state_dec[0].weight)
            self.arm_state_dec[0].weight.data *= math.sqrt(arm_btl / hidden)
            nn.init.zeros_(self.arm_state_dec[0].bias)

            # --- Grip State: chunk -> hidden -> 128 -> hidden ---
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
            self.grip_state_dec = nn.Sequential(
                nn.Linear(grip_btl, hidden),
                nn.LayerNorm(hidden, eps=1e-6),
            )
            nn.init.orthogonal_(self.grip_state_dec[0].weight)
            self.grip_state_dec[0].weight.data *= math.sqrt(grip_btl / hidden)
            nn.init.zeros_(self.grip_state_dec[0].bias)

        self.arm_ctx_state_emb = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.grip_ctx_state_emb = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

    def _pad_and_chunk(self, states, dim_size):
        # ... 原有逻辑 ...

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.mode == "unified":
            # ... 前文逻辑 ...

        else:  # "separated"
            gripper_indices = list(self.state_gripper_indices)
            all_indices = list(range(states.shape[-1]))
            arm_indices = [i for i in all_indices if i not in gripper_indices]

            arm_states = states[..., arm_indices]
            grip_states = states[..., gripper_indices]

            arm_chunked = self._pad_and_chunk(arm_states, self.state_arm_dim)
            grip_chunked = self._pad_and_chunk(grip_states, self.state_gripper_dim)

            # Arm: 过完备提取 -> Bottleneck -> 回到 hidden
            arm_latent = self.arm_state_enc2(self.arm_state_enc1(arm_chunked))
            arm_tokens = self.arm_state_dec(arm_latent) + self.arm_ctx_state_emb

            # Grip: 过完备提取 -> Bottleneck -> 回到 hidden
            grip_latent = self.grip_state_enc2(self.grip_state_enc1(grip_chunked))
            grip_tokens = self.grip_state_dec(grip_latent) + self.grip_ctx_state_emb

            return torch.cat([arm_tokens, grip_tokens], dim=1)  # [B, 2*num_chunks, hidden]
```

---

## 关键约束重申

| 检查点 | 必须满足 |
|--------|---------|
| **Noise 与 Target 共享 Decoder** | `noise_latent` 通过 `action_encoder.decode()` 映射到 1024D，禁止独立投影层 |
| **v-loss 计算维度** | 必须在 `arm_dit_to_latent` / `grip_dit_to_latent` 之后，即 256D / 128D |
| **DiT 内部维度** | 保持 1024D，Adapter 负责维度转换，不改动 5-Stream Block |
| **推理 Euler 积分** | 全程在 Latent 空间（`z_t_arm`, `z_t_grip`），最后一步才 decode |
| **StateEncoder 输出** | 直接输出 1024D token（通过 dec），不经过额外 Adapter，作为 DiT condition |

---

## 修改后预期行为

- **v-loss**：不再追求 0.003，预期收敛到 **0.05–0.15**（因为 256D 空间紧凑，无冗余维度作弊）
- **arm-loss / gripper-loss**：与之前持平或略降（DiT 容量从"抹零任务"中释放）
- **rollout 成功率**：从 83.2% 向 **85–88%** 推进（DiT 全部参数服务动作流形）
- **训练稳定性**：FP32 Encoder + 2k warm-up + t 截断下，无 BF16 死锁风险

---

## Phase 2：数值稳定性与训练策略（解决 BF16 + 极小 LR）

### 2.1 Encoder 强制 FP32 隔离

**修改内容**：ActionEncoder 与 StateEncoder 整体用 FP32 计算，仅在进入 DiT 前转 BF16。

**代码要点**：

```python
# LoLAPytorch.__init__
self.action_encoder = LolaActionEncoder(config).float()
self.state_encoder = LolaStateEncoder(config).float()

# LoLAPytorch.forward
target_actions_f32 = target_actions.float()
hist_actions_f32 = hist_actions.float()
```

**理论依据**：Bottleneck 深层 MLP（SiLU + LayerNorm）在 BF16 下累积误差大，FP32 隔离保底线。

---

### 2.2 Warm-up 与 t 采样策略

**修改内容**：
- Warm-up 时长：**10%**（占总步数 10%）。
- 曲线：**Cosine warm-up**（比 linear 平滑）。
- Warm-up 期间 t 采样截断到 **[0.1, 0.9]**，避免 $t \to 0$ 时 $\frac{1}{t^2}$ 梯度爆炸。

**代码要点**：

```python
# 学习率调度
def lr_lambda(step):
    if step < total_steps * 0.1:
        return 0.5 * (1 - math.cos(math.pi * step / (total_steps * 0.1)))
    else:
        # 后续接你的 decay 策略
        return decay_ratio

# t 采样样例（训练循环内）
if step < total_steps * 0.1:
    t_raw = torch.distributions.Beta(2, 2).sample((b,))
    t = t_raw * 0.8 + 0.1  # 截断 [0.1, 0.9]
else:
    t = torch.distributions.Beta(alpha, beta).sample((b,))
```

---

### 2.3 优化器参数分组与 LR 策略

**修改内容**：DiT 主体与 VLM 基线 LR，Encoder 适度放大（上限 2x）。

**设为可调config**：

```python
base_lr = 2e-5

optimizer_params = [
    {"params": self.model.dit.parameters(), "lr": base_lr},
    {"params": self.model.vlm_bridge.parameters(), "lr": base_lr},
    
    # Encoder：配合 FP32，适度放大 1.5x（保守）或 2.0x（激进）
    # 理由：Bottleneck 参数量小，放大不会显著增加显存，但能加速深层收敛
    {"params": self.action_encoder.parameters(), "lr": base_lr * 1.5},
    {"params": self.state_encoder.parameters(), "lr": base_lr * 1.5},
    
    # VLM 微调（若开启）
    {"params": self.vlm.parameters(), "lr": base_lr if config.train_vlm else 0.0},
]
```

**注意**：若追求绝对稳定，Encoder LR 可设为 `base_lr * 1.0`（不放大），仅靠 FP32 隔离。引入 Bottleneck 主要是解决 loss 冲突，而非颠覆流程。

---

## Phase 3：Loss 权重与监控策略（基于实验证据）

### 3.1 Loss Weight 最终配置

**修改内容**：以 rollout 成功率为锚，接受 v-loss "良性恶化"。

```python
v_loss_weight = 1.0          # 保持基线，不放大
action_loss_weight = 10.0    # 已验证有效（83.2% vs 77.6%）
gripper_loss_weight = 1.0    # 已验证有效

total_loss = v_loss_weight * v_loss \
           + action_loss_weight * arm_loss \
           + gripper_loss_weight * gripper_loss
```

**理论依据**：v-loss 衡量 1024D token 重建，包含大量与物理动作无关的冗余维度；action loss 直接锚定 70D 动作精度，是任务友好性的硬约束。当 v-loss 被压低到 0.003 时，DiT 在 954D 冗余空间中模仿 ActionEncoder 的噪声风格，动作子空间反而被架空。
。

---

### 3.2 监控指标调整

**停止以 v-loss 作为模型选择标准**。改为：

| 优先级 | 指标 | 决策用途 |
|--------|------|---------|
| P0 | **Rollout Success Rate** | 唯一真理，决定 checkpoint 保存与 early stopping |
| P1 | **Val Arm Loss / Gripper Accuracy** | 辅助参考，注意 out_proj 的自适应作弊 |
| P2 | **v-loss** | 仅监控不发散（>1.0 报警），接受 0.1–0.3 区间 |
| P3 | **Latent Dead Units** | Bottleneck 引入后，观察 256D 中静默维度比例 |

**验收标准**：训练日志中 v-loss 稳定在 0.05–0.25，rollout 成功率突破 83.2% 并继续上升。

---

## Phase 4：可选进阶（后续优化）

### 4.1 Encoder 引入 1D Temporal CNN（激进选项）

**适用条件**：`chunk_size >= 10` 且需要更强时序先验。

**修改位置**：替换 `arm_enc1` / `grip_enc1` 的首层 MLP 为 1D CNN over `chunk_size`。

```python
self.arm_cnn = nn.Sequential(
    # 输入: (B, arm_dim, chunk_size)
    nn.Conv1d(self.arm_dim, hidden // 2, kernel_size=3, padding=1),
    nn.LayerNorm([hidden // 2, self.chunk_size]), nn.SiLU(),
    nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=1),
    nn.LayerNorm([hidden, self.chunk_size]), nn.SiLU(),
    nn.AdaptiveAvgPool1d(1),  # -> (B, hidden, 1)
)
# 后续接 Bottleneck（同 Phase 1）
```

**约束**：
- Noise 必须在 CNN 输出后的 `hidden` 或 `bottleneck` 空间生成，**严禁** `noise_raw → CNN`。
- 夹爪 Encoder 保持 MLP 或极浅 CNN（夹爪 chunk 内变化极少）。
- 不用 BatchNorm，用 LayerNorm over `[channels, length]`。

### 4.2 arm_out_proj 引入位置编码（替代 CNN）

**适用条件**：希望在输出端引入结构先验，但避免 CNN 的跨 chunk 耦合风险。



---

## 禁止清单（避免踩坑）

| 禁止项 | 原因 |
|--------|------|
| ❌ **Zero Init Decoder 最后一层** | BF16 + 极小 LR 下死锁，梯度永久截断 |
| ❌ **为 Noise 单独写独立投影层** | Noise 与 Target 落在不同子空间，768 维零压制卷土重来 |
| ❌ **在 arm_out_proj 中使用 CNN** | 破坏 chunk 独立性，引入未来信息泄漏，与 DiT Attention 冗余 |
| ❌ **试图压低 v-loss 到 0.003** | 会导致 DiT 在冗余维度作弊，rollout 下降 |
| ❌ **Encoder LR > 3x** | 移动靶问题，训练震荡 |
| ❌ **加 Shortcut（残差连接）** | 在当前架构下会让 Bottleneck 被架空，学不到紧凑流形 |

---

## 执行顺序总结

1. **立即执行 Phase 1**（架构重构）：这是解决维度错配的根本，所有后续优化依赖于此。
2. **同步执行 Phase 2**（数值策略）：FP32 隔离与 warm-up 策略可与 Phase 1 同时上线。
3. **上线 Phase 3**（Loss 权重）：将 action weight 拉到 10，gripper 拉到 1，接受 v-loss 恶化。
4. **验证后执行 Phase 4**（可选 CNN）：在 rollout 稳定突破 85% 后，再尝试 Encoder CNN 进一步压榨性能。