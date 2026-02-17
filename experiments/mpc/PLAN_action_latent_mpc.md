# Plan: DiT-Internal Action Latent 分离 (MoT Style) + Action Space MPC

> **方案更新**：从 IDM-External 方案升级为 DiT-Internal (MoT) 方案。
> 参考 LingBot-VA (arXiv:2601.21998) 的 Mixture-of-Transformers 架构。

---

## 0. 动机：为什么 DiT-Internal > IDM-External

### 当前 Vidarc Pipeline（两阶段，不端到端）

```
obs → [Wan DiT: 10 denoise steps] → z_video → [VAE decode] → RGB → [IDM: UNet+ResNet] → action
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~                 ~~~~~~~~~~~~~~~~~~~~~~~~
       ~2-5s (32层 transformer)        ~50ms                     ~5ms
```

**两个根本问题**：
1. **MPC 瓶颈**：MPC 每个 candidate 需要跑完整 Wan 生成（~2-5s），5个候选就是 10-25s
2. **信息损失**：IDM 从 pixel space 后验提取 action，丢失了 DiT 内部的丰富语义信息

### IDM-External 方案（旧方案）

```
训练单独的 Action Encoder + Dynamics Model 来绕过 Wan
问题：
  - 需要额外的 dynamics model（可能不准确）
  - Action latent 是从 pixel 后验提取的，不如 DiT 内部的 latent 丰富
  - 与视频生成完全解耦，无法利用 video-action 的联合信息
```

### DiT-Internal MoT 方案（新方案，LingBot-VA 已验证可行）

```
核心思路：
  action tokens 直接嵌入 DiT 的 token 序列中，
  与 video tokens 共享 self-attention（互相可以看到对方），
  但有独立的 FFN / projection / LayerNorm（各自的参数空间）。

优势：
  1. Action latent 来自生成过程本身，不是后验提取
  2. Action 和 video 通过 attention 互相影响，建模更准确
  3. 不需要单独的 IDM 模块
  4. 为 action-space MPC 提供天然的 latent space
  5. LingBot-VA 在 5B 模型上已验证此架构可行
```

---

## 1. 架构设计：WanModelCausal → WanModelCausalMoT

### 1.0 参数映射（LingBot-VA → Vidarc）

| 维度 | LingBot-VA (Wan2.2-5B) | Vidarc (Wan2.2-TI2V) | 备注 |
|------|----------------------|---------------------|------|
| d_v (video dim) | 3072 | 2048 | Vidarc 基于较小的模型 |
| d_a (action dim) | 768 | **512** | 保持 4:1 比例 |
| num_layers | 30 | 32 | 深度相近 |
| num_heads | ? | 16 | video stream |
| num_heads_a | ? | **8** | action stream (d_a/64=8) |
| ffn_dim_v | ? | 8192 | video FFN |
| ffn_dim_a | ? | **2048** | action FFN (4:1 比例) |
| action_dim | 30 (dual-arm) | **14** (single-arm) | RoboTwin 任务 |
| tau (actions/frame) | 4 | **4** | 与 VAE temporal stride 匹配 |
| patch_size | ? | (1,2,2) | 每帧 h_patches × w_patches tokens |

### 1.1 Action Token Embedding

```
raw action a_t ∈ R^{14}
  → MLP φ(a_t): Linear(14, 256) → GELU → Linear(256, 512) → a_embed ∈ R^{d_a=512}

每个 video frame 对应 tau=4 个 action tokens
（VAE temporal stride = 4，所以 1 个 latent frame ↔ 4 个物理帧 ↔ 4 个 action）
```

### 1.2 Interleaved Token Sequence

```
对于 T 个 latent frames，每帧有 N = H_p × W_p 个 video tokens:

Sequence = [
  v_{0,1}, v_{0,2}, ..., v_{0,N},     # frame 0 的 N 个 video tokens (每个 d_v=2048)
  a_{0,1}, a_{0,2}, a_{0,3}, a_{0,4}, # frame 0 的 4 个 action tokens (每个 d_a=512)
  v_{1,1}, v_{1,2}, ..., v_{1,N},     # frame 1
  a_{1,1}, a_{1,2}, a_{1,3}, a_{1,4}, # frame 1
  ...
  v_{T-1,1}, ..., v_{T-1,N},          # frame T-1
  a_{T-1,1}, ..., a_{T-1,4},          # frame T-1
]

总 token 数: T × (N + tau) = T × (N + 4)
例如 H=60, W=106 → H_p=30, W_p=53 → N=1590
  → 每帧 1594 tokens (其中 video 1590, action 4)
  → action tokens 仅占 0.25%，几乎不增加计算量
```

### 1.3 WanMoTAttentionBlock（核心修改）

```
当前 WanAttentionBlock:
  self_attn(x) → cross_attn(x, context) → FFN(x)
  所有 token 共享相同的参数

修改为 WanMoTAttentionBlock:
  ┌──────────────────────────────────────────────────┐
  │ 输入: x_unified (包含 video + action tokens)      │
  │                                                    │
  │ Step 1: 分离 modality                              │
  │   x_v = x_unified[video_mask]   # d_v = 2048      │
  │   x_a = x_unified[action_mask]  # d_a = 512       │
  │                                                    │
  │ Step 2: Modality-specific Q/K/V                    │
  │   q_v, k_v, v_v = W_Q^v(x_v), W_K^v(x_v), W_V^v(x_v)  # d_v → d_v │
  │   q_a, k_a, v_a = W_Q^a(x_a), W_K^a(x_a), W_V^a(x_a)  # d_a → d_a │
  │                                                    │
  │ Step 3: Cross-modal projection                     │
  │   q_a', k_a', v_a' = proj_up(q_a, k_a, v_a)  # d_a → d_v │
  │                                                    │
  │ Step 4: Shared Global Self-Attention (causal)      │
  │   Q = interleave(q_v, q_a')                        │
  │   K = interleave(k_v, k_a')                        │
  │   V = interleave(v_v, v_a')                        │
  │   attn_out = softmax(QK^T / sqrt(d)) V             │
  │                                                    │
  │ Step 5: Modality-specific output projection        │
  │   out_v = W_O^v(attn_out[video_mask])   # d_v → d_v │
  │   out_a = W_O^a(proj_down(attn_out[action_mask]))  # d_v → d_a │
  │                                                    │
  │ Step 6: Residual + Modality-specific LayerNorm     │
  │   x_v = x_v + out_v                                │
  │   x_a = x_a + out_a                                │
  │                                                    │
  │ Step 7: Cross-attention (仅 video tokens, 与文本)   │
  │   x_v = x_v + cross_attn(x_v, context)            │
  │   (action tokens 不做 cross-attention，或用轻量版)  │
  │                                                    │
  │ Step 8: Modality-specific FFN                      │
  │   x_v = x_v + FFN_v(norm_v(x_v))  # ffn_dim=8192  │
  │   x_a = x_a + FFN_a(norm_a(x_a))  # ffn_dim=2048  │
  │                                                    │
  │ Step 9: 合并回 unified sequence                     │
  │   x_unified = merge(x_v, x_a)                     │
  └──────────────────────────────────────────────────┘

与当前 WanAttentionBlock 的 6-component modulation 兼容：
  video: 使用原有 6-component modulation（从 time embedding）
  action: 使用独立的 6-component modulation（新增参数）
```

### 1.4 Action Output Head

```
当前 Head:
  video tokens → LayerNorm → modulate → Linear(2048, 16*1*2*2=64) → unpatchify

新增 ActionHead:
  action tokens → LayerNorm_a → modulate_a → Linear(512, 14) → action prediction
  （直接输出 14D action，无需 unpatchify）
```

### 1.5 WanModelCausalMoT Forward Pass

```python
def forward(self, x, t, context, actions=None, ...):
    # 1. Video patch embedding (不变)
    x_v = self.patch_embedding(x)  # [B, d_v, F, H_p, W_p]
    x_v = x_v.flatten(2).transpose(1, 2)  # [B, L_v, d_v]

    # 2. Action embedding (新增)
    if actions is not None:
        x_a = self.action_embedding(actions)  # [B, T*tau, d_a]
    else:
        x_a = noise_for_flow_matching  # 推理时 action tokens 从噪声开始

    # 3. Interleave video + action tokens
    x_unified, modality_mask = self.interleave(x_v, x_a, block_size)

    # 4. Time embedding (不变，但 action tokens 用独立的 modulation)
    e_v = self.time_projection(self.time_embedding(...))  # [B, L_v, 6, d_v]
    e_a = self.time_projection_a(self.time_embedding_a(...))  # [B, L_a, 6, d_a]

    # 5. Transformer blocks (MoT)
    for block in self.blocks:
        x_unified = block(x_unified, e_v, e_a, modality_mask, ...)

    # 6. Separate heads
    v_video = self.head(x_v_tokens, e_v)      # video velocity field
    v_action = self.action_head(x_a_tokens, e_a)  # action velocity field

    return v_video, v_action
```

---

## 2. 训练策略

### 2.1 Joint Flow Matching Loss

```
LingBot-VA 的训练方式：video 和 action 同时用 flow matching 去噪。

Flow matching interpolation:
  z^(s) = (1-s) · ε + s · z_clean    (s ∈ [0,1], s=0 纯噪声, s=1 干净)
  velocity target: ż^(s) = z_clean - ε

Video dynamics loss:
  L_dyn = E[ ‖v_θ^video(z_v^(s), s, context) - (z_v_clean - ε_v)‖² ]

Inverse dynamics loss (action prediction):
  L_inv = E[ ‖v_ψ^action(z_a^(s), s, context) - (a_clean - ε_a)‖² ]

Combined:
  L = L_dyn + λ · L_inv    (λ = 1.0)
```

### 2.2 与 Self-Forcing 的兼容

```
Vidarc 当前使用 Self-Forcing 训练（wrapper_causal.py）:
  - Causal attention + KV cache
  - Teacher forcing: history frames 用 GT
  - Stochastic truncation: 在随机帧处截断梯度

MoT 修改与 Self-Forcing 完全兼容：
  - KV cache 同时存储 video KV 和 action KV
  - Teacher forcing: history 的 video tokens 和 action tokens 都用 GT
  - Stochastic truncation: 只在 video loss 上截断（action loss 可以全程传梯度）
  - Causal mask: action tokens at time t 可以看到 video tokens at time t+1
    （这使得 action tokens 能做 inverse dynamics reasoning）
```

### 2.3 Noisy History Augmentation（关键优化）

```
训练时对 history 的 video tokens 加噪:
  z̃_{<=t} = (1 - s_aug) · ε + s_aug · z_{<=t}   (概率 p=0.5)
  z̃_{<=t} = z_{<=t}                               (概率 1-p=0.5)
  s_aug ~ Uniform[0.5, 1.0]

效果：
  推理时 video tokens 只需去噪到 s=0.5（而非 s=1.0）
  → video denoise steps 减半（例如 10 步 → 3-5 步）
  → action tokens 仍然做完整去噪（10 步到 s=1.0）

  这对 MPC 意义重大：
  每个 MPC candidate 的 video 生成时间从 ~2-5s 降到 ~0.6-1.5s
```

### 2.4 Variable Chunk-Size Training

```
LingBot-VA 随机采样 chunk size K ∈ [1, 4]:
  - K=1: 单步预测（快，简单）
  - K=4: 多步预测（与推理一致，需要更多计算）

Vidarc 可以类似：
  随机采样 K ∈ [1, 4]，每次预测 K 个 video frames + K*tau=K*4 个 actions
  这使得模型在任何 horizon 下都能工作
```

### 2.5 Action Stream 参数初始化

```
LingBot-VA 的初始化策略（关键！）：

action stream 的 Q/K/V/O 权重从 video stream 权重插值:
  W_a = interpolate(W_v, d_a)
  缩放因子: α = sqrt(d_v / d_a) = sqrt(2048 / 512) = 2.0

这保持了输出方差，防止 action tokens 的分布与 video tokens 差异过大，
导致 shared attention 不稳定。

FFN_a 和 norm_a 从 FFN_v 和 norm_v 插值。
action_embedding 和 action_head 随机初始化（Xavier）。
```

### 2.6 训练数据

```
与当前 Vidarc 训练完全相同的 demo 数据，只需额外提供 action 标签：
  (video_frames, text_description, action_sequence)

action_sequence: [a_0, a_1, ..., a_{T*tau-1}]，每个 a_i ∈ R^14

如果 demo 数据中没有 action 标签：
  → 用现有 IDM 离线标注 action（IDM 已经训练好，很快）
  → 这样不需要额外的人工标注
```

---

## 3. MPC 在 Action Token Space

### 3.1 为什么 MoT 天然支持更快的 MPC

```
当前 MPC 瓶颈：
  每个 candidate → 完整 Wan 生成 → VAE decode → IDM → 评分
  ~2-5s per candidate

MoT 的 3 个 MPC 加速路径：

路径 A: Partial-Denoise MPC
  因为 noisy history augmentation，video 只需去噪到 s=0.5
  → video 生成时间减半
  → 但 action tokens 可以完整去噪
  加速比: ~2×

路径 B: Action-Only Resampling MPC
  固定 video tokens 的去噪轨迹（只跑一次 video 生成）
  对 action tokens 做多次 resampling（从不同噪声初始化）
  → 只需重跑 action stream 的 FFN + head（极小计算量）
  → self-attention 中 video KV 已缓存
  加速比: ~10-50× (取决于 action 重采样的实现)

路径 C: Action Latent Space MPC（最快）
  训练一个轻量 dynamics model in action token space:
    f(z_a_t, a_t) → z_a_{t+1}
  MPC 完全在 action latent space rollout，不经过 DiT
  → 这与旧方案类似，但 z_a 来自 DiT 内部（更丰富）
  加速比: ~100-500×
```

### 3.2 推荐 MPC 策略：分层

```
┌──────────────────────────────────────────────────┐
│ Level 1: Action-Only Resampling (路径 B)          │
│   固定一次 video 生成，对 action 做 N 次重采样     │
│   评分：action 轨迹与 subgoal 的距离               │
│   适用：实时控制，每步 ~200-500ms                   │
│                                                    │
│ Level 2: Action Latent Rollout (路径 C)            │
│   在 action token latent space 做 MPPI             │
│   完全不调用 DiT                                    │
│   适用：需要更多候选时，每步 ~50-100ms               │
│                                                    │
│ Level 3: Partial-Denoise Video (路径 A)            │
│   video 去噪到 s=0.5 + action 完整去噪             │
│   适用：需要 video 质量评分的场景                    │
└──────────────────────────────────────────────────┘
```

### 3.3 Subgoal Guidance

```
在 action token space 做 subgoal guidance 比 video space 更自然：

1. Subgoal 表示:
   subgoal image → Wan VAE encode → DiT partial forward → z_a_goal
   （或直接用 action embedding 编码 target action）

2. Score function:
   Φ(z_a, g) = -‖z_a_predicted - z_a_goal‖²

3. MPPI 结合:
   candidates = sample_action_noise(M)
   for each candidate:
     rollout in action space → z_a_final
     score = -‖z_a_final - z_a_goal‖²
   weights = softmax(scores / temperature)
   optimal_action = weighted_average(candidates, weights)

4. Gradient-based guidance (类比 classifier guidance):
   a.requires_grad_(True)
   z_a_final = dit_action_forward(z_v_fixed, a_noised)
   loss = ‖z_a_final - z_a_goal‖²
   a_guided = a - lr * grad(loss, a)

   这就是 action space 版本的:
   ε̃ = ε_θ - λ · σ · ∇Φ
```

---

## 4. 需要修改的文件

### 4.1 核心模型修改

```
vidar/wan/modules/model_causal.py
├── WanAttentionBlock → WanMoTAttentionBlock
│   ├── 新增: action_norm1, action_norm2 (LayerNorm for d_a)
│   ├── 新增: action_qkv_proj (Q/K/V for action, d_a → d_a)
│   ├── 新增: action_out_proj (d_a → d_a)
│   ├── 新增: action_cross_proj_up (d_a → d_v, for shared attention)
│   ├── 新增: action_cross_proj_down (d_v → d_a, after attention)
│   ├── 新增: action_ffn (MLP: d_a → ffn_dim_a → d_a)
│   ├── 新增: action_modulation (6-component, for d_a)
│   └── 修改: forward() — 分离 video/action, shared attention, separate FFN
│
├── Head → (不变)
├── ActionHead (新增)
│   └── action tokens → LayerNorm → modulate → Linear(d_a, action_dim=14)
│
└── WanModelCausal → WanModelCausalMoT
    ├── 新增: action_embedding (MLP: 14 → d_a=512)
    ├── 新增: action_time_embedding + action_time_projection
    ├── 新增: action_head (ActionHead)
    ├── 修改: blocks 使用 WanMoTAttentionBlock
    ├── 修改: forward() — interleave, dual-stream processing
    └── 修改: KV cache — 同时缓存 video KV 和 action KV
```

### 4.2 训练修改

```
vidar-robotwin/training/models/wrapper_causal.py
├── 修改: forward_self_forcing() — 同时处理 video + action loss
├── 修改: forward_self_forcing_multistep() — multi-step 包含 action prediction
├── 新增: action_flow_matching_loss() — action tokens 的 flow matching loss
├── 新增: noisy_history_augmentation() — 对 history video tokens 加噪
└── 修改: loss computation — L = L_dyn + λ · L_inv
```

### 4.3 推理修改

```
vidar/wan/textimage2video_causal_server.py
├── 修改: generate() — 同时输出 video + action
└── 修改: denoising loop — video 去噪到 s=0.5, action 去噪到 s=1.0

vidar/server/causal_worker.py
├── 修改: 移除 IDM 依赖（action 直接从 DiT 输出）
├── 修改: inference endpoint — 返回 (video_frames, actions)
└── 简化: 不再需要 idm.py, resnet.py, unet.py

vidar/server/mpc_optimizer.py → vidar/server/mot_mpc_optimizer.py
├── 新增: Action-Only Resampling MPC (路径 B)
├── 新增: Action Latent Rollout MPC (路径 C)
└── 修改: 评分函数 — 在 action latent space
```

### 4.4 评估修改

```
vidar-robotwin/policy/AR/ar.py
├── 修改: get_actions() — 直接从 server 获取 actions（无需 IDM）
└── 修改: MPC mode — 使用新的 MoT MPC

vidar-robotwin/script/eval_policy.py
└── 修改: 配置参数（移除 IDM 相关配置）
```

---

## 5. 实施阶段

### Phase 0: 基础设施准备（~2天）

```
目标：准备 action 标注数据 + 验证数据 pipeline

步骤：
  1. 用现有 IDM 对所有 demo 数据离线标注 action
     - 输入: (obs_t, obs_{t+1}) pairs
     - 输出: action_t ∈ R^14
     - 存储格式: 与 video 数据对齐

  2. 修改 dataloader 同时加载 video + action
     - 每个 video frame 对应 tau=4 个 action
     - 处理 action 的 normalization (learned mean/std from IDM)

  3. 验证数据对齐
     - video frame timestamps 与 action timestamps 匹配
     - 可视化检查: overlay action direction on video
```

### Phase 1: WanModelCausalMoT 实现（~5天）

```
目标：实现 MoT 架构修改

步骤：
  1. Day 1-2: WanMoTAttentionBlock
     - 实现 modality-specific Q/K/V/O projections
     - 实现 cross-modal projection (d_a ↔ d_v)
     - 实现 modality-specific FFN 和 LayerNorm
     - 实现 action modulation (6-component)
     - 单元测试: 验证前向传播 shape 正确

  2. Day 3: WanModelCausalMoT
     - 实现 action_embedding, action_head
     - 实现 interleave / de-interleave 逻辑
     - 实现 dual-stream time embedding
     - 修改 KV cache 支持 action tokens
     - 单元测试: 完整前向传播

  3. Day 4: 参数初始化
     - 实现从 pretrained video weights 插值 action weights
     - 缩放因子 α = sqrt(d_v / d_a) = 2.0
     - 验证: 初始化后 action stream 输出分布与 video 相近

  4. Day 5: 与 Self-Forcing 集成
     - 修改 wrapper_causal.py
     - 实现 joint flow matching loss
     - 实现 noisy history augmentation
     - 验证: 训练 loop 跑通（小 batch，1 epoch）
```

### Phase 2: 训练（~5天）

```
目标：在 RoboTwin demo 数据上训练 MoT 模型

训练配置（参考 LingBot-VA）:
  - 初始化: Vidarc Stage 2 checkpoint + action stream 插值初始化
  - 学习率: 1e-5 (post-training, lower than pretraining)
  - Batch size: 按 GPU 数量调整
  - Chunk size: 随机 K ∈ [1, 4]
  - Loss: L = L_dyn + L_inv (λ=1.0)
  - Noisy history: p=0.5, s_aug ~ U[0.5, 1.0]

训练策略:
  Day 1-2: 冻结 video stream，只训练 action stream
    - 让 action stream 先学会从 video context 提取 action 信息
    - 验证: action prediction loss 下降

  Day 3-4: 解冻全部参数，joint training
    - Video + action 联合训练
    - 监控: video loss 不应退化
    - 验证: 生成的 video 质量保持

  Day 5: 评估 + 调参
    - 评估 action prediction accuracy (vs IDM baseline)
    - 评估 video generation quality (vs Vidarc baseline)
    - 调整 λ 和 learning rate

成功标准:
  - Action prediction MSE 接近 IDM baseline (±20%)
  - Video quality (FVD/SSIM) 不退化 (±5%)
```

### Phase 3: 推理 + 基础评估（~3天）

```
目标：实现推理 pipeline，对比 Vidarc baseline

步骤：
  1. 修改 textimage2video_causal_server.py
     - 实现 partial denoise for video (s → 0.5)
     - 实现 full denoise for action (s → 1.0)
     - 输出同时包含 video frames 和 action sequence

  2. 修改 causal_worker.py
     - 移除 IDM 调用
     - 直接从 DiT 输出获取 action

  3. 基础评估
     - RoboTwin benchmark: 直接用 MoT 输出的 action 控制
     - 对比: Vidarc + IDM vs MoT (无 MPC)
     - 指标: task success rate, action smoothness
```

### Phase 4: Action Space MPC（~4天）

```
目标：实现 3 个层次的 MPC

Day 1-2: 路径 B — Action-Only Resampling MPC
  - 跑一次完整的 video + action 生成
  - 缓存 video tokens 的 KV
  - 对 action tokens 做 M 次重采样（从不同噪声）
  - 用 action latent distance 评分
  - 选择最优 action sequence

Day 3: 路径 C — Action Latent Dynamics MPC
  - 从 DiT 中间层提取 action token features
  - 训练轻量 MLP dynamics model: f(z_a_t, a_t) → z_a_{t+1}
  - MPPI in action latent space
  - 完全不调用 DiT

Day 4: 集成 + Subgoal Guidance
  - 实现分层 MPC 策略 (Level 1/2/3)
  - 实现 subgoal-conditioned scoring in action space
  - 评估: MPC 加速比 + 任务成功率

预期结果:
  路径 B: ~200-500ms per MPC step (vs 10-25s 当前)
  路径 C: ~50-100ms per MPC step
```

---

## 6. 风险分析

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| Action stream 初始化不稳定，影响 video 质量 | 中 | 高 | 先冻结 video stream 训练 action stream；α=sqrt(d_v/d_a) 缩放；监控 video loss |
| Action prediction 精度不如 IDM | 中 | 中 | 增大 d_a；加更多训练数据；IDM 标注兜底 |
| KV cache + MoT 导致显存不够 | 中 | 中 | action tokens 很少（4/frame），KV 增量极小；可以 offload |
| Shared attention 中 action tokens 被 video tokens 淹没 | 中 | 中 | 调整 attention scale；对 action tokens 使用 attention bias |
| Noisy history augmentation 导致训练不稳定 | 低 | 中 | 从 p=0.2 开始逐步增加到 0.5；监控 loss 曲线 |
| RoPE 位置编码与 interleaved tokens 冲突 | 低 | 高 | Action tokens 使用独立的位置编码（学习的，不用 RoPE） |
| 训练数据量不够（demo 太少） | 中 | 高 | 用 IDM 先标注大量 action；使用 data augmentation |

---

## 7. 对比总结

```
                    当前 Vidarc          MoT Vidarc
                    ─────────────        ───────────────
Action 提取方式      IDM (后验, pixel)    DiT 内部 (联合生成)
IDM 依赖             需要                 不需要
视频-动作关系         解耦                 耦合 (shared attention)
MPC 每步耗时          10-25s              0.05-0.5s
Subgoal 引导空间      video latent        action latent (更自然)
额外参数量            IDM ~25M            action stream ~350M*
端到端               否                   是

* 350M = 32 layers × (action QKV + action FFN + action norm + action modulation)
  ≈ 32 × (512×512×3 + 512×2048×2 + 512×3) ≈ 70M
  实际可能更少，因为 d_a=512 比 LingBot 的 768 小
```

---

## 8. 时间线总览

```
Week 1:  Phase 0 (数据准备) + Phase 1 前半 (MoT Block 实现)
Week 2:  Phase 1 后半 (集成 Self-Forcing) + Phase 2 前半 (训练 action stream)
Week 3:  Phase 2 后半 (joint training) + Phase 3 (推理 + 基础评估)
Week 4:  Phase 4 (Action Space MPC + subgoal guidance)

总计: ~4 周
里程碑:
  Week 1 end: MoT 前向传播跑通
  Week 2 end: Action stream 训练 loss 下降
  Week 3 end: 基础评估可比较
  Week 4 end: MPC 加速 + 任务成功率对比
```

---

## 9. 关键设计决策清单

| # | 决策 | 推荐选择 | 替代方案 | 需要验证 |
|---|------|---------|---------|---------|
| 1 | d_a (action hidden dim) | 512 | 256, 768 | Phase 2 调参 |
| 2 | tau (actions per frame) | 4 | 1, 2, 8 | 与 VAE stride 匹配 |
| 3 | Action tokens 的 cross-attention | 不做 | 做轻量版 | Phase 2 消融 |
| 4 | Action tokens 的位置编码 | 学习的 | RoPE | Phase 1 实验 |
| 5 | Video partial denoise level | s=0.5 | s=0.3, s=0.7 | Phase 3 实验 |
| 6 | 训练策略 | 先冻结 video | 直接 joint | Phase 2 比较 |
| 7 | MPC 主策略 | 路径 B (action resampling) | 路径 C (latent rollout) | Phase 4 |
| 8 | Action normalization | IDM 的 learned mean/std | per-dataset normalization | Phase 0 |
