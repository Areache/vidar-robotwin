# Survey: SSM / Structured Latent 用于 Action Modeling

> 调研日期: 2025-02
> 目的: 梳理现有将 State Space Model 或结构化 latent 用于 action 建模的工作，为 video model 中嵌入 action latent SSM pathway 提供参考。

---

## 1. SSM 作为 VLA Backbone（替代 Transformer）

### 1.1 RoboMamba (NeurIPS 2024)

- **论文**: [RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation](https://arxiv.org/abs/2406.04339)
- **核心思路**: 用 Mamba (selective SSM) 替换 VLA 中的 Transformer backbone

**原理**:

SSM 不直接建模 action dynamics，而是作为 vision-language 的 sequence encoder。action 来自最后的 MLP head。

```
Image → CLIP encoder → f_v ∈ R^{B×N×1024}
                          ↓ MLP projection
                        f_v^L ∈ R^{B×N×2560}
                          ↓ concat
Text → Tokenizer → f_t ∈ R^{B×N×2560} → [f_v^L ; f_t]
                                              ↓
                                    Mamba SSM blocks (selective scan)
                                              ↓
                                    output tokens T_a
                                              ↓ global pooling
                                    global token g ∈ R^{2560}
                                           ↓          ↓
                                    MLP_pos → a_pos   MLP_dir → a_dir
                                    ∈ R^3 (xyz)       ∈ R^{3×3} (rotation)
```

**SSM 在其中的角色**:

Mamba 的 selective SSM 处理拼接后的 vision-language token 序列：

```
h_t = Ā · h_{t-1} + B̄ · x_t        (SSM recurrence)
y_t = C · h_t                        (readout)

其中:
  Ā = exp(Δ · A),  B̄ = (ΔA)^{-1}(exp(ΔA) - I) · ΔB
  Δ, B, C 均为 input-dependent (selective mechanism)
```

hidden state `h_t` 逐 token 累积 vision-language context。最终对 output token 做 pooling 得到 global token，输入 policy MLP。

**关键特点**:
- SSM 的作用是**高效序列编码**（线性复杂度），不是 action trajectory 建模
- action 建模 = global pooling + 两个独立 MLP，无时序结构
- 两阶段训练：Stage 1 训 vision-language alignment（冻 Mamba），Stage 2 只训 policy head（冻全部，仅 3.7M / 0.1% 参数）
- Loss: `L_pos = (1/N)Σ|a_pos - a_pos^gt|`, `L_dir = (1/N)Σ arccos((Tr(R_gt^T R) - 1)/2)`

**结果**: 推理速度比 Transformer-based VLA 快 7×，fine-tune 仅需 20 分钟

**与我们的关系**: SSM 用在 vision-language token 上，不在 action trajectory 上。**没有结构化的 action latent**。但证明了 Mamba 作为 VLA backbone 的可行性和效率优势。

---

### 1.2 SpatialVLA-Mamba (2025)

- **论文**: [SpatialVLA-Mamba: Efficient State-Space Models with Self-Refinement for Spatially-Grounded Robotic Control](https://openreview.net/forum?id=sTn4EqE49A)
- **核心思路**: 用 Mamba decoder 替换 Transformer decoder，**SSM 直接 decode action 序列**

**原理**:

与 RoboMamba 的关键区别：Mamba 不是用在 encoder 端，而是用在 **action decoder 端**。

```
                     Spatial-Aware Encoder
                     ┌─────────────────────┐
RGB image ──────────→│                     │
Depth map ──────────→│ 融合 RGB + depth +  │→ spatial features
Geometric primitives→│ geometric primitives│
                     └─────────────────────┘
                              ↓
                     ┌─────────────────────┐
Language instruction→│ VLA encoder         │→ multimodal context
                     └─────────────────────┘
                              ↓
                     ┌─────────────────────┐
                     │ Mamba SSM Decoder    │
                     │                     │
                     │ h_t = Ā_t·h_{t-1} + B̄_t·x_t  ← SSM recurrence
                     │ a_t = readout(h_t)             ← action per step
                     │                     │
                     │ 沿 action horizon   │
                     │ autoregressively    │
                     │ decode action chunk │
                     └─────────────────────┘
                              ↓
                     a_{1:H} (action sequence)
                              ↓
                     ┌─────────────────────┐
                     │ CoT-RL Self-Refine  │
                     │ 1. 生成候选轨迹文本摘要│
                     │ 2. CLIPScore 评估    │
                     │ 3. PPO 更新 policy   │
                     └─────────────────────┘
```

**SSM 在 decoder 中的角色**:

SSM 的 hidden state 在 action 序列维度上递推，为 action 序列提供时序归纳偏置：
- `h_t` 编码了到当前步为止的 action 历史
- 线性复杂度：O(T) vs Transformer decoder 的 O(T²)
- 长 horizon 时显存和速度优势显著

**关键特点**:
- Spatial-aware encoder 引入 depth + geometric primitives，提供厘米级精度
- Mamba decoder **直接在 action sequence 上做 SSM recurrence**，SSM hidden state 携带轨迹动力学信息
- CoT-RL：policy 生成候选轨迹 → 文本摘要 → CLIPScore 打分 → PPO 自我优化（不依赖外部 LLM）
- 结果: spatial error 降低 35%+，unseen task 成功率 67.3%

**与我们的关系**: **比 RoboMamba 更接近我们的目标**——SSM 直接在 action 序列上建模，hidden state 隐式编码轨迹动力学。但没有显式的 action latent space，也没有与 video model 的 cross-attention。

---

## 2. SSM 用于 In-Context RL (状态空间方程建模序列决策)

### 2.1 S4 for In-Context RL (NeurIPS 2023)

- **论文**: [Structured State Space Models for In-Context Reinforcement Learning](https://arxiv.org/abs/2303.03982)
- **核心思路**: 用 S4（structured state space sequence model）替代 Transformer/RNN 做 RL 中的序列建模
- **关键技术挑战**: S4 训练时用固定卷积核（parallel scan），不能像 RNN 那样在 episode 边界 reset hidden state → 作者提出修改版 S4 支持并行 reset
- **结果**: 比 RNN 快 5×+，在 POMDP 上优于 RNN 和 Transformer
- **与我们的关系**: 证明了 **SSM 的 hidden state 天然适合做 RL/控制中的 "belief state"**，即压缩历史信息的状态表征。这和我们想用 SSM hidden state 做 action latent dynamics 的想法一致。但该工作不涉及 video generation。

---

## 3. Latent Action World Model（学习通用 action latent space）

### 3.1 Motus (清华 + 北大, 2025)

- **论文**: [Motus: A Unified Latent Action World Model](https://arxiv.org/abs/2512.13030)
- **核心思路**: 用光流学习 latent action，统一 world model / VLA / IDM / video generation
- **架构**: Mixture-of-Transformers (MoT)
  - 三个 expert: understanding expert, video generation expert, action expert
  - 共享 multi-head self-attention（Tri-model Joint Attention），分别有独立 FFN
  - UniDiffuser-style scheduler 灵活切换建模模式
- **Latent Action 定义**:
  - DPFlow: 从光流计算 pixel-level displacement → DC-AE 压缩 → lightweight encoder → latent action
  - 本质是 **运动信息的压缩表征**，不是关节空间 action
  - Embodiment-agnostic：跨 embodiment 通用
- **训练**: 三阶段 pipeline + 六层数据金字塔
- **结果**: RoboTwin 2.0 上 87.02% 成功率（+15% over X-VLA，+45% over π₀.₅）
- **⭐ 与我们的关系**: **最接近我们想做的事情。** 关键区别:
  - Motus 用 **Transformer attention** 做 action expert，我们想用 **SSM**
  - Motus 的 latent action 来自**光流**，我们的可能来自 **IDM feature**
  - Motus 是从头训练，我们想在已有 WAN 上加 action pathway
  - Motus 的 MoT 架构（共享 attention + 独立 FFN）值得借鉴

### 3.2 LAC-WM (2025)

- **论文**: [Latent Action Robot Foundation World Models for Cross-Embodiment Adaptation](https://openreview.net/forum?id=vEZgPr1deb)
- **核心思路**: 学习统一 latent action space，在其中做 forward dynamics prediction
- **架构**: 4 个组件
  1. Inverse Dynamics Model (IDM): 从 (o_t, o_{t+1}) 推断 latent action
  2. Forward Dynamics Model (FDM): 从 (o_t, latent_action) 预测 o_{t+1}
  3. Motion Decoder: latent action → 具体 embodiment 的 action
  4. Action Projector: 具体 action → latent action
- **结果**: 比 explicit-action-conditioned world model (EAC-WM) 提升 46.7%；latent action space 的下游性能随 pretraining embodiment 数量正向 scale
- **⭐ 与我们的关系**: LAC-WM 的 IDM + FDM 结构 **和我们 PLAN 中的方案非常相似**（IDM encoder + dynamics model）。但 LAC-WM 没有 SSM 结构，dynamics 是 Transformer-based。

### 3.3 UWM — Unified World Models (RSS 2025, TRI/UW)

- **论文**: [Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets](https://arxiv.org/abs/2504.02792)
- **核心思路**: 在统一 Transformer 中耦合 video diffusion 和 action diffusion，用独立的 diffusion timestep 控制两个模态
- **关键创新**:
  - 同一模型同时做: policy / forward dynamics / inverse dynamics / video generation
  - 通过控制各模态的 diffusion timestep 切换功能模式
  - 缺失 action 的视频数据：将 action 设为纯噪声即可参与训练
- **结果**: 比纯 imitation learning 更 generalizable，可利用无 action 标注的视频数据
- **⭐ 与我们的关系**: **UWM 的 "video + action 双 diffusion" 思路最值得借鉴。** 它证明了：
  - video 和 action 可以在同一模型内用不同的 diffusion process 建模
  - 不需要把 video model 和 action model 完全分开
  - 关键设计: **独立 diffusion timestep** 让两个模态解耦训练
  - 但 UWM 的 action 建模仍然是 diffusion-based Transformer，没有 SSM 结构

---

## 4. SSM + Diffusion 混合 World Model

### 4.1 StateSpaceDiffuser (2025)

- **论文**: [StateSpaceDiffuser: Bringing Long Context to Diffusion World Models](https://arxiv.org/abs/2505.22246)
- **核心思路**: 用 SSM (Mamba) backbone 给 diffusion world model 提供 long-range context
- **架构**: SSM 编码完整交互历史 → diffusion 生成当前帧
  - 将全局 context modeling（SSM）和高保真生成（diffusion）解耦
  - SSM 在 O(1) 额外开销下保持任意长度上下文
- **结果**: horizon=50 时，PSNR 比最强 diffusion baseline (DIAMOND) 提升 8.9（40.6 vs 25.4）
- **⭐ 与我们的关系**: **架构思路最接近我们的设计。** SSM 处理时序 context + diffusion 处理生成，和我们的 "SSM action pathway + DiT video pathway" 是同一个精神。区别:
  - StateSpaceDiffuser 的 SSM 编码的是 observation history，不是 action latent
  - 它用于 game world model，不是 robot manipulation
  - 但 **SSM + Diffusion 的混合架构已被验证有效**

### 4.2 EDELINE (2025)

- **论文**: [EDELINE: Enhancing Memory in Diffusion-based World Models via Linear-Time Sequence Modeling](https://arxiv.org/abs/2502.00466)
- **核心思路**: 将 Mamba SSM 融入 diffusion world model，增强记忆能力
- **关键创新**:
  1. Mamba-based recurrent embedding module — 增强长程记忆
  2. 统一框架直接条件化 reward/termination prediction
  3. Dynamic Loss Harmonization — 自适应 loss 权重
- **结果**: Atari100K, MiniGrid, VizDoom 上 SOTA
- **与我们的关系**: 进一步验证 SSM + Diffusion 混合架构的有效性。EDELINE 的 "recurrent embedding module" 和我们想加的 SSM action pathway 在精神上类似。

---

## 5. 总结与定位

### 5.1 现有工作的分类

| 类别 | 代表工作 | SSM 的角色 | Action 建模方式 | 与 Video Model 的关系 |
|------|---------|-----------|----------------|---------------------|
| SSM 作为 VLA backbone | RoboMamba, SpatialVLA-Mamba | 替换 Transformer 做 sequence encoding | MLP head 回归 | 无 video generation |
| SSM 用于 RL 序列建模 | S4 for In-Context RL | Hidden state 作为 belief state | 隐式 (在 SSM state 中) | 无 |
| Latent Action World Model | Motus, LAC-WM, UWM | 未使用 SSM | Transformer / Diffusion | video + action 联合建模 |
| SSM + Diffusion World Model | StateSpaceDiffuser, EDELINE | 提供 long-range context | 未显式建模 action | SSM 编码历史, Diffusion 生成 |

### 5.2 空白 (Gap) — 我们的切入点

**目前没有工作同时做到以下三点:**

1. ✅ 在 **video diffusion model 内部/旁边** 加入 action 建模（UWM, Motus 做到了）
2. ✅ 用 **SSM 结构** 为 action latent 提供时序动力学归纳偏置（RoboMamba, SpatialVLA-Mamba 用了 SSM，但不在 video model 内）
3. ✅ 用 **global attention** 实现轨迹级推理（Motus 的 Joint Attention 做到了）

**我们的提案: Video DiT + SSM Action Latent Pathway (global cross-attn + SSM recurrence)**

```
现有最接近的组合:
  Motus 的 MoT 架构  (video + action 联合, shared attention)
  + StateSpaceDiffuser 的 SSM context  (SSM 提供时序结构)
  + 我们已有的 WAN DiT  (不从头训练, 在现有模型上加 pathway)
```

### 5.3 各工作的具体启示

| 工作 | 我们可以借鉴什么 |
|------|----------------|
| **Motus** | MoT 架构: shared attention + independent FFN; 用光流/IDM 定义 latent action; 三阶段训练 |
| **UWM** | 独立 diffusion timestep 解耦 video 和 action; 无 action 数据也能训练 |
| **StateSpaceDiffuser** | SSM backbone + Diffusion head 的混合架构; SSM 提供 long-range context 零额外开销 |
| **LAC-WM** | IDM → latent action → FDM 的 pipeline; latent action space 跨 embodiment scale |
| **SpatialVLA-Mamba** | Mamba 做 action sequence decoder, 线性复杂度 |
| **S4 for RL** | SSM hidden state 天然适合 belief state / dynamics state |

### 5.4 风险评估（基于文献）

| 我们的设计选择 | 文献中的证据 | 风险等级 |
|--------------|------------|---------|
| SSM 建模 action 时序 | SpatialVLA-Mamba, S4-RL 验证 SSM 在 action/RL 上有效 | 低 |
| Cross-attention 连接 video 和 action | Motus 的 Joint Attention 验证有效 | 低 |
| 在已有 DiT 上加 pathway (而非从头训练) | 无直接先例 (Motus/UWM 都是从头训练) | **中** |
| Stop-gradient 保护 WAN | 常见做法 (adapter/LoRA 思路), 但在此场景无直接验证 | **中** |
| IDM feature 作为 latent action | LAC-WM 验证 IDM-based latent action 有效 | 低 |

---

## 6. 我们的设计方案

### 6.1 共同内核

两个方案共享相同的 action latent pathway 内部结构，区别在于它和 WAN 的耦合方式。

每一层 action latent layer 包含三个组件，职责分离：

```
┌─────────────────────────────────────────────────────────┐
│              Action Latent Layer (× N_layers)           │
│                                                         │
│  1. Cross-Attn to Video                                 │
│     z_t ← CrossAttn(Q=z_t, KV=video_features.detach()) │
│     职责: 从 video 中提取 action-relevant 信息            │
│     特点: 全局 (不受 causal mask 限制)                    │
│                                                         │
│  2. Self-Attn among z_{1:T}                             │
│     z_{1:T} ← SelfAttn(z_{1:T})                        │
│     职责: 双向轨迹级推理 (知道终点才能规划起点)             │
│     特点: T 很小 (8-16), 开销可忽略                       │
│                                                         │
│  3. SSM Recurrence                                      │
│     h_t = A_t · h_{t-1} + B_t · z_t                    │
│     z_t = z_t + Linear(h_t)                             │
│     职责: 动力学归纳偏置 (平滑/惯性/接触切换)              │
│     特点: A_t, B_t input-dependent (selective)           │
│           h_t 只编码动力学状态, 不混入 context             │
│                                                         │
│  4. FFN                                                 │
│     z_t = z_t + MLP(z_t)                                │
│                                                         │
│  Output: action_head(z_t) → a_t ∈ R^14                  │
└─────────────────────────────────────────────────────────┘
```

**为什么三个组件缺一不可**:

| 去掉什么 | 后果 |
|---------|------|
| 去掉 Cross-Attn | action pathway 看不到 video, 变成盲人规划 |
| 去掉 Self-Attn | 只有单向因果信息 (SSM), 无法做双向轨迹优化 |
| 去掉 SSM | 和 Motus 一样纯 attention, 没有动力学归纳偏置 |

### 6.2 方案 A: 嵌入每层 DiT (深度交互)

```
WAN DiT block l:
  x_video = video_self_attn(x_video)       ← 原有, 不改
  x_video = video_ffn(x_video)             ← 原有, 不改
  z_action = action_latent_layer(z_action, x_video.detach())  ← 新增

  重复 32 层 (和 WAN DiT 同层数)
```

- action pathway 在每一层都读到当前层的 video features
- 浅层 video features 编码低级运动, 深层编码语义 → action pathway 逐层获取多尺度信息
- 参数量: 32 层 × (~2M/层) ≈ 64M (WAN 本身 ~1.3B, 占比 ~5%)

### 6.3 方案 B: 独立模块 (最小改动)

```
WAN DiT (完全不改, 冻结):
  x_video = WAN_DiT(x_noisy, t, text_emb)    ← 正常跑完所有层

Action Latent Module (新增, 独立):
  video_features = x_video.detach()            ← 只读, 不回传梯度
  z_action = ActionLatentModule(z_action, video_features)  ← 4 层
  actions = action_head(z_action)
```

- WAN 完全冻结, 零风险
- Action module 只有 4 层, 参数量 ~8M
- 只能读到 WAN 最后一层的 video features (不是多尺度)

### 6.4 两方案对比

| | 方案 A (深度交互) | 方案 B (独立模块) |
|---|---|---|
| WAN 改动 | 每层加一个 action latent layer | 零改动 |
| WAN 破坏风险 | 中 (需 stop-gradient) | 零 |
| 多尺度 video features | 有 (浅层→深层) | 无 (只有最后一层) |
| action 表征质量 | 预期更好 | 可能 underfit |
| 训练复杂度 | 需仔细控制梯度 | 简单, 独立训练 |
| 推理额外开销 | ~0.1% | ~0.03% |
| 实现难度 | 中 (改 DiT forward) | 低 (加后处理模块) |

---

## 7. 风险与缓解

### 7.1 风险 1: WAN 被 action loss 梯度破坏

**触发条件**: 方案 A 中 L_action 的梯度通过 cross-attention 回传进 video tokens

**后果**: video features 偏向 action prediction, 视频生成质量下降

**缓解 — 分级梯度隔离策略**:

```
Level 0: WAN 完全冻结, action module 独立训练
  z = CrossAttn(Q=z_action, KV=x_video.detach())
  → 验证 action pathway 在冻结 features 上能否 work
  → 如果 action MSE 可接受, 到此为止

Level 1: WAN 冻结 + LoRA (rank 4-16)
  → action loss 只更新 LoRA 参数, WAN 原始权重不动
  → video quality 几乎不受影响
  → action prediction 可能提升

Level 2 (推荐起点): WAN 可训练, 但 cross-attn KV stop-gradient
  z = CrossAttn(Q=z_action, KV=sg(x_video))
  → L_action 不回传到 video, L_video 正常训练
  → 两个 loss 完全解耦, 互不干扰
  → action pathway 仍然读到不断改进的 video features

Level 3: 联合训练, 梯度缩放
  L = L_video + α · L_action, α = 0.01-0.1
  → 监控 video FVD/FID, 一旦下降就回退到 Level 2
```

### 7.2 风险 2: Action pathway underfit

**触发条件**: action latent 维度 D 太小, 或层数不够, 或 cross-attn 提取不到有效信息

**检测方式**: 单步 action prediction MSE vs IDM baseline
- 如果 MSE_ours > 2 × MSE_idm → underfit

**缓解**:
- 增大 D (256 → 512)
- 增加层数 (4 → 8)
- 方案 B 换成方案 A (获取多尺度 features)
- 加 IDM 蒸馏 loss: L_distill = ‖z_action - IDM_feature(obs)‖²

### 7.3 风险 3: SSM 多步 rollout 误差累积

**触发条件**: 用 SSM hidden state 做 multi-step dynamics rollout (MPC 场景)

**缓解**:
- Residual prediction: 预测 Δh = h_{t+1} - h_t 而非 h_{t+1}
- Horizon 限制: H ≤ 8
- Closed-loop: 每步重新 encode observation, 不纯依赖 rollout
- Scheduled training: 训练时逐步增加 rollout 长度

### 7.4 风险 4: 在已有 DiT 上加 pathway (无直接先例)

**本质问题**: Motus/UWM 都是从头联合训练, 我们是在 pretrained WAN 上加 pathway

**缓解**:
- 先用方案 B (完全独立, 验证 WAN features 是否 action-informative)
- 再考虑方案 A (需要先确认 features 本身是有用的)
- 参考 LoRA/adapter 的成功经验: 在 pretrained model 上加轻量模块是成熟范式

---

## 8. 实验路线

### Step 0: WAN video features 的 action 信息量验证 (1-2 天)

**目的**: 回答最根本的问题 — WAN DiT 的中间 features 里到底有没有 action 信息？如果没有，后面的设计都是空中楼阁。

```
实验设计:
  1. 用现有 WAN 跑 inference, 在 demo 数据上生成 video
  2. 提取 DiT 各层的 video features (layer 8, 16, 24, 32)
  3. 对每一帧的 video features 做空间 average pooling → f_l ∈ R^{1536}
  4. 线性探针: f_l → Linear → action_pred ∈ R^{14}
     - 训练: L2 loss vs GT action
     - 评估: R² score

期望结果:
  R² > 0.7 (某些层) → features 包含 action 信息 → 继续
  R² < 0.3 (所有层) → features 不含 action 信息 → 需要联合训练 (方案 A Level 3)

额外验证:
  - 哪一层最 action-informative? (决定方案 B 该读哪一层)
  - 空间 pooling vs attention pooling 哪个更好? (决定 cross-attn 的必要性)

所需资源: 1 GPU, demo 数据, 无需训练 WAN
```

### Step 1: 方案 B 最小可行原型 (3-5 天)

**前提**: Step 0 验证通过

```
实现:
  1. ActionLatentModule: 4 层 (cross-attn + self-attn + SSM + FFN)
     - D = 256, T = 8 action tokens
     - action_head: Linear(256, 14)
  2. WAN 完全冻结, 只训 ActionLatentModule
  3. 训练数据: demo (obs, action) pairs
     - 用冻结 WAN 提取 video features (离线, 一次性)
     - 训练 action module on features → action
  4. Loss = L_action = ‖action_head(z_T) - a_GT‖²

评估:
  - Action MSE vs IDM baseline
  - 各组件消融: 去掉 SSM / 去掉 self-attn / 去掉 cross-attn
  - SSM hidden state 可视化: h_t 在不同任务阶段的变化

所需资源: 1 GPU, ~2h 训练
```

### Step 2: 方案 A + 梯度隔离 (1 周)

**前提**: Step 1 验证 action pathway 本身能 work

```
实现:
  1. 修改 WAN DiT forward, 在每层加 action latent layer
  2. Level 2 训练: KV stop-gradient
  3. Loss = L_video + α · L_action

评估:
  - Action MSE (是否比方案 B 好)
  - Video FVD/FID (是否和 baseline WAN 一致)
  - 如果 video 质量下降 → 回退 Level 0/1
```

### Step 3: MPC 集成 (1 周)

**前提**: Step 2 确认 action 预测质量可用

```
用 SSM hidden state 做 latent dynamics rollout:
  h_{t+k} = SSM_forward(h_{t+k-1}, a_candidate_k)
  score = -‖h_{t+H} - h_goal‖²

对比实验:
  - 当前 WAN MPC (slow baseline)
  - Action Latent MPC (our method)
  - No MPC (direct policy)
```

---

## 9. 参考文献

1. RoboMamba — https://arxiv.org/abs/2406.04339
2. SpatialVLA-Mamba — https://openreview.net/forum?id=sTn4EqE49A
3. S4 for In-Context RL — https://arxiv.org/abs/2303.03982
4. Motus — https://arxiv.org/abs/2512.13030
5. LAC-WM — https://openreview.net/forum?id=vEZgPr1deb
6. UWM — https://arxiv.org/abs/2504.02792
7. StateSpaceDiffuser — https://arxiv.org/abs/2505.22246
8. EDELINE — https://arxiv.org/abs/2502.00466
