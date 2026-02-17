# Subgoal Planning Model 方案对比分析

> 核心问题：Wan 视频模型 horizon 有限（64/160帧），需要长程规划能力生成 visual subgoal keyframes

---

## 一、现有方案与候选方案总览

| 方案 | 类型 | 模型规模 | 分辨率 | 单次生成帧数 | 延迟 | 长程能力 |
|------|------|----------|--------|-------------|------|----------|
| **LIBERO (当前)** | U-Net diffusion | 201M | 128x128 | 7帧 | 0.5-2s/subgoal | 差（迭代累积误差） |
| **SuSIE** | InstructPix2Pix | ~1B (SD v1.5) | 256x256 | 1帧(subgoal) | ~1-3s/subgoal | 中（每次从当前观测重新生成） |
| **CoT-VLA** | 统一 VLA | 7B | tokenized | 1帧(subgoal) | 7x action-only | 中（autoregressive re-plan） |
| **VPP** | SVD features | 1.5B+320M | 256x256 | 16帧(internal) | <160ms | 强（隐式长程表征） |
| **Wan (当前)** | DiT diffusion | 14B | 640x736 | 81帧(4n+1) | ~10s+ | 差（global attention O(N²)） |
| **MAGI-1** | Chunk AR DiT | 24B | 高分辨率 | 24帧/chunk | 恒定/chunk | 强（chunk pipeline） |
| **SSM Video Diffusion** | SSM替换attention | ~同backbone | 32x32(验证) | 400帧+ | 线性 | 强（线性memory） |

---

## 二、详细分析

### 2.1 LIBERO（当前系统使用的 subgoal 模型）

**架构：** AVDC (Learning to Act from Actionless Videos) 的 factorized space-time U-Net

```
输入: 当前帧(128x128) + 任务文本(CLIP编码)
      ↓
  U-Net diffusion (100步 or DDIM 10步)
      ↓
输出: 7帧未来预测 → 取最后一帧作为 subgoal
      ↓
  迭代: 将 subgoal 作为下一轮输入 → 生成下一个 subgoal
```

**为什么不适合长程规划：**
1. **迭代累积误差**：每轮预测 7 帧，取末帧再预测。10 个 subgoal = 10 次迭代，误差指数累积
2. **Domain mismatch**：在 LIBERO 数据集（8个任务, 160个demo）上训练，不是 RoboTwin 域
3. **低分辨率**：128x128，丢失精细空间信息
4. **延迟过高**：10 个 subgoal × 0.5-2s = 5-20s 总生成时间
5. **代码中实际只用前 2 帧**：`use_vid_first_n_frames=2`，说明后续帧质量不可靠

**结论：LIBERO 模型本质上不具备长程规划能力，只是短程视频预测的迭代拼接。**

---

### 2.2 SuSIE（Image Editing → Subgoal）

**架构：** InstructPix2Pix (Stable Diffusion v1.5 UNet) finetuned

```
当前观测(256x256) + "grasp the bottle"
      ↓
  InstructPix2Pix (finetuned 40K steps)
      ↓
  subgoal 图像(256x256) — 场景中物体位置已改变
      ↓
  低层 goal-conditioned policy → actions
```

**关键特性：**
- 每 20 步从**当前真实观测**重新生成 subgoal → 不累积误差
- 在 BridgeData V2 + Something-Something V2 上 finetune
- **超越 RT-2-X (55B)** 和 **GT goal image oracle**
- 开源：[github.com/kvablack/susie](https://github.com/kvablack/susie)

**CALVIN 结果：**
| 连续任务数 | 成功率 |
|-----------|--------|
| 1 | 87.7% |
| 3 | 49.8% |
| 5 | 33.7% |
| 平均长度 | 2.80 |

**长程规划能力分析：**
- 优势：每次从当前观测重新生成，天然适应 closed-loop
- 劣势：**依赖语言 subgoal 分解** — 谁来产生 "grasp bottle" 这种中间指令？
- 劣势：单图生成，无时序一致性保证
- 延迟：~1-3s/subgoal

**与 Wan/VidarC 集成可行性：** 中等
- 可替换当前 LIBERO `vm_subgoal_generator.py`
- 需要在 RoboTwin 数据上 finetune InstructPix2Pix
- 仍需解决语言 subgoal 的自动生成

---

### 2.3 CoT-VLA（Visual Chain-of-Thought）

**架构：** 7B VLA (VILA-U backbone)，统一生成 subgoal image + actions

```
[观测 tokens] + [指令 tokens]
      ↓  causal attention
  生成 256 个 image tokens → 解码为 subgoal 图像
      ↓  full attention
  生成 action chunk (10步)
```

**关键创新：** 混合注意力机制
- **Causal attention**: 生成 subgoal image tokens（autoregressive，逐 token）
- **Full attention**: 生成 action tokens（全连接，关节协调）

**长程规划能力分析：**
- 优势：subgoal 生成与 action 预测在同一模型，端到端优化
- 劣势：**7x 延迟开销**（256 image tokens + 10 action tokens vs 仅 10 action tokens）
- 劣势：7B 模型，训练成本 11,000 A100-hours
- 劣势：需要替换整个 policy 架构，与现有 Wan/VidarC 不兼容

**LIBERO 结果：** 平均 81.13%（优于 OpenVLA 76.5%）

**与 Wan/VidarC 集成可行性：** 低
- 需要完全重构 policy 架构
- 无法复用现有 Wan 视频模型
- 延迟不适合实时控制

---

### 2.4 VPP（Video Prediction Policy）

**架构：** SVD internal features + Diffusion Transformer Policy

```
当前观测 + 噪声
      ↓
  SVD UNet 单步前向 (不做完整去噪!)
      ↓
  提取中间层 features (Layer 9 附近最优)
      ↓
  VideoFormer: T×L learnable tokens
  ├── spatial attention → 每帧独立空间关注
  └── temporal attention → 跨时间步关注
      ↓
  Diffusion Transformer Policy → action chunk (10步)
```

**核心洞察：不需要生成像素级 subgoal，内部表征更好**
- 单步前向提取特征 vs 50步去噪生成像素 → **延迟 <160ms**
- 中间层 features 编码了空间+时序动态信息
- "即使单步预测不完美，仍传递了关于物体运动和机械臂运动的有价值信息"

**CALVIN 结果（当前 SOTA）：**
| 连续任务数 | 成功率 |
|-----------|--------|
| 1 | 95.7% |
| 3 | 86.3% |
| 5 | 75.0% |
| **平均长度** | **4.29** |

对比：SuSIE 2.80，GR-1 3.06，3D Diffuser Actor 3.35

**长程规划能力分析：**
- 优势：**隐式长程表征** — features 编码了 16 帧的未来动态
- 优势：7-10 Hz 控制频率，实时可用
- 优势：模块化设计，可替换 video backbone
- 劣势：features 是隐式的，不如显式 visual keyframe 可解释
- 劣势：需要在 RoboTwin 数据上训练 VideoFormer

**与 Wan/VidarC 集成可行性：** 高
- 最自然的集成方式：从 Wan DiT 中间层提取 features 作为 predictive representations
- 不需要像素级 subgoal，避免了 guidance 信号质量问题
- 现有 `subgoal_guidance_scale` 机制可改为 feature-level guidance
- 开源：[github.com/roboterax/video-prediction-policy](https://github.com/roboterax/video-prediction-policy)

---

## 三、SSM-Based 长程记忆分析

### 3.1 SSM Meets Video Diffusion 架构

**论文：** arXiv 2403.07711 (ICLR 2024 Workshop)
**开源：** [github.com/shim0114/SSM-Meets-Video-Diffusion-Models](https://github.com/shim0114/SSM-Meets-Video-Diffusion-Models)

**核心替换：保留空间 attention，temporal attention → 双向 SSM**

```
原始 Video U-Net:
  每个 block = [空间层(2D conv + spatial attention)] + [时域层(temporal attention)]
                                                          ↑ 替换这里
替换后:
  每个 block = [空间层(2D conv + spatial attention)] + [时域层(Bidirectional SSM)]
```

**Factored 设计（关键）：**
- 空间层：输入 `(B×L) × (H×W) × C`，每帧独立处理空间
- 时域层：输入 `(B×H×W) × L × C`，每个空间位置独立处理时间
- **SSM 只处理时间维度**，不处理完整的 F×H×W 展平序列

### 3.2 SSM 中保留的长程信息是什么

**SSM 状态方程：**
```
s_k = A_bar · s_{k-1} + B_bar · u_k    (状态更新)
y_k = C_bar · s_k                        (输出)
```

其中 `s_k ∈ R^N` (N=512) 是隐状态向量。

**SSM 隐状态 = N 个独立的指数衰减记忆通道**

在 S4D (对角化 S4) 中，矩阵 A 是对角矩阵，每个对角元素是一个复数特征值：
- **衰减慢的通道**（特征值接近 0）：保留远距离时序趋势 — 全局场景变化、持久物体存在、整体运动方向
- **衰减快的通道**（大负特征值）：捕获近期局部动态 — 帧间运动、快速变化

可以理解为 **512 个并行的"记忆频道"**，每个有不同的时域感受野，从近期到远期。

**HiPPO 初始化**确保这些通道的衰减率均匀分布在对数尺度上，覆盖从短期到长期的完整时域范围。

### 3.3 SSM 保留 vs 丢失了什么

| SSM 保留的信息 | SSM 丢失的信息 |
|---------------|---------------|
| 平滑的时序趋势（渐变光照、全局运动） | 精确的帧间对应关系（"第23帧像素(x,y)的颜色"） |
| 周期性模式（往复运动、振荡） | 任意回溯能力（attention 可以精确查看任意历史帧） |
| 低频时域结构（场景整体动态） | 复杂的多实体时序交互 |
| 运动方向和速度的统计特征 | 物体在具体位置的精确 identity |

**根本瓶颈：** 隐状态大小固定（N=512），无论处理了多少帧。100帧的信息压缩到512维 vs 1000帧的信息也压缩到512维。而 attention 的 KV cache 随帧数线性增长，可以精确保留所有历史信息。

### 3.4 Selectivity 机制（Mamba/S6）

在 Mamba 变体中，B、C 和步长 Δ 变为**输入依赖函数**：
```
B_k = Linear(x_k)         # 输入投影
C_k = Linear(x_k)         # 输出投影
Δ_k = softplus(Linear(x_k))  # 输入依赖的步长
```

- **小 Δ_k**：忽略当前输入，保留已有状态（跳过冗余/静态帧）
- **大 Δ_k**：强烈整合当前输入（新物体出现、重要运动）

**但实验发现 S4D 优于 Mamba（在视频扩散任务中）：**

| 架构 | UCF101 FVD | MineRL-64 FVD |
|------|-----------|--------------|
| Mamba (单向) | 669.572 | 1722.097 |
| Bi-Mamba + MLP | 243.638 | 1138.779 |
| **S4D 双向 (论文方法)** | **226.447** | **1132.982** |

**原因：** 视频扩散去噪是全局操作（同时处理所有帧），HiPPO 初始化的固定动态更适合这种平滑时域结构。

### 3.5 双向性至关重要

| 配置 | UCF101 FVD |
|------|-----------|
| 双向 SSM + 2层MLP | **226.447** |
| 单向 SSM + 2层MLP | 669.582 |

单向 SSM 性能暴跌 3x！因为扩散模型同时去噪所有帧，需要前后文的双向上下文。

### 3.6 长程结果

| 帧数 | Attention | SSM |
|------|-----------|-----|
| 16帧 | 272 | **226** ← SSM 更好 |
| 64帧 | **1073** | 1132 |
| 200帧 | **1032** | 1116 |
| 400帧 | **OOM** | **972** ← attention 内存溢出 |

SSM 在 400 帧时 attention 完全无法运行，而 SSM 的 FVD 反而随帧数增加而**改善**（1132→972）。

### 3.7 Hybrid SSM + Local Attention（ICCV 2025）

**论文：** Long-Context State-Space Video World Models (Stanford/Adobe)

**架构：** Mamba blocks + 局部因果注意力（k=10帧窗口）

```
每层 = [Mamba SSM block] → [局部 attention (仅看最近10帧)]
         ↑                      ↑
   全局长程压缩记忆        精确近期帧细节
```

**"恒定每帧推理速度"的原因：**
- Mamba 状态：固定大小，不随帧数增长
- 局部 attention KV cache：固定窗口 k=10，不随帧数增长
- → 无论生成第10帧还是第1000帧，计算量相同

**结果（Memory Maze, 400帧）：**

| 方法 | SSIM |
|------|------|
| Causal Transformer (192帧上下文) | 0.829 |
| Mamba2 only | 0.735 |
| **Hybrid SSM + Local Attn** | **0.898** |
| Full-context Transformer (上界) | 0.914 |

Hybrid 达到 full-context transformer 的 **98%** 性能，同时保持恒定推理成本。

---

## 四、MAGI-1 vs Wan 架构对比

### 4.1 Wan 的架构与 horizon 限制

```
Wan (当前):
  输入: [F×H×W 完全展平] → global attention → O(N²)

  81帧 × 88×160(latent) = ~1.1M tokens
  attention 复杂度: O(1.1M²) ≈ O(1.2T)
  KV cache: 线性增长，每新帧 +88×160=14K tokens
```

**Wan causal generation 流程：**
1. Prefill: 处理条件帧，建立 KV cache
2. Chunk prefill: 每次生成新帧，将新 KV 拼接到 cache
3. 随着生成帧数增加，KV cache 不断膨胀 → 内存和计算线性增长

**Wan 的 horizon 瓶颈：**
- 不是模型能力问题（40层 DiT, 5120 dim 足够强）
- 是 **内存和计算的物理限制**：160帧已经接近 A100 80GB 的上限
- 更长视频 → KV cache 溢出 or 注意力质量下降

### 4.2 MAGI-1 的 Chunk Autoregressive 方案

**论文：** MAGI-1: Autoregressive Video Generation at Scale (Sand AI, 2025)
**开源：** [github.com/SandAI-org/MAGI-1](https://github.com/SandAI-org/MAGI-1)
**规模：** 24B 参数

```
MAGI-1:
  [Chunk 1: 24帧] → [Chunk 2: 24帧] → [Chunk 3: 24帧] → ...
       ↓                  ↓                  ↓
    bidirectional      bidirectional      bidirectional
    (chunk 内全局)     (chunk 内全局)     (chunk 内全局)

  chunk 间: block-causal attention (只看前面的 chunks)
```

### 4.3 MAGI-1 vs Wan 的关键区别

| 维度 | Wan | MAGI-1 |
|------|-----|--------|
| **生成方式** | 一次性生成全部帧 or 逐帧 causal | 按 chunk (24帧) 分段生成 |
| **attention 范围** | 全局（所有帧互相看） | chunk 内全局 + chunk 间因果 |
| **KV cache** | 随总帧数线性增长 | 固定为当前 chunk + 历史 chunks 的压缩表示 |
| **峰值计算** | 随视频长度增长 | **恒定**（每 chunk 计算量相同） |
| **时序一致性** | 全局 attention 自然保证 | block-causal + overlap 保证 |
| **流式生成** | 不支持（需等全部生成完） | 支持（chunk pipeline 并行） |
| **最大长度** | ~160帧（实际） | 理论无限（已验证 4M tokens） |

### 4.4 MAGI-1 为什么可以长程规划

**三个关键机制：**

**1. Chunk Pipeline 并行**
```
时间 →
Chunk 1: [去噪完成] ─────────────────────
Chunk 2:        [去噪中] ──────────────────
Chunk 3:              [去噪中] ────────────
Chunk 4:                    [开始去噪] ────

最多 4 个 chunk 同时处理，吞吐量 4x
```
当前 chunk 还没完全去噪完成，下一个 chunk 已经开始去噪。这利用了"partially denoised frames 已包含足够的结构信息"的事实。

**2. Block-Causal Attention**
```
        Chunk1  Chunk2  Chunk3  Chunk4
Chunk1:   ✓      ✗      ✗      ✗     (只看自己)
Chunk2:   ✓      ✓      ✗      ✗     (看 Chunk1 + 自己)
Chunk3:   ✓      ✓      ✓      ✗     (看所有前面的)
Chunk4:   ✓      ✓      ✓      ✓
```
每个 chunk 可以看到所有前面的 chunks，但不能看后面的。chunk 内部是 bidirectional（全局 attention）。

**3. Chunk-Wise Prompting**
每个 chunk 可以接收不同的 text prompt → 支持场景切换和长叙事。
```
Chunk 1: "机器人接近瓶子"
Chunk 2: "机器人抓取瓶子"
Chunk 3: "机器人放置瓶子"
```

**对比 Wan：**
- Wan 只能给整个视频一个 prompt
- MAGI-1 可以给每个 chunk 不同的 prompt → **天然的 subgoal conditioning**

### 4.5 MAGI-1 与 Self-Forcing 的关系

你们的 Self-Forcing 已经在做 autoregressive chunk generation：
- 训练时用 self-generated frames 作为条件（而非 GT）
- 推理时 KV cache 实现 causal generation

MAGI-1 的核心改进：
1. **Pipeline 并行** → 吞吐量提升 4x
2. **Block-causal 而非 token-causal** → chunk 内保留 bidirectional quality
3. **Chunk-wise prompting** → 长程语义控制
4. **更大规模** (24B) → 更强的表达能力

---

## 五、综合对比与建议

### 5.1 长程规划能力排序

```
MAGI-1 Chunk AR ≈ Hybrid SSM+Attn >> VPP (隐式) > SuSIE (reactive) >> LIBERO (迭代) ≈ Wan (直接生成)
```

### 5.2 与现有系统集成难度排序（从易到难）

```
VPP (提取 Wan 中间层 features) < SuSIE (替换 vm_subgoal_generator)
< MAGI-1 Chunk AR (改 Wan 为 chunk 生成) < SSM Hybrid (替换 temporal attention)
< CoT-VLA (完全重构)
```

### 5.3 建议路线

| 阶段 | 方案 | 工作量 | 预期效果 |
|------|------|--------|----------|
| **短期 (1-2周)** | VPP 思路：从 Wan DiT 中间层提取 predictive features，替换像素级 subgoal guidance | 中 | 延迟 <200ms，隐式长程信息 |
| **中期 (1-2月)** | MAGI-1 思路：将 Wan 改为 chunk autoregressive + block-causal attention + chunk-wise prompting | 大 | 根本解决 horizon 限制，支持任意长度 |
| **长期 (研究)** | Hybrid SSM+Attention：temporal attention → Mamba + 局部 attention | 大 | O(N) 复杂度，恒定推理成本 |

### 5.4 短期方案具体思路（VPP 式集成）

```python
# 当前: 完整去噪 → 像素 → 编码 → guidance
subgoal_frames = [pixel_image_1, pixel_image_2, ...]
guidance = subgoal_guidance_scale * gradient(latent_distance(current, subgoal_pixel))

# 改进: 单步前向 → 中间层 features → feature-level guidance
wan_features = wan_dit.forward_single_step(current_obs + noise)  # 单步，不完整去噪
predictive_repr = extract_mid_layers(wan_features)  # Layer ~20 (40层DiT的中间)
guidance = feature_guidance_scale * gradient(feature_distance(current_repr, predictive_repr))
```

优势：
- 复用现有 Wan 模型，不需要额外模型
- 单步前向 << 完整去噪（40-50步），延迟大幅降低
- 中间层 features 包含隐式的时域+空间动态信息
- 不需要解决"谁生成语言 subgoal"的问题

---

## 六、实验记录与排查日志

### 6.1 Wan2.2 TI2V-5B 生成质量验证 (2025-02-11)

**目标：** 验证预训练 WanTI2V 能否在 robot 场景生成可用的 plan 视频

#### Exp A: 低分辨率生成 (verify_plan_quality.py)

| 参数 | 值 |
|------|-----|
| 输入 | cam_high 单视角首帧 (从 HDF5) |
| max_area | 122,880 (320×384) |
| frame_num | 121 |
| steps | 20 |
| guide_scale | 5.0 |
| shift | 5.0 (默认) |
| prompt | 英文 ("A top-down view of an aloha robot...") |

**结果：完全崩溃 — 生成随机噪声/色块**

**排查过程：**

1. **初始假设：multi-view unified_image 超出分布** → 改为 cam_high 单视角 → 仍然崩溃 ❌
2. **代码 diff vidar fork vs 原始 Wan2.2** → 发现 vidar fork 删除了 negative prompt 默认值 → 修复 → 未重测（用户质疑此项不足以造成完全崩溃）
3. **隔离测试（verify_plan_quality_orig.py）** → 使用原始 Wan2.2 代码路径，同样参数 → **确认同样崩溃**
4. **初步结论：标记为 domain gap** → ❌ 此结论可能有误！

#### 根因再分析 (CRITICAL)

深入排查发现 **三个叠加因素**，其中分辨率是最可能的主因：

| # | 严重性 | 问题 | 影响 |
|---|--------|------|------|
| **1** | **致命** | `max_area=122,880` 比模型训练分辨率 `901,120` (704×1280) 小 **7.3x** | DiT 位置编码、注意力模式、噪声调度全部 OOD |
| **2** | **严重** | `shift=5.0` 用于 720p，480p 应用 `shift=3.0`，320p 更应降低 | 噪声调度 SNR 轨迹完全错误 |
| **3** | **中等** | vidar fork 缺失 negative prompt 默认值 | CFG 无条件分支退化，质量下降但不致崩溃 |

**证据：**
- Wan2.2 官方 `SUPPORTED_SIZES` 仅支持 `704*1280` 和 `1280*704`
- Wan2.2 i2v 方法注释明确写道：*"If you want to generate a 480p video, set shift to 3.0"*
- latent 空间维度对比：我们 26×18 (468 tokens/帧) vs 官方 80×44 (3520 tokens/帧)

**被排除的假设：**
- ❌ 输入图像格式错误 → PIL Image RGB uint8，与原始代码一致
- ❌ prompt 语言问题 → umt5-xxl 多语言，官方示例也用英文
- ❌ VAE 版本错误 → TI2V-5B 使用 Wan2_2_VAE，代码正确
- ❌ 代码逻辑不匹配 → i2v() 方法字节级一致（除 neg prompt）

#### 待验证实验

| Exp | 描述 | 参数变更 | 状态 |
|-----|------|----------|------|
| B | 720p 原生分辨率 | `max_area=901120, shift=5.0, steps=50, frame_num=49` | **✅ 通过** |
| C | 480p + 低 shift | `max_area=~400000, shift=3.0, steps=50` | **待测** |
| D | Vidar Stage 1 权重 (vidar.pt) | 加载 `pt_dir=vidar.pt`，720p | **待测** |
| E | Vidar Stage 1 权重 + 低分辨率 | `pt_dir=vidar.pt`，480p, shift=3.0 | **待测** |

#### Exp B 结果 (2025-02-11)

**结果：视频连贯，无伪影和异常色彩 ✅**

**但**：生成内容不符合双臂 aloha robot 单视角场景 setting — 预训练模型不理解 robot manipulation 域

**结论：**
1. ✅ **分辨率假设确认** — `max_area=122,880` 是之前崩溃的根本原因，NOT domain gap
2. ⚠️ **Domain gap 依然存在但表现不同** — 不是随机噪声，而是"内容不对"（生成通用视频而非 robot 场景）
3. → **下一步：Exp D（vidar.pt 权重）** — Stage 1 已在 robot data 微调，应能生成正确的 robot 场景

---

### 6.2 v1_subgoal 系统分析

**v1_subgoal 使用的模型：LIBERO/AVDC (非 Wan2.2)**

| 组件 | 模型 | 规模 |
|------|------|------|
| Subgoal 生成 | AVDC factorized U-Net | 201M |
| 分辨率 | 128×128 | - |
| 每次输出 | 7帧，取末帧作为下一轮输入 | - |
| 迭代方式 | 自回归，末帧→首帧 | 10轮 = 10个subgoal |

**v1_subgoal 的问题（为什么不行）：**
1. **128×128 分辨率过低** — 丢失精细空间信息，robot 零件/物体细节不可见
2. **迭代累积误差** — 每轮预测 7 帧取末帧，10 个 subgoal = 10 次迭代，误差指数积累
3. **Domain mismatch** — 在 LIBERO 数据集训练，不是 RoboTwin 域
4. **实际只用前 2 帧** — `use_vid_first_n_frames=2`，说明后续帧质量不可靠
5. **延迟过高** — 10 subgoal × 0.5-2s = 5-20s
6. **无长程一致性** — 模型没有全局视野，每轮独立预测

**结论：LIBERO 本质是短程视频预测的迭代拼接，不具备长程规划能力。**

---

### 6.3 代码修复记录

| 日期 | 文件 | 修复内容 | 影响 |
|------|------|----------|------|
| 2025-02-11 | `vidar/wan/textimage2video.py` t2v() | 补回 `if n_prompt == "": n_prompt = self.sample_neg_prompt` | CFG 质量提升 |
| 2025-02-11 | `vidar/wan/textimage2video.py` i2v() | 同上 | 同上 |
| 2025-02-11 | `verify_plan_quality.py` | cam_high 替换 unified_image | 排除 multi-view OOD |
| 2025-02-11 | `verify_plan_quality.py` | imageio+ffmpeg 替换 cv2.VideoWriter | 视频兼容性 |

---

## 七、2025-2026 新增 Robot 视频规划模型

### 7.1 Large Video Planner (最直接相关!)

**基座模型：Wan 2.1 14B** — 与我们的 Wan 2.2 同源

| 属性 | 值 |
|------|-----|
| 论文 | "Large Video Planner Enables Generalizable Robot Control" (Dec 2025) |
| 基座 | **Wan 2.1 14B** (开源视频生成模型) |
| 方法 | Diffusion Forcing 微调 |
| 训练数据 | Ego4D + Epic-Kitchens + Panda |
| 长程能力 | 支持超出训练 horizon 的因果生成 |
| 开源 | **是** — [github.com/buoyancy99/large-video-planner](https://github.com/buoyancy99/large-video-planner) |

**关键价值：** 证明了在 Wan 上微调做 robot planning 是可行的。Diffusion Forcing 方法可能直接移植到 Wan 2.2 / vidar。

### 7.2 Cosmos Policy (NVIDIA, Jan 2026)

| 属性 | 值 |
|------|-----|
| 基座 | Cosmos-Predict2-2B |
| 方法 | **不改架构**，actions/states/values 编码为额外 latent frames |
| 结果 | LIBERO 98.5%, RoboCasa 67.1% |
| 开源 | **是** — [NVIDIA Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/) |

**关键价值：** 证明了"零架构修改"微调视频基座模型做 robot control 是可行的。

### 7.3 VideoVLA (NeurIPS 2025)

| 属性 | 值 |
|------|-----|
| 基座 | **CogVideoX** |
| 方法 | Multi-modal DiT，联合建模 video + language + action |
| 推理 | 预测 4 future latents (13帧) + 6 actions/step，~3Hz on H100 |
| 结果 | 12 tasks 平均 63.0% (超 CogACT, pi0, SpatialVLA) |

### 7.4 RoboEnvision (IROS 2025)

| 属性 | 值 |
|------|-----|
| 方法 | VLM 分解子任务 → 关键帧生成 → 帧间插值 |
| 长程 | **非自回归** — 避免误差累积 |
| 关键 | 语义保持注意力模块，两阶段（keyframe + interpolation） |

**关键价值：** 与我们的方案B(VLM分解+检索)思路相似，但用生成替代检索。

### 7.5 模型能力对比更新

| 模型 | 基座 | 分辨率 | 长程帧数 | 开源 | Robot 微调 | 与 Vidar 集成 |
|------|------|--------|----------|------|-----------|--------------|
| LIBERO/AVDC (v1当前) | 自训练 U-Net | 128×128 | 7帧/轮 | 是 | LIBERO域 | 已集成 |
| **Wan2.2 TI2V-5B** | 预训练 | 704×1280 | 81-121帧 | 是 | ❌ 需微调 | 可直接用 |
| **Vidar Stage 1** | Wan2.2 微调 | 736×640 | 81帧 | 内部 | ✅ robot data | **最佳候选** |
| Large Video Planner | Wan 2.1 14B | 高分辨率 | 超训练长度 | 是 | Ego4D等 | 高 (同源) |
| Cosmos Policy | Cosmos-2B | 多分辨率 | 多种 | 是 | 多域 | 中 (需适配) |
| VideoVLA | CogVideoX | 多分辨率 | 13帧/步 | 部分 | 12 tasks | 低 (不同架构) |
| VPP | SVD features | 256×256 | 16帧(隐式) | 是 | 多域 | 高 (feature级) |

---

## 八、关键参考文献（更新版）

| 方法 | 论文 | 代码 |
|------|------|------|
| LIBERO/AVDC | Ko et al., ICLR 2024 | [github.com/flow-diffusion/AVDC](https://github.com/flow-diffusion/AVDC) |
| SuSIE | Black et al., ICLR 2025 | [github.com/kvablack/susie](https://github.com/kvablack/susie) |
| GHIL-Glue | Hatch et al., ICRA 2025 | [github.com/kyle-hatch-tri/ghil-glue](https://github.com/kyle-hatch-tri/ghil-glue) |
| CoT-VLA | Zhao et al., CVPR 2025 | [cot-vla.github.io](https://cot-vla.github.io/) |
| VPP | ICML 2025 Spotlight | [github.com/roboterax/video-prediction-policy](https://github.com/roboterax/video-prediction-policy) |
| SSM Video Diffusion | Oshima et al., ICLR 2024 WS | [github.com/shim0114/SSM-Meets-Video-Diffusion-Models](https://github.com/shim0114/SSM-Meets-Video-Diffusion-Models) |
| Long-Context SSM WM | Po et al., ICCV 2025 | [ryanpo.com/ssm_wm](https://ryanpo.com/ssm_wm/) |
| MAGI-1 | Sand AI, 2025 | [github.com/SandAI-org/MAGI-1](https://github.com/SandAI-org/MAGI-1) |
| CausVid | CVPR 2025 | [github.com/tianweiy/CausVid](https://github.com/tianweiy/CausVid) |
| Context Forcing | 2025 | [github.com/TIGER-AI-Lab/Context-Forcing](https://github.com/TIGER-AI-Lab/Context-Forcing) |
| **Large Video Planner** | Boyuan Chen et al., Dec 2025 | [github.com/buoyancy99/large-video-planner](https://github.com/buoyancy99/large-video-planner) |
| **Cosmos Policy** | NVIDIA, Jan 2026 | [NVIDIA Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/) |
| **VideoVLA** | NeurIPS 2025 | [videovla-nips2025.github.io](https://videovla-nips2025.github.io/) |
| RoboEnvision | IROS 2025 | [arXiv:2506.22007](https://arxiv.org/abs/2506.22007) |
| CoVAR | Dec 2025 | [arXiv:2512.16023](https://arxiv.org/abs/2512.16023) |
| mimic-video | Dec 2025 | [mimic-video.github.io](https://mimic-video.github.io/) |
| Vid2World | 2025 | [arXiv:2505.14357](https://arxiv.org/abs/2505.14357) |
| AVID | Microsoft, RLC 2025 | [arXiv:2410.12822](https://arxiv.org/abs/2410.12822) |
| UWM | RSS 2025 | [weirdlabuw.github.io/uwm](https://weirdlabuw.github.io/uwm/) |
| GR-2 / GR-3 | ByteDance 2024-2025 | [arXiv:2410.06158](https://arxiv.org/abs/2410.06158) |

---

## 九、下一步行动计划

### 优先级 1：排除分辨率假设（最快验证）
1. **Exp B**: 用原生 720p (max_area=901120, shift=5.0, steps=50) 跑 WanTI2V 预训练模型
   - 成功 → 确认是分辨率问题，NOT domain gap
   - 失败 → 确认是 domain gap

### 优先级 2：测试 Vidar Stage 1 权重
2. **Exp D**: 加载 vidar.pt (Stage 1 robot 微调权重) 到 WanTI2V
   - 成功 → 直接用 vidar.pt 作为 non-causal planner，无需额外训练
   - 失败 → 需要专门为 planning 任务微调

### 优先级 3：参考 Large Video Planner
3. 研究 Large Video Planner 的 Diffusion Forcing 微调方法
   - 同为 Wan 基座，方法可直接移植
   - 支持超出训练 horizon 的因果生成
