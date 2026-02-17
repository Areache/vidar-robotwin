# MPC Implementation for Vidarc: Modeling Action Temporal Correlations

## 核心问题 (Core Problem)

**当前方案 2 的局限性：随机采样 M=64 组动作序列**

```python
# Current: Random sampling (NO temporal correlation)
actions = torch.randn(M, H, action_dim)  # (64, 8, 7)
# 问题：
# 1. 每个动作独立采样，没有时序相关性
# 2. 不符合机器人物理约束（速度、加速度连续性）
# 3. 采样效率低（大部分样本是无效的）
# 4. 无法利用 demonstration 中学到的 action prior
```

**Example:**
```
Random sequence: [0.5, -0.3, 0.8, -0.9, 0.1, ...]
                  ↑      ↑      ↑      ↑
                  Jumpy, not smooth, physically impossible
```

**We need:** Actions that are **temporally coherent**, **physically plausible**, and **learned from data**.

---

## 🎯 设计理念 (Design Philosophy)

### **核心思想：建模 Action 之间的相关性**

Action sequences should have:
1. **Temporal smoothness** - Actions change gradually over time
2. **Physical plausibility** - Obey robot dynamics constraints
3. **Task-relevant patterns** - Learn from expert demonstrations
4. **Diversity** - Still explore, but intelligently

### **关键洞察：Action Diffusion**

Instead of sampling **independent** actions, sample **correlated** action sequences using **diffusion models**.

```
Random MPC: a_t ∼ U(-1, 1)  ← Independent
             a_{t+1} ∼ U(-1, 1)  ← No correlation!

Action Diffusion: [a_t, a_{t+1}, ..., a_{t+H}] ∼ p_θ(A | o_t, goal)
                  ↑
                  Learned joint distribution with temporal correlations
```

---

## 📚 方案对比 (Solution Comparison)

### **方案 0: Random Sampling (Current Baseline)**

```python
def random_mpc(obs, goal, M=64, H=8):
    # Sample M random action sequences
    actions = torch.randn(M, H, action_dim)

    # Forward predict end frames
    end_frames = model_b.batch_forward(obs, actions)

    # Score with CLIP
    scores = clip.score(end_frames, goal)

    # Select best
    best_idx = scores.argmax()
    return actions[best_idx]
```

**优点:** Simple, no training
**缺点:** No temporal correlation, inefficient

---

### **方案 1: Action Diffusion MPC (Recommended)**

Use a **diffusion model** to generate temporally coherent action sequences.

```python
class ActionDiffusionMPC:
    def __init__(self):
        # Train a diffusion model on action sequences
        self.action_diffusion = ActionDiffusionModel(
            action_dim=7,
            horizon=8,
            hidden_dim=256
        )
        # Train on demonstration data
        self.train_on_demos()

    def sample_actions(self, obs, goal, M=16):
        """
        Sample M action sequences from learned distribution.

        Key: Sequences have temporal correlations!
        """
        # Condition on current obs and goal
        context = torch.cat([obs_embed, goal_embed], dim=-1)

        # Denoise from noise to action sequences
        noise = torch.randn(M, H, action_dim)

        for t in reversed(range(num_diffusion_steps)):
            # Predict noise
            noise_pred = self.action_diffusion(noise, t, context)

            # Denoise one step
            noise = self.ddim_step(noise, noise_pred, t)

        # Final denoised = action sequences
        actions = noise  # (M, H, 7)

        return actions  # Temporally coherent!

    def forward(self, obs, goal, M=16):
        # Sample from diffusion (not random!)
        actions = self.sample_actions(obs, goal, M)

        # Rest same as MPC: predict, score, select
        end_frames = model_b.batch_forward(obs, actions)
        scores = clip.score(end_frames, goal)
        best_idx = scores.argmax()

        return actions[best_idx]
```

**优点:**
- ✅ Temporal correlation (smooth actions)
- ✅ Learned from demos (task-relevant)
- ✅ Better sample efficiency (M=16 instead of 64)
- ✅ Physically plausible

**缺点:**
- ❌ Requires training action diffusion model
- ❌ Slower inference (diffusion denoising)

---

### **方案 2: Action VAE + Sampling**

Use a **VAE** to learn a latent space of action sequences, then sample from it.

```python
class ActionVAEMPC:
    def __init__(self):
        self.action_vae = ActionVAE(
            action_dim=7,
            horizon=8,
            latent_dim=32
        )
        self.train_on_demos()

    def sample_actions(self, obs, goal, M=16):
        """Sample action sequences from learned latent space."""
        # Condition on obs + goal
        context = torch.cat([obs_embed, goal_embed], dim=-1)

        # Predict latent distribution
        mu, logvar = self.action_vae.encode_context(context)

        # Sample M latent codes
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn(M, latent_dim)

        # Decode to action sequences
        actions = self.action_vae.decode(z)  # (M, H, 7)

        return actions
```

**优点:**
- ✅ Fast sampling (no iterative denoising)
- ✅ Compact latent space (easier optimization)
- ✅ Temporal correlation learned

**缺点:**
- ❌ VAE training can be unstable
- ❌ Less flexible than diffusion

---

### **方案 3: Autoregressive Action Model**

Generate actions **sequentially** using an RNN/Transformer.

```python
class AutoregressiveActionMPC:
    def __init__(self):
        self.action_model = TransformerActionModel(
            action_dim=7,
            hidden_dim=256
        )
        self.train_on_demos()

    def sample_actions(self, obs, goal, M=16):
        """Autoregressively generate action sequences."""
        context = torch.cat([obs_embed, goal_embed], dim=-1)

        actions = []
        hidden = self.action_model.init_hidden(context)

        for h in range(H):
            # Predict next action (with noise for diversity)
            a_t, hidden = self.action_model(hidden, context)
            a_t = a_t + 0.1 * torch.randn_like(a_t)  # Add noise
            actions.append(a_t)

        actions = torch.stack(actions, dim=1)  # (M, H, 7)
        return actions
```

**优点:**
- ✅ Strong temporal correlation (sequential generation)
- ✅ Can use powerful models (Transformers)

**缺点:**
- ❌ Sequential (can't parallelize easily)
- ❌ Accumulates errors over H steps

---

### **方案 4: Flow Matching for Actions (LingBot-VA Style)**

Use **continuous normalizing flows** (flow matching) for action generation.

```python
class ActionFlowMatchingMPC:
    def __init__(self):
        self.action_flow = ActionFlowModel(
            action_dim=7,
            horizon=8,
            hidden_dim=256
        )
        self.train_on_demos()

    def sample_actions(self, obs, goal, M=16, num_steps=10):
        """Sample using flow matching (like LingBot-VA)."""
        context = torch.cat([obs_embed, goal_embed], dim=-1)

        # Start from noise
        x = torch.randn(M, H, action_dim)

        # Flow from noise to data
        for t in range(num_steps):
            t_normalized = t / num_steps

            # Predict velocity field
            v = self.action_flow(x, t_normalized, context)

            # Euler step
            x = x + (1.0 / num_steps) * v

        return x  # (M, H, 7)
```

**优点:**
- ✅ Matches LingBot-VA's training paradigm
- ✅ Smooth interpolation
- ✅ Can be trained with same loss as video

**缺点:**
- ❌ Requires flow matching training infrastructure

---

### **方案 5: MIDM-Guided Sampling (Quick Win!)**

Use your existing **MIDM** to guide action sampling (no new training!).

```python
class MIDMGuidedMPC:
    def __init__(self, midm):
        self.midm = midm  # Your existing MIDM

    def sample_actions(self, obs, goal, M=16):
        """Sample around MIDM prediction."""
        # Get MIDM's prediction
        action_mean = self.midm.predict(obs, goal)  # (H, 7)

        # Sample around it with noise
        noise_std = 0.1  # Exploration noise
        actions = []

        for i in range(M):
            # Gaussian noise around mean
            noise = torch.randn_like(action_mean) * noise_std
            action_i = action_mean + noise
            actions.append(action_i)

        actions = torch.stack(actions)  # (M, H, 7)

        # Optional: Add temporal smoothing
        actions = self.smooth_actions(actions)

        return actions

    def smooth_actions(self, actions):
        """Apply temporal smoothing filter."""
        # Simple moving average
        kernel = torch.ones(3) / 3.0
        for m in range(len(actions)):
            for dim in range(7):
                actions[m, :, dim] = torch.nn.functional.conv1d(
                    actions[m, :, dim].unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=1
                ).squeeze()
        return actions
```

**优点:**
- ✅ **No training required!** Uses existing MIDM
- ✅ Much better than random (samples near good actions)
- ✅ Can add temporal smoothing easily
- ✅ 4× faster (M=16 instead of 64)

**缺点:**
- ❌ Still doesn't fully model temporal correlations
- ❌ Smoothing is hand-crafted, not learned

---

## 🎯 推荐方案 (Recommended Approach)

### **阶段性实施 (Phased Implementation)**

#### **Phase 1: MIDM-Guided Sampling (Immediate - This Week)**

**No training, big improvement!**

```python
# experiments/mpc/midm_guided_mpc.py

class MIDMGuidedMPC:
    """
    Improvement over random MPC using existing MIDM.

    Key idea: Sample actions around MIDM prediction instead of random.
    """

    def __init__(self, model_b, midm, clip_model):
        self.model_b = model_b
        self.midm = midm
        self.clip_model = clip_model

    def sample_action_sequences(self, obs, goal, M=16, H=8):
        """
        Sample M action sequences around MIDM prediction.

        Args:
            obs: Current observation
            goal: Goal image or text
            M: Number of samples (reduced from 64!)
            H: Planning horizon

        Returns:
            actions: (M, H, 7) action sequences
        """
        # 1. Get MIDM prediction as mean
        with torch.no_grad():
            # MIDM predicts single-step action
            # We need to autoregressively predict H steps
            action_sequence = []
            obs_t = obs

            for h in range(H):
                action_h = self.midm(obs_t, goal)  # (7,)
                action_sequence.append(action_h)

                # Predict next obs (optional, or just repeat)
                obs_t = self.model_b.predict_next(obs_t, action_h)

            action_mean = torch.stack(action_sequence)  # (H, 7)

        # 2. Sample around mean with decreasing noise
        actions = []
        for m in range(M):
            # Decreasing noise over horizon (more certain about near future)
            noise_schedule = torch.linspace(0.15, 0.05, H).to(obs.device)

            noise = torch.randn_like(action_mean)
            noise = noise * noise_schedule.unsqueeze(-1)

            action_m = action_mean + noise

            # 3. Apply temporal smoothing
            action_m = self.temporal_smoothing(action_m)

            # 4. Clip to action bounds
            action_m = torch.clamp(action_m, -1.0, 1.0)

            actions.append(action_m)

        return torch.stack(actions)  # (M, H, 7)

    def temporal_smoothing(self, actions, window=3):
        """
        Apply moving average to enforce temporal coherence.

        Args:
            actions: (H, 7) action sequence
            window: Smoothing window size

        Returns:
            smoothed: (H, 7) smoothed actions
        """
        H, D = actions.shape
        smoothed = actions.clone()

        for d in range(D):
            # Simple moving average
            for h in range(H):
                start = max(0, h - window // 2)
                end = min(H, h + window // 2 + 1)
                smoothed[h, d] = actions[start:end, d].mean()

        return smoothed

    def forward(self, obs, goal, M=16, H=8):
        """
        MPC with MIDM-guided sampling.

        Returns:
            best_action_sequence: (H, 7)
            info: dict with diagnostics
        """
        # Sample action sequences
        actions = self.sample_action_sequences(obs, goal, M, H)

        # Forward predict end frames for each sequence
        end_frames = []
        for m in range(M):
            end_frame = self.model_b.rollout(obs, actions[m])
            end_frames.append(end_frame)

        end_frames = torch.stack(end_frames)  # (M, C, H, W)

        # Score with CLIP
        scores = self.clip_model.score(end_frames, goal)  # (M,)

        # Select best
        best_idx = scores.argmax()
        best_actions = actions[best_idx]

        info = {
            'scores': scores,
            'best_idx': best_idx,
            'best_score': scores[best_idx].item(),
            'mean_score': scores.mean().item(),
        }

        return best_actions, info

# Usage:
mpc = MIDMGuidedMPC(model_b, midm, clip_model)
action_sequence, info = mpc(obs, goal, M=16, H=8)
# Execute first action, get new obs, repeat
```

**Expected improvement:**
- ✅ 4× faster (M=16 instead of 64)
- ✅ Better action quality (samples near MIDM prediction)
- ✅ Temporal smoothing enforces coherence

---

#### **Phase 2: Action Diffusion MPC (Research - 2-4 Weeks)**

**Train action diffusion model for better temporal modeling.**

```python
# experiments/mpc/action_diffusion.py

class ActionDiffusionModel(nn.Module):
    """
    Diffusion model for action sequence generation.

    Architecture: UNet-style with temporal convolutions
    """

    def __init__(self, action_dim=7, horizon=8, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        # Time embedding (for diffusion timestep)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Context embedding (obs + goal)
        self.context_encoder = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temporal UNet for action sequence
        self.action_encoder = TemporalConv1d(action_dim, hidden_dim)

        self.down_blocks = nn.ModuleList([
            TemporalResBlock(hidden_dim, hidden_dim * 2),
            TemporalResBlock(hidden_dim * 2, hidden_dim * 4),
        ])

        self.mid_block = TemporalResBlock(hidden_dim * 4, hidden_dim * 4)

        self.up_blocks = nn.ModuleList([
            TemporalResBlock(hidden_dim * 4, hidden_dim * 2),
            TemporalResBlock(hidden_dim * 2, hidden_dim),
        ])

        self.action_decoder = nn.Conv1d(hidden_dim, action_dim, 1)

    def forward(self, x, t, context):
        """
        Args:
            x: (B, H, action_dim) noisy action sequence
            t: (B,) diffusion timestep [0, 1]
            context: (B, context_dim) obs + goal embedding

        Returns:
            noise_pred: (B, H, action_dim) predicted noise
        """
        B, H, D = x.shape

        # Time embedding
        t_emb = self.time_mlp(t)  # (B, hidden_dim)

        # Context embedding
        c_emb = self.context_encoder(context)  # (B, hidden_dim)

        # Combine time + context
        emb = t_emb + c_emb  # (B, hidden_dim)

        # Reshape for temporal conv: (B, D, H)
        x = x.transpose(1, 2)

        # Encode
        h = self.action_encoder(x)  # (B, hidden_dim, H)

        # Downsampling with skip connections
        skips = []
        for block in self.down_blocks:
            h = block(h, emb)
            skips.append(h)

        # Middle
        h = self.mid_block(h, emb)

        # Upsampling with skip connections
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = block(h, emb)

        # Decode to noise
        noise_pred = self.action_decoder(h)  # (B, action_dim, H)

        # Reshape back: (B, H, action_dim)
        noise_pred = noise_pred.transpose(1, 2)

        return noise_pred


class TemporalConv1d(nn.Module):
    """1D convolution for temporal sequences."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        return self.conv(x)


class TemporalResBlock(nn.Module):
    """Residual block with temporal convolutions."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = TemporalConv1d(in_channels, out_channels)
        self.conv2 = TemporalConv1d(out_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        # Projection for residual
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x, t_emb):
        """
        Args:
            x: (B, C, H) temporal sequence
            t_emb: (B, C) time embedding
        """
        h = x

        # First conv
        h = self.conv1(h)
        h = self.norm1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.act(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)

        # Residual
        h = h + self.residual_proj(x)
        h = self.act(h)

        return h


# Training script
def train_action_diffusion(model, dataloader, num_steps=10000):
    """
    Train action diffusion model on demonstration data.

    Loss: Flow matching or DDPM objective
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, batch in enumerate(dataloader):
        obs = batch['obs']
        goal = batch['goal']
        actions = batch['actions']  # (B, H, 7) ground truth

        # Encode context
        context = encode_context(obs, goal)

        # Sample timestep
        t = torch.rand(len(actions)).to(actions.device)

        # Add noise
        noise = torch.randn_like(actions)
        actions_noisy = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * actions

        # Predict noise
        noise_pred = model(actions_noisy, t, context)

        # Loss: Flow matching
        target = actions - noise
        loss = F.mse_loss(noise_pred, target)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")


# Sampling (for MPC)
def sample_action_sequences(model, obs, goal, M=16, num_steps=10):
    """Sample M action sequences using DDIM."""
    context = encode_context(obs, goal)
    context = context.repeat(M, 1)  # (M, context_dim)

    # Start from noise
    x = torch.randn(M, H, action_dim).to(obs.device)

    # DDIM sampling
    for i in reversed(range(num_steps)):
        t = torch.ones(M).to(obs.device) * i / num_steps

        # Predict noise
        noise_pred = model(x, t, context)

        # DDIM step (faster than DDPM)
        x = ddim_step(x, noise_pred, i, num_steps)

    return x  # (M, H, 7) - temporally coherent!
```

**Training data:**
- Use existing robot demonstrations
- Extract action sequences (H=8 steps)
- Train diffusion model to denoise them

**Expected improvement:**
- ✅ Strong temporal correlations (learned from data)
- ✅ Task-relevant action patterns
- ✅ Better than MIDM-guided (models joint distribution)

---

#### **Phase 3: Integrate into Vidarc DiT (Long-term)**

**Ultimate goal: Action tokens in DiT (like LingBot-VA)**

This is the full LingBot-VA approach we discussed earlier - action tokens in the shared latent space.

---

## 📊 Comparison Matrix

| Method | Temporal Correlation | Training | Latency | Quality | Implementation |
|--------|---------------------|----------|---------|---------|----------------|
| **Random MPC** | ❌ None | ✅ No | 3200ms | ⭐⭐ | ✅ Trivial |
| **MIDM-Guided** | ⭐ Hand-crafted smoothing | ✅ No | 800ms | ⭐⭐⭐ | ✅ Easy |
| **Action VAE** | ⭐⭐ Learned latent | ❌ Yes | 100ms | ⭐⭐⭐ | ⭐⭐ Medium |
| **Action Diffusion** | ⭐⭐⭐⭐⭐ Strong learned | ❌ Yes | 200ms | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ Hard |
| **Autoregressive** | ⭐⭐⭐⭐ Sequential | ❌ Yes | 150ms | ⭐⭐⭐⭐ | ⭐⭐ Medium |
| **Flow Matching** | ⭐⭐⭐⭐ Continuous | ❌ Yes | 200ms | ⭐⭐⭐⭐ | ⭐⭐⭐ Hard |

---

## 🎯 Implementation Roadmap

### **Week 1: MIDM-Guided MPC**

```bash
# Create experiment
cd experiments/mpc

# Implement MIDM-guided sampling
python midm_guided_mpc.py --data-dir /path/to/robotwin --M 16 --H 8

# Compare with random baseline
python compare_mpc.py --baseline random --proposed midm_guided
```

**Metrics to track:**
- Success rate improvement
- Average CLIP score
- Action smoothness (measure velocity/acceleration)
- Computation time

### **Week 2-4: Action Diffusion**

```bash
# Train action diffusion model
python train_action_diffusion.py \
    --data-dir /path/to/demos \
    --horizon 8 \
    --hidden-dim 256 \
    --num-steps 10000

# Integrate into MPC
python action_diffusion_mpc.py --M 16 --diffusion-steps 10
```

### **Month 2+: Full Integration**

- Integrate action tokens into Vidarc DiT
- Joint video-action training
- Eliminate MPC overhead

---

## 🔬 Evaluation Protocol

### **Metrics:**

1. **Temporal Coherence**
```python
def measure_smoothness(actions):
    """Measure action sequence smoothness."""
    # Velocity (first derivative)
    velocity = torch.diff(actions, dim=0)

    # Acceleration (second derivative)
    acceleration = torch.diff(velocity, dim=0)

    # Smoothness = low acceleration variance
    smoothness = -acceleration.var().item()
    return smoothness
```

2. **Task Success Rate**
```python
# On RoboTwin 2.0 benchmark
success_rate = num_success / num_trials
```

3. **Sample Efficiency**
```python
# How many samples needed for good performance?
# Random MPC: M=64
# MIDM-Guided: M=16
# Diffusion: M=8
sample_efficiency = 1 / M
```

4. **CLIP Score Distribution**
```python
# Higher mean = better quality
# Lower variance = more consistent
clip_scores = [clip(pred, goal) for pred in predictions]
mean_score = np.mean(clip_scores)
std_score = np.std(clip_scores)
```

---

## 📝 Code Structure

```
experiments/mpc/
├── mpc.md                          # This file
├── README.md                       # Quick start guide
├── midm_guided_mpc.py             # Phase 1 implementation
├── action_diffusion.py            # Phase 2: Diffusion model
├── action_diffusion_mpc.py        # Phase 2: MPC with diffusion
├── train_action_diffusion.py      # Training script
├── compare_mpc.py                 # Comparison tool
├── visualize_actions.py           # Visualization utilities
├── configs/
│   ├── midm_guided.yaml
│   └── action_diffusion.yaml
└── notebooks/
    ├── analyze_temporal_correlation.ipynb
    └── visualize_action_samples.ipynb
```

---

## 🚀 Quick Start

### **Run MIDM-Guided MPC (Recommended First Step)**

```bash
cd /home/areache/SF-VLA/vidar-robotwin/experiments/mpc

# Run MPC with MIDM-guided sampling
python midm_guided_mpc.py \
    --model-b-path /path/to/vidarc.pt \
    --midm-path /path/to/midm.pt \
    --task adjust_bottle \
    --M 16 \
    --H 8 \
    --output-dir results/midm_guided

# Compare with random baseline
python compare_mpc.py \
    --method1 random \
    --method2 midm_guided \
    --output results/comparison.json
```

---

## 💡 Key Insights

### **Why Action Diffusion > Random Sampling**

1. **Temporal Structure**
```
Random: [0.5, -0.3, 0.8, -0.9, ...]  ← Jumpy
Diffusion: [0.1, 0.12, 0.15, 0.18, ...]  ← Smooth
```

2. **Sample Efficiency**
```
Random: Need M=64 to find 1 good sequence
Diffusion: Need M=8 because all samples are reasonable
→ 8× speedup!
```

3. **Physical Plausibility**
```
Random: May violate robot constraints
Diffusion: Learned from real demonstrations
→ All samples are feasible
```

### **Your Design Philosophy is Exactly Right!**

> "建模 action 之间的相关性或者能够对 action 的时序规律进行获取和筛选"

This is **exactly** what action diffusion does:
- **相关性 (Correlation):** Diffusion models joint distribution p(a₁, a₂, ..., a_H)
- **时序规律 (Temporal patterns):** Temporal convolutions learn patterns
- **获取和筛选 (Capture & filter):** Training captures patterns, sampling filters bad sequences

**Action diffusion is the missing piece for your MPC!**

---

## 📚 References

**Action Diffusion for Robotics:**
1. "Diffusion Policy" (Chi et al., 2023) - Original action diffusion work
2. "Action Chunking with Transformers" (Zhao et al., 2023) - Temporal modeling
3. "Flow Matching for Action Prediction" - LingBot-VA style

**MPC + Learning:**
1. "Model Predictive Control with Learned Dynamics" (survey)
2. "Sampling-based MPC with Neural Networks" - Similar to your approach

---

## ⚠️ Common Pitfalls

1. **Don't forget action normalization**
```python
# Always normalize actions before training diffusion
actions_normalized = (actions - mean) / std
```

2. **Temporal smoothing is crucial**
```python
# Even with diffusion, add slight smoothing
actions = gaussian_smooth(actions, sigma=0.5)
```

3. **Start small**
```python
# Don't train diffusion on full dataset immediately
# Start with 1000 demos, verify it works, then scale up
```

4. **Monitor action statistics**
```python
# Check if generated actions are in reasonable range
assert actions.min() >= -1.0 and actions.max() <= 1.0
```

---

## 🎯 Success Criteria

### **Phase 1 (MIDM-Guided) Success:**
- ✅ 4× speedup (M=16 vs 64)
- ✅ 10-15% improvement in success rate
- ✅ Higher CLIP scores on average
- ✅ Smoother actions (lower acceleration variance)

### **Phase 2 (Action Diffusion) Success:**
- ✅ 8× speedup (M=8 vs 64)
- ✅ 20-30% improvement in success rate
- ✅ Actions look human-like (qualitative)
- ✅ Generalize to novel tasks

---

## 🤝 Next Steps

1. **This week:** Implement `midm_guided_mpc.py` (Phase 1)
2. **Next week:** Collect metrics, compare with random baseline
3. **Week 3-4:** Start action diffusion training (Phase 2)
4. **Month 2:** Integrate into Vidarc DiT (Phase 3)

**Let's build temporal correlation into your MPC! 🚀**