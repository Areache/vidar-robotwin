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

**Let's build temporal correlation into your MPC!**

---
---

# Part 2: 深入讨论 — Action 时序建模与 Vidarc 架构融合

## 背景：与现有 Subgoal 框架的关系

你的系统已有清晰的层级设计（来自 `theory.md` 和 `subgoal_plan.md`）：

```
Level 3: Task Memory    — VLM context window (unlimited horizon)
Level 2: Visual Memory  — WanTI2V / Model A (2-5s per round)
Level 1: State Memory   — Current observation (instantaneous)

Model A (高层规划): "Where should we go?" → subgoal g_k
Model B (底层执行): "How do we get there?" → action sequence

连接：ε̃_θ(xt) = ε_θ(xt) - λ · σ_t · ∇_xt Φ(x_t, g_k)
```

**MPC (方案 2) 在这个框架中的位置：**

```
Without MPC:
  Model A → g_k → Φ → Model B → actions (直接生成)

With MPC:
  Model A → g_k → 采样 M 组 action → Model B predict → score → select
                   ↑
                   这里需要 action temporal correlation！
```

**核心问题重新定义：**

MPC 的 action 采样模块需要 **action prior**，它既要：
1. 产生时序连贯的 action chunk
2. 与 Vidarc 的 classifier guidance 兼容
3. 能够条件化在 subgoal g_k 上

---

## 方案 6: Subgoal-Conditioned Action Diffusion（推荐研究方向）

**核心思想：** Action diffusion model 以 `(obs, subgoal)` 为条件生成 action chunk，
然后用 Model B 的 video prediction 做 re-ranking。

**这是 action diffusion 与你的 classifier guidance 理论的完美结合。**

```
┌─────────────────────────────────────────────────────────┐
│ Model A: Subgoal Generator                              │
│ "Next phase: gripper should be above the cup"           │
│ Output: g_k (keyframe image or latent)                  │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Action Diffusion Prior (NEW!)                           │
│ p_θ(a_{1:H} | obs, g_k)                                │
│                                                         │
│ Input: (current obs, subgoal g_k)                       │
│ Output: M temporally-coherent action sequences          │
│                                                         │
│ Architecture: Temporal UNet or Transformer              │
│ Training: On demonstration (obs, subgoal, actions) tuples│
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Model B (Vidarc): Forward Prediction + Scoring          │
│                                                         │
│ For each candidate action sequence:                     │
│   1. Rollout Model B with actions → predicted video     │
│   2. Score: Φ(predicted_end_frame, g_k)                │
│   3. Select best action sequence                        │
│                                                         │
│ OR: 直接用 classifier guidance 修正最佳候选             │
│   ε̃ = ε_θ(xt) - λ · σ_t · ∇_xt Φ(x_t, g_k)          │
└─────────────────────────────────────────────────────────┘
```

### 为什么这比纯 Action Diffusion 更好？

```python
# 纯 Action Diffusion:
actions = action_diffusion.sample(obs, goal)  # 直接输出 actions
# 问题：没有 video model 验证，可能产生物理不可行的动作

# Subgoal-Conditioned + Model B Re-ranking:
candidates = action_diffusion.sample(obs, subgoal, M=8)  # 8 组候选
futures = [model_b.rollout(obs, c) for c in candidates]   # 验证
scores = [Φ(f, subgoal) for f in futures]                  # 评分
best = candidates[argmax(scores)]                           # 选最优
# 优点：action prior 提供好的候选，model B 提供物理验证
```

### 关键设计：双重条件化

```python
class SubgoalConditionedActionDiffusion(nn.Module):
    """
    Action diffusion conditioned on BOTH:
    1. Current observation (where am I?)
    2. Subgoal image (where should I go?)

    This models: p(a_{1:H} | o_t, g_k)
    """

    def __init__(self, action_dim=7, horizon=8, hidden_dim=256,
                 obs_encoder=None, subgoal_encoder=None):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        # Visual encoders (can share weights or be separate)
        # 重要：用 frozen encoder，和 Vidarc 的 Φ 共享表示空间
        self.obs_encoder = obs_encoder or FrozenResNet18()
        self.subgoal_encoder = subgoal_encoder or FrozenResNet18()

        # Combine obs + subgoal + diffusion timestep
        obs_dim = 512  # ResNet18 output
        self.context_fuser = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),  # obs + subgoal
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temporal denoising network
        self.denoiser = TemporalUNet(
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
        )

    def forward(self, noisy_actions, timestep, obs, subgoal):
        """
        Args:
            noisy_actions: (B, H, 7)
            timestep: (B,) in [0, 1]
            obs: (B, 3, H, W) current observation
            subgoal: (B, 3, H, W) subgoal image from Model A
        """
        # Encode visual context
        with torch.no_grad():
            obs_feat = self.obs_encoder(obs)       # (B, 512)
            sg_feat = self.subgoal_encoder(subgoal) # (B, 512)

        # Fuse context
        context = self.context_fuser(
            torch.cat([obs_feat, sg_feat], dim=-1)
        )  # (B, hidden_dim)

        # Predict noise/velocity
        return self.denoiser(noisy_actions, timestep, context)

    @torch.no_grad()
    def sample(self, obs, subgoal, M=8, num_steps=10):
        """Sample M action sequences conditioned on (obs, subgoal)."""
        B = obs.shape[0]
        device = obs.device

        # Expand for M samples
        obs_M = obs.repeat(M, 1, 1, 1)          # (M*B, 3, H, W)
        sg_M = subgoal.repeat(M, 1, 1, 1)       # (M*B, 3, H, W)

        # Start from noise
        x = torch.randn(M * B, self.horizon, self.action_dim, device=device)

        # Flow matching sampling
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(M * B, device=device) * (i / num_steps)
            v = self.forward(x, t, obs_M, sg_M)
            x = x + dt * v

        # Reshape: (M*B, H, 7) → (M, B, H, 7)
        x = x.view(M, B, self.horizon, self.action_dim)
        return x  # All M sequences are temporally coherent!
```

### 训练数据构造

```python
def prepare_training_data(demo_dataset, subgoal_interval=8):
    """
    从 demonstration 中构造 (obs, subgoal, actions) 训练样本。

    Key: subgoal 就是 demo 中 H 步后的 frame！
    """
    training_samples = []

    for episode in demo_dataset:
        obs_frames = episode['observations']   # (T, H, W, 3)
        actions = episode['actions']            # (T, 7)

        for t in range(0, len(actions) - subgoal_interval):
            sample = {
                'obs': obs_frames[t],                     # 当前 obs
                'subgoal': obs_frames[t + subgoal_interval], # H 步后的帧
                'actions': actions[t:t + subgoal_interval],  # 中间的 H 步 actions
            }
            training_samples.append(sample)

    return training_samples

# 训练时：
# diffusion 学会: "给定当前 obs 和目标 subgoal，中间应该执行什么 action 序列？"
# 这正是 inverse dynamics 的 chunk 版本！
```

---

## 方案 7: Action Chunking Transformer (ACT Style)

**灵感来源：** ACT (Action Chunking with Transformers, Zhao et al. 2023)

**核心思想：** 用 CVAE (Conditional VAE) 建模 action chunk 的多模态分布，
Transformer 解码器负责建模 chunk 内的时序相关性。

```
┌─────────────────────────────────────────────────────┐
│ Training (CVAE):                                    │
│                                                     │
│ Encoder: (obs, subgoal, GT_actions) → z ~ q(z|...)  │
│ Decoder: (obs, subgoal, z) → predicted_actions      │
│                                                     │
│ z 是 action chunk 的 "style" latent                 │
│ 同一个 (obs, subgoal) 可以有多种到达方式            │
│ z 编码了 "哪种方式"                                 │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Inference (Sample from prior):                      │
│                                                     │
│ z ~ N(0, I)   ← 采样 M 组不同的 z                   │
│ Decoder: (obs, subgoal, z_i) → action_sequence_i   │
│                                                     │
│ 每组 z 给出一种 temporally-coherent 的执行方式      │
│ 然后用 Model B 做 re-ranking                        │
└─────────────────────────────────────────────────────┘
```

```python
class ActionChunkingCVAE(nn.Module):
    """
    CVAE for action chunk generation.
    Models multimodal action distributions with temporal coherence.
    """

    def __init__(self, action_dim=7, horizon=8, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Visual encoder
        self.obs_encoder = FrozenResNet18()
        self.subgoal_encoder = FrozenResNet18()

        # CVAE Encoder: q(z | obs, subgoal, actions)
        self.cvae_encoder = nn.Sequential(
            nn.Linear(512 * 2 + action_dim * horizon, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Transformer Decoder: p(actions | obs, subgoal, z)
        self.action_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=4,
        )

        # Embeddings
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_proj = nn.Linear(512 * 2, hidden_dim)
        self.pos_embed = nn.Embedding(horizon, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode(self, obs, subgoal, actions):
        """Encode to latent z (training only)."""
        obs_feat = self.obs_encoder(obs)
        sg_feat = self.subgoal_encoder(subgoal)
        actions_flat = actions.flatten(1)

        x = torch.cat([obs_feat, sg_feat, actions_flat], dim=-1)
        h = self.cvae_encoder(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def decode(self, obs, subgoal, z):
        """Decode z to action sequence."""
        B = z.shape[0]
        H = self.pos_embed.num_embeddings

        obs_feat = self.obs_encoder(obs)
        sg_feat = self.subgoal_encoder(subgoal)

        # Context: obs + subgoal + z
        context = self.context_proj(torch.cat([obs_feat, sg_feat], dim=-1))
        z_feat = self.z_proj(z)
        memory = torch.stack([context, z_feat], dim=1)  # (B, 2, hidden)

        # Autoregressive action decoding
        pos = self.pos_embed(torch.arange(H, device=z.device))
        tgt = pos.unsqueeze(0).expand(B, -1, -1)  # (B, H, hidden)

        # Causal mask for temporal ordering
        causal_mask = nn.Transformer.generate_square_subsequent_mask(H).to(z.device)

        decoded = self.action_decoder(
            tgt, memory, tgt_mask=causal_mask
        )  # (B, H, hidden)

        actions = self.action_head(decoded)  # (B, H, 7)
        return actions

    def forward(self, obs, subgoal, actions):
        """Training forward: encode + reparameterize + decode."""
        mu, logvar = self.encode(obs, subgoal, actions)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Decode
        actions_pred = self.decode(obs, subgoal, z)

        # Losses
        recon_loss = F.l1_loss(actions_pred, actions)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        return actions_pred, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, obs, subgoal, M=8):
        """Sample M diverse action sequences."""
        B = obs.shape[0]

        all_actions = []
        for _ in range(M):
            z = torch.randn(B, self.latent_dim, device=obs.device)
            actions = self.decode(obs, subgoal, z)
            all_actions.append(actions)

        return torch.stack(all_actions)  # (M, B, H, 7)
```

**与 Action Diffusion 的对比：**

| 方面 | Action Diffusion | ACT (CVAE) |
|------|-----------------|------------|
| **采样速度** | 慢 (需要多步 denoise) | **快 (一次 decode)** |
| **多样性** | 高 (diffusion 天然多样) | 中 (z 的方差控制) |
| **时序建模** | Temporal Conv | **Transformer (更强)** |
| **训练稳定性** | 中等 | 中等 (KL collapse 风险) |
| **表达能力** | 高 (任意分布) | 中 (近似高斯后验) |

**ACT 的优势：采样速度快！**
- Diffusion 需要 10 步 denoise → 10 次 forward pass
- CVAE 只需要 1 次 decode → **10× faster**
- 对于 MPC re-ranking 场景，速度非常重要

---

## 方案 8: MPPI (Model Predictive Path Integral)

**动机：** 比 random shooting 更优雅的 MPC 变体，
利用 action prior 做 importance-weighted sampling。

**核心思想：** 不是 "采样 → 选最好的"，
而是 "采样 → 加权平均所有候选"。

```python
class MPPI:
    """
    Model Predictive Path Integral Control.

    Unlike random shooting MPC (select argmax), MPPI does
    weighted averaging of all candidates.

    Key advantage: Smoother actions (no discontinuous switching)
    """

    def __init__(self, model_b, scorer, action_dim=7, horizon=8):
        self.model_b = model_b
        self.scorer = scorer
        self.action_dim = action_dim
        self.horizon = horizon

        # Running mean: warm-start from previous plan
        self.action_mean = None  # (H, 7)
        self.temperature = 1.0   # λ in MPPI

    def step(self, obs, subgoal, M=32, noise_std=0.3):
        """
        MPPI step: sample, score, weighted average.

        Unlike argmax MPC, MPPI computes:
          a* = Σ_i w_i * a_i  where w_i ∝ exp(score_i / λ)

        This gives SMOOTH action sequences naturally!
        """
        device = obs.device

        # 1. Warm-start: use previous plan shifted by 1
        if self.action_mean is None:
            self.action_mean = torch.zeros(self.horizon, self.action_dim,
                                           device=device)
        else:
            # Shift: drop first action, repeat last
            self.action_mean = torch.cat([
                self.action_mean[1:],
                self.action_mean[-1:],
            ], dim=0)

        # 2. Sample perturbations around mean
        # 关键：这里可以用 action diffusion 替换 random noise！
        noise = torch.randn(M, self.horizon, self.action_dim,
                            device=device) * noise_std

        # Temporally-correlated noise (simple low-pass filter)
        # 这已经在建模 action 之间的相关性！
        for m in range(M):
            for d in range(self.action_dim):
                noise[m, :, d] = self._temporal_filter(noise[m, :, d])

        candidates = self.action_mean.unsqueeze(0) + noise  # (M, H, 7)
        candidates = torch.clamp(candidates, -1.0, 1.0)

        # 3. Score each candidate
        scores = []
        for m in range(M):
            end_frame = self.model_b.rollout(obs, candidates[m])
            score = self.scorer(end_frame, subgoal)
            scores.append(score)
        scores = torch.stack(scores)  # (M,)

        # 4. MPPI weighted average (NOT argmax!)
        # w_i ∝ exp((score_i - max_score) / temperature)
        weights = torch.softmax(
            (scores - scores.max()) / self.temperature,
            dim=0
        )  # (M,)

        # Weighted average of all candidates
        weighted_actions = (
            weights.unsqueeze(-1).unsqueeze(-1) * candidates
        ).sum(dim=0)  # (H, 7)

        # 5. Update running mean
        self.action_mean = weighted_actions.detach()

        # 6. Return first action
        return weighted_actions[0]  # Execute first step only

    def _temporal_filter(self, signal, alpha=0.7):
        """
        Simple exponential moving average for temporal correlation.

        alpha=0.7: Strong correlation between adjacent actions
        alpha=0.0: Independent (original noise)
        """
        filtered = torch.zeros_like(signal)
        filtered[0] = signal[0]
        for t in range(1, len(signal)):
            filtered[t] = alpha * filtered[t-1] + (1 - alpha) * signal[t]
        return filtered
```

**MPPI vs Random Shooting MPC 的关键区别：**

```
Random Shooting:
  candidates = [a1, a2, ..., aM]
  scores = [s1, s2, ..., sM]
  action = candidates[argmax(scores)]  ← 离散选择，可能跳变

MPPI:
  candidates = [a1, a2, ..., aM]
  scores = [s1, s2, ..., sM]
  weights = softmax(scores / λ)
  action = Σ weights_i * candidates_i  ← 加权平均，天然平滑！

MPPI + Warm-start:
  上一轮的 plan 作为本轮的 mean
  新噪声只是小扰动
  → 时间步之间的连续性自然保证！
```

**MPPI 天然解决 action correlation 问题：**
1. **Warm-start** → 前后两步的 plan 高度相关
2. **Weighted average** → 输出平滑（不是离散跳变）
3. **Temporal filter on noise** → 采样本身就有时序相关性
4. **无需训练！** → 纯算法层面的改进

---

## 方案 9: Action Tokenization (VQ-VAE + GPT)

**动机：** 类似 LLM 的方式，将 action 离散化后用 autoregressive model 生成。

**核心思想：**
1. 用 VQ-VAE 将 action chunk 压缩成离散 token
2. 用 GPT-style model 生成 token 序列
3. Token 序列 decode 回 action chunk

```
Training:
  action_chunk (H, 7) → VQ-VAE Encoder → [tok_1, tok_2, ..., tok_K]
  token_sequence → GPT → next_token_prediction

Inference:
  obs, subgoal → GPT → [tok_1, tok_2, ..., tok_K]
  tokens → VQ-VAE Decoder → action_chunk (H, 7)
```

```python
class ActionTokenizer(nn.Module):
    """VQ-VAE for action chunk tokenization."""

    def __init__(self, action_dim=7, horizon=8, codebook_size=512,
                 num_tokens=4, embed_dim=64):
        super().__init__()
        self.num_tokens = num_tokens  # K tokens per chunk

        # Encoder: action chunk → K tokens
        self.encoder = nn.Sequential(
            nn.Linear(action_dim * horizon, 256),
            nn.GELU(),
            nn.Linear(256, num_tokens * embed_dim),
        )

        # Vector Quantization
        self.codebook = nn.Embedding(codebook_size, embed_dim)

        # Decoder: K tokens → action chunk
        self.decoder = nn.Sequential(
            nn.Linear(num_tokens * embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim * horizon),
        )

    def encode(self, actions):
        """Actions (B, H, 7) → Tokens (B, K)"""
        B = actions.shape[0]
        x = actions.flatten(1)  # (B, H*7)
        z = self.encoder(x).view(B, self.num_tokens, -1)  # (B, K, E)

        # Quantize
        distances = torch.cdist(z, self.codebook.weight)  # (B, K, codebook)
        token_ids = distances.argmin(dim=-1)  # (B, K)
        z_q = self.codebook(token_ids)  # (B, K, E)

        return token_ids, z_q

    def decode(self, z_q):
        """Tokens (B, K, E) → Actions (B, H, 7)"""
        B = z_q.shape[0]
        x = z_q.flatten(1)  # (B, K*E)
        actions = self.decoder(x).view(B, -1, 7)  # (B, H, 7)
        return actions


class ActionGPT(nn.Module):
    """
    GPT-style model for action token sequence generation.

    Sequence: [obs_token, subgoal_token, action_tok_1, ..., action_tok_K]
    """

    def __init__(self, codebook_size=512, num_action_tokens=4,
                 hidden_dim=256, num_layers=4):
        super().__init__()

        vocab_size = codebook_size + 2  # +2 for obs/subgoal special tokens

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(num_action_tokens + 2, hidden_dim)

        # Transformer decoder
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=hidden_dim * 4, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers)

        self.output_head = nn.Linear(hidden_dim, codebook_size)

    def forward(self, obs_emb, subgoal_emb, action_tokens=None):
        """
        Autoregressive generation of action tokens.

        Args:
            obs_emb: (B, hidden_dim)
            subgoal_emb: (B, hidden_dim)
            action_tokens: (B, K) for training, None for inference
        """
        B = obs_emb.shape[0]
        device = obs_emb.device

        if action_tokens is not None:
            # Training: teacher forcing
            token_emb = self.token_embed(action_tokens)  # (B, K, D)
            seq = torch.cat([
                obs_emb.unsqueeze(1),
                subgoal_emb.unsqueeze(1),
                token_emb,
            ], dim=1)  # (B, K+2, D)

            pos = self.pos_embed(torch.arange(seq.shape[1], device=device))
            seq = seq + pos

            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq.shape[1]
            ).to(device)

            out = self.transformer(seq, seq, tgt_mask=causal_mask)
            logits = self.output_head(out[:, 2:])  # (B, K, codebook_size)
            return logits

        else:
            # Inference: autoregressive sampling
            tokens = []
            seq = torch.cat([
                obs_emb.unsqueeze(1),
                subgoal_emb.unsqueeze(1),
            ], dim=1)

            for k in range(self.num_action_tokens):
                pos = self.pos_embed(torch.arange(seq.shape[1], device=device))
                seq_pos = seq + pos

                out = self.transformer(seq_pos, seq_pos)
                logits = self.output_head(out[:, -1])
                next_token = logits.argmax(dim=-1)  # Greedy
                tokens.append(next_token)

                next_emb = self.token_embed(next_token).unsqueeze(1)
                seq = torch.cat([seq, next_emb], dim=1)

            return torch.stack(tokens, dim=1)  # (B, K)
```

**为什么 Action Tokenization 适合建模时序相关性：**

```
连续 action space:
  a = [0.123, -0.456, 0.789, ...]  ← 连续值，相关性隐式

离散 token space:
  tokens = [42, 167, 23, 89]  ← 离散 token，GPT 学习序列规律
  "token 42 之后经常出现 167" ← 这就是时序相关性！

类比 NLP：
  "the" 后面常跟名词 → GPT 学会了语法
  "reach" 后面常跟 "grasp" → Action GPT 学会了操作顺序
```

---

## 方案 10: Consistency Model for Actions（最快推理）

**动机：** Diffusion 的 M 步 denoise 太慢，
Consistency Model 可以 **1 步** 直接从噪声生成 action。

```python
class ActionConsistencyModel(nn.Module):
    """
    Consistency Model for ultra-fast action generation.

    Key: Trained to map ANY noise level directly to clean actions.
    → 1-step generation (no iterative denoising!)
    """

    def __init__(self, action_dim=7, horizon=8, hidden_dim=256):
        super().__init__()
        self.denoiser = TemporalUNet(action_dim, horizon, hidden_dim)

    def forward(self, noisy_actions, timestep, context):
        return self.denoiser(noisy_actions, timestep, context)

    @torch.no_grad()
    def sample(self, obs, subgoal, M=16):
        """
        1-step generation!

        Normal diffusion: noise → 10 steps → actions (slow)
        Consistency model: noise → 1 step  → actions (10× faster!)
        """
        context = encode_context(obs, subgoal)
        context = context.repeat(M, 1)

        # Start from noise
        noise = torch.randn(M, self.horizon, self.action_dim)

        # ONE step directly to clean actions!
        t = torch.ones(M) * 0.999  # High noise level
        actions = self.forward(noise, t, context)

        return actions  # (M, 16, 7) - done!

    # Training: Consistency distillation from action diffusion
    # See Song et al. "Consistency Models" (2023)
```

**速度对比：**

| Method | Denoise Steps | Sampling Time (M=16) |
|--------|--------------|---------------------|
| Action Diffusion | 10 | ~200ms |
| Action CVAE | 1 | ~20ms |
| **Consistency Model** | **1** | **~20ms** |
| Random (baseline) | 0 | ~5ms |

---

## 深度讨论：哪个方案最适合 Vidarc？

### 你的需求分析

```
需求 1: 建模 action 之间的相关性 (temporal correlation)
  → Diffusion ✅, ACT ✅, MPPI ✅, Tokenization ✅, Consistency ✅

需求 2: 与 subgoal classifier guidance 兼容
  → Subgoal-conditioned Diffusion ✅✅✅ (最佳)
  → ACT ✅✅ (支持 subgoal conditioning)
  → MPPI ✅✅ (scorer 可以用 Φ)

需求 3: 不需要大量训练
  → MPPI ✅✅✅ (零训练)
  → MIDM-Guided ✅✅ (零训练)
  → Others ❌ (需要训练)

需求 4: 作为 Vidarc 的补充（不修改 Model B）
  → 所有方案都满足 ✅ (action prior 是独立模块)

需求 5: 可以逐步升级
  → Phase 1: MPPI (零训练)
  → Phase 2: Subgoal-conditioned Diffusion (轻量训练)
  → Phase 3: 集成到 Vidarc DiT (重训练)
```

### 最终推荐路径

```
Week 1: MPPI + Temporal Filtering
├─ 零训练
├─ Warm-start 自然给出时序连续性
├─ Weighted average 自然平滑
└─ 预期：比 random shooting 提升 15-20%

Week 2-3: Subgoal-Conditioned Action Diffusion
├─ 用 demo 数据训练轻量 diffusion (256 hidden, ~2h on 1 GPU)
├─ 条件化在 (obs, subgoal)
├─ M=8 即可，结合 MPPI weighted average
└─ 预期：比 MPPI 再提升 10-15%

Month 2: Action Chunking CVAE (ACT)
├─ 如果 diffusion 推理太慢
├─ CVAE 单步 decode，10× faster
├─ 适合高频闭环控制
└─ 预期：推理速度提升 10×，精度持平

Month 3+: 集成到 Vidarc DiT
├─ Action tokens interleaved with video tokens
├─ Joint flow matching (LingBot-VA style)
├─ 消除 MPC overhead
└─ 预期：接近 LingBot-VA 水平
```

---

## 与 Classifier Guidance 的融合

**你的理论框架：**
```
ε̃_θ(xt) = ε_θ(xt) - λ · σ_t · ∇_xt Φ(x_t, g_k)
```

**Action Diffusion 可以类似地加 guidance：**
```
ã_θ(at) = a_θ(at) - λ · σ_t · ∇_at Ψ(a_t, g_k)
```

其中 `Ψ` 是 action-level 的 potential function：

```python
def action_guidance_potential(actions, obs, subgoal, model_b, scorer):
    """
    Action-level classifier guidance.

    Ψ(a) = Φ(ModelB(obs, a), subgoal)

    即：用 Model B 预测执行 actions 后的结果，
    然后评估结果与 subgoal 的距离。
    """
    # 1. Forward predict
    end_frame = model_b.rollout(obs, actions)

    # 2. Score against subgoal
    score = scorer(end_frame, subgoal)

    return score


def guided_action_diffusion_step(
    action_diffusion, actions_t, t, obs, subgoal,
    model_b, scorer, guidance_scale=1.0
):
    """
    Action diffusion with classifier guidance.

    Combines:
    1. Action prior: p_θ(a | obs, subgoal) from diffusion
    2. Action guidance: ∇_a Ψ(a, g_k) from Model B

    This is the action-space analog of your video-space
    classifier guidance ε̃ = ε_θ - λ · σ · ∇Φ
    """
    # Unconditional score (from action diffusion)
    with torch.no_grad():
        score_uncond = action_diffusion(actions_t, t, obs, subgoal)

    # Guidance gradient
    actions_t.requires_grad_(True)
    potential = action_guidance_potential(actions_t, obs, subgoal,
                                          model_b, scorer)
    grad = torch.autograd.grad(potential, actions_t)[0]
    actions_t.requires_grad_(False)

    # Guided score
    guided_score = score_uncond + guidance_scale * grad

    return guided_score
```

**这形成了双层 guidance 体系：**

```
Level 1 (Video Space):
  ε̃_θ(xt) = ε_θ(xt) - λ_v · σ_t · ∇_xt Φ(x_t, g_k)
  Model B 的 video generation 被 subgoal 引导

Level 2 (Action Space):  ← NEW!
  ã_θ(at) = a_θ(at) - λ_a · σ_t · ∇_at Ψ(a_t, g_k)
  Action diffusion 的 action generation 被 subgoal 引导

Both:
  ε̃ = ε_θ - λ_v · ∇Φ_video - λ_a · ∇Ψ_action
  Video 和 Action 同时被 subgoal 引导！
```

---

## 总结：Action 时序建模的核心选择

| 方案 | 时序建模方式 | 训练 | 速度 | 与你的框架兼容性 |
|------|------------|------|------|----------------|
| **MPPI** | Warm-start + filtering | 无 | 快 | ✅ 直接替换 random shooting |
| **Subgoal Diffusion** | Temporal UNet | 轻量 | 中 | ✅✅✅ Subgoal conditioning |
| **ACT (CVAE)** | Transformer decoder | 轻量 | **最快** | ✅✅ 支持 subgoal |
| **Action GPT** | Autoregressive tokens | 中等 | 中 | ✅ 可条件化 |
| **Consistency Model** | 1-step denoise | 需蒸馏 | **最快** | ✅✅ 同 diffusion |
| **Guided Diffusion** | Diffusion + ∇Ψ | 轻量 | 慢 | ✅✅✅ 完美融合 Φ 理论 |

**我的建议：先 MPPI（零成本验证），再 Subgoal-Conditioned Diffusion（最适合你的理论框架）。**
