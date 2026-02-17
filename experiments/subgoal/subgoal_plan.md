# Subgoal-Guided Action Generation: Plan

## 1. Goal

Build a two-level system where a high-level planner (Model A) provides subgoal guidance to a low-level action policy (Model B / VIDAR), without modifying Model B's weights.

**Core constraint:** Validate each component's representation quality through controlled experiments BEFORE building the full system.

---

## 2. Core Theory

We reshape the energy landscape of Model B's generation distribution using a subgoal potential:

```
ε̃_θ(xt) = ε_θ(xt) - λ · σ_t · ∇_xt Φ
           ~~~~~~~~   ~~~~~~~~~~~~~~~~~~~
           base policy   subgoal guidance
```

where `Φ(x_1:t, g)` measures how far the current trajectory is from the subgoal `g`.

This is classifier guidance (Dhariwal & Nichol 2021) applied to trajectory space:

```
∇_xt log p(xt | g) = ∇_xt log p(xt) + ∇_xt log p(g | xt)
```

**Key insight:** Model B is untouched. All guidance comes from `Φ`'s gradient. If `Φ` is wrong, the system degrades gracefully (λ=0 recovers the original policy).

### Potential Function Candidates

| Φ Definition | Formula | When to Use |
|---|---|---|
| Latent L2 | `min_k \|\|Enc(xt) - z_gk\|\|^2` | Shared encoder space (Method 1) |
| Reachability | `-log R(z_t, z_gk)` | When L2 ≠ reachability |
| CLIP cosine | `1 - cos(CLIP(xt), CLIP(gk))` | Cross-domain / language goals |

---

## 3. Architecture

```
Model A: Subgoal Generator              Model B: Action Policy (VIDAR)
"Where should we go?"                   "How do we get there?"
────────────────────                    ──────────────────────
Coarse-grained, long-horizon            Fine-grained, short-horizon
Semantic planning                       Motor control
Open-loop imagination                   Closed-loop with real observations
Low-frequency (once per phase)          High-frequency (every action chunk)
```

Data flow:

```
     Model A → g_k (subgoal)
                  │
                  ▼
            Φ(x_t, g_k)        potential function
                  │
                  ▼
            ∇_xt Φ              gradient signal
                  │
                  ▼
     Model B denoising corrected:  ε̃ = ε_θ(xt) - λ · σ_t · ∇_xt Φ
```

### Model A Candidates

| Candidate | Source | Long-Horizon | Training Cost | Status |
|-----------|--------|-------------|---------------|--------|
| Lightweight Planning Head (Method 1) | Frozen encoder + MLP/Transformer | Limited by single-frame input | Low (hours) | Not implemented |
| MPC Rollout (Method 2) | Model B short-horizon rollout | H steps (short) | Zero | Validated concept |
| **WanTI2V Non-Causal (Method 3)** | Wan 2.2 TI2V 5B, full bidirectional attention | **5s per round, unlimited via chaining** | **Zero (model already in codebase)** | `run_keyframe_model.py` exists |
| **VLM + GT Retrieval (Method B)** | VLM phase detection + keyframe library from GT demos | **Unlimited (VLM context window)** | **Zero** | Not yet implemented |
| **VLM + WanTI2V (Method C)** | VLM task decomposition + WanTI2V per-subtask plan | **Unlimited (VLM manages transitions)** | **Zero (both models pretrained)** | Not yet implemented |

**Recommended progression:** Method B first (validate VLM pipeline with GT subgoals), then Method C (upgrade to generated subgoals for generalization). Method 3 (standalone WanTI2V) is subsumed by Method C. See Section 6 and [theory.md](theory.md) for design principles.

---

## 4. Validation Plan: Controlled Experiments (表征验证)

### Design Principles

- **Goal:** Validate representation quality, NOT maximize task success
- **Method:** Change exactly ONE variable per experiment (控制变量)
- **Criterion:** Each experiment answers a binary question (pass/fail)
- **Order:** Follow the dependency graph — stop early if a gate fails

### Dependency Graph

```
Exp 1.1 (Baseline)  ─┐
                      ├─► Exp 1.2 (Oracle) ──► GATE: Does oracle help?
                      │                              │
                      │                         NO → diagnose with Exp 3.x, 4.x
                      │                              │
                      │                         YES ─┤
                      │                              ▼
                      │                    Exp 1.3 (λ ablation)
                      │                              │
                      │                              ▼
                      │                    Exp 1.4 (interval ablation)
                      │                              │
                      │                              ▼
                      │                    Exp 2.x (keyframe strategies)
                      │                              │
                      │              ┌───────────────┴──────────────┐
                      │              ▼                              ▼
                      │    Exp 3.x (latent diagnostics)   Exp 4.x (gradient)
                      │              │                              │
                      │              └───────► Exp 5.x (switching) ◄┘
                      │                              │
                      │                              ▼
                      └──────────────────► Build Planning Head
```

---

### Series 1: Oracle Guidance Baseline (验证指导信号的上界)

**Question:** Does providing perfect future-frame guidance improve policy behavior at all?

This is the most important gate. If GT keyframes don't help, stop and fix Φ before proceeding.

#### Exp 1.1 — No Guidance Baseline

| Item | Value |
|------|-------|
| **Variable** | (none — this is the reference) |
| **Config** | `use_libero_subgoal: false`, `subgoal_guidance_scale: 0.0` |
| **Measure** | Success rate, trajectory smoothness, action reversal count |
| **Purpose** | Reference for all subsequent comparisons |

#### Exp 1.2 — Oracle GT Keyframes

| Item | Value |
|------|-------|
| **Variable** | Add GT keyframes as subgoals (everything else identical to 1.1) |
| **Config** | `use_libero_subgoal: true`, `extraction_strategy: uniform`, `keyframe_interval: 8`, `subgoal_guidance_scale: 0.5` |
| **Measure** | Success rate delta vs 1.1, per-step distance to GT subgoal, trajectory Frechet distance |
| **Pass** | Success rate > baseline + 10% |
| **Fail** | Success rate unchanged or worse → STOP, diagnose Φ (jump to Series 3/4) |

#### Exp 1.3 — Guidance Strength (λ) Ablation

| Item | Value |
|------|-------|
| **Variable** | `subgoal_guidance_scale` ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0} |
| **Fixed** | Everything else from Exp 1.2 |
| **Measure** | Success rate vs. λ curve, action magnitude (jerky?), latent tracking accuracy |
| **Pass** | Inverted-U curve with clear optimum λ* |
| **Fail** | Monotonic decrease → gradient direction is wrong |

#### Exp 1.4 — Temporal Density Ablation

| Item | Value |
|------|-------|
| **Variable** | `keyframe_interval` ∈ {4, 8, 16, 32, 64} |
| **Fixed** | λ = λ* from Exp 1.3, `extraction_strategy: uniform` |
| **Measure** | Success rate vs. interval, gradient magnitude vs. interval |
| **Pass** | Inverted-U — too dense is useless, too sparse loses critical states |
| **Fail** | Flat line → guidance is ineffective regardless of density |

---

### Series 2: Keyframe Extraction Methods (验证信号源的语义质量)

**Question:** Which keyframe extraction method produces the most task-relevant subgoals?

**Fixed across all:** λ = λ* from 1.3, `max_keyframes: 20`

#### 2A: Rule-Based Strategies (Keyframe Method 1)

| Exp | Strategy | Signal | HDF5? | Key Metric |
|-----|----------|--------|-------|------------|
| 2.1 | `uniform` | fixed interval | No | (= Exp 1.2, baseline) |
| 2.2 | `visual_change` | pixel MSE > 0.05 | No | Keyframe distribution over task phases |
| 2.3 | `gripper_change` | action[6:13] delta | Yes | # keyframes (expected 2-5), gap sizes |
| 2.4 | `action_milestone` | action velocity | Yes | Alignment with approach/grasp/transport/place |
| 2.5 | `composite` | gripper + visual infill | Yes | Success ≥ max(2.2, 2.3)? |

**Cross-strategy analysis:**
- Compute Jaccard similarity between keyframe sets from all strategies
- High overlap → robust signal; low overlap → semantics are ambiguous

#### 2B: Encoder-Based Event Detection (Keyframe Method 2)

| Exp | Stage | What | Key Metric |
|-----|-------|------|------------|
| 2.6 | Stage 0 only | Pseudo-keyframes from Delta_t + A_t (no training) | Overlap with gripper_change GT, success rate vs 2.3 |
| 2.7 | Stage 0+1 | Trained scorer on pseudo-labels | Precision/recall vs Stage 0, temporal sparsity |
| 2.8 | Stage 0+1+2 | Scorer + planning head soft coupling | Success rate vs hard-pipeline baseline |

**Exp 2.6 is the critical gate for Method 2:**
- If Stage 0 pseudo-keyframes overlap >70% with gripper_change (Jaccard) AND achieve comparable success rate → encoder latent space is good enough, proceed to Stage 1
- If overlap <50% → encoder doesn't capture manipulation events well → stay with Method 1 rule-based strategies

**Exp 2.7 vs 2.6 delta:**
- If trained scorer (2.7) has higher precision than Stage 0 heuristics (2.6) at same recall → learning helps
- If no improvement → Stage 0 is already saturated, skip Stage 1 and use pseudo-labels directly

**Exp 2.8 vs hard-pipeline baseline:**
- Hard pipeline: scorer → fixed keyframes → planning head (no gradient flow)
- Soft pipeline: scorer → score-weighted attention → planning head (joint optimization)
- If soft > hard by >5% success → coupling matters, invest in Option B/C
- If soft ≈ hard → hard pipeline is fine, simpler to maintain

---

### Series 3: Latent Space Quality (验证潜在空间的适配性)

**Question:** Is Model B's encoder latent space suitable for Φ-guidance?

**Method:** Pure diagnostic tests. No task execution.

#### Exp 3.1 — Neighborhood Consistency

| Item | Value |
|------|-------|
| **Setup** | Encode all frames from GT trajectories. For each z_t, find K=10 nearest neighbors across the dataset |
| **Measure** | Phase consistency (% neighbors from same task phase), temporal locality (avg \|t - t'\|) |
| **Pass** | >70% phase consistency |
| **Fail** | <50% → encoder is texture-dominated → need projection head |

#### Exp 3.2 — Gradient Direction Validation

| Item | Value |
|------|-------|
| **Setup** | For (z_t, z_g) pairs from GT, compute `z' = z_t - ε·∇Φ`, check if z' is closer to actual intermediate frame |
| **Measure** | % of gradient steps that reduce distance to the real midpoint |
| **Pass** | >60% |
| **Fail** | <40% → L2 distance ≠ reachability → need reachability predictor |

#### Exp 3.3 — Interpolation Smoothness

| Item | Value |
|------|-------|
| **Setup** | Linearly interpolate `z(α) = (1-α)·z_t + α·z_{t+k}`, find nearest-neighbor frame index for each α |
| **Measure** | Monotonicity (does NN frame index increase with α?), coverage (does it hit all intermediate frames?) |
| **Pass** | Monotonic in >80% of segments |
| **Fail** | Non-monotonic → topology gaps in latent space |

---

### Series 4: Guidance Mechanism (验证梯度传播的正确性)

**Question:** Is the guidance gradient numerically stable and semantically useful?

#### Exp 4.1 — Gradient Magnitude Monitoring

| Item | Value |
|------|-------|
| **Setup** | During Exp 1.2, log `\|\|∇Φ\|\|`, `\|\|z_t - z_g\|\|`, action magnitude at each step |
| **Measure** | Gradient-distance correlation, guidance-induced action change |
| **Pass** | Gradient ∝ sqrt(distance), smooth decay near subgoal |
| **Fail** | Gradient saturates or vanishes → need adaptive λ or gradient clipping |

#### Exp 4.2 — Guidance vs. Base Policy Conflict

| Item | Value |
|------|-------|
| **Setup** | Measure cosine similarity between base action `a_base` and guided action `a_guided` |
| **Measure** | Success when guidance agrees (cos > 0.5) vs. conflicts (cos < 0) |
| **Pass** | Higher success when guidance agrees → guidance is helpful correction |
| **Fail** | Lower success when guidance agrees → Φ is adversarial to base policy |

---

### Series 5: Subgoal Switching (验证切换策略)

**Question:** When should the policy advance to the next subgoal?

**Fixed:** Best strategy from Series 2, λ* from 1.3

| Exp | Strategy | Control Variable | Key Metric |
|-----|----------|------------------|------------|
| 5.1 | Distance-based | `threshold ∈ {0.1, 0.5, 1.0, 2.0}` | # subgoals reached, time per subgoal |
| 5.2 | Time-based | `frames_per_subgoal` | Does policy reach subgoal before forced switch? |
| 5.3 | Soft weighting | `Φ = Σ_k w_k·\|\|z_t - z_gk\|\|^2` with `w_k = exp(-β·d_k)` | Gradient stability, success rate |

**Interpretation:**
- Soft weighting wins → task benefits from multi-subgoal context
- Hard switching wins → task has clear phase boundaries

---

## 5. Keyframe Extraction Methods

Two approaches for extracting subgoal keyframes from demonstration data. Method 1 provides training labels for Method 2, and both feed into the implementation methods (Section 6).

### Keyframe Method 1: Rule-Based Signal Processing (Existing Code)

**Status:** Implemented in `experiments/gt_keyframe_test/extract_keyframes.py`

Five strategies operating on raw signals (pixel values, action vectors). No learned components.

```
Strategy            Signal                          Detection Rule
──────────────     ──────────────                  ──────────────
uniform             fixed interval                  frame_idx % N == 0
visual_change       pixel MSE(f_t, f_{t-1})         MSE > threshold (0.05)
gripper_change      |action[t,6] - action[t-1,6]|   delta > 0.3
action_milestone    ||action_velocity||              velocity crosses threshold (0.1)
semantic            motion_score + change_score      motion stop OR visual change
```

**Strengths:**
- Zero training cost
- Gripper_change is highly precise for manipulation (directly detects grasp/release)
- Action_milestone captures phase boundaries (approach → grasp → transport → place)
- Already integrated into AR policy with non-uniform lookahead (ar.py:1267-1284)

**Weaknesses:**
- All pixel-based strategies use raw MSE, not task-relevant features
  (lighting change = same score as gripper contact)
- Gripper_change is too sparse (2-5 keyframes, 30-40 frame gaps with no guidance)
- Strategies are mutually exclusive in current config (no composition)
- Thresholds are task-agnostic and manually set
- Requires HDF5 action data for best strategies (gripper, milestone)

**Role in the system:**
- Serves as GT labels for training Keyframe Method 2
- Composite strategy (gripper anchors + visual infill) recommended as default GT
- Ablation across strategies validates which signals matter (Series 2 experiments)

---

### Keyframe Method 2: Encoder-Based Event Detection (New)

**Status:** Not yet implemented. Depends on Method 1 for training labels.

A learned keyframe detector operating in Model B's encoder latent space. Three stages:

#### Stage 0: Pseudo-Keyframe Generation (Rule-Based Bootstrapping)

Use frozen Model B encoder to build event-driven pseudo-labels. This is the **critical bootstrap step** — 90% of the value comes from here.

**0.1 — First-order latent change (sustained dynamics):**

```
For each frame in a demo trajectory:
  z_t = Enc(x_t)                          # frozen Model B encoder
  Delta_t = ||z_t - z_{t-1}||^2           # latent temporal difference

Keyframe if:
  Delta_t > percentile(Delta, 90%)         # top 10% of changes
  OR
  Delta_t > mean(Delta) + k * std(Delta)   # k=2 recommended
```

Unlike pixel MSE (Method 1's visual_change), this operates in encoder feature space:
- Suppresses task-irrelevant visual variation (lighting, shadows)
- Amplifies task-relevant state changes (gripper contact, object displacement)
- Natively aligned with the Phi potential space

**0.2 — Second-order latent change (instantaneous events):**

Many contact events are not large Delta_t but sudden *changes* in Delta_t:

```
A_t = |Delta_t - Delta_{t-1}|             # acceleration of latent change

Keyframe if:
  A_t > threshold
```

Intuition:
- Delta captures sustained change (arm moving continuously)
- A captures instantaneous events (contact snap, object state flip)
- Together they cover both gradual transitions and abrupt events

**0.3 — Score fusion + temporal NMS:**

```
score_t = w1 * normalize(Delta_t) + w2 * normalize(A_t)

# w1=0.6, w2=0.4 recommended (sustained change slightly more important)

Then:
  1. Temporal NMS: within +/- k frames (k=3), keep only highest score
  2. Cap at max_keyframes (e.g. 15)
  3. Always include first and last frame
```

**Output:** A set of pseudo-keyframe indices per trajectory, ready as training labels.

#### Stage 1: Learned Keyframe Scorer

Train a lightweight event detector using Stage 0 pseudo-labels as weak supervision.

**Architecture (must be lightweight — this is a scoring problem, not generation):**

```
Input:   z_t, z_{t-1}, z_{t-2}                 # 3-frame context window
Feature: [z_t, z_t - z_{t-1}, z_{t-1} - z_{t-2}]  # absolute + first/second diff
Model:   MLP (2-3 layers, hidden=256)
         OR 1-layer Transformer (if variable-length context needed)
Output:  s_t in [0, 1]                         # event probability
```

**Training labels:**

```
Positive:  frames selected by Stage 0 pseudo-labeling
Negative:  frames where BOTH Delta_t AND A_t are below median
           AND at least tau frames away from any positive (tau=5)
```

This is weak supervision — Stage 0 rules are noisy, but the pattern
(contact events → high latent change) is strong enough for a simple scorer.

**Loss (precision-biased, not standard BCE):**

```
L = BCE(s_t, y_t) + lambda_sparse * sum_t(s_t)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                     temporal sparsity regularizer:
                     penalizes the model for firing too often
                     forces it to only activate at true events
```

lambda_sparse = 0.01-0.1 (tune to get ~5-15 keyframes per trajectory).

The sparsity term is critical: without it, the model learns to predict
s_t > 0.5 for any "interesting" frame, producing 30+ keyframes that are
no better than visual_change. With it, the model is forced to discriminate
between "somewhat interesting" and "truly event-critical."

**What this gives you:**

```
KeyframeScorer(z_1:T) -> {(t1, s_t1), (t2, s_t2), ...}

Properties:
  - Operates in encoder latent space (aligned with Phi)
  - Captures both sustained + instantaneous events
  - Generalizes beyond hard thresholds (learns soft boundaries)
  - Runs in <1ms per trajectory (MLP on pre-computed latents)
```

#### Stage 2: Coupling with Planning Head

**The wrong way (hard pipeline):**

```
KeyframeScorer -> fixed keyframe set -> Planning Head trains on these
```

Problem: the planning head is locked to whatever the scorer decided.
If the scorer makes a mistake (missed event, false positive), the
planning head has no way to recover.

**The right way (soft coupling):**

```
Option A: Score-weighted subgoal attention

  Planning Head receives ALL frame latents, weighted by scorer:
    attention_weight_t = softmax(s_t / temperature)

  Planning Head learns to attend to high-score frames
  BUT can also attend to low-score frames if they help planning

  -> Scorer provides prior, planning head has final say

Option B: Joint training (end-to-end)

  L_total = L_plan + alpha * L_keyframe

  L_plan:      standard planning head loss (predict future subgoals)
  L_keyframe:  BCE + sparsity from Stage 1

  Gradient from L_plan flows back through score-weighted attention
  -> Scorer learns "which frames help the planner most"
  -> This is the theoretically cleanest approach

  alpha = 0.1 (keyframe loss as regularizer, not primary objective)

Option C: Iterative refinement (practical compromise)

  Round 1: Train scorer with Stage 0 pseudo-labels
  Round 2: Train planning head using scorer's keyframes
  Round 3: Re-train scorer using planning head's loss as signal
           (frames where planning head has high reconstruction error
            → these are informative frames the scorer should have caught)

  -> Avoids end-to-end training complexity
  -> 2-3 rounds typically sufficient
```

**Recommended path:** Start with Option A (score-weighted attention).
If the planning head consistently ignores the scorer's suggestions,
escalate to Option C (iterative refinement).

---

### Keyframe Methods: Comparison

```
                        Method 1 (Rule-Based)           Method 2 (Encoder-Based)
                        ─────────────────────          ──────────────────────────
Signal space            raw pixels / action vectors     encoder latent space
Task relevance          gripper_change: high            high (learned to filter noise)
                        visual_change: low (noisy)
Threshold tuning        manual per-task                 learned (pseudo-label + train)
Contact detection       gripper: direct                 first-order Delta_t
Instantaneous events    none                            second-order A_t
Composability           strategies are independent      fused score (w1*Delta + w2*A)
Training cost           zero                            Stage 0: zero, Stage 1: ~1hr
Phi alignment           not aligned (pixel space)       natively aligned (same encoder)
HDF5 dependency         gripper/milestone: yes          no (only needs video frames)
Planning head coupling  hard (fixed keyframe set)       soft (score-weighted attention)
```

---

## 6. Implementation Methods (Action Generation)

### Method 1: Frozen Encoder + Lightweight Planning Head

**When to use:** After Series 1-5 validate that Φ-guidance works with GT keyframes.

```
  +---------------------------+
  |  Model B Visual Encoder    |
  |  (FROZEN)                  |
  +---------------------------+
               |
               v
     z_t = Enc(o_t)
               |
               v
  +---------------------------+
  |  Lightweight Planning Head |
  |  (Transformer / MLP)       |
  |  Input: z_t + task_embed   |
  |  Output: {z_g1, ..., z_gK} |
  +---------------------------+
               |
               v
    Φ = ||Enc(x_t) - z_gk||^2   →   Guide Model B
```

**Training:**

```
Step 1: Data prep
  Video:    [f_0, f_1, ..., f_T]
  Sample:   [f_0, f_N, f_2N, ..., f_KN]   (N from Exp 1.4 optimal interval)
  Encode:   [z_0, z_N, z_2N, ..., z_KN]    (frozen encoder)

Step 2: Train planning head
  Input:   z_0 + task_embedding
  Target:  [z_N, z_2N, ..., z_KN]
  Loss:    L = Σ_k ||predicted_z_gk - actual_z_gk||^2

Step 3: Inference
  1. z_t = Enc(o_t)
  2. {z_g1, ..., z_gK} = PlanHead(z_t, task)
  3. Φ = ||Enc(x_t) - z_gk*||^2    (k* = nearest subgoal)
  4. ε̃ = ε_θ(xt) - λ · σ_t · ∇_xt Φ
  5. Execute actions, observe, repeat
```

**Planning head options:**
- **MLP**: Simplest. `[z_0; task_embed] → K * latent_dim`.
- **Transformer decoder**: K learnable queries, cross-attend to task embedding. Better for variable K.
- **Latent diffusion**: Model distribution over subgoal sequences. Preserves multi-modality.

**Key design decisions:**

| Decision | Informed by | Recommended Default |
|----------|-------------|---------------------|
| Keyframe interval N | Exp 1.4 | Start with optimal from ablation |
| Number of subgoals K | Task length analysis | 5-8 for manipulation |
| Switching strategy | Exp 5.x | Best from ablation |
| Guidance strength λ | Exp 1.3 | λ* from ablation |

### Method 2: MPC-Style Rollout (Zero-Training Baseline)

**When to use:** First, to validate the Φ-guidance framework before training any new model.

```
Algorithm:
  1. Sample M action sequences from prior
  2. Rollout each through Model B: ô_{t+H} = ModelB.predict(o_t, a_i)
  3. Score: s_i = similarity(Enc(ô_{t+H}), task_goal)
  4. Best rollout endpoint → next subgoal g_next
  5. Φ = ||Enc(x_t) - Enc(g_next)||^2
  6. Use Φ to guide Model B, execute, repeat
```

**Variants:**

| Variant | Method | Planning Horizon |
|---------|--------|------------------|
| Random Shooting | `a ~ Uniform(a_min, a_max)` | H steps |
| CEM | Iterative mean/variance refinement | H steps |
| Chained | Sequence rollouts g_1→g_2→...→g_K | K×H steps |

**Pros:** Zero training, validates framework immediately, subgoals are real predicted images.
**Cons:** Short horizon (H steps), M forward passes per planning step, compute-heavy.

---

### Method 3: WanTI2V Non-Causal Planner + Haar Temporal Decomposition

**When to use:** As the primary Model A for long-horizon planning. Does not require training — uses the existing Wan 2.2 TI2V 5B model and checkpoint already in the codebase.

#### 3.1 Motivation: Why Methods 1 & 2 Cannot Do Long-Horizon

Methods 1 and 2 both suffer from the same bottleneck — they lack a world dynamics model:

```
Method 1 (Planning Head):
  Input:  z_0 (single frame) + task_text
  Output: K future subgoal latents
  Problem: Must "hallucinate" future world states from a snapshot.
           The planning head has no physics model, no dynamics.
           It can only memorize trajectory patterns from training data.
           → Prediction variance explodes beyond ~20 frames.

Method 2 (MPC Rollout):
  Input:  current observation + M random action sequences
  Output: best rollout endpoint as subgoal
  Problem: Planning horizon = H model rollout steps (short).
           Chaining rollouts compounds prediction error.
           → Effective horizon limited to ~8-16 frames.

Both miss: a model that "knows how the world evolves over time"
```

The long-horizon signal must come from a model that has learned temporal dynamics from large-scale video data. WanTI2V (non-causal, TI2V 5B) is exactly this model — and it already exists in the codebase.

#### 3.2 Core Idea: "Think" with WanTI2V, "Act" with Vidarc

```
WanTI2V (non-causal, 5B)              Vidarc (causal, few-step)
"Plan the full trajectory"             "Execute frame by frame"
─────────────────────────              ──────────────────────────
Full bidirectional attention           Causal attention + KV cache
All frames see each other              Only sees past frames
One-shot: 49 frames (2s plan)          Autoregressive: 1 frame/step
50 denoising steps (high quality)      10 steps (fast, few-step)
Slow (~1-12s per plan)                 Fast (~0.1s per frame)
Open-loop imagination                  Closed-loop with real obs
```

**Key insight:** WanTI2V's full bidirectional attention guarantees that frame 1 and frame 49 are globally coherent — something Vidarc's causal generation with KV cache pop fundamentally cannot achieve.

#### 3.3 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: PLAN (slow, but has global temporal vision)           │
│                                                                  │
│  obs_t + task_text                                               │
│    → WanTI2V.generate(frame_num=49, size=(320,368), steps=20)   │
│    → plan_video [3, 49, 368, 320]                                │
│    → ~0.5-1.5s on A100 (low-res + few steps)                    │
│                                                                  │
│  Source: run_keyframe_model.py (SubgoalGenerator, already coded) │
│  Checkpoint: TI2V-5B (already available)                         │
│  Code change: adjust frame_num, size, sampling_steps             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌─────────────────┐ ┌──────────────────┐
│ A: Pixel-space    │ │ B: Latent-space │ │ C: Low-freq      │
│    keyframes      │ │    subgoals     │ │    trajectory     │
│                   │ │                 │ │                   │
│ extract_keyframes │ │ WanVAE.encode   │ │ WanVAE.encode    │
│ (5 strategies,    │ │ → z_plan        │ │ → z_plan          │
│  already coded)   │ │ → slice at      │ │ → Haar decompose │
│                   │ │   keyframe idx  │ │ → z_LF trajectory │
└────────┬─────────┘ └───────┬─────────┘ └────────┬─────────┘
         │                   │                     │
         └───────────┬───────┘                     │
                     ▼                             ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ Phase 2: GUIDE               │  │ Phase 2': MONITOR             │
│                               │  │                               │
│ inject_gt_subgoals(policy,    │  │ divergence = ||z_LF_obs       │
│   keyframes)                  │  │              - z_LF_plan||    │
│                               │  │                               │
│ Vidarc + Φ-guidance:          │  │ if divergence > threshold:    │
│   ε̃ = ε_θ - λ·σ_t·∇Φ        │  │   → trigger re-plan          │
│                               │  │   → back to Phase 1           │
│ Already coded:                │  │                               │
│   ar.py + server.py           │  │ Low-freq is smooth →          │
│                               │  │   stable divergence signal    │
└──────────────────────────────┘  └──────────────────────────────┘
```

#### 3.4 Haar Temporal Decomposition on Wan-VAE Latents

The temporal frequency decomposition does NOT require modifying Wan 2.2 or adding WF-VAE. It is a pure post-processing operation on Wan-VAE's existing latent output.

**Why not WF-VAE?** WF-VAE has incompatible latent space (different z_dim, different spatial stride, different distribution statistics, non-causal temporal alignment). Integrating it would require retraining the entire DiT. Instead, we apply Haar wavelets directly on Wan-VAE latents — zero training cost, perfect causal alignment.

**Implementation (~10 lines):**

```python
import math
import torch

def haar_temporal_decompose(z: torch.Tensor, levels: int = 1):
    """Haar wavelet temporal decomposition on Wan-VAE latents.

    Args:
        z: [B, C, T, H, W] — Wan-VAE encoded latent
        levels: number of decomposition levels (each halves T)

    Returns:
        z_low:  [B, C, T//(2^levels), H, W] — temporal low-frequency
        z_highs: list of high-frequency bands per level
    """
    z_highs = []
    for _ in range(levels):
        z_even = z[:, :, 0::2, :, :]
        z_odd  = z[:, :, 1::2, :, :]
        T_min = min(z_even.shape[2], z_odd.shape[2])
        z_even, z_odd = z_even[:, :, :T_min], z_odd[:, :, :T_min]
        z_low  = (z_even + z_odd) / math.sqrt(2)
        z_high = (z_even - z_odd) / math.sqrt(2)
        z_highs.append(z_high)
        z = z_low
    return z_low, z_highs
```

**What each level captures:**

```
Input: z_plan from WanVAE.encode(plan_video)
       shape [1, 48, 13, H', W']  (for 49-frame plan)

Level 1:
  z_LF:  [1, 48, 6, H', W']   each frame ≈ 8 original frames (0.33s)
  z_HF:  [1, 48, 6, H', W']   frame-to-frame motion detail

Level 2:
  z_LF:  [1, 48, 3, H', W']   each frame ≈ 16 original frames (0.67s)
  z_HF1: [1, 48, 6, H', W']   fine motion
  z_HF2: [1, 48, 3, H', W']   coarse motion

For a 49-frame (2s) plan:
  3 low-freq subgoals, each covering ~0.67s of task
  → Enough to represent approach / grasp / transport phases
```

**Why this helps Φ-guidance (addresses Problems 1-3 from subgoal.md):**

| Problem | Without Haar | With Haar |
|---------|-------------|-----------|
| P1: Keyframe granularity | N choice is critical and task-dependent | Low-freq subgoals are adaptive (wavelet-determined) |
| P2: Planning head generalization | Must predict far-future from single frame | WanTI2V provides the plan; Haar just filters it |
| P3: Latent misalignment | Pixel-level L2 ≠ reachability | Low-freq filters visual noise (lighting, texture); distance reflects spatial structure |

#### 3.5 Fast Planning: Low-Resolution + Fewer Steps

Planning does not need pixel-perfect quality. It only needs correct spatial structure (where objects are, where the arm goes).

**Optimization parameters:**

| Setting | Default (high quality) | Fast plan | Speedup |
|---------|----------------------|-----------|---------|
| Resolution | 640×736 | 320×368 | ~16× (O(n²) attention) |
| frame_num | 121 (5s) | 49 (2s) | ~6.8× |
| sampling_steps | 50 | 20 (DPM++) | 2.5× |
| **Total** | **~50s on A100** | **~0.5s on A100** | **~100×** |

```python
# Fast plan generation (in SubgoalGenerator or new wrapper)
plan_video = model.generate(
    input_prompt=task_prompt,
    img=observation,
    frame_num=49,              # 2s plan, covers 1 subtask
    size=(320, 368),           # low-res for speed
    sampling_steps=20,         # DPM++ 20 steps
    sample_solver='dpm++',
    guide_scale=5.0,
    seed=seed,
)
```

#### 3.6 Closed-Loop Re-Planning

Instead of generating one long plan, generate short plans and re-plan when execution diverges.

```
Algorithm: Closed-Loop Plan-Execute-Replan

  plan_LF ← None
  subgoals ← []
  sg_idx ← 0

  loop:
    obs ← get_observation()

    # Re-plan if needed
    if plan_LF is None OR diverged(obs, plan_LF, sg_idx):
      plan_video ← WanTI2V.generate(obs, task, frame_num=49, ...)  # ~0.5s
      z_plan ← WanVAE.encode(plan_video)
      plan_LF, _ ← haar_temporal_decompose(z_plan, levels=2)
      subgoals ← extract_keyframes(plan_video, strategy="composite")
      sg_idx ← 0

    # Execute with subgoal guidance
    current_sg ← subgoals[sg_idx]
    action ← vidarc_step(obs, current_sg, λ=λ*)

    # Advance subgoal
    if reached(obs, current_sg):
      sg_idx += 1
      if sg_idx >= len(subgoals):
        plan_LF ← None  # Force re-plan for next phase

    execute(action)
```

**Divergence detection in low-frequency space:**

```python
def diverged(obs, plan_LF, sg_idx, threshold=0.3):
    """Check if execution has diverged from the plan."""
    z_obs = wan_vae.encode(obs)
    # Compare low-freq representation of current state vs expected
    # plan_LF[:, :, sg_idx] is the expected low-freq state at this phase
    if sg_idx >= plan_LF.shape[2]:
        return True  # Past end of plan
    z_expected = plan_LF[:, :, sg_idx:sg_idx+1, :, :]
    # Spatial average for robust comparison
    z_obs_avg = z_obs.mean(dim=(-2, -1))
    z_exp_avg = z_expected.mean(dim=(-2, -1))
    return (z_obs_avg - z_exp_avg).norm() > threshold
```

**Properties:**
- Each re-plan starts from the **current real observation**, not the imagined future → error does not accumulate
- Low-freq divergence metric is smooth and stable (filters out frame-level noise)
- Effective horizon is **unlimited**: re-plan at each subtask boundary, each covering 2s
- Latency: ~0.5s re-plan + real-time execution = acceptable for manipulation

#### 3.7 Longer Horizon via Chained Plans

For tasks longer than 2s (49 frames), chain multiple plan rounds:

```
Round 1:  obs_0 → WanTI2V → plan_0 (frames 0-48, 2s)
          Take plan_0's last frame as obs_1

Round 2:  obs_1 → WanTI2V → plan_1 (frames 49-96, 2s)

Round 3:  obs_2 → WanTI2V → plan_2 (frames 97-144, 2s)

Full plan: plan_0 + plan_1 + plan_2 = 6s
```

**Haar decomposition smooths chain boundaries:**

```
z_full = cat(z_plan_0, z_plan_1, z_plan_2)   # [1, 48, 39, H', W']
z_LF, _ = haar_temporal_decompose(z_full, levels=2)  # [1, 48, 9, H', W']

# Low-freq averaging naturally smooths boundary discontinuities:
#   z_LF[boundary] ≈ average(last frames of plan_k, first frames of plan_{k+1})
#   → Smooth transition instead of hard cut
```

**When to chain vs. re-plan:**
- **Pre-plan chaining:** Generate full plan before execution (offline, for visualization/debugging)
- **Online re-planning:** Generate one 2s plan at a time during execution (recommended for deployment, handles perturbations)

#### 3.8 Integration with Existing Code

| Component | Existing Code | Change Needed |
|-----------|--------------|---------------|
| WanTI2V model | `wan/textimage2video.py:WanTI2V` | None |
| TI2V-5B checkpoint | Already downloaded | None |
| SubgoalGenerator | `experiments/subgoal/run_keyframe_model.py:SubgoalGenerator` | Adjust `frame_num`, `size`, `sampling_steps` |
| Keyframe extraction | `experiments/gt_keyframe_test/extract_keyframes.py` | None |
| Subgoal injection | `ar.py:inject_gt_subgoals()` | None |
| Vidarc server guidance | `textimage2video_causal_server.py:880-910` | Replace linear interpolation with Φ-gradient (optional, Phase 2) |
| **Haar decomposition** | **Not yet implemented** | **~10 lines, new utility function** |
| **Divergence detection** | **Not yet implemented** | **~15 lines, new utility function** |
| **Re-plan loop** | **Not yet implemented** | **~30 lines, new control loop** |

**Total new code: ~55 lines. No model training. No architecture changes.**

#### 3.9 Comparison: Method 3 vs Methods 1 & 2

```
                        Method 1              Method 2             Method 3
                        (Planning Head)       (MPC Rollout)        (WanTI2V Planner)
                        ────────────────     ────────────────     ────────────────────
Model A                 MLP/Transformer       Model B itself       WanTI2V (non-causal)
Training cost           Hours (head only)     Zero                 Zero
Long-horizon            Limited (ill-posed)   H steps (short)      2s/round, unlimited
World dynamics          None (memorized)      Short-horizon only   Pretrained on 100M+ videos
Frequency decomposition None                  None                 Haar on Wan-VAE latents
Re-planning             Possible but costly   Natural (per-step)   Natural (per-subtask)
Interpretability        Latent (opaque)       Image (visible)      Image (visible)
Compute per plan        <1ms                  M×H forward passes   ~0.5s (fast mode)
Failure mode            Wrong subgoals        Myopic planning      Plan doesn't match scene
Mitigation              Re-planning           Longer H             Fine-tune WanTI2V (LoRA)
```

**Note:** Method 3 is subsumed by Method C below. Method 3's components (WanTI2V generation, Haar decomposition, divergence detection) are reused within Method C, but long-range task structure is managed by VLM instead of chaining.

---

### Method B: VLM + GT Demo Retrieval (Recommended First Implementation)

**When to use:** As the first complete pipeline implementation. Validates the VLM-based planning framework using highest-quality subgoals (GT demos). See [theory.md](theory.md) Section 6 for why B before C.

**Evaluation target:** 160 frames @ 10fps = 16s tasks.

#### B.1 Offline: Build Keyframe Library

```
For each GT demo HDF5:
  1. Load images + actions
  2. Extract keyframes: extract_keyframes_from_hdf5(strategy="composite")
     → gripper anchors + visual infill (existing code)
  3. Encode each keyframe: z_kf = vidarc_encoder(img)
  4. Auto-label phase from action signals:
     gripper closing → "grasp", gripper open + moving → "approach", etc.
  5. Store: library[task][phase].append({image, z_latent, progress, demo_id})

Library structure:
  keyframe_library/
  ├── pick_and_place_red_block/
  │   ├── approach/    (10-20 keyframes from different demos)
  │   ├── grasp/       (5-10 keyframes)
  │   ├── transport/   (15-25 keyframes)
  │   ├── place/       (5-10 keyframes)
  │   └── retreat/     (5-10 keyframes)
  └── ...
```

**Phase auto-labeling (from action signals, no VLM needed):**

```python
def assign_phase(kf_idx, actions, gripper_col=6):
    gripper = actions[:, gripper_col]
    velocities = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
    g = gripper[kf_idx]
    v = velocities[min(kf_idx, len(velocities)-1)]
    dg = abs(g - gripper[max(0, kf_idx-1)]) if kf_idx > 0 else 0

    if dg > 0.3 and g < gripper[max(0, kf_idx-1)]:   return "grasp"
    elif dg > 0.3:                                      return "place"
    elif g < 0.5 and v > 0.05:                          return "transport"
    elif g >= 0.5 and v > 0.05:                         return "approach"
    else:                                                return "idle"
```

#### B.2 Online: VLM Phase Detection + Retrieval

```
┌──────────────────────────────────────────────────────────────┐
│ Execution Loop                                                │
│                                                               │
│ obs_t ──┬──→ VLM Phase Detector (async prefetch)              │
│         │       input:  obs_t + task_text + phase_history     │
│         │       output: current_phase, subtask_complete       │
│         │       freq:   event-driven (5-8 calls/episode)      │
│         │                                                     │
│         ├──→ Keyframe Retrieval                               │
│         │       z_obs = encoder(obs_t)                        │
│         │       candidates = library[task][current_phase]     │
│         │       subgoal = nearest_forward(z_obs, candidates)  │
│         │                                                     │
│         └──→ Vidarc + Φ-guidance                              │
│               inject_gt_subgoals(policy, [subgoal])           │
│               action = vidarc_step(obs_t, subgoal, λ=λ*)     │
│                                                               │
│ VLM calling strategy (event-driven, async prefetch):          │
│   Trigger at: subgoal reached / no progress for N frames /    │
│               subtask start                                   │
│   Prefetch at 70% subtask progress → result ready before need │
│   Only first call (task decomposition) is blocking (~0.5-2s)  │
│   Total VLM overhead: ~2s blocking + 0s async                 │
└──────────────────────────────────────────────────────────────┘
```

**VLM prompt (phase detection):**

```
Task: {task_text}
Phase history: {phase_history}
Look at the observation. Answer:
1. Current phase: [approach/grasp/transport/place/retreat/done]
2. Subtask complete? (yes/no)
Reply JSON: {"phase": "...", "subtask_complete": true/false}
```

**Retrieval: nearest-forward in latent space:**

```python
def retrieve_subgoal(z_obs, library_phase, current_progress, k=3):
    # Only consider forward-looking candidates
    candidates = [e for e in library_phase if e['progress'] > current_progress + 0.02]
    if not candidates:
        return None  # → trigger VLM re-check

    # Top-k nearest in latent space
    dists = [(e, (z_obs - e['z_latent']).norm()) for e in candidates]
    top_k = sorted(dists, key=lambda x: x[1])[:k]

    # From top-k, pick smallest progress delta (nearest "next step")
    return min(top_k, key=lambda x: x[0]['progress'])[0]
```

#### B.3 VLM Choice

| Model | Size | VRAM | Latency | Recommendation |
|-------|------|------|---------|----------------|
| Qwen2.5-VL-7B | 7B | ~14GB | 0.1-0.2s | **Primary (local)** |
| Qwen2.5-VL-3B | 3B | ~6GB | ~0.1s | Memory-constrained fallback |
| GPT-4o API | - | 0 | 0.5-1.5s | Phase 0 validation only |

**Validation path:** GPT-4o API first (verify approach) → Qwen2.5-VL-7B (deploy locally).

#### B.4 Limitations

```
1. Generalization: only retrieves from GT library
   → New object positions / configurations → no matching keyframe
   → Upgrade to Method C when this becomes a bottleneck

2. Progress estimation: frame_idx / 160 is coarse
   → Improve: latent distance to task goal

3. Phase boundary ambiguity: VLM may oscillate at transitions
   → Hysteresis: require N consecutive confirmations to advance
```

---

### Method C: VLM + WanTI2V Collaboration (Generalization Upgrade)

**When to use:** When Method B's GT library cannot cover the evaluation scenarios (new object configurations, unseen task variants). Replaces GT retrieval with WanTI2V visual planning per subtask.

**Key simplification vs standalone Method 3:**
- No chaining (VLM manages cross-subtask transitions)
- No cross-round memory (VLM context window)
- No chain boundary smoothing
- WanTI2V generates only 1 short plan (49 frames) per subtask
- Haar decomposition only within a single plan (for divergence detection)

#### C.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Level 3: VLM Task Planner                                     │
│   Input: obs + task_text + conversation history               │
│   Output: subtask list + subtask prompts + completion criteria│
│   Memory: VLM context window (all past obs + decisions)       │
│   Calls: 1 (decompose) + 4-8 (completion checks)             │
├──────────────────────────────────────────────────────────────┤
│ Level 2: WanTI2V Short-Horizon Planner                        │
│   Input: obs + subtask_prompt (from VLM)                      │
│   Output: plan_video (49 frames, 2s) → keyframes + z_LF      │
│   Calls: 1 per subtask (+ re-plan on divergence)              │
├──────────────────────────────────────────────────────────────┤
│ Level 1: Vidarc + Φ-guidance                                  │
│   Input: obs + subgoal_frames (from Level 2)                  │
│   Output: actions                                             │
│   Runs: every frame (160 times)                               │
└──────────────────────────────────────────────────────────────┘
```

#### C.2 VLM Task Decomposition

```
VLM prompt (task decomposition, called once at episode start):

  "You are planning a robot manipulation task.
   Task: {task_text}
   [image: initial observation]
   Decompose into 3-6 sequential subtasks.
   For each: name, prompt (for video model), completion_criterion.
   Reply JSON."

Example output:
  [
    {"name": "approach",  "prompt": "robot arm moves toward the red block",
     "completion_criterion": "gripper directly above the block"},
    {"name": "grasp",     "prompt": "robot gripper closes on the red block",
     "completion_criterion": "block lifted off the table"},
    {"name": "transport", "prompt": "robot carries block toward the shelf",
     "completion_criterion": "block is above the shelf"},
    {"name": "place",     "prompt": "robot places block on shelf and opens gripper",
     "completion_criterion": "block rests on shelf, gripper open"}
  ]

→ Each subtask.prompt feeds directly into WanTI2V.generate()
→ VLM context window accumulates all obs + decisions = long-term memory
```

#### C.3 Per-Subtask Visual Planning

```python
# For each subtask:
plan_video = wantiv.generate(
    input_prompt=subtask['prompt'],   # from VLM decomposition
    img=current_obs,
    frame_num=49,                     # 2s plan (single subtask)
    size=(320, 368),
    sampling_steps=20,
    sample_solver='dpm++',
)

z_plan = wan_vae.encode(plan_video)
z_LF, _ = haar_temporal_decompose(z_plan, levels=1)  # divergence monitor
keyframes = extract_latent_keyframes(z_plan, max_keyframes=6)
```

#### C.4 Execution Loop

```
Algorithm:
  subtasks ← VLM.decompose(obs_0, task)          # blocking, ~2s
  for subtask in subtasks:
    keyframes, z_LF ← WanTI2V_plan(obs, subtask.prompt)  # ~1s
    sg_idx ← 0
    for frame_idx in range(budget):
      action ← vidarc_step(obs, keyframes[sg_idx])
      obs ← env.step(action)

      if reached(obs, keyframes[sg_idx]):
        sg_idx += 1

      if sg_idx >= len(keyframes) OR diverged(obs, z_LF):
        complete ← VLM.check_completion(obs, subtask)  # async
        if complete → break to next subtask
        else → re-plan: keyframes, z_LF ← WanTI2V_plan(obs, subtask.prompt)
```

#### C.5 Timing for 160-Frame Episode

```
Frame:  0         40        60        100       140    160
VLM:    [D 2s]    [C]       [C]       [C]       [C]
WanTI2V:  [P 1s]    [P]       [P]       [P]
Vidarc: ·····██████████████████████████████████████████

D = decompose (blocking)
C = completion check (async prefetch, ~0 blocking)
P = per-subtask plan (~1s, overlaps with VLM check)

Total blocking overhead: ~2s (first VLM call)
Total async overhead: 0s (all prefetched)
```

#### C.6 When Method C Over Method B

```
Use Method B when:    GT library covers evaluation scenarios
                      Simplest pipeline, highest subgoal quality

Use Method C when:    New object configurations not in GT library
                      Tasks with more variability than demos cover
                      Need to generalize beyond training distribution
```

---

## 7. Roadmap

```
Phase 0: Validate Φ-Guidance + VLM Pipeline
  │
  ├── Week 0.1: Oracle baseline (does Φ-guidance help at all?)
  │   ├── Exp 1.1: No guidance baseline (160 frames)
  │   ├── Exp 1.2: GT keyframes (uniform) + Φ-guidance
  │   ├── Exp 1.2b: GT keyframes (composite) + Φ-guidance
  │   │
  │   ├── GATE: GT keyframes improve success rate by >10%?
  │   │   YES → proceed to 0.2
  │   │   NO  → fix Φ (Series 3/4 diagnostics), do NOT build planner
  │   │
  │   └── Parallel: VLM feasibility check
  │       ├── Use GPT-4o API on 5 episodes
  │       ├── Check: task decomposition reasonable?
  │       ├── Check: phase detection accurate?
  │       └── Check: completion criteria work?
  │
  ├── Week 0.2: Method B implementation
  │   ├── Build keyframe library from GT demos (offline, ~1hr)
  │   │   ├── extract_keyframes(composite) for all demos
  │   │   ├── Encode with Vidarc encoder → z_latent
  │   │   └── Auto-label phases from action signals
  │   │
  │   ├── Implement VLM phase detection + async prefetch
  │   ├── Implement nearest-forward retrieval
  │   ├── Wire into Vidarc execution loop
  │   │
  │   ├── Test: Method B on 5-10 episodes
  │   │
  │   └── GATE: Method B > no-guidance baseline?
  │       YES → Method B is viable, proceed to Phase 1
  │       NO  → debug: is VLM wrong? is retrieval wrong? is Φ wrong?
  │
  └── Week 0.3: Local VLM deployment
      ├── Deploy Qwen2.5-VL-7B, compare accuracy vs GPT-4o
      ├── Benchmark: latency, GPU memory, phase detection accuracy
      └── If 7B insufficient → try 72B or stay with API

Phase 1: Full Evaluation with Method B
  │
  ├── Week 1: Series 1 (λ ablation, interval ablation)
  │   ├── Exp 1.3: λ ablation with Method B subgoals
  │   ├── Exp 1.4: keyframe density ablation
  │   └── Establish optimal λ* and keyframe density
  │
  ├── Week 2: Series 2 (keyframe strategies) + Series 3 (latent diagnostics)
  │   ├── Compare retrieval with different library strategies
  │   │   (uniform vs composite vs gripper_change libraries)
  │   ├── Run Exp 2.6 (Stage 0 pseudo-keyframes) → GATE for Keyframe Method 2
  │   └── Run 3.x in parallel — pure diagnostic, no task execution
  │
  ├── Week 3: Series 4-5 (gradient + switching) + Method B full eval
  │   ├── Gradient analysis with retrieved subgoals
  │   ├── Subgoal switching strategy ablation
  │   ├── Full 160-frame evaluation across all tasks
  │   └── Identify failure cases: where does retrieval fail?
  │
  └── Week 4: Analysis + decision
      ├── Method B success rate across all tasks
      ├── Failure case analysis: which episodes have no good library match?
      │
      └── GATE: Method B sufficient for all evaluation scenarios?
          YES → ship Method B, proceed to Phase 3 refinement
          NO  → proceed to Phase 2 (Method C upgrade)

Phase 2: Method C — WanTI2V Upgrade (only if Method B insufficient)
  │
  ├── Week 5: WanTI2V plan quality validation
  │   ├── Run WanTI2V on Method B's failure cases
  │   ├── Human inspect: plan video quality on RoboTwin scenes
  │   ├── GATE: WanTI2V generates plausible robot plans?
  │   │   YES → proceed to integration
  │   │   NO  → LoRA fine-tune on RoboTwin data, re-check
  │   │
  │   └── Parallel: implement Haar decomposition + divergence detection
  │
  ├── Week 6: Method C implementation
  │   ├── VLM task decomposition → subtask prompts
  │   ├── WanTI2V per-subtask planning (frame_num=49, fast mode)
  │   ├── Latent-space keyframe extraction from plan
  │   ├── Re-plan on divergence
  │   └── Full pipeline integration
  │
  └── Week 7: Method C evaluation
      ├── Compare: Method C vs Method B on all tasks
      ├── Focus on Method B's failure cases: does Method C fix them?
      └── Hybrid: Method B for in-distribution, Method C for OOD

Phase 3: Iterate & Scale
  │
  ├── Φ variant experiments (latent L2 vs reachability vs CLIP)
  ├── Adaptive λ scheduling (large when far, small when near)
  ├── Guidance upgrade: Φ in Haar z_LF space vs full latent space
  ├── Pipeline parallelism (VLM + WanTI2V async)
  ├── If Keyframe Method 2 + soft coupling works:
  │   joint end-to-end scorer training (Option B)
  └── WanTI2V LoRA fine-tuning (if plan quality is bottleneck)
```

---

## 8. Related Documents

- **[theory.md](theory.md)** — Design principles, motivation, and architectural rationale for the three-level planning hierarchy
- **[subgoal.md](subgoal.md)** — Theoretical foundation: Φ-guidance, potential functions, and open problems analysis

## 9. References

- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021 — classifier guidance
- Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis", ICML 2022 — diffusion as planner
- Luo & Du, "Grounding Video Models to Actions through Goal Conditioned Exploration", ICLR 2025 — video-to-action with subgoal pursuit
- Chi et al., "Diffusion Policy", RSS 2023 — diffusion-based action generation
- Ko et al., "Learning to Act from Actionless Videos through Dense Correspondences", ICLR 2024 — AVDC
