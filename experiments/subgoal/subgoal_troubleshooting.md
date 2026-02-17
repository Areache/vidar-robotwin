# Subgoal System: Troubleshooting & Problem Analysis

Companion document to `subgoal_plan.md`. Use this when an experiment fails or a component doesn't behave as expected.

---

## Quick Reference: Experiment Failure → Root Cause

| Failed Experiment | Most Likely Problem | Go To |
|---|---|---|
| Exp 1.2 (oracle doesn't help) | Φ-guidance is broken end-to-end | [Check 0: Plumbing](#check-0-plumbing-verification) |
| Exp 1.3 (no good λ) | Latent space misalignment | [Problem 3](#problem-3-latent-space-misalignment) |
| Exp 1.4 (interval doesn't matter) | Guidance too weak or task doesn't need planning | [Problem 1 → small N regime](#the-core-tension) |
| Exp 2.x (all strategies equal) | Keyframe quality is not the bottleneck | [Problem 1 → check if Φ itself is broken](#gap-analysis) |
| Exp 3.1 (low phase consistency) | Encoder is texture-dominated | [Problem 3 → Solution a](#solution-a-projection-head) |
| Exp 3.2 (gradients wrong direction) | L2 ≠ reachability | [Problem 3 → Solution b](#solution-b-reachability-predictor) |
| Exp 3.3 (non-monotonic interpolation) | Topology gaps in latent space | [Problem 3 → Solution c](#solution-c-encoder-fine-tuning) |
| Exp 4.1 (gradient saturates/vanishes) | Φ is numerically ill-conditioned | [Gradient Fixes](#gradient-fixes) |
| Exp 4.2 (guidance conflicts with policy) | Φ is adversarial to base policy | [Problem 3 → fundamental misalignment](#the-fundamental-issue) |

---

## Check 0: Plumbing Verification

Before diagnosing representation issues, verify the data pipeline is correct.

**Checklist:**

```
[ ] subgoal_frames is non-empty when passed to vidar server
    → Log in ar.py around line 1308: print(f"subgoal_frames: {len(subgoal_frames)}")

[ ] Server actually receives and processes subgoal_frames
    → Check causal_worker.py logs for subgoal_frames parameter

[ ] Subgoal images are valid (not blank, not corrupted)
    → Save subgoal_frames[0] to disk, visually inspect

[ ] Guidance gradient is computed (not zero)
    → Log ||∇Φ|| in the denoising loop

[ ] Guidance gradient is applied to denoising
    → Verify ε̃ = ε_θ + λ·σ·∇Φ, not just ε̃ = ε_θ

[ ] subgoal_guidance_scale is actually read (not hardcoded to 0)
    → Check version_registry.py and deploy_policy.yml
```

If any of these fail, the problem is engineering (not representation). Fix the plumbing before running experiments.

---

## Problem 1: Keyframe Extraction Granularity

### Current Implementation

Five strategies exist in `experiments/gt_keyframe_test/extract_keyframes.py`:

```
Strategy            Signal Source       HDF5?   Detects
────────────        ──────────────      ─────   ────────────
uniform             fixed interval      No      nothing (baseline)
visual_change       pixel MSE           No      any visual difference > threshold
gripper_change      action[6,13] diff   Yes     grasp/release events
action_milestone    action velocity     Yes     motion start/stop transitions
semantic            motion + MSE        No      motion stops + visual changes
```

AR policy (`ar.py:1267-1284`) supports non-uniform keyframe indices via lookahead.

### The Core Tension

Keyframe interval N controls two things that conflict:

```
N too small (N=3):
  → Subgoals ≈ next few frames → no planning benefit
  → Planning head degenerates to "be a worse Model B"
  → Φ gradients are tiny → guidance has no effect

N too large (N=80):
  → Subgoals skip critical intermediate states
  → Prediction variance explodes (many plausible futures)
  → Φ points to unreachable state → misleading gradient
```

Example for pick-and-place:

```
Actual dynamics:
  Phase 1 (approach):  30 frames, smooth, low info density
  Phase 2 (grasp):      5 frames, contact, VERY high info density
  Phase 3 (transport): 40 frames, smooth
  Phase 4 (release):    3 frames, contact, high info density

uniform N=10:
  [f_0, f_10, f_20, f_30, f_40, f_50, f_60, f_70]
  → grasp gets 1 sample, release might be missed entirely

gripper_change:
  [f_0, f_32, f_33, f_75, f_76]
  → 5 keyframes, all at semantically meaningful moments
  → BUT: 30-frame gaps with no guidance
```

### Gap Analysis

**(1) No strategy comparison metrics**

All strategies exist but have never been compared. Run the 5-strategy comparison from Exp 2.x first — it requires zero code changes.

**(2) visual_change uses raw pixel MSE**

```python
# extract_keyframes.py:359
diff = np.mean((frame_gray - prev_frame) ** 2)   # raw pixel MSE
```

Detects lighting flicker and camera shake equally with gripper contact. For Φ-guidance, what matters is task-relevant state change, not pixel difference.

**(3) gripper_change too sparse**

Only fires on grasp/release → 2-5 keyframes per trajectory. 30-40 frame gaps with zero guidance.

**(4) No strategy composition**

Strategies are mutually exclusive. Ideal: combine gripper anchors + visual infill.

### Solutions

**Solution a) Strategy ablation** (do first, zero code change)

Run all 5 strategies through Exp 2.x. This tells you:
- Whether keyframe quality matters at all
- Which signal is most informative

**Solution b) Composite strategy**

```
1. Extract gripper_change → anchor keyframes at contact events
2. For gaps > 20 frames between anchors:
     Insert visual_change keyframes (encoder-based)
3. Result: semantic anchors + smooth-phase infill
```

**Solution c) Encoder-based visual change**

Replace pixel MSE with encoder feature distance:

```
# Instead of:  np.mean((frame_gray - prev_frame) ** 2)
# Use:         ||Enc(f_t) - Enc(f_{t-1})||^2
```

Produces keyframes natively aligned with Φ space.

**Solution d) Hierarchical multi-scale**

```
Level 0 (coarse): gripper_change, K=3-5    → phase anchors
Level 1 (fine):   encoder visual_change     → within-phase infill

Φ = Φ_coarse + Φ_fine
```

**Solution e) Random-N training** (planning head robustness)

```
During training: N ~ Uniform(5, 50) per video
During inference: choose N based on task complexity
```

**Priority:** a → b → c → d (stop when experiments pass)

---

## Problem 2: Planning Head Generalization

### Root Cause

The planning head sees `(z_0, task_embed) → {z_g1, ..., z_gK}` during training. This is ill-posed:

```
Input:   z_0 (single frame) + task_embed
Output:  K future subgoal latents

Missing:
  - Scene context beyond one frame
  - Physical properties (weight, friction)
  - Robot proprioception (joint config, gripper state)
```

A deterministic head (MLP + L2 loss) learns the **mean** of valid paths, which may itself be invalid:

```
"put object in box":
  Path A: approach left  → grasp → move left  → place
  Path B: approach right → grasp → move right → place
  Mean:   approach center → grasp → move center → collide with obstacle
```

### Failure Modes

| Generalization Type | Failure Mode | Cause |
|---|---|---|
| New initial state | Subgoals assume training-distribution poses | z_0 is OOD for planning head |
| New task instruction | Blends subgoals from similar seen tasks | CLIP embedding space is smooth |
| Longer task horizon | Runs out of subgoals before goal | Fixed K can't adapt |
| New object categories | Subgoals ignore novel affordances | Encoder lacks features for novel geometry |

### Mitigations

**Solution a) Conditional diffusion planning head**

```
Replace: PlanHead(z_0, task) → {z_g1, ..., z_gK}          (deterministic)
With:    PlanHead(z_0, task, noise) → {z_g1, ..., z_gK}   (diffusion)

Training: standard diffusion loss on subgoal latent sequences
Inference: sample multiple sequences, score each, pick best

+ Preserves multi-modality
- Slower inference, harder to train
```

**Solution b) Autoregressive re-planning** (strongest mitigation)

```
1. Generate only g_1 = PlanHead(z_t, task)
2. Execute Model B until reaching g_1 (or timeout)
3. Observe real o_new
4. Generate g_2 = PlanHead(Enc(o_new), task)
5. Repeat

+ Closed-loop at planning level
+ Each prediction is 1-step from real state (not K-step from imagined state)
- Model A runs K times instead of once
```

This converts open-loop planning into closed-loop, dramatically reducing generalization burden.

**Solution c) Context enrichment**

```
Replace: PlanHead(z_0, task)
With:    PlanHead(z_0, z_{-1}, z_{-2}, ..., task, proprio)

Add: observation history, proprioception, task progress indicator
Architecture: Transformer decoder with cross-attention to history
```

**Solution d) Data augmentation**

```
- Camera viewpoint perturbation
- Object position/color randomization
- Language paraphrasing
- Trajectory stitching (approach-from-A + place-from-B)
```

**Recommendation:** Start with (b) autoregressive re-planning. If tasks are multi-modal, add (a) diffusion head.

---

## Problem 3: Latent Space Misalignment

### The Fundamental Issue

Model B's encoder was trained for short-horizon action-conditioned prediction:

```
Encoder trained to optimize: ||predicted_next_frame - actual_next_frame||^2

What this produces:
  ✅ Fine-grained local differences well-represented (arm moved 2cm → distinct)
  ❌ Task-irrelevant details preserved (lighting → different latent)
  ❌ Long-range semantic similarity NOT guaranteed
     ("object grasped" from different angles may be far apart)

What Φ-guidance needs:
  ✅ Smooth gradient landscape (nearby states → nearby latents)
  ✅ Task-relevant features dominate distance
  ✅ Long-range semantic proximity (same phase → close)
```

### Concrete Failure Scenario

```
Subgoal: z_gk = "gripper holding cup above plate"
Current: z_t  = "gripper approaching cup from left"

∇Φ = 2(z_t - z_gk) points toward minimizing PIXEL DISTANCE to subgoal.

Result: "make the image look like cup-above-plate ASAP"
  → Teleport the cup (physically impossible)
  → Move the camera (wrong solution)

Desired: "move gripper toward cup, grasp, lift"
  → Semantically correct state sequence
```

### Three Manifestations

```
(1) Gradient direction is wrong
    Φ gradient → "looks similar" not "is reachable"
    → Misleading guidance, degrades below no-guidance baseline

(2) Gradient magnitude is miscalibrated
    Small task-relevant change (gripper open→closed) → tiny distance
    Large irrelevant change (lighting shift) → large distance
    → Φ overwhelmed by visual noise

(3) Topology gaps
    Latent space doesn't interpolate smoothly between states A and C
    → Gradient points into regions that don't correspond to real states
```

### Diagnostics (Exp 3.1-3.3)

```
Test 1 (Exp 3.1): Neighborhood consistency
  Find K-NN for each z_t → are they semantically similar?
  Pass: >70% same-phase neighbors

Test 2 (Exp 3.2): Gradient direction
  Step z' = z_t - ε·∇Φ → is z' closer to actual intermediate?
  Pass: >60% of steps improve

Test 3 (Exp 3.3): Interpolation smoothness
  z(α) = lerp(z_t, z_{t+k}) → NN frame index monotonic?
  Pass: >80% monotonic segments
```

### Solution a) Projection Head

```
Don't use Enc(o_t) directly. Add learned projection:

  z_task = Proj(Enc(o_t))    Proj = MLP (2-3 layers)

Train with contrastive objective:
  Positive: (o_t, o_{t+k}) same trajectory, k ∈ [10, 50]
  Negative: (o_t, o_random) different trajectories
  Loss:     InfoNCE / SimCLR

Architecture:
  Enc(o_t) → [frozen, d=512] → Proj → [trainable, d=128] → z_task

Φ = ||Proj(Enc(x_t)) - Proj(Enc(g_k))||^2

Gradient flows through Proj (trainable) but NOT through Enc (frozen)

+ Lightweight (hours to train)
+ Doesn't touch Model B
+ Directly reshapes "nearby" from visual similarity to reachability
```

### Solution b) Reachability Predictor

```
Instead of: Φ = ||z_t - z_gk||^2

Train R(z_t, z_g) → [0, 1]:
  R = 1: z_g is reachable from z_t within H steps
  R = 0: not reachable

Φ = -log R(z_t, z_gk)

Training data:
  Positive: (z_t, z_{t+H}) from actual trajectories
  Negative: (z_t, z_random) from different trajectories
  Hard neg: (z_t, z_{t+H'}) where H' >> H

+ Φ respects dynamics, not just visual similarity
- Extra model (but small — MLP on latent pairs)
```

### Solution c) Encoder Fine-Tuning

```
Unfreeze encoder with regularization:

  L_total = L_planning + α · ||Enc_finetuned(o) - Enc_frozen(o)||^2

Or use LoRA on encoder (limit drift).

⚠ Risk: may degrade Model B if decoder depends on exact encoder features
  Mitigation: fine-tune a COPY of encoder for Model A only
```

**Recommendation:** Try (a) first. Escalate to (b) if projected distances still don't correlate with reachability. Reserve (c) for when encoder fundamentally lacks needed features.

---

## Gradient Fixes

If Exp 4.1 shows gradient issues:

**Gradient vanishes (far from subgoal):**
```
Cause: ||z_t - z_g|| is very large but ∇Φ = 2(z_t - z_g) saturates in high-D space
Fix:   Normalize gradient: ∇Φ_norm = ∇Φ / ||∇Φ|| (unit direction, constant magnitude)
       Or: use log-distance: Φ = log(1 + ||z_t - z_g||^2)
```

**Gradient explodes (near subgoal):**
```
Cause: small distance amplified by large λ
Fix:   Clamp: ||∇Φ|| = min(||∇Φ||, max_grad)
       Or: smooth Φ: Φ = ||z_t - z_g||^2 / (||z_t - z_g||^2 + ε)
```

**Gradient is noisy (fluctuates wildly between steps):**
```
Cause: encoder features are noisy for intermediate diffusion timesteps
Fix:   EMA smoothing: ∇Φ_smooth = β·∇Φ_prev + (1-β)·∇Φ_current
       Or: only apply guidance at low-noise timesteps (t < T/2)
```

**Adaptive λ schedule:**
```
λ(d) = λ_max · clip(d / d_ref, 0, 1)

  d large (far from subgoal) → λ = λ_max (strong guidance needed)
  d small (near subgoal)     → λ → 0 (let base policy handle precision)
  d_ref = median distance in training data
```

---

## Data Collection Protocol

**Log these for every experiment (for reproducibility):**

```yaml
logs_per_episode:
  - task_name
  - episode_id
  - success (binary)
  - keyframe_indices (list)
  - subgoal_switch_times (list)
  - guidance_scale (float)

logs_per_step:
  - timestep
  - latent_distance_to_subgoal (float)
  - gradient_magnitude (float)
  - action_before_guidance (7D array)
  - action_after_guidance (7D array)
  - base_policy_loss (optional)
```

**Storage:** `experiments/subgoal/results/{exp_id}/{task}/{episode}.json`

**Visualization per experiment:**
1. Success rate bar chart (compare to baseline)
2. Latent distance over time (should decrease toward subgoal)
3. Gradient magnitude heatmap (time x subgoal_index)
