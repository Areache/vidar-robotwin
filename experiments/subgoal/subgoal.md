# Subgoal-Guided Action Generation via Trajectory Potential

## Core Idea

Instead of directly conditioning the action generation model on subgoals, we reshape the **energy landscape** of the generation distribution:

```
log p'(xt | ·) = log p(xt | ·) - λ · Φ(x_1:t, g)
```

where:
- `p(xt | ·)` is the base action generation policy (VIDAR, unchanged)
- `Φ` is the subgoal potential function
- `g = {g_1, g_2, ..., g_K}` are subgoal states
- `λ` controls guidance strength (λ=0: no guidance, λ→∞: hard constraint)

This is the Bayesian posterior decomposition from classifier guidance (Dhariwal & Nichol 2021), applied to trajectory space:

```
∇_xt log p(xt | g) = ∇_xt log p(xt)  +  ∇_xt log p(g | xt)
                      ~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~
                      base policy score   guidance from Φ
```

The denoising process of Model B is modified as:

```
ε̃_θ(xt) = ε_θ(xt) - λ · σ_t · ∇_xt Φ
```

### Potential Function Candidates

| Φ Definition | Formula | Pros | Cons |
|---|---|---|---|
| Latent distance | `Φ = min_k \|\|f(xt) - gk\|\|^2` | Simple, differentiable | Requires shared latent space |
| Reachability prediction | `Φ = error(x̂_{t+Δ} → gk)` | Dynamics-aware | Needs extra reachability model |
| Semantic embedding | `Φ = 1 - cos(CLIP(xt), CLIP(gk))` | Cross-domain transfer | CLIP gradients noisy on noisy inputs |

---

## Architecture Overview

The system consists of two models with distinct roles:

```
Model A: Subgoal Generator         Model B: Closed-Loop Action Policy (VIDAR)
  "Where should we go?"              "How do we get there?"
  ─────────────────────              ──────────────────────────────────────────
  Coarse-grained, long-horizon       Fine-grained, short-horizon
  Semantic planning                  Motor control
  Open-loop imagination              Closed-loop with real observations
  Tolerates spatial error            Must be physically precise
  Low-frequency (once per task)      High-frequency (every action chunk)
```

Connection:

```
Model A outputs g_k
       |
       v
  Φ(x_1:t, g_k)     defines potential field
       |
       v
  ∇_xt Φ             gradient signal
       |
       v
  Model B denoising is corrected:
       ε̃_θ(xt) = ε_θ(xt) - λ · σ_t · ∇_xt Φ
```

---

## Method 1: Frozen Encoder + Lightweight Planning Head

### Motivation

Model B (VIDAR) is trained as a short-horizon, high-feedback next-frame generator. It cannot be directly used as Model A because:

- Short horizon vs. long horizon needed for subgoals
- Requires action input vs. planning phase has no actions
- Relies on real observation feedback vs. subgoals require open-loop imagination

However, Model B's **visual encoder has learned a strong state representation** during training. We reuse this encoder and train only a lightweight planning head on top.

### Architecture

```
                    +---------------------------+
                    |  Model B Visual Encoder    |
                    |  (FROZEN)                  |
                    +---------------------------+
                               |
                               v
                     z_t = Enc(o_t)        current state latent
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
                      K subgoal latents
                               |
                               v
              Φ = ||Enc(x_t) - z_gk||^2    (fed to Model B's guidance)
```

### Training Procedure

**Step 1: Prepare data**

From demonstration videos (no action labels needed):
- Sample keyframes every N frames (N tunable, e.g. 10-50)
- Encode all keyframes with Model B's frozen encoder to get latent sequences

```
Video:    [f_0, f_1, ..., f_T]
Sample:   [f_0, f_N, f_2N, ..., f_KN]
Encode:   [z_0, z_N, z_2N, ..., z_KN]   using frozen Enc()
```

**Step 2: Train planning head**

```
Input:    z_0 (first frame latent) + task_embedding
Target:   [z_N, z_2N, ..., z_KN]  (K subgoal latents)
Loss:     L = Σ_k ||predicted_z_gk - actual_z_gk||^2
```

Options for the planning head:
- **MLP**: Simplest. Input `[z_0; task_embed]`, output `K * latent_dim`.
- **Transformer decoder**: Input `z_0` as context, K learnable queries, cross-attend to task embedding. Better for variable-length subgoal sequences.
- **Diffusion in latent space**: Model the distribution over subgoal sequences. Preserves multi-modality (multiple valid paths).

**Step 3: Inference**

```
1. Encode current observation:           z_t = Enc(o_t)
2. Generate subgoal latents:             {z_g1, ..., z_gK} = PlanHead(z_t, task)
3. Define potential for closest subgoal:  Φ = ||Enc(x_t) - z_gk*||^2
                                          where k* = argmin_k ||z_t - z_gk||
4. Guide Model B denoising:              ε̃ = ε_θ(xt) - λ · σ_t · ∇_xt Φ
5. Execute action chunk, get new obs, repeat from 1
```

### Key Design Decisions

**Keyframe sampling interval N:**
- Too small (N=2): subgoals are trivially close, no planning benefit
- Too large (N=100): subgoals miss critical intermediate states
- Recommended: start with N=20, tune based on average task length

**Number of subgoals K:**
- Fixed K: simpler, works when tasks have similar length
- Variable K: use Transformer decoder with early-stop token
- Recommended: K=5-8 for typical manipulation tasks

**Subgoal switching strategy:**
- Switch to next g_{k+1} when `||z_t - z_gk|| < threshold`
- Or: time-based, allocate equal budget per subgoal
- Or: soft weighting, `Φ = Σ_k w_k(t) · ||z_t - z_gk||^2` with time-decaying weights

### Pros and Cons

```
+ Reuses Model B's learned visual representation (no wasted training)
+ Planning head is lightweight, low training cost
+ Subgoals live in same latent space as Φ (native compatibility)
+ Long-horizon planning in a single forward pass
+ Only needs video data, no action annotations

- Requires collecting/accessing demonstration videos for planning head training
- Fixed encoder may not capture all task-relevant features
- Latent-space subgoals are not human-interpretable (harder to debug)
```

### Open Problems Analysis

#### Problem 1: Keyframe Extraction Granularity

##### What We Already Have

Five extraction strategies are implemented in `vidar-robotwin/experiments/gt_keyframe_test/extract_keyframes.py`:

```
Strategy            Signal Source       HDF5 Required   What It Detects
──────────────     ──────────────      ─────────────   ──────────────
uniform             fixed interval      No              nothing (baseline)
visual_change       pixel MSE           No              any visual difference > threshold
gripper_change      action[6,13] diff   Yes             grasp/release events
action_milestone    action velocity     Yes             motion start/stop transitions
semantic            motion + MSE        No              motion stops + visual changes
```

The AR policy (`policy/AR/ar.py:1267-1284`) already supports non-uniform keyframe indices via a lookahead strategy: given GT keyframe indices, it finds the next un-reached keyframe and uses it as the current subgoal target.

Current config (`config.yml`) uses `uniform` with `interval=8`, `max_keyframes=20`.

##### The Core Tension

Keyframe interval N simultaneously determines **subgoal semantic granularity** and **planning head learning difficulty**, and these two objectives conflict.

```
Task: pick up cup, move to plate, place down

Actual dynamics:

  Phase 1 (approach):  30 frames, visually smooth, low information density
  Phase 2 (grasp):      5 frames, contact event, extremely high information density
  Phase 3 (transport): 40 frames, smooth again
  Phase 4 (release):    3 frames, contact event, high density

uniform N=10:
  [f_0, f_10, f_20, f_30, f_40, f_50, f_60, f_70]
         ~~~approach~~~   grasp  ~~~transport~~~  release
                          ^                        ^
                     gets 1 sample              might be missed entirely

gripper_change (what we actually want):
  [f_0, f_32, f_33, f_75, f_76]
         ^     ^      ^     ^
      pre-grasp grasp pre-release release
  → 5 keyframes, all at semantically meaningful moments
```

The information density across a trajectory is **highly non-uniform**. Contact events, state transitions, and object interactions happen in a few frames but carry most of the task-relevant information. Uniform sampling either:
- Over-samples smooth phases (wasting subgoal budget on redundant waypoints)
- Under-samples critical transitions (missing the moments that actually matter)

**Deeper issue — granularity affects what the planning head must learn:**

```
N too small (e.g. N=3):
  - Subgoals are nearly identical to their neighbors
  - Planning head degenerates to "predict next few frames" = Model B's job
  - No actual planning benefit, just a worse version of Model B
  - Φ gradients are tiny (nearby latents), guidance has no effect

N too large (e.g. N=80):
  - Subgoals skip over critical intermediate states
  - Planning head must predict very distant futures from a single frame
  - Prediction variance explodes (many plausible futures at long horizons)
  - Φ points to a state that is unreachable without passing through
    un-specified intermediate states — the policy has no guidance
    for "how to get from here to that distant goal"
```

##### Gap Analysis: What's Missing in Current Implementation

**(1) No strategy comparison metrics**

All five strategies are implemented, but there is no evaluation framework to compare them. We don't know which strategy actually produces the best subgoals for downstream task success. The critical missing experiment:

```
For each strategy S in {uniform, visual_change, gripper_change, action_milestone, semantic}:
  For each task T:
    Extract GT keyframes with strategy S
    Inject into policy via inject_gt_subgoals()
    Run eval → measure task success rate

Compare: which strategy gives highest success rate?
         This directly answers "what granularity matters"
```

This is the **highest priority experiment** — it tells us whether the entire Φ-guidance approach is sensitive to keyframe quality, and which signal (gripper state vs. visual change vs. velocity) matters most.

**(2) visual_change uses raw pixel MSE, not encoder features**

The current `visual_change` strategy computes:
```python
# extract_keyframes.py:359
diff = np.mean((frame_gray - prev_frame) ** 2)   # raw pixel MSE
```

This detects **any** visual change (camera shake, lighting flicker, irrelevant background motion) with equal weight as task-relevant changes (gripper contact, object displacement). For the planning head and Φ-guidance, what matters is **task-relevant state change**, not pixel difference.

Proposed improvement: replace raw pixel MSE with encoder-based change detection:
```
d(t) = ||Enc(f_t) - Enc(f_{t-1})||    # Model B encoder features
```

The encoder has learned to suppress task-irrelevant visual variation during action prediction training. Its feature space naturally emphasizes task-relevant state changes. This maps directly to our proposed Φ space — keyframes detected in encoder space will be the ones where Φ gradients are most meaningful.

**(3) gripper_change produces too few keyframes for long tasks**

`gripper_change` only fires on grasp/release events. For a typical RoboTwin task, this yields 2-5 keyframes. Between two gripper events, the robot may travel a long distance (approach phase, transport phase), and the policy receives **no intermediate guidance** during these phases.

```
gripper_change output for a pick-and-place task:
  f_0 ─────────────── f_32 ── f_33 ─────────────── f_75 ── f_76
       30 frames gap          40 frames gap
       NO subgoal guidance    NO subgoal guidance
       policy is flying blind policy is flying blind
```

**(4) No strategy composition**

The strategies are mutually exclusive in the current config (`extraction_strategy: uniform`). But the ideal set of keyframes combines signals:

```
gripper_change:     captures WHAT (grasp, release)
action_milestone:   captures WHEN (motion phase transitions)
visual_change:      captures WHERE (significant state changes)

Best keyframe set = union of all three, deduplicated by min_interval
```

##### Recommended Solutions (Updated)

**(a) Strategy ablation experiment (do this first):**

Run the 5-strategy comparison described above. This requires zero code changes — just modify `config.yml` and run `run_with_gt_subgoals.py` for each strategy. The results will tell us:
- Whether non-uniform strategies actually outperform uniform (validates the theory)
- Which signal matters most for manipulation tasks
- Whether `max_keyframes=20` is too many or too few

**(b) Composite strategy (gripper_change + visual_change infill):**

```
Algorithm:
  1. Extract gripper_change keyframes → anchor keyframes at contact events
  2. For each gap between consecutive anchors:
     if gap > max_gap_frames (e.g. 20):
       Insert visual_change keyframes within the gap
       (using encoder-based d(t) instead of pixel MSE)
  3. Result: semantic anchors + infill for long smooth phases
```

This solves the "gripper_change is too sparse" problem while keeping contact events as hard anchors.

**(c) Encoder-based visual change (upgrade existing visual_change):**

Replace the pixel MSE in `extract_keyframes_visual_change` with:

```
# Instead of:
diff = np.mean((frame_gray - prev_frame) ** 2)

# Use:
z_curr = frozen_encoder(frame_rgb)
z_prev = frozen_encoder(prev_frame_rgb)
diff = ||z_curr - z_prev||^2
```

This produces keyframes that are **natively compatible** with Φ — the same encoder space is used for both keyframe detection and guidance gradient computation.

**(d) Hierarchical multi-scale subgoals (if composite strategy works):**

```
Level 0 (coarse):  gripper_change, K=3-5   → task-phase-level anchors
Level 1 (fine):    encoder-based visual_change, K=5-10  → within-phase infill

Φ = Φ_coarse + Φ_fine

Coarse Φ provides long-range direction (which phase are we in?)
Fine Φ provides local precision (are we approaching the object correctly?)
```

**(e) Random-N training (for planning head robustness):**

Instead of hand-designing the sampling rule, train planning head with **random N per sample**:

```
During training:
  For each video, sample N ~ Uniform(N_min, N_max)  e.g. (5, 50)
  Extract keyframes at this N
  Planning head learns to output subgoals at varying granularity

During inference:
  Choose N based on task complexity or remaining horizon
```

This forces the planning head to be robust to granularity, but may hurt prediction accuracy at any single scale. Use this as a fallback if the composite strategy doesn't generalize.

##### Priority Order

```
Step 1: Run 5-strategy ablation (zero code change, just config + eval)
        → Establishes: does keyframe quality matter? which signal wins?

Step 2: Implement composite strategy (b)
        → Combines gripper anchors + visual infill

Step 3: Upgrade visual_change to encoder-based (c)
        → Aligns keyframe detection with Φ space

Step 4: If tasks vary in length → hierarchical (d) or random-N (e)
```

---

#### Problem 2: Planning Head Generalization

The planning head sees `(z_0, task_embed) → {z_g1, ..., z_gK}` during training. The question is whether it generalizes to:
- Unseen initial states (new object poses, new scenes)
- Unseen tasks (new language instructions)
- Unseen combinations of seen elements

**Root cause of poor generalization — the problem is ill-posed:**

```
Input:   z_0 (single frame latent) + task_embed
Output:  K future subgoal latents

What's missing:
  - Scene context beyond a single frame (occluded objects, spatial layout)
  - Task-relevant physical properties (object weight, friction)
  - Robot proprioceptive state (joint configuration, gripper state)
```

From a single image + language, there are potentially many valid subgoal sequences. A deterministic planning head (MLP/Transformer with L2 loss) will learn the **mean** of these valid sequences, which may itself be invalid:

```
Example: "put object in box"

Valid path A: approach from left  → grasp → move left → place
Valid path B: approach from right → grasp → move right → place

Mean of A and B: approach from center → grasp → move... center?
  → This averaged path may collide with obstacles
  → L2 loss cannot distinguish "valid average" from "average of valid"
```

**Failure modes by generalization type:**

```
Generalization type        Failure mode                          Why it happens
─────────────────         ───────────────                       ──────────────
New initial state          Subgoals assume training-distribution  z_0 is out-of-distribution for
                           object positions                       the planning head
New task instruction       Task embed maps to nearest seen task   CLIP embedding space is smooth,
                           → subgoals are a blend of seen tasks   similar instructions map nearby
Longer task horizon        Runs out of subgoals, last g_K         Fixed K cannot represent tasks
                           is still far from goal                  with more phases than training data
New object categories      Subgoals don't respect object          Encoder features may not capture
                           affordances                             novel object geometry
```

**Mitigation strategies:**

**(a) Conditional diffusion planning head (instead of deterministic):**

```
Replace:  PlanHead(z_0, task) → {z_g1, ..., z_gK}          (deterministic)
With:     PlanHead(z_0, task, noise) → {z_g1, ..., z_gK}   (diffusion/flow)

Training: standard diffusion loss on subgoal latent sequences
Inference: sample multiple subgoal sequences, score each, pick best

Benefit: preserves multi-modality, avoids "mean of valid = invalid" problem
Cost: slower inference (need denoising steps), harder to train
```

**(b) Autoregressive generation with re-planning:**

```
Instead of generating all K subgoals at once from z_0:

1. Generate only g_1 = PlanHead(z_t, task)        → next subgoal
2. Execute Model B until reaching g_1 (or timeout)
3. Get real observation o_new
4. Generate g_2 = PlanHead(Enc(o_new), task)       → re-plan from reality
5. Repeat

Benefit: each subgoal is conditioned on actual state, not imagined state
         inherently closed-loop at the planning level
Cost: Model A runs K times instead of once
```

This is the **single strongest mitigation** — it converts the open-loop planning problem into a closed-loop one, dramatically reducing the generalization burden on the planning head. Each step only needs to predict **one subgoal from the current real state**, not an entire sequence from a possibly out-of-distribution starting point.

**(c) Context enrichment:**

```
Replace:  PlanHead(z_0, task)
With:     PlanHead(z_0, z_{-1}, z_{-2}, ..., task, proprio)

Add:
  - History of recent observations (trajectory context)
  - Robot proprioceptive state (joint angles, gripper width)
  - Task progress indicator (how many subgoals already achieved)

Architecture: Transformer decoder with cross-attention to observation history
```

**(d) Training data augmentation:**

```
- Random camera viewpoint perturbation
- Object position/color randomization (if sim data available)
- Language paraphrasing for task descriptions
- Trajectory stitching: combine approach-from-A with place-from-B
```

**Our recommendation:** Use autoregressive re-planning (b) as the primary defense — it reduces the generalization problem from "predict K distant futures from one frame" to "predict one next subgoal from current reality." Combine with a conditional diffusion head (a) if multi-modality matters for your tasks.

---

#### Problem 3: Latent Space Misalignment

This is the most subtle and potentially the most damaging problem. The frozen encoder was trained as part of Model B (VIDAR), optimized for **short-horizon action-conditioned prediction**. We are now asking it to serve as the representation space for **long-horizon goal specification**. These two objectives shape the latent space differently.

**The fundamental issue:**

```
Model B encoder was trained to optimize:
  L_B = ||predicted_next_frame - actual_next_frame||^2

What this encourages in latent space:
  - Fine-grained local differences are well-represented
    (robot arm moved 2cm → distinct latents, because prediction needs this)
  - Task-irrelevant visual details are preserved
    (lighting change → different latent, because pixel loss penalizes this)
  - Long-range semantic similarity is NOT guaranteed
    (two frames showing "object grasped" from different angles
     may be far apart in latent space)

What Φ-guidance needs from latent space:
  - Smooth gradient landscape between current state and subgoal
    (for ∇_xt Φ to be meaningful, nearby states should have nearby latents)
  - Task-relevant features dominate distance
    (gripper-object contact matters more than shadow position)
  - Long-range semantic proximity
    ("object grasped" frames should cluster regardless of exact pose)
```

**Concrete failure scenario:**

```
Subgoal: z_gk encodes "gripper holding cup above plate"

Current state: gripper approaching cup from the left
  → Enc(o_t) is far from z_gk in every direction

The gradient ∇_xt Φ = 2 * (Enc(x_t) - z_gk) points in a direction
that minimizes PIXEL-LEVEL distance to the subgoal image.

This might mean:
  "Make the image look like cup-above-plate as quickly as possible"
  → Teleport the cup (physically impossible)
  → Move the camera (wrong solution)

Instead of:
  "Move gripper toward cup, grasp, lift"
  → Semantically correct sequence of states
```

The gradient in a pixel-reconstruction-trained latent space does not respect physical dynamics. It points toward **visual similarity**, not **dynamical reachability**.

**Three manifestations of misalignment:**

```
(1) Gradient direction is wrong
    Φ gradient points toward "looks similar" not "is reachable"
    → Model B receives misleading guidance, degrades below no-guidance baseline

(2) Gradient magnitude is miscalibrated
    Small task-relevant changes (gripper open→closed) → tiny latent distance
    Large task-irrelevant changes (lighting shift)    → large latent distance
    → Φ guidance overwhelmed by irrelevant visual noise

(3) Latent space has topology gaps
    Training data covers states A→B→C, but latent space may not
    interpolate smoothly between A and C.
    → Subgoal at C, current at A, gradient points to a latent region
       that doesn't correspond to any real physical state
```

**Diagnostic — how to detect this problem:**

```
Test 1: Latent neighborhood consistency
  For a state z_t, find its K-nearest neighbors in latent space
  Check: are they semantically similar states? Or visually similar but semantically different?
  Metric: retrieval precision (fraction of neighbors that share task phase)

Test 2: Gradient direction validation
  Given (z_t, z_gk), compute gradient direction ∇_zt Φ
  Take a small step: z' = z_t - ε * ∇Φ
  Decode z' back to image (if possible)
  Check: does the image look like a physically plausible intermediate state?

Test 3: Interpolation smoothness
  Linearly interpolate between z_t and z_gk: z(α) = (1-α)*z_t + α*z_gk
  Decode each z(α)
  Check: do intermediate images form a plausible trajectory?
  If they show visual artifacts or impossible states → topology gap
```

**Mitigation strategies:**

**(a) Projection head (lightweight, recommended first attempt):**

```
Don't use Enc(o_t) directly. Add a learned projection:

  z_task = Proj(Enc(o_t))    where Proj is a small MLP (2-3 layers)

Train Proj with a contrastive objective:

  Positive pairs: (o_t, o_{t+k}) from same trajectory, k in [10, 50]
                  → should be close (same task phase, reachable)
  Negative pairs: (o_t, o_random) from different trajectories
                  → should be far

Loss: InfoNCE / SimCLR style

This reshapes the frozen encoder's latent space into one where
"nearby" means "dynamically reachable" rather than "visually similar"
```

```
Architecture:
  Enc(o_t)  →  [frozen, d=512]  →  Proj  →  [trainable, d=128]  →  z_task

Φ is now defined on the projected space:
  Φ = ||Proj(Enc(x_t)) - Proj(Enc(g_k))||^2

The gradient flows through Proj (trainable) but NOT through Enc (frozen)
  → Proj learns to reshape distances without touching Model B
```

**(b) Action-conditioned reachability metric (replace L2 distance):**

```
Instead of:  Φ = ||z_t - z_gk||^2   (assumes L2 = reachability, often wrong)

Train a reachability predictor R(z_t, z_g) → [0, 1]:
  R = 1 means "z_g is reachable from z_t within H steps"
  R = 0 means "z_g is not reachable"

Φ = -log R(z_t, z_gk)

Training data: from Model B rollouts
  Positive: (z_t, z_{t+H}) pairs from actual trajectories, label=1
  Negative: (z_t, z_random) pairs, label=0
  Hard negatives: (z_t, z_{t+H'}) where H' >> H, label=0

Benefit: Φ now respects physical dynamics, not just visual similarity
Cost: extra model to train, but small (MLP on latent pairs)
```

**(c) Encoder fine-tuning with regularization (heavier, use if (a) fails):**

```
Unfreeze the encoder but constrain it:

  L_total = L_planning + α * L_distillation

  L_planning:     normal planning head loss
  L_distillation: ||Enc_finetuned(o) - Enc_frozen(o)||^2
                  keeps the encoder close to its original representation

  Or use LoRA:
    Only train low-rank adaptors on Enc, keeping most weights frozen
    This limits the representation drift while allowing task-relevant adjustment

Risk: fine-tuning may degrade Model B's action generation
      (if Model B's decoder depends on specific encoder features)
Mitigation: only fine-tune a copy of the encoder used by Model A
            Model B keeps its original frozen encoder
```

**Our recommendation:** Start with projection head (a). It is lightweight (a few hours of training), doesn't touch Model B at all, and directly addresses the core issue (L2 distance in encoder space ≠ reachability). If the diagnostic tests show that even projected distances don't correlate with reachability, escalate to (b). Reserve (c) for when (a) and (b) both fail — it means the encoder fundamentally lacks the features needed for planning, and needs structural change.

---

## Method 2: MPC-Style Short-Horizon Rollout (Zero-Training Baseline)

### Motivation

Before investing in training a new planning head, we can validate the trajectory potential framework using Model B's existing short-horizon prediction capability. This requires **zero additional training**.

### Algorithm

```
Algorithm: MPC Subgoal Generation via Model B Rollout

Input:  current observation o_t
        task goal descriptor (e.g., CLIP embedding of target)
        Model B (frozen, used as forward model)
        M: number of action samples (e.g., 64)
        H: rollout horizon in steps (e.g., 8)

Output: next subgoal g_next

1.  z_task = encode_task(task_description)    # e.g., CLIP text embedding

2.  For i = 1 to M:
3.      Sample action sequence:  a_i = {a_1, ..., a_H} ~ prior(a)
4.      Rollout Model B:         ô_{t+H}^(i) = ModelB.predict(o_t, a_i)
5.      Score:  s_i = similarity(Enc(ô_{t+H}^(i)), z_task)

6.  Select best: i* = argmax_i s_i
7.  g_next = ô_{t+H}^(i*)

8.  Define Φ = ||Enc(x_t) - Enc(g_next)||^2
9.  Use Φ to guide Model B's closed-loop action generation
10. After executing action chunk, get new o_t, go to step 1
```

### Variants

**Variant A: Random Shooting (simplest)**

```
Action prior: a ~ Uniform(a_min, a_max)
No iterative refinement
```

**Variant B: CEM (Cross-Entropy Method)**

```
1. Initialize: μ = 0, σ = σ_init
2. For iter = 1 to N_iters:
   a. Sample M sequences from N(μ, σ)
   b. Rollout and score each
   c. Select top-K (elite) sequences
   d. Update: μ = mean(elites), σ = std(elites)
3. g_next = rollout with final μ
```

**Variant C: Multi-Step Chained Subgoals**

```
For k = 1 to K:
    g_k = MPC_rollout(o = g_{k-1}, horizon=H)     # chain rollouts
    (use g_0 = o_t)

Result: {g_1, ..., g_K} covers K*H steps into the future
```

This extends the planning horizon from H to K*H, but compounds prediction error.

### Scoring Functions

| Scorer | Formula | When to Use |
|---|---|---|
| CLIP similarity | `cos(CLIP_img(ô), CLIP_txt(task))` | Language-specified tasks |
| Goal image distance | `-\|\|Enc(ô) - Enc(o_goal)\|\|^2` | Goal-image-specified tasks |
| Task progress heuristic | Domain-specific (e.g., gripper-object distance) | When domain knowledge available |

### Pros and Cons

```
+ Zero additional training — validate Φ-guidance immediately
+ Uses Model B as-is, no architecture changes
+ Physically grounded — rollouts respect short-term dynamics
+ Easy to implement and debug (subgoals are actual predicted images)

- Planning horizon limited to H steps (short-sighted)
- M forward passes per planning step (compute-heavy at inference)
- Action prior quality matters — random sampling may miss good trajectories
- Compounding error if chaining rollouts (Variant C)
- Not a scalable long-term solution
```

---

## Recommended Workflow

**See [subgoal_plan.md](subgoal_plan.md) for the full experiment plan and roadmap.**

```
Phase 0: Validate Φ-guidance + VLM pipeline
  ├── Oracle baseline: do GT keyframes help?
  ├── VLM feasibility: can VLM detect task phases?
  └── Method B: VLM + GT keyframe retrieval

Phase 1: Full evaluation with Method B
  ├── λ and keyframe density ablation
  ├── Keyframe strategy comparison
  └── Identify Method B failure cases

Phase 2: Method C upgrade (if Method B insufficient)
  ├── WanTI2V per-subtask visual planning
  ├── Haar temporal decomposition + divergence detection
  └── Hybrid: Method B for in-distribution, Method C for OOD

Phase 3: Iterate & scale
  ├── Φ variants, adaptive λ, guidance upgrades
  └── End-to-end training (if applicable)
```

---

## Evolution: From Single-Model Planning to Three-Level Hierarchy

### Why Methods 1 & 2 Are Insufficient for Long Horizon

Methods 1 (planning head) and 2 (MPC rollout) were designed assuming short-to-medium horizon tasks. For the actual evaluation target (160 frames, 16s), both fail:

```
Method 1: single-frame input → must hallucinate 16s of future
           → prediction variance explodes beyond ~20 frames

Method 2: H-step rollout horizon → effective range 8-16 frames
           → chaining compounds error

Neither has a world dynamics model for long-range prediction.
```

### Method 3: WanTI2V as Planner (Intermediate Step)

We explored using WanTI2V (non-causal, 5B parameters, full bidirectional attention) as a standalone long-horizon planner. Key findings:

```
✓ WanTI2V provides world dynamics knowledge (pretrained on 100M+ videos)
✓ Full bidirectional attention guarantees intra-plan consistency
✓ Already exists in codebase (run_keyframe_model.py)
✓ Haar wavelet decomposition on Wan-VAE latents enables frequency separation

✗ WanTI2V covers only 2-5s per round → needs chaining for 16s
✗ Chained rounds have zero cross-round attention → coherence breaks
✗ Cross-round memory requires an external Level 3 module (VLM or state machine)
✗ WanTI2V plan quality on RoboTwin is unvalidated (pretrained on internet video)
✗ Total engineering complexity: 8+ components
```

**Critical realization:** Even with WanTI2V, a Level 3 module is unavoidable. If Level 3 is required anyway, it should be the primary planner.

### Three-Level Architecture (Current Design)

```
Level 3: VLM (task decomposition, phase tracking, long-term memory)
         → Unlimited horizon via context window
         → Outputs: subtask descriptions + completion criteria

Level 2: Subgoal Provider (visual targets for Vidarc)
         → Method B: GT demo retrieval (highest quality, no generation risk)
         → Method C: WanTI2V per-subtask plan (generalization, needs validation)

Level 1: Vidarc (frame-by-frame action generation with Φ-guidance)
         → Unchanged weights, guidance through ε̃ = ε_θ - λ·σ_t·∇Φ
```

**See [theory.md](theory.md) for design principles and detailed motivation.**

---

## Experimental Design: Representation Validation (表征验证)

### Design Principles

**Goal:** Validate that each component of the subgoal system produces meaningful representations, NOT to maximize task success rate.

**Methodology:** Controlled variable comparison (控制变量对比) — change exactly ONE variable per experiment, keep all others fixed.

**Success criteria:** Each experiment should answer a specific yes/no question about representation quality.

---

### Experiment Series 1: Oracle Guidance Baseline (验证指导信号的上界)

**Research Question:** Does providing perfect visual guidance improve policy behavior?

This establishes the upper bound — if GT keyframes don't help, the entire Φ-guidance approach is flawed.

#### Exp 1.1: No Guidance Baseline
**Config:**
```yaml
use_libero_subgoal: false
subgoal_guidance_scale: 0.0
```

**What to measure:**
- Task success rate (baseline reference)
- Trajectory smoothness (action velocity variance)
- Number of recovery behaviors (detect via action reversals)

**Expected outcome:** Establishes baseline performance without any subgoal guidance.

---

#### Exp 1.2: Oracle Visual Guidance (Uniform Sampling)
**Config:**
```yaml
use_libero_subgoal: true
extraction_strategy: uniform
keyframe_interval: 8
subgoal_guidance_scale: 0.5
max_keyframes: 20
```

**Control variables:**
- Use SAME task, SAME episodes as Exp 1.1
- Only difference: inject GT keyframes from successful demonstrations

**What to measure:**
- Success rate delta vs. Exp 1.1
- Trajectory deviation from GT trajectory (Frechet distance)
- Per-step distance to GT subgoal (does it actually approach the target?)

**Validation criteria:**
- ✅ **Pass:** Success rate increases by >10% → visual guidance helps
- ❌ **Fail:** Success rate unchanged or decreases → Problem:
  - Either `subgoal_guidance_scale` is wrong
  - Or latent space is misaligned (Problem 3)
  - Or guidance gradients are too noisy

**Interpretation:**
- If this fails, STOP — don't build planning head yet, fix Φ-guidance first
- If this passes, continue to test guidance parameters

---

#### Exp 1.3: Guidance Strength Ablation
**Config:** Vary ONLY `subgoal_guidance_scale`
```yaml
subgoal_guidance_scale: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
# Keep all other settings from Exp 1.2
```

**What to measure:**
- Success rate vs. λ curve
- Visual subgoal tracking accuracy (L2 distance in latent space over time)
- Action magnitude (does strong guidance cause jerky movements?)

**Validation criteria:**
- ✅ **Pass:** Exists an optimal λ* where success > baseline
- ❌ **Fail:** Monotonic decrease with λ → gradient direction is wrong

**Expected λ* range:** 0.3-0.7 (if outside this, indicates representation issue)

---

#### Exp 1.4: Temporal Density Ablation
**Config:** Vary ONLY `keyframe_interval`
```yaml
keyframe_interval: [4, 8, 16, 32, 64]
subgoal_guidance_scale: 0.5  # fixed at value from Exp 1.3
extraction_strategy: uniform
```

**What to measure:**
- Success rate vs. interval
- Average frames to reach each subgoal
- Guidance gradient magnitude (does it vanish when interval is too large?)

**Validation criteria:**
- ✅ **Pass:** Inverted-U curve (too dense → no planning benefit, too sparse → missing critical states)
- ❌ **Fail:** Flat line → temporal granularity doesn't matter (unexpected, suggests guidance is ineffective)

**Interpretation:**
- Optimal interval reveals task timescale
- If interval=4 wins: smooth tasks, fine-grained control needed
- If interval=32 wins: discrete phase transitions, coarse planning sufficient

---

### Experiment Series 2: Keyframe Extraction Strategies (验证不同信号源的语义质量)

**Research Question:** Which signal (pixel, gripper, action, encoder) extracts the most task-relevant keyframes?

**Fixed variables across all Exp 2.x:**
```yaml
subgoal_guidance_scale: 0.5  # from Exp 1.3
max_keyframes: 20
```

#### Exp 2.1: Uniform (Baseline)
```yaml
extraction_strategy: uniform
keyframe_interval: 8  # from Exp 1.4
```
*(Already measured in Exp 1.2)*

---

#### Exp 2.2: Visual Change (Pixel-Based)
```yaml
extraction_strategy: visual_change
visual_change_threshold: 0.05
min_interval: 4
```

**What to measure:**
- Number of keyframes extracted (is it too many/few?)
- Keyframe distribution over trajectory phases (are they clustered at contact events?)
- Success rate vs. Exp 2.1

**Validation:**
- If success > uniform → pixel change correlates with task semantics
- If success < uniform → pixel MSE is too noisy (lighting, irrelevant motion)

---

#### Exp 2.3: Gripper State Change
```yaml
extraction_strategy: gripper_change
# Requires HDF5 data with action[6:13]
```

**What to measure:**
- Number of keyframes (expected: 2-5 for pick-and-place)
- Gap sizes between keyframes (are there >30 frame gaps?)
- Success rate vs. Exp 2.1

**Expected outcome:**
- Higher precision (keyframes at semantically critical moments)
- Lower recall (misses intermediate waypoints in long smooth phases)

---

#### Exp 2.4: Action Velocity Milestones
```yaml
extraction_strategy: action_milestone
# Detects motion start/stop via action velocity
```

**What to measure:**
- Keyframe alignment with task phases (approach/grasp/transport/place)
- Success rate vs. Exp 2.1

---

#### Exp 2.5: Composite Strategy
```yaml
extraction_strategy: composite
# Union of gripper_change + visual_change (encoder-based)
max_gap_frames: 20  # infill threshold
```

**Validation criteria:**
- ✅ **Pass:** Success ≥ max(Exp 2.2, Exp 2.3) → combining signals helps
- ❌ **Fail:** Success < both → too many keyframes dilutes guidance

**Key insight to extract:**
- Compare keyframe sets from all strategies
- Measure overlap (Jaccard similarity)
- Check: do different strategies agree on which frames are important?
- If agreement is high → signal is robust
- If agreement is low → task semantics are not well-defined in any modality

---

### Experiment Series 3: Latent Space Quality (验证潜在空间的适配性)

**Research Question:** Is Model B's encoder latent space suitable for Φ-guidance?

**Method:** Diagnostic tests, NOT task performance.

#### Exp 3.1: Latent Neighborhood Consistency
**Setup:**
- Extract GT trajectory from successful demonstration
- Encode each frame: `z_t = Enc(o_t)`
- For each `z_t`, find K=10 nearest neighbors in latent space (from all trajectories in dataset)

**What to measure:**
- **Phase consistency:** % of neighbors from same task phase (approach/grasp/transport/place)
- **Trajectory consistency:** % of neighbors from trajectories of same task
- **Temporal locality:** Average |t - t'| for neighbors (are neighbors temporally close?)

**Validation criteria:**
- ✅ **Pass:** >70% phase consistency → encoder space groups semantically similar states
- ❌ **Fail:** <50% → encoder is texture-dominated, not task-aware → need projection head (Problem 3, Solution a)

---

#### Exp 3.2: Gradient Direction Validation
**Setup:**
- Take (current state `z_t`, subgoal `z_g`) pairs from GT trajectories
- Compute gradient direction: `∇Φ = 2(z_t - z_g)`
- Take small step: `z' = z_t - ε * ∇Φ` for ε = 0.1
- Measure: does `z'` move closer to actual intermediate frames?

**Metric:**
- Frechet distance: `d(z', z_{actual_midpoint})` vs. `d(z_t, z_{actual_midpoint})`
- Expected: `d(z', ·) < d(z_t, ·)` → gradient points toward plausible path

**Validation criteria:**
- ✅ **Pass:** >60% of gradients reduce distance to actual path
- ❌ **Fail:** <40% → gradients point toward visual similarity, not reachability → need reachability predictor (Problem 3, Solution b)

---

#### Exp 3.3: Interpolation Smoothness
**Setup:**
- For trajectory segment [z_t, z_{t+1}, ..., z_{t+k}], linearly interpolate:
  `z(α) = (1-α)*z_t + α*z_{t+k}` for α ∈ [0, 1]
- Decode interpolated latents (if decoder available) OR
- Measure: nearest-neighbor frame index for each z(α)

**What to measure:**
- **Monotonicity:** Does nearest-frame index increase monotonically with α?
- **Coverage:** Do interpolated points hit all intermediate frames?

**Validation criteria:**
- ✅ **Pass:** Monotonic in >80% of segments → latent space has smooth path structure
- ❌ **Fail:** Non-monotonic or jumps → topology gaps, need to fix encoder

---

### Experiment Series 4: Guidance Mechanism (验证梯度传播的正确性)

**Research Question:** Is the guidance gradient ∇_xt Φ numerically stable and semantically meaningful?

#### Exp 4.1: Gradient Magnitude Monitoring
**Setup:**
- During Exp 1.2, log:
  - `||∇_xt Φ||` at each diffusion timestep
  - Distance to subgoal: `||z_t - z_g||`
  - Action magnitude after guidance

**What to measure:**
- Gradient-distance correlation (does gradient vanish when far from subgoal? explode when close?)
- Guidance-induced action change (does guidance alter actions significantly?)

**Validation criteria:**
- ✅ **Pass:** Gradient magnitude ∝ sqrt(distance) → well-calibrated potential
- ❌ **Fail:** Gradient saturates or vanishes → need adaptive λ scheduling

---

#### Exp 4.2: Guidance vs. Base Policy Conflict
**Setup:**
- Measure angle between:
  - Base policy action: `a_base = IDM(x_t)`
  - Guided action: `a_guided = IDM(x_t - λ∇Φ)`

**What to measure:**
- Cosine similarity: `cos(a_base, a_guided)`
- Success rate when guidance agrees (cos > 0.5) vs. conflicts (cos < 0)

**Validation criteria:**
- ✅ **Pass:** Success higher when guidance agrees → guidance provides useful correction
- ❌ **Fail:** Success lower when guidance agrees → guidance is fighting the base policy, making it worse

---

### Experiment Series 5: Subgoal Switching Strategy (验证切换时机的影响)

**Research Question:** When should the policy switch to the next subgoal?

**Fixed:** Use best strategy from Exp 2.x, best λ from Exp 1.3

#### Exp 5.1: Distance-Based Switching
```python
if ||z_t - z_{g_k}|| < threshold:
    switch to g_{k+1}
```
**Vary:** `threshold ∈ [0.1, 0.5, 1.0, 2.0]` (in latent space units)

**What to measure:**
- Number of subgoals actually reached per episode
- Time spent per subgoal (is it balanced?)

---

#### Exp 5.2: Time-Based Switching
```python
if current_frame >= k * frames_per_subgoal:
    switch to g_{k+1}
```

**What to measure:**
- Does policy always reach subgoal before forced switch?
- Success rate vs. Exp 5.1

---

#### Exp 5.3: Soft Weighting (No Hard Switch)
```python
Φ = Σ_k w_k(t) * ||z_t - z_{g_k}||^2
w_k(t) = exp(-β * ||z_t - z_{g_k}||)  # closer subgoals have higher weight
```

**What to measure:**
- Success rate vs. hard switching
- Gradient stability (does soft weighting reduce noise?)

**Validation criteria:**
- If soft weighting wins → task benefits from multi-subgoal context
- If hard switching wins → task has clear phase boundaries

---

### Experiment Priority & Dependency Graph

```
START
  │
  ├─► Exp 1.1 (Baseline) ──────────────────────────┐
  │                                                  │
  └─► Exp 1.2 (Oracle) ────► GATE: Does it help? ───┤
                              │                     │
                              NO → STOP, fix Φ      │
                              │                     │
                              YES                   │
                              ▼                     │
                        Exp 1.3 (λ ablation)        │
                              │                     │
                              ▼                     │
                        Exp 1.4 (interval)          │
                              │                     │
                              ▼                     │
                        Exp 2.x (strategies) ───────┤
                              │                     │
         ┌────────────────────┴─────────────┐       │
         ▼                                  ▼       │
   Exp 3.x (latent diagnosis)      Exp 4.x (gradient) │
         │                                  │       │
         └──────────► Exp 5.x (switching) ◄─┘       │
                              │                     │
                              ▼                     │
                         ALL COMPLETE ◄─────────────┘
                              │
                              ▼
                    Build Planning Head (Method 1)
```

---

### Negative Result Interpretation Guide

If an experiment fails, it tells you WHERE the problem is:

| Failed Exp | Problem Location | Next Action |
|------------|------------------|-------------|
| Exp 1.2 | Φ-guidance doesn't work at all | Check: (1) Is subgoal_frames actually passed to server? (2) Is guidance gradient computed? (3) Run Exp 3.x to diagnose latent space |
| Exp 1.3 | No good λ exists | Latent space misalignment (Exp 3.1/3.2) or guidance conflicts with base policy (Exp 4.2) |
| Exp 1.4 | Interval doesn't matter | Either: guidance is too weak (increase λ), or task doesn't need planning (rare) |
| Exp 2.x | All strategies fail equally | Keyframe quality is not the bottleneck, issue is in Φ or guidance mechanism |
| Exp 3.1 | Low phase consistency | Encoder is not task-aware → add projection head (Problem 3, Solution a) |
| Exp 3.2 | Gradients point wrong way | L2 ≠ reachability → switch to reachability predictor (Problem 3, Solution b) |
| Exp 4.2 | Guidance conflicts hurt performance | Base policy is already optimal, or λ is too large, or Φ is adversarial |

---

### Data Collection Protocol

**For reproducibility, log these for every experiment:**

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
  - base_policy_loss (optional, if available)
```

**Storage:** Save to `experiments/subgoal/results/{exp_id}/{task}/{episode}.json`

**Visualization:** For each experiment, generate:
1. Success rate bar chart (compare to baseline)
2. Latent distance trajectory (does it decrease toward subgoal?)
3. Gradient magnitude heatmap (over time × subgoal_index)

---

## Related Documents

- **[theory.md](theory.md)** — Design principles, motivation, three-level architecture rationale
- **[subgoal_plan.md](subgoal_plan.md)** — Experiment plan, Methods B/C implementation, roadmap

## References

- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021 — classifier guidance
- Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis", ICML 2022 — diffusion as planner
- Luo & Du, "Grounding Video Models to Actions through Goal Conditioned Exploration", ICLR 2025 — video-to-action with subgoal pursuit
- Chi et al., "Diffusion Policy", RSS 2023 — diffusion-based action generation
- Ko et al., "Learning to Act from Actionless Videos through Dense Correspondences", ICLR 2024 — AVDC