# Design Theory: Hierarchical Planning for Long-Horizon Robot Manipulation

## 1. Problem Statement

SF-VLA (Vidarc) is a causal video-to-action model. It generates the next video frame autoregressively and extracts actions via an inverse dynamics model. Its core limitation:

```
Vidarc's causal attention + KV cache pop
  → Only sees recent past frames
  → No global temporal coherence
  → Cannot plan beyond its receptive field (~1-2s)

Evaluation horizon: 160 frames @ 10fps = 16s
Vidarc's effective planning horizon: ~10-20 frames = 1-2s

Gap: 14s of task structure that Vidarc cannot reason about
```

**The question: how to provide Vidarc with long-horizon planning signal without modifying its weights?**

---

## 2. Design Principles

### Principle 1: Separation of Planning and Execution

```
Planning (slow, global):  "What should happen over the next 16s?"
Execution (fast, local):  "What action to take right now?"

These are fundamentally different computational problems:
  - Planning requires global temporal reasoning (bidirectional)
  - Execution requires real-time reactivity (causal)
  - No single model architecture can do both optimally

→ Use separate models for each, connected by a guidance signal
```

### Principle 2: Model B (Vidarc) is Untouched

All guidance enters through the potential function Phi:

```
ε̃_θ(xt) = ε_θ(xt) - λ · σ_t · ∇_xt Φ
           ~~~~~~~~   ~~~~~~~~~~~~~~~~~~~
           frozen       external guidance
```

If the guidance is wrong, λ=0 recovers the original policy. This is a **safety constraint** — we never risk degrading the base model.

### Principle 3: Memory Lives Outside the Video Model

Video generation models (WanTI2V, Vidarc) have finite temporal receptive fields. Long-term memory must be maintained by an external module:

```
Level 3: Task Memory    — VLM context window (unlimited horizon)
Level 2: Visual Memory  — WanTI2V global attention (2-5s per round)
Level 1: State Memory   — Current observation image (instantaneous)
```

No single level is sufficient. The system requires all three.

### Principle 4: Validate Before Building

Every component must pass a controlled experiment before being integrated:

```
GT keyframes don't help?     → Stop, fix Φ before building planner
Latent space is misaligned?  → Stop, add projection head before training
WanTI2V plans look wrong?    → Stop, fine-tune before deploying
VLM phases are inaccurate?   → Stop, fix prompts before closing loop
```

Follow the dependency graph. Don't build downstream components on unvalidated assumptions.

### Principle 5: Complexity Budget

Each additional component must justify its existence:

```
Baseline (no guidance):           0 extra components
+ GT keyframe injection:          1 component  (extract_keyframes)
+ VLM phase detection:            2 components (VLM + phase logic)
+ VLM + GT retrieval (Method B):  3 components (VLM + library + retrieval)
+ VLM + WanTI2V (Method C):       4 components (VLM + WanTI2V + Haar + keyframe extraction)
+ Standalone WanTI2V (Method 3):  8+ components (WanTI2V + Haar + chain + Level 3 + ...)

Each level must demonstrate >10% success rate improvement
to justify the added complexity.
```

---

## 3. The Long-Horizon Problem: Three Bottlenecks

### Bottleneck 1: Vidarc's Causal Attention Cannot See the Future

```
WanModelCausal architecture:
  - patch_size = (1, 2, 2)
  - Causal attention: frame t only sees frames 0..t
  - KV cache pop (model_causal.py:282): discards old frames to save memory

  → Frame 100 cannot attend to frame 0
  → No global trajectory coherence
  → Prediction drifts over time
```

### Bottleneck 2: No World Dynamics Model

```
Planning Head (Method 1):
  Input: z_0 (single frame) + task_text
  Problem: must "hallucinate" future from a snapshot
  → No physics, no dynamics, just pattern memorization
  → Variance explodes beyond ~20 frames

MPC Rollout (Method 2):
  Input: current obs + random action sequences
  Problem: horizon = H rollout steps (short)
  → Chaining compounds errors
  → Effective range: ~8-16 frames
```

### Bottleneck 3: Cross-Round Memory Loss

```
WanTI2V full bidirectional attention only holds within a single generation:
  Round 0: frames 0-48   ← global attention ✓
  Round 1: frames 0-48   ← global attention ✓ (but independent of Round 0)

  Round 1 has zero attention to Round 0
  → Each plan round starts from scratch
  → 16s task requires 3-4 rounds with no cross-round coherence
```

---

## 4. Motivation: Why a Three-Level Architecture

### Why not just WanTI2V? (Method 3 alone is insufficient)

We initially proposed using WanTI2V as the sole long-horizon planner (Method 3 in subgoal_plan.md). Analysis revealed escalating complexity:

```
To make WanTI2V cover 16s:
  1. WanTI2V video generation (2-5s per round)
  2. Haar temporal decomposition for divergence
  3. Latent-space keyframe extraction
  4. Chained multi-round planning
  5. Chain boundary smoothing
  6. Cross-round memory (needs Level 3 anyway!)
  7. Dynamic prompt updating
  8. Possibly LoRA fine-tuning on RoboTwin

And critically: WanTI2V was pretrained on internet video, NOT robot scenes.
Plan quality on RoboTwin is completely unvalidated.
```

**The fundamental issue: even with WanTI2V, we still need an external Level 3 module for cross-round memory. If we need Level 3 anyway, it should be the primary planner.**

### Why VLM as Level 3?

```
VLM provides exactly what WanTI2V lacks:
  - Unlimited context window → natural long-term memory
  - Reasoning ability → task decomposition
  - Visual understanding → phase detection from observation
  - Structured output → subtask descriptions, completion criteria
  - Zero training → works out of the box

What VLM lacks (and WanTI2V provides):
  - Visual trajectory planning (where the arm should move)
  - Spatial subgoals (pixel-level targets)
  → But these can come from GT demo retrieval instead (Method B)
```

### The Three-Level Design

```
┌────────────────────────────────────────────────────┐
│ Level 3: VLM Task Planner                           │
│ Horizon: entire task (16s+)                         │
│ Frequency: 5-8 calls per episode                    │
│ Memory: conversation context window                 │
│ Output: subtask descriptions + completion criteria  │
│ Role: WHAT to do and WHEN it's done                 │
├────────────────────────────────────────────────────┤
│ Level 2: Subgoal Provider                           │
│ Horizon: per subtask (2-5s)                         │
│ Frequency: per subtask transition                   │
│ Source: GT demo retrieval (B) or WanTI2V gen (C)    │
│ Output: visual subgoal frames                       │
│ Role: WHERE to go (spatial targets)                 │
├────────────────────────────────────────────────────┤
│ Level 1: Vidarc Action Policy                       │
│ Horizon: per frame (~0.1s)                          │
│ Frequency: every frame                              │
│ Memory: causal KV cache (short)                     │
│ Output: actions                                     │
│ Role: HOW to get there (motor control)              │
└────────────────────────────────────────────────────┘

Information flow:
  Level 3 → subtask description → Level 2
  Level 2 → visual subgoal     → Level 1 (via Φ-guidance)
  Level 1 → actions             → environment
  environment → observation     → Level 3 (for completion check)
```

---

## 5. Two Concrete Methods

### Method B: VLM + GT Demo Retrieval (Recommended First)

```
Core idea: Don't generate new subgoals — retrieve the best matching ones
           from a library of GT demonstration keyframes.

Why:
  - GT keyframes have the highest quality (real images from successful demos)
  - Zero risk of generating physically impossible or off-domain plans
  - RoboTwin has sufficient GT demos to build the library
  - Simplest engineering (VLM + retrieval, no generative model)

Limitation:
  - Cannot generalize beyond GT demo coverage
  - New object configurations → library may not have matching keyframes
```

**Architecture:**

```
Offline (once):
  GT demos → extract_keyframes(composite) → encode with Vidarc encoder
  → Build library indexed by (task, phase)

Online (per episode):
  1. VLM decomposes task into subtasks
  2. For each subtask:
     a. VLM identifies current phase
     b. Retrieve nearest-forward GT keyframe from library
     c. Inject as subgoal → Vidarc executes with Φ-guidance
     d. VLM checks: subtask complete? → advance
```

### Method C: VLM + WanTI2V Collaboration (Generalization Upgrade)

```
Core idea: VLM handles long-range structure, WanTI2V handles short-range
           visual planning. Each operates within its competence.

Why:
  - When GT library doesn't cover the current scenario
  - WanTI2V only generates 2s plans (within its attention range)
  - No chaining needed (VLM manages cross-subtask transitions)
  - Much simpler than standalone Method 3

Additional requirement:
  - WanTI2V plan quality must be validated on RoboTwin
  - May need LoRA fine-tuning if pretrained model generates poor robot scenes
```

**Architecture:**

```
Online (per episode):
  1. VLM decomposes task into subtasks with specific prompts
  2. For each subtask:
     a. WanTI2V.generate(obs, subtask_prompt, frame_num=49)  ← 2s plan
     b. Extract keyframes from plan video (latent-space)
     c. Haar decompose for divergence monitoring
     d. Vidarc executes with Φ-guidance
     e. If diverged → re-plan same subtask from current obs
     f. VLM checks: subtask complete? → advance to next
```

---

## 6. Why Method B Before Method C

```
Method B validates:
  ✓ Does VLM phase detection work on RoboTwin?
  ✓ Does GT subgoal injection improve 160-frame success?
  ✓ Is the VLM → subgoal → Vidarc pipeline sound?
  ✓ What VLM calling frequency is needed?
  ✓ Does async prefetch work in practice?

If Method B works → the pipeline is validated, subgoal quality is the variable.
If Method B fails → the issue is NOT subgoal quality (GT is optimal),
                     but the pipeline itself → fix before adding WanTI2V.

Method C only adds value when:
  ✗ GT library doesn't cover the scenario
  ✗ Need to generalize to unseen configurations
  ✗ Method B's retrieval fails to find good matches

→ Method B is the necessary first step regardless.
```

---

## 7. Comparison with Alternative Approaches

### vs. Video-to-Action (Luo & Du, ICLR 2025)

```
V2A: lightweight video model (per-env from scratch) → goal frames →
     goal-conditioned policy (CNN DP) → actions via self-exploration

Differences:
  - V2A trains video model from scratch per environment (we use pretrained)
  - V2A requires 250+ rollouts self-exploration (we require zero rollouts)
  - V2A has no re-planning mechanism (open-loop)
  - V2A doesn't need action annotations (we do, but RoboTwin has them)

Conclusion: V2A solves a different problem (no-action-annotation learning).
            For RoboTwin with full action labels, our approach is simpler.
```

### vs. WF-VAE Frequency Decomposition

```
WF-VAE: wavelet-based VAE with explicit temporal frequency separation

Why we don't use it:
  1. Incompatible latent space (z_dim, spatial stride, distribution)
  2. Non-causal temporal alignment (vs Wan-VAE's causal encoding)
  3. Integration requires retraining the entire DiT

Instead: Haar wavelet decomposition directly on Wan-VAE latents
  → Zero extra model, zero training
  → Perfect causal alignment
  → ~10 lines of code
```

### vs. Hierarchical Φ (subgoal.md original proposal)

```
Original subgoal.md proposed:
  Φ = Φ_coarse + Φ_fine  (multi-scale potential)

This is a good theoretical framework but:
  - Φ_coarse requires accurate long-range subgoals (where do they come from?)
  - Planning head (Method 1) cannot predict long-range accurately
  - MPC rollout (Method 2) is horizon-limited

Our solution: VLM provides the coarse structure (task decomposition),
              GT retrieval or WanTI2V provides the visual subgoals.
              Φ_coarse ≈ VLM phase tracking
              Φ_fine   ≈ visual subgoal Φ-guidance
```

---

## 8. Key Technical Decisions

### Decision 1: VLM Choice

```
Recommended: Qwen2.5-VL-7B (local deployment)
  - 14GB bf16, fits alongside Vidarc on A100
  - 0.1-0.2s per call
  - Sufficient for phase detection (binary classification)

Validation path:
  Step 1: GPT-4o API (verify VLM approach feasibility)
  Step 2: Qwen2.5-VL-7B (local, match GPT-4o accuracy?)
  Step 3: Qwen2.5-VL-3B (if memory constrained)
```

### Decision 2: Async VLM Execution

```
Only the first VLM call (task decomposition) is blocking.
All subsequent calls use async prefetch:

  Trigger prefetch at 70% subtask progress
  → VLM runs in background thread
  → Result ready before Vidarc needs it
  → Zero blocking overhead after initialization

Total VLM overhead: ~2s (first call only)
```

### Decision 3: Keyframe Library Construction

```
Offline, per task:
  - Extract keyframes from all GT demos (composite strategy)
  - Encode each with Vidarc encoder → z_latent
  - Auto-label phase from action signals (gripper state + velocity)
  - Index by (task, phase) for O(1) lookup

Retrieval: nearest-forward in latent space
  - Must be forward-looking (subgoal ahead of current state)
  - Use latent L2 distance for matching
  - Top-k nearest → select smallest progress delta
```

### Decision 4: When to Upgrade from B to C

```
Method B failure modes → Method C solutions:

  B fails: "no good match in library for this configuration"
  C fixes: WanTI2V generates a plan from current observation

  B fails: "task has more phases than any single demo covers"
  C fixes: WanTI2V can imagine novel phase transitions

  B works fine:
  → Stay with B. Simpler is better.
```

---

## 9. Evaluation Criteria

### 160-Frame Episode Metrics

```
Primary:
  - Task success rate (binary)
  - Success rate vs no-guidance baseline (delta)

Secondary:
  - Number of subgoal switches per episode
  - Average frames to reach each subgoal
  - VLM phase detection accuracy (vs human labels)
  - Re-plan trigger count (Method C only)
  - Total inference time per episode

Diagnostic:
  - Latent distance to subgoal over time (should decrease)
  - Guidance gradient magnitude (should not explode/vanish)
  - VLM subtask completion accuracy (binary correctness)
```

### Ablation Structure

```
Ablation 0: No guidance (baseline)
Ablation 1: GT keyframes, uniform, no VLM (oracle upper bound)
Ablation 2: GT keyframes, composite, no VLM (better oracle)
Ablation 3: VLM + GT retrieval (Method B)
Ablation 4: VLM + WanTI2V (Method C)

Expected ranking: 2 ≥ 1 > 3 > 4 > 0
(Oracle always best; B beats C because GT quality > generated quality;
 both beat no guidance)
```

---

## 10. References

- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021 — classifier guidance
- Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis", ICML 2022 — diffusion as planner
- Luo & Du, "Grounding Video Models to Actions through Goal Conditioned Exploration", ICLR 2025 — video-to-action
- Chi et al., "Diffusion Policy", RSS 2023 — diffusion-based action generation
- Ko et al., "Learning to Act from Actionless Videos through Dense Correspondences", ICLR 2024 — AVDC
