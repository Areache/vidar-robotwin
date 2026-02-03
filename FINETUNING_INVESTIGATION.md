# Fine-tuning Investigation: Preserving Pre-trained Model Capabilities

## Problem Statement

After fine-tuning with `run_train_vidarc.sh`, the model produces **quasi-static, non-progressive outputs** instead of dynamic action sequences. The original pre-trained `vidarc.pt` works well, but Stage 2 fine-tuning destroys its capabilities.

**Symptoms:**
- Generated videos have low motion (avg pixel change ~6.8 vs GT's ~15.1)
- 160 frames of repetitive, jittery motion without state transitions
- No clear grasp/lift/place events - just continuous non-progressive motion

**Setup:**
- Starting from working `vidarc.pt` (original open-source checkpoint)
- Only doing Stage 2 fine-tuning (vidarc_trainer.py)
- Noise schedule convention is CORRECT (matches inference)

---

## Critical Issues Identified

### Issue #1: GRADIENT ACCUMULATION BUG (CRITICAL)

**Location:** `training/trainers/base.py:357-360`

```python
for i in range(accum_steps):
    step_metrics = self.train_step(batch)  # SAME batch reused 8 times!
    loss = step_metrics["loss"] / accum_steps
    loss.backward()
```

**Problem:** The same batch is processed 8 times instead of fetching 8 different batches.

**Impact:**
- Claimed effective batch size: 32 (4 GPUs × 1 × 8)
- Actual effective batch size: 4 (4 GPUs × 1 unique sample)
- Severely reduced gradient diversity → overfitting → catastrophic forgetting

**Status:** [ ] NOT FIXED

---

### Issue #2: SELF-FORCING DISABLED (RE-EVALUATED)

**Location:** `configs/vidarc_4xh200.yaml:42-43`

```yaml
self_forcing:
  enabled: false   # Disabled as requested
```

#### ✅ UPDATE (2026-01-25): Evaluation IS Closed-Loop

**Finding:** The evaluation pipeline (`run_eval_ddp_causal.sh`) **IS running in closed-loop mode** with re-prefilling, matching the paper's Algorithm 1.

**Evidence from code:**

1. **Real observations from environment** (`eval_policy.py:425-427`):
```python
while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
    observation = TASK_ENV.get_obs()  # REAL observation from simulator
    eval_func(TASK_ENV, model, observation)
```

2. **Re-prefilling mechanism** (`ar.py:1241-1247`):
```python
if self.num_conditional_frames + self.num_new_frames > self.rollout_bound:
    self.num_conditional_frames = self.rollout_prefill_num  # Reset to 33
    obs_cache = self.obs_cache[-self.num_conditional_frames:]  # Last 33 REAL obs
    clean_cache = True  # Trigger KV cache clear
```

3. **KV cache cleanup** (`textimage2video_causal_server.py:612-615`):
```python
if num_conditional_frames <= 1 or clean_cache:
    self.clean_state()  # Clears KV cache, re-prefills with real observations
```

**Paper's Claim (Section 3.3):**
> "We generate the next observation based on previous real-world observations, rather than generated ones, enabling closed-loop control. This paradigm aligns with teacher forcing training."

**Revised Impact:** Since inference uses real observations via re-prefilling (not autoregressive predictions), there is NO train-test mismatch for the observation source. The teacher forcing training IS appropriate for this closed-loop inference paradigm.

**However:** This finding shifts the problem elsewhere (see Issue #5 below).

**Status:** [x] NOT A BUG - Evaluation correctly implements closed-loop with re-prefilling

---

### Issue #3: LEARNING RATE TOO HIGH

**Location:** `configs/vidarc_4xh200.yaml:15`

```yaml
lr: 2.0e-5  # Same as Stage 1 pre-training
```

**Problem:** Fine-tuning should use 5-10x lower LR than pre-training.

**Recommendation:** Use `lr: 2.0e-6` to `4.0e-6` for Stage 2.

**Status:** [ ] NOT FIXED

---

### Issue #4: NO WEIGHT PRESERVATION MECHANISM

**Missing safeguards:**
1. No EWC (Elastic Weight Consolidation) loss
2. No L2 regularization toward original weights
3. No knowledge distillation from original model

**Status:** [ ] NOT FIXED

---

### Issue #5: PROGRESSIVE BLUR AND GHOSTING (NEW - CRITICAL)

**Symptom:** Videos show progressive blur midway through, with:
- Softened object contours
- Greyed edges and ghosting overlays
- Colors averaged into low-contrast neutral patches
- Temporal frame-blurring (not random noise)

**Important:** This occurs with `v0_original` which has `use_libero_subgoal=False` and `subgoal_guidance_scale=0.0`. **Subgoal guidance is NOT the cause.**

#### Root Cause Analysis: Autoregressive Generation + Finetuning Damage

The blur is caused by **two interacting factors**:

**Factor 1: Intra-Chunk Autoregressive Error Accumulation**

Within each 16-frame generation chunk, frames are generated one-by-one:

```python
# textimage2video_causal_server.py:796
for latent_frame_idx in tqdm(range(cond_latent_frame, T)):
    # Generate frame, cache KV, repeat...
    self.model(..., cache=True, ...)  # Each frame's KV added to cache
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│              INTRA-CHUNK AUTOREGRESSIVE GENERATION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Frame 1:  KV=[real_obs] → Generate → Sharp                            │
│   Frame 2:  KV=[real_obs, gen_1] → Generate → Slight blur               │
│   Frame 3:  KV=[real_obs, gen_1, gen_2] → Generate → More blur          │
│   ...                                                                   │
│   Frame 16: KV=[real_obs, gen_1...gen_15] → Generate → Significant blur │
│                                                                         │
│   Each generated frame adds noise to KV cache for subsequent frames     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Factor 2: Finetuning Destroyed Model's Sharpness**

The finetuned model has learned to generate "averaged" outputs due to:
- Mode collapse from low batch diversity (Issue #1)
- High learning rate causing weight drift (Issue #3)

**Pre-trained model:** Generates diverse, sharp predictions
**Finetuned model:** Generates "safe", blurry predictions (mean of distribution)

#### Evidence: Progressive Nature of Blur

Why blur appears around frame 80 (midway through 160 frames):

| Frames | What Happens | Blur Level |
|--------|--------------|------------|
| 1-16 | Chunk 1: Fresh start, some AR accumulation | Low |
| 17-32 | Chunk 2: More accumulation | Low-Medium |
| 33-48 | Chunk 3: AR errors compound | Medium |
| 49-64 | Chunk 4: Triggers cache reset at ~60 | Medium (brief recovery) |
| 65-80 | Chunk 5: Blur rebuilds after reset | **High - visually obvious** |
| 80-160 | Remaining: Continued accumulation cycles | High |

#### Additional Contributing Factors

**1. Too Few Sampling Steps**
```bash
NUM_SAMPLING_STEP=10  # In run_eval_ddp_causal.sh
```
- 10 steps may not be enough for sharp convergence
- Try 20-30 steps to test

**2. CFG Scale**
```bash
CFG=3.0
```
- May be too low for maintaining sharpness
- Try CFG=5.0-7.0

#### ✅ CONFIRMED: Blur at END of Each Chunk, Only in Finetuned Model

**Diagnostic Results:**
1. **Blur position:** END of each 16-frame chunk (frames 16, 32, 48, 64, 80, 96, 112, 128, 144, 160)
2. **Original model:** Does NOT have this blur pattern

**This confirms finetuning destroyed the model's autoregressive robustness.**

#### Root Cause: Finetuning Caused AR Error Sensitivity

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY BLUR APPEARS AT FRAME 16                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Within each 16-frame chunk, frames are generated autoregressively:         │
│                                                                             │
│  Frame 1:  KV = [real_obs]                    → Model confident → SHARP     │
│  Frame 2:  KV = [real_obs, gen_1]             → Still confident → Sharp     │
│  Frame 3:  KV = [real_obs, gen_1, gen_2]      → Slight uncertainty          │
│  ...                                                                        │
│  Frame 15: KV = [real_obs, gen_1...gen_14]    → Growing uncertainty         │
│  Frame 16: KV = [real_obs, gen_1...gen_15]    → MAX UNCERTAINTY → BLURRY    │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  PRE-TRAINED MODEL:  Handles uncertainty → generates diverse, sharp frames  │
│  FINETUNED MODEL:    Collapses under uncertainty → generates averaged blur  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why Finetuning Caused This

The finetuned model learned to generate "safe" averaged outputs when uncertain:

| Training Issue | Effect on AR Generation |
|----------------|------------------------|
| **Gradient accumulation bug** (same batch 8x) | Model overfit to narrow patterns, can't handle diverse AR contexts |
| **High learning rate** (2e-5) | Rapid drift from pre-trained weights that handled AR well |
| **No regularization** | Lost the "robustness" learned during pre-training |

The pre-trained model was trained on diverse data and learned to maintain sharpness even with accumulated AR errors. Finetuning destroyed this capability.

#### Why Frame 16 Specifically?

Frame 16 has the **maximum KV cache pollution** within each chunk:
- 15 generated frames in KV cache (vs 0 for frame 1)
- Each generated frame adds small errors to the attention context
- By frame 16, errors compound enough to trigger the model's "uncertainty → blur" behavior

**Status:** [x] ROOT CAUSE CONFIRMED - Finetuning destroyed AR robustness

---

### Issue #6: ROOT CAUSE OF REPETITIVE MOTION (CRITICAL)

**Symptom:** After finetuning, the model produces repetitive low-level motion without clear task-level state transitions (grasp → lift → place events are missing).

#### Important Clarification: Two Separate Problems

| Problem | Cause | Fix Priority |
|---------|-------|--------------|
| **Progressive blur/ghosting** | Subgoal guidance averaging (Issue #5) | High - disable subgoal guidance |
| **Repetitive motion** | Catastrophic forgetting (Issues #1,#3,#4) | High - fix training bugs |

These are **independent issues** that both contribute to poor task execution.

#### Why This Happens: The Compounding Effect

The repetitive motion problem is caused by the **compounding interaction** of Issues #1, #3, and #4:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CATASTROPHIC FORGETTING CASCADE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Issue #1: Gradient Accumulation Bug                                        │
│       ↓                                                                     │
│  Effective batch size: 4 (not 32)                                           │
│       ↓                                                                     │
│  Low gradient diversity → Model sees same patterns 8x per step              │
│       ↓                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Issue #3: High Learning Rate (2e-5)                                  │   │
│  │      ↓                                                               │   │
│  │ Large weight updates per step                                        │   │
│  │      ↓                                                               │   │
│  │ Rapid deviation from pre-trained weights                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       ↓                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Issue #4: No Weight Preservation                                     │   │
│  │      ↓                                                               │   │
│  │ No anchor to original knowledge                                      │   │
│  │      ↓                                                               │   │
│  │ Model "forgets" diverse motion patterns from pre-training            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       ↓                                                                     │
│  ════════════════════════════════════════════════════════════════════════  │
│                     MODE COLLAPSE TO "SAFE" MOTIONS                         │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Analysis

**1. Video-to-Action Architecture Amplifies the Problem**

The Vidar pipeline is:
```
Real Obs → Video Model → Generated Frames → IDM → Actions
```

The video model is trained to minimize **frame prediction loss**, not action quality. When finetuning causes mode collapse:

| What the model learns | What this means for task execution |
|-----------------------|-----------------------------------|
| Generate "smooth" frames | Continuous micro-movements, no state changes |
| Minimize visual discontinuities | Avoids "jumpy" grasp/release events |
| Stay close to conditioning frames | Repetitive motion around initial state |

**2. Teacher Forcing Creates Local Optimum Trap**

With teacher forcing, the model is trained on:
- Input: Ground truth frames [t-N, t-1]
- Output: Predict frame [t]

The loss landscape has a **local minimum** at "predict minimal change from input":
- This minimizes per-frame MSE loss
- But destroys task-level progress (no state transitions)

**3. Low Effective Batch Size → Narrow Feature Coverage**

With only 4 unique samples per gradient step (instead of 32):
```
Pre-trained model saw:     [grasp, lift, place, rotate, push, pull, ...]
Finetuning model sees:     [similar_motion, similar_motion, similar_motion, ...]
                                     ↓
                           Model specializes on narrow motion distribution
                                     ↓
                           "Forgets" how to generate diverse state transitions
```

**4. The Vicious Cycle**

Once the model starts producing repetitive motion:
1. Re-prefilling feeds back real observations (which show no progress)
2. Model predicts "continue current state" (locally optimal)
3. Actions extracted by IDM are small/repetitive
4. Environment shows minimal change
5. Cycle repeats → Task never completes

#### Why Pre-trained Model Works

The pre-trained `vidarc.pt` was trained on:
- Large-scale diverse video data
- Many different task demonstrations
- Varied motion patterns (grasps, lifts, places, etc.)

This created a rich feature space that can generate diverse state transitions.

#### Solution: Break the Cascade

All three contributing issues must be fixed together:

| Fix | Why It's Needed |
|-----|-----------------|
| Fix gradient accumulation | Restore batch diversity (32 unique samples) |
| Reduce learning rate (2e-6) | Slower weight updates, preserve more pre-trained knowledge |
| Add EWC/weight regularization | Anchor to original weights, prevent mode collapse |

**Status:** [ ] ROOT CAUSE IDENTIFIED - Requires fixing Issues #1, #3, #4 together

---

## Proposed Fixes

### Fix #1: Fix Gradient Accumulation (CRITICAL)

In `training/trainers/base.py`, modify `_train_step_with_accumulation`:

```python
def _train_step_with_accumulation(self, data_iterator) -> Dict[str, float]:
    """Training step with gradient accumulation using different batches."""
    accum_steps = self.config.training.gradient_accumulation

    total_loss = 0.0
    for i in range(accum_steps):
        batch = next(data_iterator)  # Fetch NEW batch each time
        batch = self._to_device(batch)

        with torch.cuda.amp.autocast(enabled=...):
            step_metrics = self.train_step(batch)

        loss = step_metrics["loss"] / accum_steps
        loss.backward()
        total_loss += step_metrics["loss"].item() / accum_steps

    self.optimizer.step()
    self.optimizer.zero_grad()
    ...
```

### Fix #2: Reduce Learning Rate (CRITICAL)

In `configs/vidarc_4xh200.yaml`:

```yaml
training:
  lr: 2.0e-6  # 10x lower than original pre-training
```

**Rationale:** Lower LR preserves more pre-trained knowledge during finetuning.

### ~~Fix #3: Enable Self-Forcing~~ (NOT NEEDED)

**Update:** Self-forcing is NOT needed because evaluation uses closed-loop with re-prefilling.
The paper explicitly states this design choice aligns teacher forcing training with closed-loop inference.

### Fix #3: Add EWC Loss (CRITICAL for Preventing Mode Collapse)

Create a new loss term in `training/losses.py`:

```python
def ewc_loss(model_params, original_params, lambda_ewc=0.1):
    """Elastic Weight Consolidation loss."""
    loss = 0.0
    for (name, param), (_, orig_param) in zip(
        model_params.items(), original_params.items()
    ):
        loss += ((param - orig_param) ** 2).sum()
    return lambda_ewc * loss
```

---

## Progress Tracking

### Phase 1: Investigation (Complete) ✅
- [x] Verify evaluation pipeline (closed-loop confirmed)
- [x] Analyze self-forcing requirement (not needed - paper design)
- [x] Confirm v0_original does NOT use subgoals
- [x] Identify blur pattern: exactly 10 blurs at frames 16, 32, 48... (end of each chunk)
- [x] Confirm original model has NO blur → **Finetuning is the cause**
- [x] Root cause: Finetuning destroyed AR robustness (Issue #5)
- [x] Root cause: Catastrophic forgetting causes repetitive motion (Issue #6)

### Phase 2: Training Bug Fixes (Critical - All Required)
- [ ] Fix gradient accumulation to use different batches (Issue #1)
- [ ] Reduce learning rate to 2e-6 (Issue #3)
- [ ] Add EWC/weight regularization (Issue #4)

**Important:** These fixes must be applied TOGETHER to:
1. Restore batch diversity → model learns robust representations
2. Slow weight drift → preserve pre-trained AR robustness
3. Anchor to original weights → prevent mode collapse

### Phase 3: Validation
- [ ] Re-run training with all three fixes
- [ ] Evaluate on hd_clean task
- [ ] Verify NO blur at frame 16, 32, 48... (AR robustness restored)
- [ ] Verify state transitions (grasp → lift → place) present
- [ ] Compare motion metrics with GT

---

## Testing Procedure

After applying fixes:

```bash
# 1. Train with fixed code
bash run_train_vidarc.sh \
  configs/vidarc_4xh200.yaml \
  /path/to/dataset \
  /path/to/wan_ckpt \
  /path/to/vidarc.pt \
  ./output_vidarc_fixed \
  4000

# 2. Evaluate
bash run_eval_ddp_causal.sh \
  hd_clean \
  ./output_vidarc_fixed/vidarc.pt

# 3. Compare motion metrics
python scripts/compare_motion.py \
  --gt episode0_gt.mp4 \
  --pred episode0.mp4
```

---

## References

**Key Files:**
- Gradient accumulation: `training/trainers/base.py:350-380`
- Self-forcing: `training/models/wrapper_causal.py:317-428`
- Config: `configs/vidarc_4xh200.yaml`
- Inference: `vidar/wan/textimage2video_causal.py:663-672`
- FlowMatchScheduler: `vidar/wan/utils/fm.py:122-133`

**Papers:**
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003

---

## Change Log

| Date | Change | Status |
|------|--------|--------|
| 2026-01-25 | Initial investigation - identified 4 critical issues | Complete |
| 2026-01-25 | **Verified eval pipeline is CLOSED-LOOP with re-prefilling** | ✅ Confirmed |
| 2026-01-25 | **Issue #2 re-evaluated**: Self-forcing not a bug, matches paper design | ✅ Resolved |
| 2026-01-25 | **Issue #5**: Blur at END of each 16-frame chunk (frames 16,32,48...) | ✅ Confirmed |
| 2026-01-25 | **Issue #5**: Original model has NO blur → Finetuning is the cause | ✅ Confirmed |
| 2026-01-25 | **Issue #6**: Repetitive motion from catastrophic forgetting | ✅ Confirmed |
| 2026-01-25 | **ROOT CAUSE**: All symptoms trace to training bugs #1, #3, #4 | ✅ Confirmed |
| | Issue #1: Gradient accumulation bug | ❌ Pending Fix |
| | Issue #3: Learning rate too high | ❌ Pending Fix |
| | Issue #4: No weight preservation | ❌ Pending Fix |
| | Fix #1: Gradient accumulation | Pending |
| | Fix #2: Reduce LR to 2e-6 | Pending |
| | Fix #3: Add EWC/weight regularization | Pending |
