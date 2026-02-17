# Experiment: GT Keyframe Subgoal Training

**Goal:** Determine whether injecting GT keyframes (extracted from HDF5 demos) as subgoal conditioning during training improves policy task success at evaluation time.

**Core question:** Does training with explicit subgoal signals produce a policy that better follows future-state guidance?

**Depends on:** `subgoal_keyframe_rule.md` (keyframe extraction strategies), `subgoal_plan.md` (Exp 1.1/1.2 oracle guidance baseline)

---

## 1. Problem Statement

VIDAR currently trains on (first frame, instruction) -> full video. At eval, subgoals can be injected via `subgoal_frames` in the causal worker, but the model has **never seen subgoal conditioning during training**. This creates a train-eval mismatch:

```
Training:  first_frame + instruction  -->  video
Eval:      first_frame + instruction + subgoal_frames  -->  video (with guidance)
```

If we instead train with GT keyframes as conditioning:

```
Training:  first_frame + instruction + gt_keyframes  -->  video
Eval:      first_frame + instruction + subgoal_frames  -->  video (with guidance)
```

The model learns to *use* subgoal information, closing the gap.

---

## 2. Setup

### 2.1 Keyframe Extraction

Extract GT keyframes from HDF5 training demos using the semantic strategy from `extract_keyframes.py`:

```python
from extract_keyframes import extract_keyframes_from_hdf5

keyframes = extract_keyframes_from_hdf5(
    hdf5_path="episode_000001.hdf5",
    strategy="semantic",          # motion stops + visual changes
    max_keyframes=20,
    use_cache=True,
)
```

**Why semantic strategy:**
- Combines motion-stop detection (robot pauses at grasp/release) with visual change detection
- No HDF5 action data required (works on images only, portable to new datasets)
- Produces denser keyframes than `gripper_change` (which yields only 2-5) but sparser than `uniform`
- Detects task phase boundaries without manual threshold tuning per-task

**Extraction code:** `vidar-robotwin/experiments/gt_keyframe_test/extract_keyframes.py:649-802` (`extract_keyframes_semantic` / `_detect_semantic_keyframes`)

**Detection logic:**
```
Pass 1: Compute frame-to-frame signals
  - motion_score[t] = mean(|gray_t - gray_{t-1}|)    (mean absolute difference)
  - change_score[t] = mean((gray_t - gray_{t-1})^2)   (MSE)

Pass 2: Detect events
  - Motion stop: smoothed_motion transitions high -> low  (robot pauses)
  - Visual change: change_score > threshold               (object state change)
  - min_interval=5 between keyframes
```

### 2.2 Data Paths

```
HDF5 data:    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed
Wan2.2 ckpt:  /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B
Vidarc ckpt:  /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/vidar_ckpts/vidarc.pt
IDM ckpt:     /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/vidar_ckpts/idm.pt
```

### 2.3 Training Config

Base config: `configs/vidarc_2xh200.yaml`

Key training parameters:
```yaml
model:
  ckpt_dir: checkpoints/Wan2.2-TI2V-5B
  model_class: WanModelCausal

training:
  num_steps: 4000
  batch_size: 1               # per GPU
  gradient_accumulation: 8    # effective batch = 16 (2 GPUs)
  lr: 2.0e-5
  freeze: [t5, vae]

data:
  num_frames: 16
  resolution: [736, 640]
  fps: 10
  cfg_prob: 0.1               # 10% classifier-free guidance dropout

loss:
  type: causal_flow_matching
  embodiment_aware: true
  eta: 3.0                    # robot region weighted 3x

self_forcing:
  enabled: true
  causal: true
  chunk_size: 1
```

---

## 3. Experiment Design

### 3.1 Conditions

| Condition | Training | Eval Subgoals | Purpose |
|-----------|----------|---------------|---------|
| **A: Baseline** | Standard (no keyframes) | None | Reference performance |
| **B: Baseline + Oracle Eval** | Standard (no keyframes) | GT keyframes injected at eval | Does oracle guidance help an untrained-for-subgoals model? (= Exp 1.2 from subgoal_plan.md) |
| **C: Trained + Oracle Eval** | With GT keyframe conditioning | GT keyframes injected at eval | Does training with keyframes improve guidance effectiveness? |
| **D: Trained, No Eval Subgoals** | With GT keyframe conditioning | None | Does keyframe-conditioned training hurt when subgoals are absent? (robustness check) |

### 3.2 Training Commands

**Condition A (Baseline):**
```bash
bash run_train_vidarc.sh \
    configs/vidarc_2xh200.yaml \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/vidar_ckpts/vidarc.pt \
    ./output_vidarc
```

**Condition C (With GT keyframes):**
```bash
bash run_train_vidarc.sh \
    configs/vidarc_2xh200_subgoal.yaml \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/vidar_ckpts/vidarc.pt \
    ./output_vidarc_subgoal
```

*(Requires creating `vidarc_2xh200_subgoal.yaml` and modifying the data pipeline — see Section 5.)*

### 3.3 Evaluation Command

**All conditions use the same eval command** (with/without subgoal injection toggled):

```bash
bash run_eval_ddp_causal.sh \
    hd_clean \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/checkpoints/vidarc_2x_aligned_20.pt \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/vidar_ckpts/idm.pt \
    4x_35 > log_eval_org.log2 2>&1
```

For conditions with oracle subgoal injection at eval (B, C), modify the eval to use `v1_subgoal` version:
```bash
bash run_eval_ddp_causal.sh \
    hd_clean \
    <MODEL_CHECKPOINT> \
    /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/vidar_ckpts/idm.pt \
    4x_35 \
    16 10 3.0 \
    true \            # USE_VIDEO_SUBGOALS
    ... \
    v1_subgoal        # VERSION
    > log_eval_subgoal.log 2>&1
```

---

## 4. Metrics

### 4.1 Primary Metrics

| Metric | How | Pass Criteria |
|--------|-----|---------------|
| **Task success rate** | Per-task binary success over N episodes | C > A by >10% |
| **Success rate delta (B vs A)** | Oracle eval improvement over baseline | B > A → oracle guidance helps |
| **Success rate delta (C vs B)** | Training improvement over eval-only injection | C > B → training with keyframes is better than just injecting at eval |

### 4.2 Diagnostic Metrics

| Metric | What it reveals |
|--------|-----------------|
| Trajectory smoothness (action velocity variance) | Does guidance cause jerky movements? |
| Action reversal count | Does the policy oscillate? |
| Per-step latent distance to subgoal | Does the model actively approach subgoals? |
| Condition D success rate vs A | If D << A, model became dependent on subgoals (fragile) |
| Training loss curve (A vs C) | Does keyframe conditioning change convergence? |

---

## 5. Implementation Requirements

### 5.1 What Exists

- [x] `extract_keyframes_from_hdf5()` with semantic strategy (`extract_keyframes.py:952-1099`)
- [x] `subgoal_frames` parameter in causal worker server
- [x] `inject_gt_subgoals()` for eval-time injection (`run_with_gt_subgoals.py`)
- [x] Version system (`v0_original`, `v1_subgoal`) for toggling eval behavior
- [x] HDF5 data pipeline (`HDF5VLADataset`)
- [x] Self-forcing trainer (`VidarCausalTrainer`)

### 5.2 What Needs Implementation

**5.2.1 — Training data pipeline: add GT keyframes to each batch**

Modify `HDF5VLADataset` to:
1. Extract keyframes from each episode (using semantic strategy) during `__init__` or `__getitem__`
2. Return `subgoal_frames` alongside video frames and instruction
3. Cache extracted keyframes to avoid re-extraction

```python
# In HDF5VLADataset.__getitem__:
keyframes = extract_keyframes_from_hdf5(
    hdf5_path, strategy="semantic", max_keyframes=self.max_subgoals,
    use_cache=True
)
subgoal_images = [base64_to_numpy(kf.image_b64) for kf in keyframes]
# Subsample to match training frame indices
subgoal_frames = select_subgoals_for_window(subgoal_images, start_idx, num_frames)
```

**5.2.2 — Trainer: pass subgoals through self-forcing forward pass**

Modify `VidarCausalTrainer.forward_self_forcing_aligned()` to:
1. Accept `subgoal_latents` parameter
2. Encode subgoal frames with frozen VAE (same as conditioning frames)
3. Inject into KV cache or as cross-attention conditioning
4. Ensure subgoal conditioning is dropped with `cfg_prob` for classifier-free guidance

**5.2.3 — Config: `vidarc_2xh200_subgoal.yaml`**

```yaml
# Extends vidarc_2xh200.yaml with:
data:
  use_gt_keyframes: true
  keyframe_strategy: semantic
  max_keyframes: 5
  keyframe_dropout: 0.1    # Drop keyframes 10% of the time (learn to work without)

subgoal:
  enabled: true
  injection_method: concat_conditioning  # or: cross_attention, latent_guidance
  guidance_scale: 0.5
```

**5.2.4 — Causal worker: ensure subgoal_frames used consistently in train and eval**

Verify that the `subgoal_frames` processing path in `causal_worker.py` matches the training injection path. Train and eval must encode/inject subgoals identically.

---

## 6. Execution Checklist

```
Phase 1: Baseline (Condition A)
[ ] Run baseline training with provided command
[ ] Run baseline eval → log_eval_baseline.log
[ ] Record per-task success rates

Phase 2: Oracle Eval Only (Condition B)
[ ] Extract GT keyframes for eval episodes (semantic strategy)
[ ] Run eval with GT keyframe injection (v1_subgoal) using SAME baseline model
[ ] Record per-task success rates
[ ] Compare B vs A → GATE: does oracle guidance help?
    [ ] If NO: stop, diagnose Φ (jump to Series 3/4 in subgoal_plan.md)
    [ ] If YES: proceed to Phase 3

Phase 3: Implement Training-Time Injection (Condition C)
[ ] Modify HDF5VLADataset to extract and return GT keyframes
[ ] Modify VidarCausalTrainer to accept subgoal conditioning
[ ] Create vidarc_2xh200_subgoal.yaml config
[ ] Run training with GT keyframes
[ ] Run eval WITH subgoal injection → compare C vs B
[ ] Run eval WITHOUT subgoal injection (Condition D) → robustness check

Phase 4: Analysis
[ ] Compare all 4 conditions
[ ] Fill in summary with findings
[ ] Decide next steps based on results
```

---

## 7. Pass/Fail Criteria

| Gate | Pass | Fail | Implication |
|------|------|------|-------------|
| B > A (oracle helps) | Success +10% | Success unchanged | Φ-guidance is broken → fix potential function before training changes |
| C > B (training helps) | Success +5% | Success unchanged | Training with keyframes adds value beyond eval-only injection |
| D ~ A (robust) | Within 5% of A | D << A (>10% drop) | Model became dependent on subgoals → add keyframe_dropout |
| C > A (overall improvement) | Success +15% | No improvement | The full pipeline (train + eval with keyframes) works end-to-end |

---

## 8. Key Decision This Experiment Informs

```
IF B > A (oracle guidance helps at eval):
  → Φ-guidance framework is viable
  → Proceed with planning head or MPC for generating subgoals
  → The "which model generates subgoals" question (subgoal_keyframe_model.md) becomes relevant

IF C > B (training with keyframes helps more):
  → Subgoal-aware training is necessary, not just eval-time injection
  → Future work: replace GT keyframes with predicted keyframes (planning head)

IF B = A (oracle guidance doesn't help):
  → The fundamental Φ-guidance mechanism is broken
  → Diagnose: latent space misalignment (Exp 3.x), gradient issues (Exp 4.x)
  → Do NOT invest in keyframe extraction, generation models, or planning heads yet
```

---

## 9. Connection to Other Experiments

**This experiment is the CRITICAL GATE for the entire subgoal research line.**

```
subgoal_train.md (THIS)
  ├── Phase 1-2 (A vs B) = Exp 1.1 + 1.2 from subgoal_plan.md
  │   └── GATE: Does Φ-guidance work?
  │         ├── YES → Phase 3 (train with keyframes)
  │         │         └── subgoal_plan.md Exp 1.3 (λ ablation)
  │         │         └── subgoal_plan.md Exp 1.4 (interval ablation)
  │         │         └── subgoal_plan.md Exp 2.x (keyframe strategies)
  │         └── NO  → subgoal_plan.md Exp 3.x (latent diagnostics)
  │                  → subgoal_plan.md Exp 4.x (gradient analysis)
  │
  ├── subgoal_keyframe_rule.md → provides extraction strategies
  │   (semantic strategy validated here)
  │
  └── subgoal_keyframe_model.md → DEFERRED until this gate passes
      (pixel-space generation is irrelevant if Φ-guidance itself doesn't work)
```

---

## 10. Output Structure

```
experiments/subgoal/results/train/
├── baseline/
│   ├── training_log.json           # Condition A training metrics
│   ├── log_eval_baseline.log       # Condition A eval
│   └── eval_results.json           # Per-task success rates
├── oracle_eval/
│   ├── log_eval_oracle.log         # Condition B eval (same model as A, GT keyframes at eval)
│   ├── eval_results.json
│   └── keyframes_used/             # Extracted GT keyframes per episode
├── trained_subgoal/
│   ├── training_log.json           # Condition C training metrics
│   ├── log_eval_with_sg.log        # Condition C eval (with subgoals)
│   ├── log_eval_no_sg.log          # Condition D eval (without subgoals)
│   └── eval_results.json
└── summary.md                      # Fill in after all conditions complete
```
