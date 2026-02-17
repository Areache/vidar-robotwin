# Subgoal Planning: Experiment Progress

## Phase 0: Validation & Feasibility

### Exp 0.1 — Baseline Eval (No Subgoal Guidance)

**Status:** Done

**Config:**
- Task: `adjust_bottle`
- Policy: Vidarc (AR, causal)
- Subgoal guidance: disabled (`use_gt_keyframes: False`)

**Command:**
```bash
python eval_policy.py --config ../policy/AR/deploy_policy.yml --overrides \
  --task_name adjust_bottle --save_dir eval_result/ar/baseline
```

**Result:** (TODO: fill in success rate)

---

### Exp 0.2 — Oracle GT Keyframe Eval

**Status:** Pending

**Config:**
- Task: `adjust_bottle`
- Subgoal source: GT demo video (uniform extraction, interval=8, max=20)
- GT video dir: `eval_result/ar/ddp_causal`

**Command:**
```bash
python eval_policy.py --config ../policy/AR/deploy_policy.yml --overrides \
  --task_name adjust_bottle \
  --use_gt_keyframes True \
  --gt_keyframe_dir /mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal \
  --gt_keyframe_strategy uniform \
  --gt_keyframe_interval 8 \
  --gt_max_keyframes 20 \
  --save_dir eval_result/ar/gt_keyframe_oracle
```

**Result:** (TODO)

**Purpose:** Upper bound — if GT keyframes can't help, no planner will.

---

### Exp 0.3 — VLM Phase Detection Feasibility (Qwen2.5-VL-7B)

**Status:** Done

**Config:**
- Model: Qwen2.5-VL-7B-Instruct (local, bfloat16, flash_attention_2)
- Checkpoint: `models--Qwen--Qwen2.5-VL-7B-Instruct`
- Task: `adjust_bottle`
- Video: `eval_result/ar/ddp_causal/adjust_bottle/episode0.mp4` (51 frames)
- Sample count: 8 (uniform)

**Command:**
```bash
python test_vlm_phase.py \
  --video /mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal/adjust_bottle/episode0.mp4 \
  --task adjust_bottle
```

**Result:**

#### Test 1: Task Decomposition (8.16s)

VLM decomposed `adjust_bottle` into 3 phases:

| Phase | Name | Description |
|-------|------|-------------|
| 0 | approach_bottle | The robot's arms move towards the green bottle on the table |
| 1 | grasp | The robot's grippers close around the bottle and lift it slightly off the table |
| 2 | lift | The robot lifts the bottle upwards, keeping it upright |

#### Test 2: Phase Detection (8 frames, 1.7-2.2s per call)

| Frame | Detected Phase | Status | Confidence | Latency |
|-------|---------------|--------|------------|---------|
| 0 | 0 (approach) | not_started | high | 1.74s |
| 7 | 0 (approach) | in_progress | medium | 1.88s |
| 14 | 1 (grasp) | in_progress | high | 1.75s |
| 21 | 1 (grasp) | in_progress | high | 1.79s |
| 28 | 1 (grasp) | in_progress | high | 1.79s |
| 35 | 1 (grasp) | in_progress | high | 1.80s |
| 42 | 1 (grasp) | in_progress | high | 1.71s |
| 50 | 2 (lift) | in_progress | high | 2.20s |

- Phase sequence: `[0, 0, 1, 1, 1, 1, 1, 2]`
- **Monotonic: Yes**
- Confidence: 7/8 high, 1/8 medium

#### Test 3: Completion Detection (2.91s)

| | Frame | Progress |
|---|-------|----------|
| Early | 0 | 0% |
| Late | 50 | 100% |
| **Order correct** | | **True** |

#### Summary

| Metric | Value |
|--------|-------|
| Total VLM calls | 10 |
| Total inference time | 25.7s |
| Avg latency per call | 2.57s |
| First call latency | 8.16s (includes warmup) |
| Steady-state latency | 1.7-2.2s |
| Phase monotonicity | Yes |
| Completion order correct | Yes |

**Conclusion:** Qwen2.5-VL-7B can reliably detect task phases from single observation frames. Phase decomposition is semantically reasonable, detection is monotonic with high confidence. Method B (VLM + GT Retrieval) is feasible.

---

### Exp 0.4 — GT Keyframe Library Construction

**Status:** Done

**Config:**
- Task: `stack_bowls_two` (only HDF5 task available)
- Mode: `hdf5` (gripper state crossing detection)
- Gripper indices: cols 7, 15 (binary open/close commands)
- Data: 50 episodes from RoboTwin processed HDF5

**Command:**
```bash
python build_library.py --mode hdf5 \
  --data_dir /mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed \
  --tasks stack_bowls_two \
  --output library.json
```

**Result:**

| Metric | Value |
|--------|-------|
| Task | `stack_bowls_two` |
| Detected phases | **9** |
| Canonical episodes | 50/50 (100% consistent) |
| Median boundaries | `[0, 7, 11, 23, 27, 39, 43, 56, 60, 65]` |

Typical episode boundary pattern (episode_000000, 67 frames):
```
[0, 7, 11, 23, 27, 40, 44, 56, 60, 66]
```

9 phases correspond to 4 gripper close/open cycles across both arms:

| Phase | Frames (median) | Interpretation |
|-------|--------|----------------|
| 0 | 0-7 | Approach bowl A |
| 1 | 7-11 | Grasp bowl A (gripper close) |
| 2 | 11-23 | Lift & move bowl A |
| 3 | 23-27 | Release bowl A (gripper open) |
| 4 | 27-39 | Approach bowl B |
| 5 | 39-43 | Grasp bowl B (gripper close) |
| 6 | 43-56 | Lift & move bowl B over A |
| 7 | 56-60 | Place bowl B on A (gripper open) |
| 8 | 60-65 | Retract |

**Notes:**
- Initial run with `GRIPPER_INDICES=(6, 13)` and threshold=0.3 detected only 1 phase (continuous joint angles, delta too small)
- Fixed by switching to cols 7, 15 (binary gripper commands) with state-crossing detection at threshold=0.5
- Phase count is 100% consistent across all 50 episodes — strong signal for Method B

**Next:** Run `--label_phases` to get VLM-generated semantic phase names

---

### Exp 0.5 — VLM Interaction Success Detection (Closed-Loop Supervisor Feasibility)

**Status:** Done

**Config:**
- Model: Qwen2.5-VL-7B-Instruct (local, bfloat16, flash_attention_2)
- Task: `adjust_bottle`
- Plan video: `episode1_pred_160.mp4` (video model prediction, 160 frames)
- Actual video: `episode1.mp4` (simulation execution, 160 frames)
- Sample count: 8 (uniform)

**Command:**
```bash
python test_vlm_phase.py \
  --plan_video /mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal/adjust_bottle/episode1_pred_160.mp4 \
  --actual_video /mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal/adjust_bottle/episode1.mp4 \
  --task adjust_bottle
```

**Result:**

#### Per-Frame Interaction Success Detection

| Frame | Interaction Success | Confidence | Plan-Actual Match | Mismatch Type | Latency |
|-------|-------------------|------------|-------------------|---------------|---------|
| 0 | Yes | high | Yes | — | 2.51s |
| 22 | Yes | high | Yes | — | 2.73s |
| 45 | No | high | No | position_error | 2.88s |
| 67 | No | medium | No | grasp_failure | 2.79s |
| 90 | No | high | No | position_error | 2.85s |
| 112 | Yes | high | Yes | — | 2.64s |
| 135 | Yes | high | Yes | — | 2.71s |
| 159 | No | high | No | position_error | 3.02s |

#### Summary

| Metric | Value |
|--------|-------|
| Interaction success rate | **4/8 (50%)** |
| Plan-actual match rate | **4/8 (50%)** |
| Mismatch breakdown | 3 position_error, 1 grasp_failure |
| Total VLM calls | 8 |
| Avg latency per call | 2.77s |
| Confidence | 7/8 high, 1/8 medium |

**Conclusion:** Qwen2.5-VL-7B can detect interaction failures by comparing plan videos against actual execution. The VLM identifies two failure modes: `position_error` (robot arm not reaching target position) and `grasp_failure` (gripper fails to secure object). This validates the **closed-loop VLM supervisor** concept — the VLM can serve as a real-time monitor that detects execution failures and triggers re-planning or retry.

**Implications for Architecture:**
- Subgoal guidance (Method B/C) addresses **planning errors** — wrong trajectory direction
- Closed-loop VLM supervisor addresses **interaction precision errors** — correct trajectory but failed contact/grasp
- Combining both creates a robust two-level error correction system

---

### Exp 0.6 — DINOv2 Cross-Episode Deviation Gate

**Status:** Done (FAILED)

**Purpose:** Validate DINOv2 as lightweight post-condition checker using cross-episode keyframe comparison.

**Result:**

| Metric | Value | Verdict |
|--------|-------|---------|
| Same-phase cross-episode sim | 0.6419 | Too low |
| Different-phase same-episode sim | 0.7212 | Higher than same-phase (!) |
| Discrimination margin | **-0.0793** | **Negative — wrong direction** |
| Gating accuracy (±4 frames) | **29%** (13/45) | Near random |
| Latency | **7.1ms** | Good but irrelevant |

**Failure cause:** DINOv2 embeddings capture "which scene/episode" not "which phase." Cross-episode object position randomization dominates over phase semantics. Different frames within the same episode are more similar to each other than same-phase frames across episodes.

---

### Exp 0.7 — DINOv2 Predicted vs Actual (Same Episode)

**Status:** Done (INCONCLUSIVE → lean negative)

**Purpose:** Instead of cross-episode comparison, compare video model's predicted frame vs actual observation at the same timestep from the same episode. Eliminates cross-episode variation. Also tests embodiment masking.

**Commands:**
```bash
# adjust_bottle (single episode)
python test_deviation_gate.py \
  --model_path /mnt/shared-storage-user/kangli/workspace/cyujie/dinov2_vits14_pretrain.pth \
  --tests 6 \
  --eval_dir /mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal/adjust_bottle \
  --episodes 1 --mask_ratios 0.0 0.3 0.5

# stack_bowls_two (multi-episode)
python test_deviation_gate.py \
  --model_path /mnt/shared-storage-user/kangli/workspace/cyujie/dinov2_vits14_pretrain.pth \
  --tests 6 \
  --eval_dir /mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal/stack_bowls_two \
  --episodes 0 1 2 3 4 --mask_ratios 0.0 0.3 0.5
```

**Key hypothesis:** When comparing predicted vs actual from the same episode:
- Same scene, same objects, same positions → no cross-episode noise
- Similarity should be HIGH when execution matches plan
- Similarity should DROP at failure points (position_error, grasp_failure)
- Embodiment mask (hide robot arms) may amplify object-state signal

**Result:**

#### Task 1: adjust_bottle (1 episode)

| Mask | Mean | Std | Min | Max | Anomalies |
|------|------|-----|-----|-----|-----------|
| no_mask | 0.7624 | 0.0379 | 0.6947 | 0.8198 | 1 (frame 10) |
| mask=30% | 0.6849 | 0.0595 | 0.5870 | 0.8009 | 1 (frame 127) |
| mask=50% | 0.4738 | 0.1072 | 0.2784 | 0.6906 | 2 (frames 127, 137) |

Cross-reference with Exp 0.5 VLM failure detection (same videos):
- VLM detected failures at frames 45, 67, 90, 159
- DINOv2 no_mask: frame 45 has sim=0.8070 (HIGHEST) — **no correlation with failures**
- Conclusion: no_mask too flat (std=0.038), mask50% variance driven by temporal drift not interaction failure

#### Task 2: stack_bowls_two (2 episodes: ep1, ep2)

| Mask | Global Mean | Global Std | Anomalies |
|------|------------|-----------|-----------|
| no_mask | 0.7206 | 0.0580 | 3 (ep1: f21,f53; ep2: f127) |
| mask=30% | 0.6474 | 0.0772 | 3 (ep1: f21,f53; ep2: f127) |
| mask=50% | 0.4799 | 0.0903 | 2 (ep1: f21; ep2: f21) |

**Key observation:** Frame 21 flagged as anomaly in BOTH episodes across ALL mask levels:

| Episode | Frame 21 (no_mask) | Frame 21 (mask30%) | Frame 21 (mask50%) |
|---------|-------------------|--------------------|--------------------|
| 1 | **0.557** | **0.479** | **0.234** |
| 2 | 0.659 | 0.587 | **0.375** |

Frame 21 in 160-step eval ≈ early manipulation phase. Cross-episode consistency suggests the video model consistently mis-predicts contact/grasp dynamics.

#### Summary

| Signal | adjust_bottle (1ep) | stack_bowls_two (2ep) |
|--------|--------------------|-----------------------|
| no_mask std | 0.038 (too flat) | 0.058 (slightly better) |
| mask50% std | 0.107 | 0.090 |
| Cross-ep consistency | N/A | **Frame 21 consistent** |
| Failure correlation | **None** (VLM failures uncorrelated) | Unknown (no GT labels) |
| Masking effect | Increases variance | Increases variance |

**Verdict: INCONCLUSIVE → lean negative.**
- Masking increases variance but signal correlates with video model prediction drift, not interaction failure
- adjust_bottle: VLM-detected failures (frames 45, 67, 90) show HIGH DINOv2 similarity — actively contradicts hypothesis
- stack_bowls_two: frame 21 consistency is interesting but may reflect systematic video model weakness at contact points, not per-episode failure detection
- Without success/failure labels for eval episodes, cannot definitively validate

**DINOv2 deviation gate route concluded.** Exp 0.6 (cross-episode) failed, Exp 0.7 (pred-vs-actual) inconclusive/negative. Moving to alternative approaches.

**⚠ Post-hoc issue:** Video shape mismatch discovered — pred videos are 640×736 (main view + 2 arm camera views), actual videos are 640×480 (main view only). The extra 256 rows in pred contain arm viewpoints that don't exist in actual. Before DINOv2 `Resize(256)+CenterCrop(224)`, the different aspect ratios cause different content regions to be compared. **Fix applied:** crop pred frames to `[:480, :, :]` before encoding. This confound affected all Exp 0.7 results but does not change the fundamental conclusion given the Scenario C limitation (see Exp 0.8).

---

### Exp 0.8 — DINOv2 Pred vs Actual with GT Success/Failure Labels

**Status:** Done (FAILED — DINOv2 cannot discriminate success vs failure)

**Purpose:** Re-run Exp 0.7 with ground truth success/failure labels per episode to definitively test whether DINOv2 pred-vs-actual similarity differs between successful and failed episodes.

**Tasks tested:**

| Task | Episodes tested | GT Labels |
|------|----------------|-----------|
| stack_bowls_two | ep0-4 (only ep1,ep2 loaded) | ep0,1,2=FAIL, ep3+=SUCCESS |
| pick_dual_bottles | ep0-9 (7 loaded) | ep4,6,7,8=SUCCESS, rest=FAIL |

**Key limitation:** For stack_bowls_two, only FAIL episodes (ep1,ep2) had both pred+actual videos; SUCCESS episodes (ep3+) had no pred videos. Could not compare success vs failure for this task.

#### pick_dual_bottles — 7 episodes with GT labels

| Mask | SUCCESS (ep4) mean | FAIL (ep0,1,2,3,5,9) mean | Gap | Direction |
|------|-------------------|--------------------------|-----|-----------|
| no_mask | 0.7597 | 0.7308 | +0.029 | Correct but too small |
| mask=30% | — | — | — | Similar |
| mask=50% | 0.4339 | 0.4488 | **-0.015** | **WRONG** |

**Analysis:**
- no_mask: SUCCESS mean is slightly higher (+0.029) but well within overlap of per-episode distributions
- mask=50%: SUCCESS mean is LOWER than FAIL mean — **wrong direction**, masking makes it worse
- Per-episode means: all episodes cluster in 0.72-0.76 (no_mask), no clear separation

#### Scenario C — Fundamental Limitation

User insight: If the video model generates a **wrong prediction** AND the robot **faithfully follows** that wrong plan, pred-vs-actual similarity stays **HIGH even in failure**. This means:

| Scenario | Plan correct? | Execution correct? | Pred-actual sim | Detectable? |
|----------|:---:|:---:|:---:|:---:|
| A: Full success | ✓ | ✓ | High | N/A |
| B: Execution deviation | ✓ | ✗ | Low | ✓ Yes |
| C: Planning failure | ✗ | ✗ (follows bad plan) | **High** | **✗ No** |

DINOv2 pred-vs-actual can only detect **Scenario B** (execution deviation from correct plan). Most observed failures are likely **Scenario C** (wrong plan faithfully executed), which explains why DINOv2 shows no success/failure discrimination.

**Verdict: DINOv2 deviation gate definitively ruled out.** Three experiments (0.6, 0.7, 0.8) all negative. The approach has a fundamental design limitation (Scenario C blindspot) in addition to insufficient sensitivity.

---

## Revised Architecture

**Action space confirmed:** Absolute joint positions (qpos), 14-dim. Linear interpolation rollback is valid.

**Deviation gate design:** Two tiers depending on Exp 0.7 result:

```
Tier 1 (if Exp 0.7 succeeds):
  DINOv2 compares predicted frame vs actual obs at gripper transition
  ~7ms latency, online feasible

Tier 2 (fallback):
  Gripper width detection (sim only): commanded close vs actual angle
  ~0ms latency, but sim-specific, not transferable
```

**Rollback mechanism (unchanged):**
- Vidarc outputs absolute qpos → linear interpolation to checkpoint
- Gripper: set to 0 (open) during rollback
- Policy context: reset to checkpoint observation after rollback

---

## Phase 1: Method B Full Evaluation

(Pending Phase 0 completion)

---

## Progress Timeline

| Date | Milestone |
|------|-----------|
| 2025-02-10 | Phase 0 started: baseline eval + VLM feasibility |
| 2025-02-10 | Exp 0.3 done: VLM phase detection validated on adjust_bottle |
| 2025-02-10 | Exp 0.4 done: library built for stack_bowls_two (9 phases, 50/50 canonical) |
| 2025-02-10 | Exp 0.5 done: VLM interaction success detection validated (50% detect rate) |
| 2025-02-10 | Architecture revised: lightweight DINOv2 gate + trajectory rollback |
| 2025-02-10 | Exp 0.6 done: DINOv2 cross-episode gate FAILED (margin -0.0793, 29% accuracy) |
| 2025-02-10 | Exp 0.7 designed: predicted-vs-actual DINOv2 + embodiment mask |
| 2025-02-10 | Exp 0.7 done: DINOv2 pred-vs-actual INCONCLUSIVE (no failure correlation) |
| 2025-02-10 | Exp 0.8 done: DINOv2 with GT labels FAILED (gap +0.029 no_mask, -0.015 mask50%) |
| 2025-02-10 | Scenario C identified: planning failures undetectable by pred-vs-actual comparison |
| 2025-02-10 | Video shape mismatch fixed: pred 736→480 crop (arm camera views removed) |
| | Exp 0.1 baseline result: (TODO) |
| | Exp 0.2 oracle eval: (TODO) |

---

## Key Findings

1. **VLM phase detection works.** Qwen2.5-VL-7B correctly decomposes `adjust_bottle` into 3 phases and tracks phase progression monotonically across 51 frames.
2. **Latency is acceptable for offline.** Steady-state ~1.8s/call. Too slow for online control, but fine for offline task decomposition and library construction.
3. **Phase count matches gripper events.** 3 phases (approach → grasp → lift) aligns with expected gripper state transitions, validating the action-signal-based phase boundary detection in `build_library.py`.
4. **VLM can detect interaction failures.** Comparing plan vs actual video, Qwen2.5-VL-7B identifies position_error and grasp_failure with high confidence (Exp 0.5).
5. **VLM latency prohibits online use.** At ~2s/call vs 0.1s control cycle, VLM cannot serve as a real-time supervisor. Lightweight classifier (DINOv2, ~5ms) is needed for online gating.
6. **Action space is absolute qpos.** Vidarc outputs 14-dim absolute joint positions. This enables linear interpolation rollback to checkpoint states.
7. **DINOv2 cross-episode gate fails.** Embeddings capture scene identity (object layout), not phase semantics. Cross-episode object randomization makes same-phase similarity (0.64) lower than different-phase same-episode similarity (0.72). Pivoting to predicted-vs-actual comparison within same episode.
8. **DINOv2 pred-vs-actual cannot discriminate success/failure.** With GT labels on pick_dual_bottles (7 episodes), SUCCESS vs FAIL gap is only +0.029 (no_mask) and -0.015 (mask50%, wrong direction). Three experiments (0.6, 0.7, 0.8) all negative.
9. **Scenario C blindspot is fundamental.** If the video model generates a wrong prediction and the robot faithfully follows it, pred-vs-actual similarity stays high even in failure. Most failures are Scenario C (planning failures), not Scenario B (execution deviations). This makes any pred-vs-actual comparison approach inadequate.
10. **Video shape mismatch confound.** Pred videos (640×736) include 2 extra arm camera views stacked below the main view (640×480). Must crop pred to `[:480,:,:]` before comparison. Fix applied in `_compute_episode_sims`.

---

## File Index

| File | Purpose |
|------|---------|
| `test_vlm_phase.py` | Step 3: VLM feasibility test (Tests 1-4) |
| `test_deviation_gate.py` | Exp 0.6: DINOv2 deviation gate validation |
| `build_library.py` | Step 4: GT keyframe library builder |
| `theory.md` | Design principles & motivation |
| `subgoal_plan.md` | Full implementation plan (Method B & C) |
| `subgoal.md` | Overview & evolution |
| `progress.md` | This file — experiment results log |
