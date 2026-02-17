# Experiment: TI2V Model Keyframe Generation Quality

**Goal:** Evaluate whether the existing Wan2.2-TI2V-5B model can generate visually plausible event keyframes given (first frame + text description), and how they compare to GT keyframes extracted by rules.

**Scope:** Use the model as-is (no training). Measure generation quality, NOT downstream task success.

**Depends on:** `subgoal_keyframe_rule.md` (need GT keyframes for comparison)

---

## 1. Problem Statement

The subgoal system (Model A) needs to generate future keyframe images. Before building a planning head, we can test whether the existing TI2V model already produces useful subgoal images:

1. Given first frame + "the robot grasps bowl A", does the model generate a plausible grasp state?
2. Are the generated images detailed enough to serve as visual targets for Φ-guidance?
3. Where does the model fail — wrong object, wrong pose, hallucinated geometry?

This experiment reveals the **generation quality ceiling** of our video model and identifies which failure modes a future planning head must avoid.

---

## 2. Setup

### Model

```
Wan2.2-TI2V-5B (Text-Image-to-Video)
Checkpoint: /mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B
Output: 17 frames per generation (4n+1)
Default: sampling_steps=30, guide_scale=5.0
```

### Existing Code

```
experiments/key_frame_wan/stack_bowl_two/generate_subgoals.py  → WanSubgoalGenerator class
experiments/key_frame_wan/stack_bowl_two/subgoals.json         → prompt templates
```

### Subgoal Prompt Format

Each subgoal is defined by a semantic image description:

```json
{
  "subgoal_id": 1,
  "subgoal_name": "bowl_A_grasped",
  "semantic_image_description": "The robot gripper is holding bowl {A}, lifted above the table surface. Bowl {B} remains on the table in its original position.",
  "key_conditions": [
    "Bowl {A} is gripped by the robot",
    "Bowl {A} is elevated from the table",
    "Bowl {B} is stationary on the table"
  ]
}
```

---

## 3. Experiment Design

### 3.1 Variables

| Variable | Values | Notes |
|----------|--------|-------|
| Task | Same 5 episodes from `subgoal_keyframe_rule.md` | Control: same inputs |
| First frame | From HDF5 `observations/unified_image[0]` | Input to model |
| Subgoal prompt | 3 subgoals per task (see below) | Semantic descriptions |
| Guide scale | {3.0, 5.0, 7.0} | CFG strength ablation |
| Seed | {42, 123, 456} | 3 seeds per config for variance |

### 3.2 Subgoal Prompts per Task

For each of the 5 test tasks, define 3 subgoals that match the rule-extracted GT events:

**Example (stack_bowl_two):**

| Subgoal | Prompt | Corresponding GT Event |
|---------|--------|------------------------|
| sg1 | "The robot gripper is holding bowl A, lifted above the table" | gripper_change: grasp event |
| sg2 | "Bowl A is held above bowl B, both bowls vertically aligned" | action_milestone: transport end |
| sg3 | "Bowl A rests on top of bowl B, gripper released" | gripper_change: release event |

**Example (adjust_bottle):**

| Subgoal | Prompt | Corresponding GT Event |
|---------|--------|------------------------|
| sg1 | "The robot gripper approaches the bottle on the table" | action_milestone: approach end |
| sg2 | "The robot gripper holds the bottle, slightly lifted" | gripper_change: grasp |
| sg3 | "The bottle is placed at the target position, gripper open" | gripper_change: release |

For remaining 3 tasks: write similar 3-subgoal sequences.

### 3.3 Generation Procedure

For each (episode, subgoal, guide_scale, seed):

```python
generator = WanSubgoalGenerator(checkpoint_dir, device_id=0)
generator.load_model()

# Generate 17-frame video
video = generator.generate_subgoal_video(
    first_frame=first_frame,               # from HDF5
    subgoal_description=prompt,            # semantic description
    frame_num=17,
    sampling_steps=30,
    guide_scale=guide_scale,
    seed=seed
)

# Extract last frame as subgoal image
subgoal_image = generator.extract_subgoal_frame(video, frame_idx=-1)
```

**Total generations:** 5 episodes x 3 subgoals x 3 guide_scales x 3 seeds = **135 generations**

(If compute is limited, fix guide_scale=5.0 and seed=42 first → **15 generations** for initial check)

---

## 4. Visualizations

### Vis 1: Generated vs GT Keyframe Side-by-Side

**Purpose:** Directly compare model output with the rule-extracted GT frame for the same event.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│ Episode 2: stack_bowl_two                                        │
│                                                                  │
│ Subgoal 1: "bowl_A_grasped"                                     │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│ │  First Frame  │  │  GT (f30)    │  │  Generated   │            │
│ │  (input)      │  │  (rule)      │  │  (TI2V)      │            │
│ └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                  │
│ Subgoal 2: "bowl_A_positioned_above_B"                           │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│ │  First Frame  │  │  GT (f50)    │  │  Generated   │            │
│ │  (input)      │  │  (rule)      │  │  (TI2V)      │            │
│ └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                  │
│ Subgoal 3: "bowl_A_stacked_on_B"                                 │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│ │  First Frame  │  │  GT (f63)    │  │  Generated   │            │
│ │  (input)      │  │  (rule)      │  │  (TI2V)      │            │
│ └──────────────┘  └──────────────┘  └──────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

**What to check:**
- Does the generated image show the described state (bowl grasped, stacked, etc.)?
- Are objects in the correct positions relative to GT?
- Are there hallucinated objects or missing objects?
- Is the robot arm in a physically plausible pose?

---

### Vis 2: Generated Video Strip (All 17 Frames)

**Purpose:** See the full transition the model imagines, not just the last frame.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│ Subgoal: "bowl_A_grasped" (guide_scale=5.0, seed=42)            │
│                                                                  │
│ [f0] [f1] [f2] [f3] [f4] [f5] [f6] [f7] [f8]                  │
│ [f9] [f10] [f11] [f12] [f13] [f14] [f15] [f16]                 │
│                                                                  │
│ Questions:                                                       │
│ - Does the motion sequence look physically plausible?            │
│ - At which frame does the grasp actually happen?                 │
│ - Is the last frame (f16) the best subgoal, or is an            │
│   intermediate frame more suitable?                              │
└──────────────────────────────────────────────────────────────────┘
```

**What to check:**
- Is the generated motion smooth and physically plausible?
- Does the event (grasp/release) happen at the right point in the 17-frame sequence?
- Is extracting the last frame always optimal, or should we pick a specific frame?

---

### Vis 3: Guide Scale Comparison (Same Prompt, Different CFG)

**Purpose:** Find the right CFG strength for subgoal generation.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│ Subgoal: "bowl_A_grasped" — Guide Scale Comparison               │
│                                                                  │
│           guide_scale=3.0    guide_scale=5.0    guide_scale=7.0  │
│ seed=42   [image]            [image]            [image]          │
│ seed=123  [image]            [image]            [image]          │
│ seed=456  [image]            [image]            [image]          │
│                                                                  │
│ GT (f30): [image]                                                │
│                                                                  │
│ Observation:                                                     │
│ - Low CFG (3.0): more diverse but less faithful to prompt?       │
│ - High CFG (7.0): more prompt-following but possible artifacts?  │
│ - Which CFG gives best visual match to GT?                       │
└──────────────────────────────────────────────────────────────────┘
```

**What to check:**
- Does higher CFG produce images closer to GT?
- Does higher CFG cause visual artifacts (over-saturation, unnatural poses)?
- How much variance across seeds? (high variance → model is uncertain about this subgoal)

---

### Vis 4: Failure Case Gallery

**Purpose:** Catalog the model's failure modes to understand what a planning head must fix.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│ Failure Gallery                                                  │
│                                                                  │
│ Failure Type 1: WRONG OBJECT                                     │
│ Prompt: "robot holds bowl A"                                     │
│ [generated image]  ← model moved bowl B instead                  │
│                                                                  │
│ Failure Type 2: PHYSICALLY IMPOSSIBLE POSE                       │
│ Prompt: "bowl A stacked on bowl B"                               │
│ [generated image]  ← bowl floating in mid-air, no contact        │
│                                                                  │
│ Failure Type 3: HALLUCINATED GEOMETRY                            │
│ Prompt: "gripper released, bowl on table"                        │
│ [generated image]  ← extra object appeared, table deformed       │
│                                                                  │
│ Failure Type 4: CORRECT SEMANTICS, WRONG DETAILS                │
│ Prompt: "bowl A above bowl B"                                    │
│ [generated image]  ← roughly correct but position off by much    │
│                                                                  │
│ Failure Type 5: NO CHANGE                                        │
│ Prompt: "robot approaches bottle"                                │
│ [generated image]  ← identical to first frame, nothing happened  │
└──────────────────────────────────────────────────────────────────┘
```

**What to check:**
- Which failure types are most common?
- Are failures prompt-dependent (some descriptions work, others don't)?
- Are failures task-dependent (some tasks easier to imagine than others)?

---

### Vis 5: Latent Distance Analysis (Generated vs GT in Encoder Space)

**Purpose:** Even if the generated image doesn't look identical to GT pixel-wise, it might be close enough in the encoder's latent space (which is what Φ actually uses).

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│ Latent Space Distance: Generated Subgoals vs GT Keyframes        │
│                                                                  │
│ Table:                                                           │
│ Episode  │ Subgoal  │ d(gen, GT)  │ d(first, GT)  │ Ratio       │
│ ─────────┼──────────┼─────────────┼───────────────┼─────────────│
│ ep1      │ sg1      │    0.42     │     1.85      │ 0.23 ✅     │
│ ep1      │ sg2      │    0.78     │     2.10      │ 0.37 ✅     │
│ ep1      │ sg3      │    1.65     │     2.30      │ 0.72 ⚠️     │
│ ep2      │ sg1      │    0.35     │     1.90      │ 0.18 ✅     │
│ ...      │ ...      │    ...      │     ...       │ ...         │
│                                                                  │
│ d(gen, GT):    ||Enc(generated) - Enc(GT_keyframe)||             │
│ d(first, GT):  ||Enc(first_frame) - Enc(GT_keyframe)||          │
│ Ratio < 0.5:  generated is closer to GT than starting point ✅  │
│ Ratio > 0.8:  generated is barely closer than start ⚠️          │
│                                                                  │
│ Scatter plot: d(gen, GT) vs d(first, GT) for all subgoals       │
│   Points below y=x line: model helps                             │
│   Points above y=x line: model makes things worse                │
└──────────────────────────────────────────────────────────────────┘
```

**What to check:**
- Are generated images consistently closer to GT than the first frame in latent space?
- Which subgoals are easy vs hard for the model? (early subgoals easier?)
- If ratio > 0.8 consistently → TI2V model cannot generate useful subgoals for this task

**Implementation:** Requires loading Model B's encoder to compute latent distances. If encoder is unavailable, skip this visualization and note it as future work.

---

## 5. Evaluation Metrics

### Pixel-Level Metrics (Easy to Compute, Noisy)

| Metric | Formula | Use |
|--------|---------|-----|
| PSNR | `10 * log10(MAX^2 / MSE)` | Overall image quality |
| SSIM | structural similarity | Perceptual similarity |
| LPIPS | learned perceptual distance | Better correlates with human judgment |

### Semantic Metrics (What We Actually Care About)

| Metric | Method | Use |
|--------|--------|-----|
| Object presence | Manual check or detector | Does the described object exist? |
| Object position | Bounding box IoU with GT | Is the object in the right place? |
| Gripper state match | Binary: open/closed matches description? | Is the key condition met? |
| Physical plausibility | Manual rating 1-5 | Is the generated pose physically possible? |

### Latent Metrics (If Encoder Available)

| Metric | Formula | Use |
|--------|---------|-----|
| Latent distance ratio | `d(gen, GT) / d(first, GT)` | Is generation closer to GT than start? |
| Latent direction alignment | `cos(gen - first, GT - first)` | Does generation move toward GT in latent space? |

---

## 6. Pass/Fail Criteria

| Check | Pass | Fail | Implication |
|-------|------|------|-------------|
| Visual plausibility | >60% of generated subgoals show correct event (manual review) | <40% | TI2V cannot serve as Model A → need planning head (Method 1) |
| Object consistency | Generated image preserves objects from first frame (no hallucination) in >70% cases | <50% | Model hallucinates → need latent-space planning, not pixel-space |
| Latent distance ratio | Ratio < 0.5 for >60% of subgoals | Ratio > 0.8 for >40% | Even if pixels are wrong, latent representation might still be useful for Φ |
| CFG sensitivity | Clear optimal CFG (one value is best across tasks) | All CFG values equally bad | Prompt engineering problem, not model capability |
| Cross-seed variance | Low variance across seeds (consistent generation) | High variance (model is uncertain) | Task is ambiguous for the model → need better prompts or constrained generation |

---

## 7. Output Structure

```
experiments/subgoal/results/model/
├── vis1_gen_vs_gt_ep{1-5}.png            ← generated vs GT side-by-side
├── vis2_video_strip_ep{1-5}_sg{1-3}.png  ← full 17-frame sequences
├── vis3_cfg_comparison_ep{1-5}_sg{1-3}.png ← guide scale ablation
├── vis4_failure_gallery.png              ← curated failure cases
├── vis5_latent_distance.png              ← scatter plot + table
├── generation_log.json                   ← raw data
├── generated_images/                     ← all generated subgoal images
│   ├── ep1_sg1_cfg5.0_seed42.jpg
│   ├── ep1_sg1_cfg5.0_seed42_video/      ← all 17 frames
│   └── ...
└── summary.md                            ← fill in after review
```

**generation_log.json schema:**
```json
{
  "episode_id": "episode_000001",
  "task": "stack_bowl_two",
  "subgoal_id": 1,
  "subgoal_name": "bowl_A_grasped",
  "prompt": "The robot gripper is holding bowl A...",
  "guide_scale": 5.0,
  "seed": 42,
  "gt_frame_index": 30,
  "metrics": {
    "ssim_vs_gt": 0.45,
    "lpips_vs_gt": 0.32,
    "latent_distance_to_gt": 0.42,
    "latent_distance_first_to_gt": 1.85,
    "latent_ratio": 0.23,
    "manual_plausibility_score": 4,
    "manual_object_correct": true,
    "manual_gripper_state_correct": true
  }
}
```

---

## 8. Execution Checklist

```
[ ] Write 3 subgoal prompts per task (5 tasks x 3 subgoals = 15 prompts)
[ ] Load Wan2.2-TI2V-5B model on GPU
[ ] Run initial 15 generations (5 episodes x 3 subgoals, cfg=5.0, seed=42)
[ ] Generate Vis 1 and Vis 2 → quick sanity check
[ ] If initial results look reasonable:
    [ ] Run full 135 generations (3 cfgs x 3 seeds)
    [ ] Generate Vis 3 (CFG comparison)
[ ] Curate Vis 4 (failure gallery) from all generations
[ ] If encoder is available: compute Vis 5 (latent distances)
[ ] Fill in summary.md with findings
```

---

## 9. Connection to Other Experiments

**Feeds into:**
- **subgoal_plan.md → Method 2 (MPC):** If TI2V generates decent subgoals, we can use it directly as Model A for the MPC baseline
- **subgoal_plan.md → Method 1:** If TI2V fails, it confirms we need a trained planning head in latent space (not pixel space)
- **subgoal_troubleshooting.md → Problem 3:** The latent distance analysis (Vis 5) provides early data for diagnosing latent space misalignment

**Depends on:**
- **subgoal_keyframe_rule.md:** GT keyframes from rule-based extraction are used as comparison targets
- GT keyframes must be extracted and validated BEFORE running this experiment

**Key decision this experiment informs:**

```
IF TI2V quality is high (>60% plausible):
  → Use TI2V as Model A directly
  → Jump to Exp 1.2 (Oracle) with TI2V-generated subgoals
  → Compare TI2V subgoals vs GT keyframes

IF TI2V quality is medium (30-60% plausible):
  → TI2V provides useful signal but needs refinement
  → Use TI2V as initialization for planning head training
  → Or: use TI2V for coarse subgoals + rule-based for fine subgoals

IF TI2V quality is low (<30% plausible):
  → Pixel-space generation is not viable for this task domain
  → Build planning head in latent space (Method 1)
  → TI2V failure modes inform planning head architecture
```
