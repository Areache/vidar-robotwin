# Experiment: Rule-Based Keyframe Extraction from Dataset

**Goal:** Extract event keyframes from HDF5 demonstrations using rule-based strategies, and visually verify they capture the right task-critical moments.

**Scope:** No model involved. Pure signal processing on actions/pixels. 5 visualizations for sanity check.

---

## 1. Problem Statement

We need subgoal targets extracted from demonstration data. Before building any model, we must answer:

1. Can we reliably detect task-critical events (grasp, release, phase transitions) from the raw data?
2. Which signal (gripper state, action velocity, pixel change) best captures task semantics?
3. How many keyframes does each strategy produce, and are they at the right moments?

If we can't extract meaningful keyframes from demonstrations, there's no foundation for the subgoal system.

---

## 2. Data Source

**HDF5 demonstrations:**
```
path: /mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/datasets/robotwin/processed/hdf5
```

**Per-episode structure:**
```
episode_XXXXXX.hdf5
├── observations/
│   ├── unified_image:     (T, 720, 640, 3) uint8   ← use this for visualization
│   ├── images/cam_high:   (T, H, W, 3)
│   └── qpos:              (T, state_dim)
├── action:                (T, 14)                   ← indices 6,13 = gripper states
└── instruction (attr):    string
```

**Eval videos (for comparison):**
```
path: eval_result/ar/ddp_causal/{task}/episode{N}.mp4
```

**Existing extraction code:**
```
experiments/gt_keyframe_test/extract_keyframes.py
```
Implements: `uniform`, `visual_change`, `gripper_change`, `action_milestone`, `semantic`, `from_hdf5`

---

## 3. Extraction Strategies to Test

| # | Strategy | Signal | Key Params | Expected Output |
|---|----------|--------|------------|-----------------|
| A | `uniform` | fixed interval | `interval=8` | Evenly spaced, ~10 keyframes per episode |
| B | `visual_change` | pixel MSE between consecutive frames | `threshold=0.05`, `min_interval=4` | Clusters at visually active moments |
| C | `gripper_change` | `action[6]` and `action[13]` delta | `threshold=0.3`, `min_interval=4` | 2-5 keyframes at grasp/release events |
| D | `action_milestone` | action velocity (excl. gripper dims) | `velocity_threshold=0.1`, `min_interval=8` | Phase transitions (start/stop moving) |
| E | `semantic` | motion stops + visual change (2-pass) | `motion_threshold=0.01`, `change_threshold=0.02` | Combined: pauses + visual events |

---

## 4. Experiment Steps

### Step 1: Pick 5 Episodes (Cover Task Diversity)

Select 5 episodes that cover different task structures:

| Episode | Task | Why This Episode |
|---------|------|------------------|
| ep_1 | single-arm pick-and-place (e.g. `adjust_bottle`) | Simplest: 1 grasp, 1 release |
| ep_2 | dual-arm task (e.g. `stack_bowl_two`) | Multiple grasps, sequential phases |
| ep_3 | long-horizon task | Tests if strategies handle longer trajectories |
| ep_4 | task with fine manipulation | Tests sensitivity to small but critical motions |
| ep_5 | failed/edge-case episode (if available) | Tests robustness to unexpected dynamics |

### Step 2: Run All 5 Strategies on Each Episode

For each (episode, strategy) pair:
1. Extract keyframes
2. Record: number of keyframes, frame indices, time distribution
3. Save visualization

```bash
# Example for one episode, one strategy:
python experiments/gt_keyframe_test/extract_keyframes.py \
    /path/to/episode_000001.hdf5 \
    --hdf5 \
    --strategy gripper \
    --visualize \
    --vis-output experiments/subgoal/results/rule/ep1_gripper.jpg
```

### Step 3: Generate 5 Visualizations

---

## 5. Visualizations

### Vis 1: Full Video Strip + Keyframe Overlay (per episode)

**Purpose:** See where each strategy places keyframes on the original video timeline.

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│ Episode 1: adjust_bottle (T=81 frames)                  │
│                                                         │
│ Original video (every 5th frame, small thumbnails):     │
│ [f0] [f5] [f10] [f15] [f20] [f25] [f30] ... [f75] [f80]│
│                                                         │
│ Strategy A (uniform):      ▼        ▼        ▼        ▼│
│ Strategy B (visual):    ▼     ▼▼          ▼      ▼     │
│ Strategy C (gripper):              ▼▼            ▼▼    │
│ Strategy D (milestone):    ▼         ▼       ▼    ▼    │
│ Strategy E (semantic):     ▼    ▼▼   ▼       ▼   ▼▼   │
│                                                         │
│ Timeline: [0 ──────────────────────────────────────── T]│
└─────────────────────────────────────────────────────────┘
```

**What to check:**
- Do gripper_change markers align with actual grasp/release in the video?
- Does visual_change fire on task-relevant events or on noise?
- Are there long gaps where no strategy places a keyframe?

**Generate with:**
- Top row: `visualize_keyframes()` for sampled video frames
- Bottom rows: `visualize_keyframes_timeline()` per strategy, stacked vertically

---

### Vis 2: Extracted Keyframes Side-by-Side (5 strategies x 1 episode)

**Purpose:** Directly compare which frames each strategy selects.

**Layout:**
```
┌───────────────────────────────────────────────────────────────┐
│ Episode 2: stack_bowl_two                                     │
│                                                               │
│ uniform:     [f0]  [f8]  [f16] [f24] [f32] [f40] [f48]       │
│ visual:      [f0]  [f12] [f14] [f30] [f31] [f55]             │
│ gripper:     [f0]  [f29] [f30] [f62] [f63]                   │
│ milestone:   [f0]  [f10] [f28] [f35] [f60] [f68]             │
│ semantic:    [f0]  [f10] [f29] [f30] [f55] [f62] [f63]       │
│                                                               │
│ (each cell shows the actual extracted image at that frame)    │
└───────────────────────────────────────────────────────────────┘
```

**What to check:**
- Are the gripper_change frames visually showing a grasp/release event?
- Does uniform miss critical frames that other strategies catch?
- Does semantic capture a superset of gripper + milestone events?

---

### Vis 3: Gripper Signal + Action Velocity Curve (per episode)

**Purpose:** Understand the raw signals that strategies threshold on.

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│ Episode 3: long-horizon task                         │
│                                                      │
│ Plot 1: Gripper state over time                      │
│   1.0 ─────┐              ┌─────┐              ┌────│
│             │              │     │              │    │
│   0.0       └──────────────┘     └──────────────┘    │
│         0    20    40    60    80   100   120   140   │
│              ▲              ▲     ▲              ▲   │
│              grasp          rel   grasp          rel │
│                                                      │
│ Plot 2: Action velocity (norm, excl. gripper dims)   │
│   0.5  ╱╲        ╱╲           ╱╲        ╱╲          │
│        │  ╲      │  ╲         │  ╲      │  ╲        │
│   0.0 ─┘   ╲────┘   ╲───────┘   ╲────┘   ╲────    │
│              ▲         ▲          ▲         ▲       │
│              stop      stop       stop      stop    │
│                                                      │
│ Plot 3: Pixel MSE between consecutive frames         │
│   0.1        ╱╲   ╱╲                ╱╲   ╱╲        │
│              │ ╲  │  ╲              │ ╲  │  ╲      │
│   0.0 ───────┘  ╲─┘   ╲────────────┘  ╲─┘   ╲──── │
│                                                      │
│ Bottom: Keyframe markers per strategy                │
│   gripper:     ▼         ▼          ▼         ▼     │
│   milestone:  ▼  ▼      ▼  ▼      ▼  ▼      ▼  ▼  │
│   visual:       ▼▼▼       ▼▼         ▼▼▼      ▼▼   │
└──────────────────────────────────────────────────────┘
```

**What to check:**
- Do gripper_change markers align with actual gripper state transitions?
- Do milestone markers align with velocity zero-crossings?
- Is the pixel MSE signal noisy or clean? Does visual_change threshold make sense?

**Implementation:** Use matplotlib. Plot 3 subplots + keyframe markers as vertical lines.

---

### Vis 4: Critical Event Zoom (Grasp/Release Moments)

**Purpose:** Close-up of the 3-5 frames around each detected event to verify the exact frame is correct.

**Layout:**
```
┌──────────────────────────────────────────────────────────────┐
│ Episode 2: stack_bowl_two — Event Zoom                       │
│                                                              │
│ Event 1: GRASP detected at frame 30                          │
│ [f27]  [f28]  [f29]  [*f30*]  [f31]  [f32]  [f33]          │
│                        ▲ gripper closes here                 │
│ action[6]: 1.0   1.0   0.8   [0.2]   0.0    0.0    0.0     │
│                                                              │
│ Event 2: RELEASE detected at frame 63                        │
│ [f60]  [f61]  [f62]  [*f63*]  [f64]  [f65]  [f66]          │
│                        ▲ gripper opens here                  │
│ action[6]: 0.0   0.0   0.1   [0.8]   1.0    1.0    1.0     │
│                                                              │
│ Event 3: PHASE TRANSITION (approach → grasp) at frame 28     │
│ [f25]  [f26]  [f27]  [*f28*]  [f29]  [f30]  [f31]          │
│                        ▲ velocity drops to ~0                │
│ velocity: 0.3   0.2   0.05  [0.01]  0.02   0.01   0.00     │
└──────────────────────────────────────────────────────────────┘
```

**What to check:**
- Is the detected frame exactly at the event, or off by a few frames?
- Is the gripper threshold (0.3) correct, or does it miss gradual open/close?
- Do consecutive events have enough separation (min_interval)?

---

### Vis 5: Cross-Task Keyframe Statistics

**Purpose:** Compare extraction behavior across all 5 tasks (aggregate view).

**Layout:**
```
┌──────────────────────────────────────────────────────────────┐
│ Keyframe Extraction Statistics Across Tasks                  │
│                                                              │
│ Table:                                                       │
│ Task              │ T    │ uniform│ visual│ gripper│ mile │ sem│
│ ──────────────────┼──────┼────────┼───────┼────────┼──────┼────│
│ adjust_bottle     │  81  │   10   │   7   │   2    │   4  │  5 │
│ stack_bowl_two    │  81  │   10   │   9   │   4    │   6  │  7 │
│ long_horizon_task │ 150  │   18   │  12   │   4    │   8  │  9 │
│ fine_manipulation │  81  │   10   │  14   │   6    │   5  │  8 │
│ edge_case         │  60  │    7   │   5   │   2    │   3  │  4 │
│                                                              │
│ Bar chart: # keyframes per strategy (grouped by task)        │
│                                                              │
│ Histogram: Inter-keyframe gap distribution per strategy      │
│   uniform:    [||||||||||||]  all gaps = 8 (by design)       │
│   gripper:    [||   |||||||||||||||   ||]  bimodal: short+long│
│   milestone:  [|||||||||      |||||]  moderate spread        │
│                                                              │
│ Key question: Does gripper_change have >30 frame gaps?       │
│   If yes → needs infill (composite strategy)                 │
└──────────────────────────────────────────────────────────────┘
```

**What to check:**
- Does gripper_change consistently produce 2-5 keyframes (as expected)?
- Are there tasks where visual_change produces too many keyframes (noisy)?
- Do gap distributions reveal tasks needing composite strategies?

---

## 6. Pass/Fail Criteria

| Check | Pass | Fail | Action on Fail |
|-------|------|------|----------------|
| Gripper events detected | Keyframe is within 2 frames of actual gripper state change | Misses event or fires on noise | Adjust `threshold` (try 0.1, 0.2, 0.5) |
| Phase transitions detected | milestone markers at approach/grasp/transport/place boundaries | Misses transitions or fires mid-motion | Adjust `velocity_threshold` |
| Visual change not too noisy | <15 keyframes per episode, clustered at task events | >20 keyframes or scattered randomly | Increase `threshold`, increase `min_interval` |
| Semantic is superset | semantic keyframes ⊇ gripper ∪ milestone (up to 2-frame tolerance) | semantic misses events found by other strategies | Adjust thresholds in semantic 2-pass |
| Gap coverage | No strategy leaves >30 consecutive frames without a keyframe | Long gaps in gripper/milestone strategies | Confirms need for composite strategy |

---

## 7. Output Structure

```
experiments/subgoal/results/rule/
├── vis1_timeline_ep{1-5}.png         ← video strip + strategy overlays
├── vis2_sidebyside_ep{1-5}.png       ← extracted frames comparison
├── vis3_signals_ep{1-5}.png          ← gripper/velocity/MSE curves
├── vis4_event_zoom_ep{1-5}.png       ← close-up of detected events
├── vis5_cross_task_stats.png         ← aggregate statistics
├── extraction_log.json               ← raw data for all (episode, strategy) pairs
└── summary.md                        ← fill in after running experiments
```

**extraction_log.json schema:**
```json
{
  "episode_id": "episode_000001",
  "task": "stack_bowl_two",
  "total_frames": 81,
  "strategies": {
    "uniform":    {"n_keyframes": 10, "frame_indices": [0,8,16,...], "params": {"interval": 8}},
    "gripper":    {"n_keyframes": 4,  "frame_indices": [0,29,30,63], "params": {"threshold": 0.3}},
    "...": "..."
  }
}
```

---

## 8. Execution Checklist

```
[ ] Identify 5 HDF5 episodes (cover task diversity)
[ ] Verify HDF5 structure (has action[:, 6] and action[:, 13] for gripper)
[ ] Run all 5 strategies on all 5 episodes (25 extractions)
[ ] Generate Vis 1-4 per episode (20 images)
[ ] Generate Vis 5 aggregate (1 image)
[ ] Record extraction_log.json
[ ] Review visualizations, fill in summary.md
[ ] Decide: which strategy (or combination) to use as GT keyframes for Exp 1.2 in subgoal_plan.md
```

---

## 9. Connection to subgoal_plan.md

This experiment feeds directly into:
- **Exp 2.x (Keyframe Strategies):** The best strategy from this visual inspection becomes the default for downstream experiments
- **Problem 1 (Keyframe Granularity):** Gap analysis here confirms whether composite strategy is needed
- **Planning Head Training Data:** The validated extraction strategy will be used to generate training targets for the planning head (Method 1)
