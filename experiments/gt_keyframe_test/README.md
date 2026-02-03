# GT Keyframe Subgoal Experiment

Replace model-generated subgoals with ground truth keyframes to evaluate policy performance with "perfect" visual guidance, **without finetuning or disrupting the causal generation structure**.

## Overview

This experiment tests the hypothesis that providing ground truth future frames as subgoals will improve policy performance. By extracting keyframes from successful demonstration videos and injecting them as subgoals, we can measure the gap between model-generated subgoals and ideal guidance.

### Key Constraint: Preserve Causal Structure

The implementation preserves the causal attention mechanism:

- Subgoals are passed via `subgoal_frames` parameter (separate from conditional frames)
- Vidar server processes them as **guidance**, not as part of causal attention chain
- No modification to model weights or inference logic

## File Structure

```
experiments/gt_keyframe_test/
├── README.md                     # This documentation
├── extract_keyframes.py          # Video keyframe extraction
├── run_with_gt_subgoals.py       # Test script using GT keyframes
└── config.yml                    # Configuration
```

## Source Videos

Keyframes are extracted from successful demonstration videos:

```
/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal/{task}/episode{N}.mp4
```

- Format: 640x736, ~10 FPS, H.264/MP4

## Quick Integration (Recommended)

Add this to `script/eval_policy.py`:

```python
# At the top of the file:
from experiments.gt_keyframe_test.integration_example import GTSubgoalEvalRunner

GT_RUNNER = GTSubgoalEvalRunner(
    eval_result_dir="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal",
    strategy="uniform",
    interval=8
)

# In eval_policy() function, after reset_func(model) (around line 299):
reset_func(model)

# ADD THESE LINES:
keyframes = GT_RUNNER.get_keyframes(task_name, TASK_ENV.test_num)
if keyframes:
    GT_RUNNER.inject_keyframes(model, keyframes)

# Continue with existing code...
while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
    ...
```

## Usage

### 1. Extract Keyframes (Demo)

```bash
# Test extraction from a single video
python extract_keyframes.py /path/to/video.mp4 --strategy uniform --interval 8

# Extract with visual change detection
python extract_keyframes.py /path/to/video.mp4 --strategy visual_change --threshold 0.05

# Save keyframes as images
python extract_keyframes.py /path/to/video.mp4 --output-dir ./keyframes/
```

### 2. Run Extraction Demo

```bash
# Demo extraction for all configured tasks
python run_with_gt_subgoals.py --demo

# Demo for a specific task
python run_with_gt_subgoals.py --demo --task adjust_bottle
```

### 3. Integration with AR Policy

```python
from experiments.gt_keyframe_test.run_with_gt_subgoals import (
    GTSubgoalEvaluator,
    inject_gt_subgoals
)
from experiments.gt_keyframe_test.extract_keyframes import extract_keyframes_uniform

# Option 1: Manual injection
keyframes = extract_keyframes_uniform(video_path, interval=8)
inject_gt_subgoals(policy, keyframes)

# Option 2: Use evaluator
evaluator = GTSubgoalEvaluator(config)
evaluator.prepare_policy(policy, task, episode_video_path)
```

## Extraction Strategies

### Uniform Extraction

Extracts keyframes at regular intervals:

```python
extract_keyframes_uniform(video_path, interval=8, max_keyframes=20)
```

- `interval`: Number of frames between keyframes (default: 8)
- `max_keyframes`: Maximum keyframes to extract (default: 20)

### Visual Change Detection

Extracts keyframes when significant visual change is detected:

```python
extract_keyframes_visual_change(video_path, threshold=0.05, min_interval=4)
```

- `threshold`: MSE threshold for change detection (default: 0.05)
- `min_interval`: Minimum frames between keyframes (default: 4)

## How Subgoals Preserve Causal Structure

Current flow in `ar.py` `get_actions()`:

```python
# Lines 1276-1293 - subgoals are passed SEPARATELY
data = {
    "prompt": self.prompt,
    "imgs": obs_cache,                    # Conditional frames (causal chain)
    "num_conditional_frames": ...,
    "num_new_frames": ...,
    "subgoal_frames": subgoal_frames,     # SEPARATE - guidance only
    ...
}
```

In Vidar server:
- `subgoal_frames` are VAE-encoded independently
- Applied as guidance via `subgoal_guidance_scale` (default 0.5)
- Does NOT modify the causal attention/KV cache mechanism

## Configuration

Edit `config.yml` to customize:

```yaml
# Video source
eval_result_dir: /path/to/eval_result/ar/ddp_causal

# Extraction settings
extraction_strategy: uniform  # or visual_change
keyframe_interval: 8
max_keyframes: 20
visual_change_threshold: 0.05

# Tasks to test
tasks:
  - adjust_bottle
```

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `policy/AR/ar.py` | 1250-1293 | Subgoal selection and passing to server |
| `policy/AR/ar.py` | 130-134 | Subgoal config (interval, use_libero_subgoal) |

## Verification Checklist

- [ ] Video loads correctly from eval_result path
- [ ] Keyframes extracted at correct intervals/visual changes
- [ ] Base64 JPEG format matches Vidar expectation
- [ ] Subgoals injected into `policy.current_subgoals`
- [ ] Causal structure preserved (check server logs)
- [ ] Results compared to baseline (model-generated subgoals)

## Future Work (Phase 2)

1. **HDF5-based extraction** with action data for gripper-state keyframes
2. **Task-specific extraction** using task configs
3. **Comparison experiments** across extraction strategies
4. **Ablation studies** on subgoal_guidance_scale
