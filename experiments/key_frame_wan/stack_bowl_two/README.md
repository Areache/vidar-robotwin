# Subgoal Image Generation Experiment: stack_bowls_two

Generate subgoal images using Wan I2V model for the "stack two bowls" task.

## Overview

This experiment:
1. Takes a first frame (initial state) as input
2. Uses predefined subgoal descriptions (from LLM)
3. Generates target state images using Wan text-image-to-video model

## Files

```
stack_bowl_two/
├── README.md                    # This file
├── subgoal_prompt.txt           # LLM prompt for subgoal generation
├── subgoals.json                # Predefined subgoals (LLM output)
├── generate_subgoals.py         # Main generation script
├── first_frame.jpg              # Input first frame (user-provided)
├── subgoal_01_bowl_A_grasped.jpg
├── subgoal_02_bowl_A_positioned_above_B.jpg
├── subgoal_03_bowl_A_stacked_on_B.jpg
└── subgoals_visualization.jpg   # Combined visualization
```

## Usage

### 1. Placeholder Mode (no GPU required)

```bash
# Generate placeholder subgoals (for testing)
python generate_subgoals.py

# With custom first frame
python generate_subgoals.py --first-frame /path/to/first_frame.jpg
```

### 2. Wan Model Mode (requires GPU)

```bash
# Generate using Wan I2V model
python generate_subgoals.py \
    --first-frame first_frame.jpg \
    --use-wan \
    --checkpoint-dir /path/to/Wan2.2-TI2V-5B

# From video
python generate_subgoals.py \
    --video /path/to/episode.mp4 \
    --use-wan

# From HDF5
python generate_subgoals.py \
    --hdf5 /path/to/episode.hdf5 \
    --use-wan
```

### 3. Custom Subgoals

```bash
# Use custom subgoals JSON
python generate_subgoals.py \
    --first-frame first_frame.jpg \
    --subgoals-json custom_subgoals.json \
    --use-wan
```

## Subgoal Structure

Each subgoal in `subgoals.json`:

```json
{
  "subgoal_id": 1,
  "subgoal_name": "bowl_A_grasped",
  "semantic_image_description": "The robot gripper is holding bowl {A}...",
  "key_conditions": [
    "Bowl {A} is gripped by the robot",
    "Bowl {A} is elevated from the table",
    "Bowl {B} is stationary on the table"
  ]
}
```

## LLM Prompt

The `subgoal_prompt.txt` contains the prompt used to generate subgoals from task instructions. Use this with an LLM to generate subgoals for new tasks.

## Model Details

- **Model**: Wan2.2-TI2V-5B (Text-Image to Video)
- **Input**: First frame + subgoal description
- **Output**: Short video (17 frames), last frame extracted as subgoal image
- **Guidance Scale**: 5.0 (default)
- **Sampling Steps**: 30 (default)

## Data Sources

First frame can come from:
- Direct image file (`--first-frame`)
- Video file (`--video`) - extracts frame 0
- HDF5 file (`--hdf5`) - reads `observations/unified_image[0]`

## Expected HDF5 Structure

```
episode.hdf5
├── observations/
│   ├── unified_image: (T, 720, 640, 3)  # RGB images
│   └── ...
└── action: (T, 14)
```
