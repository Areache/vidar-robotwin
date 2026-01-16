#!/usr/bin/env python3
"""
Convert RDT fine-tuning dataset to Vidar training format.

Source: https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data

Input structure:
    rdt_data/
    ├── task_1/
    │   ├── episode_1.hdf5
    │   └── expanded_instruction_gpt-4-turbo.json
    └── ...

Output structure:
    output/
    ├── hdf5/
    │   └── episode_XXXXXX.hdf5 (with unified observation)
    ├── config.json
    └── instructions.json

Usage:
    python scripts/prepare_rdt_data.py \
        --src-dir /path/to/rdt_data \
        --dst-dir /path/to/output \
        --robot-type aloha
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import numpy as np
import h5py

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Instruction template with scene description
INSTRUCTION_TEMPLATE = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed front camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: {instruction}"
)

# Camera names in RDT dataset
CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# Unified observation resolution (from Vidarc paper)
UNIFIED_RESOLUTION = (720, 640)  # (H, W)


def format_instruction(instruction: str) -> str:
    """Format instruction with scene description."""
    # Normalize
    instruction = instruction.strip()
    if instruction and instruction[0].islower():
        instruction = instruction[0].upper() + instruction[1:]
    if instruction and instruction[-1] not in ".!?":
        instruction += "."
    return INSTRUCTION_TEMPLATE.format(instruction=instruction)


def compose_unified_observation(
    cam_high: np.ndarray,
    cam_left: np.ndarray,
    cam_right: np.ndarray,
    output_size: Tuple[int, int] = UNIFIED_RESOLUTION,
) -> np.ndarray:
    """
    Compose three camera views into unified 720x640 observation.

    Layout:
        [      cam_high (360, 640)       ]
        [ cam_left (360,320) | cam_right (360,320) ]
    """
    H, W = output_size
    h_top = H // 2
    h_bottom = H - h_top
    w_half = W // 2

    unified = np.zeros((H, W, 3), dtype=np.uint8)

    # Resize and place each camera
    if HAS_CV2:
        # Top: cam_high
        unified[:h_top, :, :] = cv2.resize(cam_high, (W, h_top))
        # Bottom-left: cam_left_wrist
        unified[h_top:, :w_half, :] = cv2.resize(cam_left, (w_half, h_bottom))
        # Bottom-right: cam_right_wrist
        unified[h_top:, w_half:, :] = cv2.resize(cam_right, (W - w_half, h_bottom))
    else:
        # Nearest neighbor fallback
        def resize_nn(img, size):
            h, w = img.shape[:2]
            th, tw = size
            y_idx = (np.arange(th) * h / th).astype(int)
            x_idx = (np.arange(tw) * w / tw).astype(int)
            return img[y_idx][:, x_idx]

        unified[:h_top, :, :] = resize_nn(cam_high, (h_top, W))
        unified[h_top:, :w_half, :] = resize_nn(cam_left, (h_bottom, w_half))
        unified[h_top:, w_half:, :] = resize_nn(cam_right, (h_bottom, W - w_half))

    return unified


def process_episode(
    src_path: Path,
    dst_path: Path,
    instruction: str,
    num_frames: int = 81,
) -> bool:
    """Process single episode HDF5 file."""
    try:
        with h5py.File(src_path, "r") as src:
            # Get trajectory length
            obs = src["observations"]
            T = obs["qpos"].shape[0]

            # Load camera images
            images = obs["images"]
            cam_high = images["cam_high"][:]
            cam_left = images["cam_left_wrist"][:]
            cam_right = images["cam_right_wrist"][:]

            # Load state and actions
            qpos = obs["qpos"][:]
            action = src["action"][:]

            # Subsample if needed
            if T > num_frames:
                indices = np.linspace(0, T - 1, num_frames, dtype=int)
                cam_high = cam_high[indices]
                cam_left = cam_left[indices]
                cam_right = cam_right[indices]
                qpos = qpos[indices]
                action = action[indices]
                T = num_frames

            # Create unified observations
            unified = np.zeros((T, *UNIFIED_RESOLUTION, 3), dtype=np.uint8)
            for t in range(T):
                unified[t] = compose_unified_observation(
                    cam_high[t], cam_left[t], cam_right[t]
                )

            # Format instruction with scene description
            formatted_instruction = format_instruction(instruction)

            # Save to new HDF5
            with h5py.File(dst_path, "w") as dst:
                # Observations group
                obs_grp = dst.create_group("observations")

                # Unified image
                obs_grp.create_dataset(
                    "unified_image",
                    data=unified,
                    compression="gzip",
                    compression_opts=4,
                )

                # Per-camera images
                img_grp = obs_grp.create_group("images")
                img_grp.create_dataset(
                    "cam_high", data=cam_high,
                    compression="gzip", compression_opts=4
                )
                img_grp.create_dataset(
                    "cam_left_wrist", data=cam_left,
                    compression="gzip", compression_opts=4
                )
                img_grp.create_dataset(
                    "cam_right_wrist", data=cam_right,
                    compression="gzip", compression_opts=4
                )

                # State
                obs_grp.create_dataset("qpos", data=qpos.astype(np.float32))

                # Actions
                dst.create_dataset("action", data=action.astype(np.float32))

                # Instruction attributes
                dst.attrs["instruction"] = formatted_instruction
                dst.attrs["instruction_raw"] = instruction

            return True

    except Exception as e:
        logger.error(f"Error processing {src_path}: {e}")
        return False


def load_instructions(task_dir: Path) -> Dict[str, str]:
    """Load instructions from JSON file."""
    instructions = {}

    # Try different instruction file names
    json_files = [
        "expanded_instruction_gpt-4-turbo.json",
        "instructions.json",
        "lang.json",
    ]

    for json_name in json_files:
        json_path = task_dir / json_name
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)

            # Handle different formats
            if isinstance(data, dict):
                for ep_name, ep_data in data.items():
                    if isinstance(ep_data, dict):
                        # Use original instruction or first expanded
                        instr = ep_data.get("instruction", "")
                        if not instr and "expanded_instruction" in ep_data:
                            expanded = ep_data["expanded_instruction"]
                            if expanded:
                                instr = expanded[0] if isinstance(expanded, list) else expanded
                        instructions[ep_name] = instr
                    elif isinstance(ep_data, str):
                        instructions[ep_name] = ep_data
            break

    return instructions


def main():
    parser = argparse.ArgumentParser(
        description="Convert RDT fine-tuning dataset to Vidar format"
    )
    parser.add_argument(
        "--src-dir", type=str, required=True,
        help="Path to rdt_data directory"
    )
    parser.add_argument(
        "--dst-dir", type=str, required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--num-frames", type=int, default=81,
        help="Target frames per episode (default: 81)"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Maximum episodes to process (for testing)"
    )

    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if not src_dir.exists():
        logger.error(f"Source directory not found: {src_dir}")
        sys.exit(1)

    # Create output directories
    hdf5_dir = dst_dir / "hdf5"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    # Find all task directories
    task_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(task_dirs)} task directories")

    # Process all episodes
    episode_idx = 0
    all_instructions = {}
    processed_episodes = []

    for task_dir in tqdm(task_dirs, desc="Tasks"):
        # Load instructions for this task
        instructions = load_instructions(task_dir)

        # Find HDF5 files
        hdf5_files = sorted(task_dir.glob("episode_*.hdf5"))

        for hdf5_file in hdf5_files:
            if args.max_episodes and episode_idx >= args.max_episodes:
                break

            # Get instruction
            ep_name = hdf5_file.stem
            instruction = instructions.get(ep_name, f"{task_dir.name}: {ep_name}")

            # Process
            dst_path = hdf5_dir / f"episode_{episode_idx:06d}.hdf5"
            success = process_episode(
                hdf5_file, dst_path, instruction, args.num_frames
            )

            if success:
                all_instructions[f"episode_{episode_idx:06d}"] = {
                    "instruction": format_instruction(instruction),
                    "instruction_raw": instruction,
                    "source_task": task_dir.name,
                    "source_episode": ep_name,
                }
                processed_episodes.append(str(dst_path))
                episode_idx += 1

        if args.max_episodes and episode_idx >= args.max_episodes:
            break

    # Save config
    config = {
        "dataset_name": "rdt_ft_data",
        "robot_type": "aloha",
        "n_episodes": len(processed_episodes),
        "episode_len": args.num_frames,
        "state_dim": 14,
        "action_dim": 14,
        "camera_names": CAMERA_NAMES,
        "unified_resolution": list(UNIFIED_RESOLUTION),
        "instruction_template": INSTRUCTION_TEMPLATE,
    }

    with open(dst_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(dst_dir / "instructions.json", "w") as f:
        json.dump(all_instructions, f, indent=2)

    # Save episode list
    episodes_list = [
        {"path": f"hdf5/episode_{i:06d}.hdf5", "name": f"episode_{i:06d}"}
        for i in range(len(processed_episodes))
    ]
    with open(dst_dir / "episodes.json", "w") as f:
        json.dump(episodes_list, f, indent=2)

    logger.info(f"\nProcessed {len(processed_episodes)} episodes")
    logger.info(f"Output: {dst_dir}")
    print("\n" + "=" * 60)
    print("To train with this dataset:")
    print("=" * 60)
    print(f"""
torchrun --nproc_per_node=2 scripts/train_vidar.py \\
    --config configs/vidar_finetune.yaml \\
    --data-dir {dst_dir} \\
    --ckpt-dir /path/to/Wan2.2-TI2V-5B
""")


if __name__ == "__main__":
    main()
