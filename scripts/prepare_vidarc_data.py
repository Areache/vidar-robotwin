#!/usr/bin/env python3
"""
Prepare dataset for Vidarc Stage 2 training.

Stage 2 (Vidarc) uses the same data format as Stage 1, but typically requires
smaller, task-specific fine-tuning datasets:
- RoboTwin: ~1,000 episodes (20 per task × 50 tasks)
- Real-world: ~2,000 episodes

The key difference is that Stage 2 training:
1. Starts from Stage 1 fine-tuned weights
2. Uses causal (autoregressive) training
3. Can use embodiment-aware loss with action masks

Dataset format requirements (from Vidarc paper):
- Unified observation: 720×640 resolution
- FPS: 10 fps
- Cameras: fixed rear/front + movable left/right arm cameras
- Actions: 14-dim (for Aloha bimanual robot)

Usage:
    # Convert RoboTwin simulation data
    python scripts/prepare_vidarc_data.py \
        --src-dir /path/to/robotwin_data \
        --dst-dir /path/to/vidarc_output \
        --dataset-type robotwin

    # Convert real-world Aloha data
    python scripts/prepare_vidarc_data.py \
        --src-dir /path/to/aloha_data \
        --dst-dir /path/to/vidarc_output \
        --dataset-type aloha
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


# Unified observation resolution (from Vidarc paper)
UNIFIED_RESOLUTION = (720, 640)  # (H, W)
TARGET_FPS = 10

# Instruction templates with scene descriptions (from paper)
INSTRUCTION_TEMPLATES = {
    "robotwin": (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
        "The aloha robot is currently performing the following task: {instruction}"
    ),
    "aloha": (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed front camera, a movable left arm camera, and a movable right arm camera. "
        "The aloha robot is currently performing the following task: {instruction}"
    ),
    "agibot": (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed high camera, a movable left arm camera, and a movable right arm camera. "
        "The genie-1 robot is currently performing the following task: {instruction}"
    ),
}

# Camera configurations for different datasets
CAMERA_CONFIGS = {
    "robotwin": {
        "high": ["cam_high", "cam_rear", "cam_back", "camera_high"],
        "left": ["cam_left_wrist", "cam_left", "camera_left"],
        "right": ["cam_right_wrist", "cam_right", "camera_right"],
    },
    "aloha": {
        "high": ["cam_high", "cam_front", "camera_front"],
        "left": ["cam_left_wrist", "cam_left"],
        "right": ["cam_right_wrist", "cam_right"],
    },
    "agibot": {
        "high": ["cam_high", "high_camera"],
        "left": ["cam_left", "left_camera"],
        "right": ["cam_right", "right_camera"],
    },
}


def format_instruction(instruction: str, dataset_type: str) -> str:
    """Format instruction with scene description."""
    instruction = instruction.strip()
    if instruction and instruction[0].islower():
        instruction = instruction[0].upper() + instruction[1:]
    if instruction and instruction[-1] not in ".!?":
        instruction += "."

    template = INSTRUCTION_TEMPLATES.get(dataset_type, INSTRUCTION_TEMPLATES["aloha"])
    return template.format(instruction=instruction)


def find_camera_key(images_group: h5py.Group, camera_type: str, dataset_type: str) -> Optional[str]:
    """Find the camera key in HDF5 images group."""
    possible_keys = CAMERA_CONFIGS.get(dataset_type, CAMERA_CONFIGS["aloha"])[camera_type]

    for key in possible_keys:
        if key in images_group:
            return key

    # Fallback: search by pattern
    available_keys = list(images_group.keys())
    for key in available_keys:
        key_lower = key.lower()
        if camera_type == "high" and any(p in key_lower for p in ["high", "front", "rear", "back"]):
            return key
        elif camera_type == "left" and "left" in key_lower:
            return key
        elif camera_type == "right" and "right" in key_lower:
            return key

    return None


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize single frame."""
    if HAS_CV2:
        return cv2.resize(frame, (target_size[1], target_size[0]))
    else:
        # Nearest neighbor fallback
        h, w = frame.shape[:2]
        th, tw = target_size
        y_idx = (np.arange(th) * h / th).astype(int)
        x_idx = (np.arange(tw) * w / tw).astype(int)
        return frame[y_idx][:, x_idx]


def compose_unified_observation(
    cam_high: np.ndarray,
    cam_left: np.ndarray,
    cam_right: np.ndarray,
    output_size: Tuple[int, int] = UNIFIED_RESOLUTION,
) -> np.ndarray:
    """
    Compose three camera views into unified 720×640 observation.

    Layout:
        [      cam_high (360, 640)       ]
        [ cam_left (360,320) | cam_right (360,320) ]
    """
    H, W = output_size
    h_top = H // 2
    h_bottom = H - h_top
    w_half = W // 2

    unified = np.zeros((H, W, 3), dtype=np.uint8)

    # Top: high/front/rear camera
    unified[:h_top, :, :] = resize_frame(cam_high, (h_top, W))
    # Bottom-left: left arm camera
    unified[h_top:, :w_half, :] = resize_frame(cam_left, (h_bottom, w_half))
    # Bottom-right: right arm camera
    unified[h_top:, w_half:, :] = resize_frame(cam_right, (h_bottom, W - w_half))

    return unified


def subsample_to_fps(data: np.ndarray, src_fps: float, target_fps: float = TARGET_FPS) -> np.ndarray:
    """Subsample data to target FPS."""
    if src_fps <= target_fps:
        return data

    T = data.shape[0]
    ratio = src_fps / target_fps
    indices = np.arange(0, T, ratio).astype(int)
    indices = indices[indices < T]
    return data[indices]


def process_hdf5_episode(
    src_path: Path,
    dst_path: Path,
    instruction: str,
    dataset_type: str,
    src_fps: float = 30.0,
    num_frames: Optional[int] = None,
) -> bool:
    """Process single HDF5 episode for Stage 2."""
    try:
        with h5py.File(src_path, "r") as src:
            # Get observations group
            if "observations" in src:
                obs = src["observations"]
            elif "obs" in src:
                obs = src["obs"]
            else:
                logger.warning(f"No observations found in {src_path}")
                return False

            # Find images group
            if "images" in obs:
                images = obs["images"]
            elif "image" in obs:
                images = obs["image"]
            else:
                # Try to find camera data at top level of observations
                images = obs

            # Find camera keys
            high_key = find_camera_key(images, "high", dataset_type)
            left_key = find_camera_key(images, "left", dataset_type)
            right_key = find_camera_key(images, "right", dataset_type)

            if not all([high_key, left_key, right_key]):
                logger.warning(f"Missing cameras in {src_path}: high={high_key}, left={left_key}, right={right_key}")
                # Try to proceed with available cameras
                available = [k for k in [high_key, left_key, right_key] if k]
                if not available:
                    return False

            # Load camera data
            T = None
            cam_high = cam_left = cam_right = None

            if high_key:
                cam_high = images[high_key][:]
                T = cam_high.shape[0]
            if left_key:
                cam_left = images[left_key][:]
                T = T or cam_left.shape[0]
            if right_key:
                cam_right = images[right_key][:]
                T = T or cam_right.shape[0]

            # Use placeholder for missing cameras
            if cam_high is None:
                cam_high = np.zeros((T, 480, 640, 3), dtype=np.uint8)
            if cam_left is None:
                cam_left = np.zeros((T, 480, 640, 3), dtype=np.uint8)
            if cam_right is None:
                cam_right = np.zeros((T, 480, 640, 3), dtype=np.uint8)

            # Subsample to target FPS
            cam_high = subsample_to_fps(cam_high, src_fps, TARGET_FPS)
            cam_left = subsample_to_fps(cam_left, src_fps, TARGET_FPS)
            cam_right = subsample_to_fps(cam_right, src_fps, TARGET_FPS)
            T = cam_high.shape[0]

            # Further subsample if num_frames specified
            if num_frames and T > num_frames:
                indices = np.linspace(0, T - 1, num_frames, dtype=int)
                cam_high = cam_high[indices]
                cam_left = cam_left[indices]
                cam_right = cam_right[indices]
                T = num_frames

            # Create unified observations
            unified = np.zeros((T, *UNIFIED_RESOLUTION, 3), dtype=np.uint8)
            for t in range(T):
                unified[t] = compose_unified_observation(
                    cam_high[t], cam_left[t], cam_right[t]
                )

            # Load state (qpos)
            qpos = None
            for qpos_key in ["qpos", "joint_positions", "state"]:
                if qpos_key in obs:
                    qpos = obs[qpos_key][:]
                    qpos = subsample_to_fps(qpos, src_fps, TARGET_FPS)
                    if num_frames and len(qpos) > num_frames:
                        indices = np.linspace(0, len(qpos) - 1, num_frames, dtype=int)
                        qpos = qpos[indices]
                    break

            # Load actions
            action = None
            for action_key in ["action", "actions"]:
                if action_key in src:
                    action = src[action_key][:]
                    action = subsample_to_fps(action, src_fps, TARGET_FPS)
                    if num_frames and len(action) > num_frames:
                        indices = np.linspace(0, len(action) - 1, num_frames, dtype=int)
                        action = action[indices]
                    break

            # Format instruction with scene description
            formatted_instruction = format_instruction(instruction, dataset_type)

            # Save to new HDF5
            with h5py.File(dst_path, "w") as dst:
                # Observations group
                obs_grp = dst.create_group("observations")

                # Unified image (primary for Stage 2)
                obs_grp.create_dataset(
                    "unified_image",
                    data=unified,
                    compression="gzip",
                    compression_opts=4,
                )

                # Per-camera images (optional, for debugging)
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
                if qpos is not None:
                    obs_grp.create_dataset("qpos", data=qpos.astype(np.float32))

                # Actions
                if action is not None:
                    dst.create_dataset("action", data=action.astype(np.float32))

                # Instruction attributes
                dst.attrs["instruction"] = formatted_instruction
                dst.attrs["instruction_raw"] = instruction
                dst.attrs["dataset_type"] = dataset_type
                dst.attrs["fps"] = TARGET_FPS
                dst.attrs["num_frames"] = T

            return True

    except Exception as e:
        logger.error(f"Error processing {src_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_instructions_from_dir(task_dir: Path) -> Dict[str, str]:
    """Load instructions from JSON files in task directory."""
    instructions = {}

    # Try different instruction file patterns
    json_patterns = [
        "expanded_instruction_gpt-4-turbo.json",
        "instructions.json",
        "lang.json",
        "task_info.json",
    ]

    for pattern in json_patterns:
        json_path = task_dir / pattern
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for ep_name, ep_data in data.items():
                    if isinstance(ep_data, dict):
                        instr = ep_data.get("instruction", "")
                        if not instr and "expanded_instruction" in ep_data:
                            expanded = ep_data["expanded_instruction"]
                            if expanded:
                                instr = expanded[0] if isinstance(expanded, list) else expanded
                        if not instr and "task" in ep_data:
                            instr = ep_data["task"]
                        instructions[ep_name] = instr
                    elif isinstance(ep_data, str):
                        instructions[ep_name] = ep_data
            elif isinstance(data, str):
                # Single instruction for all episodes
                instructions["default"] = data
            break

    return instructions


def process_robotwin_dataset(
    src_dir: Path,
    dst_dir: Path,
    num_frames: Optional[int] = 81,
    max_episodes: Optional[int] = None,
) -> int:
    """Process RoboTwin simulation dataset."""
    hdf5_dir = dst_dir / "hdf5"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    # Find all task directories
    task_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(task_dirs)} task directories")

    episode_idx = 0
    all_instructions = {}
    processed_episodes = []

    for task_dir in tqdm(task_dirs, desc="Tasks"):
        # Get task name from directory
        task_name = task_dir.name

        # Load instructions
        instructions = load_instructions_from_dir(task_dir)
        default_instr = instructions.get("default", task_name.replace("_", " "))

        # Find HDF5 files
        hdf5_files = sorted(task_dir.glob("*.hdf5"))

        for hdf5_file in hdf5_files:
            if max_episodes and episode_idx >= max_episodes:
                break

            # Get instruction
            ep_name = hdf5_file.stem
            instruction = instructions.get(ep_name, default_instr)

            # Process
            dst_path = hdf5_dir / f"episode_{episode_idx:06d}.hdf5"
            success = process_hdf5_episode(
                hdf5_file, dst_path, instruction,
                dataset_type="robotwin",
                src_fps=30.0,
                num_frames=num_frames,
            )

            if success:
                all_instructions[f"episode_{episode_idx:06d}"] = {
                    "instruction": format_instruction(instruction, "robotwin"),
                    "instruction_raw": instruction,
                    "source_task": task_name,
                    "source_episode": ep_name,
                }
                processed_episodes.append(str(dst_path))
                episode_idx += 1

        if max_episodes and episode_idx >= max_episodes:
            break

    # Save metadata
    _save_metadata(dst_dir, processed_episodes, all_instructions, "robotwin", num_frames)

    return len(processed_episodes)


def process_aloha_dataset(
    src_dir: Path,
    dst_dir: Path,
    num_frames: Optional[int] = 81,
    max_episodes: Optional[int] = None,
    src_fps: float = 50.0,
) -> int:
    """Process Aloha-style dataset (real-world or simulation)."""
    hdf5_dir = dst_dir / "hdf5"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    # Find HDF5 files (may be in subdirectories)
    hdf5_files = sorted(src_dir.rglob("*.hdf5"))
    logger.info(f"Found {len(hdf5_files)} HDF5 files")

    episode_idx = 0
    all_instructions = {}
    processed_episodes = []

    # Try to load global instructions
    global_instructions = {}
    for instr_file in ["instructions.json", "task_info.json"]:
        instr_path = src_dir / instr_file
        if instr_path.exists():
            with open(instr_path, "r") as f:
                global_instructions = json.load(f)
            break

    for hdf5_file in tqdm(hdf5_files, desc="Episodes"):
        if max_episodes and episode_idx >= max_episodes:
            break

        # Determine instruction
        ep_name = hdf5_file.stem
        task_name = hdf5_file.parent.name if hdf5_file.parent != src_dir else ep_name

        instruction = global_instructions.get(ep_name, {})
        if isinstance(instruction, dict):
            instruction = instruction.get("instruction", task_name.replace("_", " "))
        elif not instruction:
            instruction = task_name.replace("_", " ")

        # Process
        dst_path = hdf5_dir / f"episode_{episode_idx:06d}.hdf5"
        success = process_hdf5_episode(
            hdf5_file, dst_path, instruction,
            dataset_type="aloha",
            src_fps=src_fps,
            num_frames=num_frames,
        )

        if success:
            all_instructions[f"episode_{episode_idx:06d}"] = {
                "instruction": format_instruction(instruction, "aloha"),
                "instruction_raw": instruction,
                "source_task": task_name,
                "source_episode": ep_name,
            }
            processed_episodes.append(str(dst_path))
            episode_idx += 1

    # Save metadata
    _save_metadata(dst_dir, processed_episodes, all_instructions, "aloha", num_frames)

    return len(processed_episodes)


def _save_metadata(
    dst_dir: Path,
    processed_episodes: List[str],
    all_instructions: Dict,
    dataset_type: str,
    num_frames: Optional[int],
):
    """Save dataset metadata files."""
    # Config
    config = {
        "dataset_name": f"vidarc_{dataset_type}",
        "dataset_type": dataset_type,
        "n_episodes": len(processed_episodes),
        "episode_len": num_frames or "variable",
        "state_dim": 14,
        "action_dim": 14,
        "fps": TARGET_FPS,
        "unified_resolution": list(UNIFIED_RESOLUTION),
        "instruction_template": INSTRUCTION_TEMPLATES.get(dataset_type),
        "stage": 2,
    }

    with open(dst_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(dst_dir / "instructions.json", "w") as f:
        json.dump(all_instructions, f, indent=2)

    # Episode list
    episodes_list = [
        {"path": f"hdf5/episode_{i:06d}.hdf5", "name": f"episode_{i:06d}"}
        for i in range(len(processed_episodes))
    ]
    with open(dst_dir / "episodes.json", "w") as f:
        json.dump(episodes_list, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Vidarc Stage 2 training"
    )
    parser.add_argument(
        "--src-dir", type=str, required=True,
        help="Source data directory"
    )
    parser.add_argument(
        "--dst-dir", type=str, required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--dataset-type", type=str, default="aloha",
        choices=["robotwin", "aloha", "agibot"],
        help="Dataset type (determines camera config and instruction template)"
    )
    parser.add_argument(
        "--num-frames", type=int, default=81,
        help="Target frames per episode (default: 81 for 8.1s at 10fps)"
    )
    parser.add_argument(
        "--src-fps", type=float, default=30.0,
        help="Source video FPS (default: 30)"
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

    logger.info(f"Processing {args.dataset_type} dataset")
    logger.info(f"Source: {src_dir}")
    logger.info(f"Output: {dst_dir}")

    # Process based on dataset type
    if args.dataset_type == "robotwin":
        n_episodes = process_robotwin_dataset(
            src_dir, dst_dir,
            num_frames=args.num_frames,
            max_episodes=args.max_episodes,
        )
    else:
        n_episodes = process_aloha_dataset(
            src_dir, dst_dir,
            num_frames=args.num_frames,
            max_episodes=args.max_episodes,
            src_fps=args.src_fps,
        )

    logger.info(f"\nProcessed {n_episodes} episodes for Stage 2")
    logger.info(f"Output: {dst_dir}")

    print("\n" + "=" * 60)
    print("Stage 2 Training Command:")
    print("=" * 60)
    print(f"""
torchrun --nproc_per_node=2 scripts/train_vidarc.py \\
    --config configs/vidarc_causal.yaml \\
    --data-dir {dst_dir} \\
    --ckpt-dir /path/to/Wan2.2-TI2V-5B \\
    --pt-dir /path/to/vidar.pt \\
    --max-steps 4000 \\
    --chunk-size 16
""")


if __name__ == "__main__":
    main()
