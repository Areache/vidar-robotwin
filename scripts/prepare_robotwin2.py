#!/usr/bin/env python3
"""
Prepare RoboTwin2.0 dataset for Vidarc Stage 2 training.

RoboTwin2.0 data format:
- HDF5 files with observations and actions
- Images stored as bit streams (JPEG/PNG encoded)
- Multiple camera views (head_camera, wrist cameras)

Usage:
    # First, unzip the downloaded files
    unzip franka_clean_50.zip -d ./data/robotwin2/stack_bowls_two/clean
    unzip franka_randomized_500.zip -d ./data/robotwin2/stack_bowls_two/randomized

    # Explore data structure (run this first to see what's inside)
    python scripts/prepare_robotwin2.py \
        --src-dir ./data/robotwin2/stack_bowls_two \
        --explore

    # Process for Stage 2 training
    python scripts/prepare_robotwin2.py \
        --src-dir ./data/robotwin2/stack_bowls_two \
        --dst-dir ./data/vidarc_robotwin2 \
        --task-name "stack bowls"

    # With custom instruction
    python scripts/prepare_robotwin2.py \
        --src-dir ./data/robotwin2/stack_bowls_two \
        --dst-dir ./data/vidarc_robotwin2 \
        --instruction "Stack two bowls on top of each other"
"""

import os
import sys
import json
import argparse
import logging
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import h5py

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not installed. Install with: pip install opencv-python")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Unified observation resolution (from Vidarc paper)
UNIFIED_RESOLUTION = (720, 640)  # (H, W)
TARGET_FPS = 10

# Instruction template for RoboTwin (Franka is single-arm, but we use similar format)
INSTRUCTION_TEMPLATE = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed head camera, a movable left wrist camera, and a movable right wrist camera. "
    "The franka robot is currently performing the following task: {instruction}"
)

# For dual-arm (Aloha-style) robots
INSTRUCTION_TEMPLATE_DUAL = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: {instruction}"
)

# Camera key patterns to search for
CAMERA_PATTERNS = {
    "head": ["head_camera", "head", "front_camera", "front", "cam_high", "cam_front", "cam_rear", "overhead"],
    "left": ["left_wrist_camera", "left_wrist", "left_camera", "left", "cam_left_wrist", "cam_left", "wrist_camera_left"],
    "right": ["right_wrist_camera", "right_wrist", "right_camera", "right", "cam_right_wrist", "cam_right", "wrist_camera_right"],
    "wrist": ["wrist_camera", "wrist", "hand_camera"],  # For single-arm robots
}

# RoboTwin2.0 specific paths
ROBOTWIN2_CAMERA_PATHS = {
    "head": "observation/head_camera/rgb",
    "left": "observation/left_camera/rgb",
    "right": "observation/right_camera/rgb",
    "third": "third_view_rgb",
}

ROBOTWIN2_ACTION_PATH = "joint_action/vector"
ROBOTWIN2_QPOS_PATHS = ["joint_action/vector", "endpose"]


def decode_image(image_data: np.ndarray) -> np.ndarray:
    """
    Decode image from bit stream format used in RoboTwin2.0.

    Args:
        image_data: Either raw image array or JPEG/PNG encoded bytes

    Returns:
        Decoded image as numpy array (H, W, 3) in RGB format
    """
    if image_data is None:
        return None

    # If already a proper image array
    if len(image_data.shape) == 3 and image_data.shape[-1] == 3:
        return image_data

    # If it's a 2D array with shape (H, W), it might be grayscale
    if len(image_data.shape) == 2 and image_data.shape[0] > 100:
        return np.stack([image_data] * 3, axis=-1)

    # Bit stream encoded image
    if HAS_CV2:
        try:
            # Handle different input types
            if isinstance(image_data, np.ndarray):
                if image_data.dtype == np.uint8:
                    buf = image_data.tobytes()
                else:
                    buf = image_data.astype(np.uint8).tobytes()
            else:
                buf = bytes(image_data)

            # Decode
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        except Exception as e:
            logger.debug(f"Failed to decode as bit stream: {e}")

    # If decoding failed, check if it's raw bytes that represent pixel values
    if len(image_data.shape) == 1:
        # Try to reshape as image
        total_pixels = len(image_data)
        # Common resolutions
        for h, w in [(480, 640), (720, 1280), (360, 640), (240, 320)]:
            if total_pixels == h * w * 3:
                return image_data.reshape(h, w, 3).astype(np.uint8)
            elif total_pixels == h * w:
                gray = image_data.reshape(h, w).astype(np.uint8)
                return np.stack([gray] * 3, axis=-1)

    logger.warning(f"Could not decode image with shape {image_data.shape}")
    return None


def find_camera_data(group: h5py.Group, camera_type: str) -> Tuple[Optional[str], Optional[h5py.Dataset]]:
    """
    Find camera data in HDF5 group.

    Args:
        group: HDF5 group to search in
        camera_type: Type of camera ("head", "left", "right", "wrist")

    Returns:
        Tuple of (key_name, dataset) or (None, None) if not found
    """
    patterns = CAMERA_PATTERNS.get(camera_type, [camera_type])

    def search_in_group(g, prefix=""):
        for key in g.keys():
            full_key = f"{prefix}/{key}" if prefix else key
            key_lower = key.lower()

            # Check if this key matches any pattern
            for pattern in patterns:
                if pattern in key_lower:
                    item = g[key]
                    if isinstance(item, h5py.Dataset):
                        return full_key, item
                    elif isinstance(item, h5py.Group):
                        # Recurse into group
                        result = search_in_group(item, full_key)
                        if result[0]:
                            return result

            # Recurse into subgroups
            if isinstance(g[key], h5py.Group):
                result = search_in_group(g[key], full_key)
                if result[0]:
                    return result

        return None, None

    return search_in_group(group)


def explore_hdf5(file_path: Path, max_depth: int = 4) -> Dict[str, Any]:
    """
    Explore HDF5 file structure.

    Args:
        file_path: Path to HDF5 file
        max_depth: Maximum depth to explore

    Returns:
        Dictionary describing the structure
    """
    structure = {}

    def visit(name, obj, depth=0):
        if depth > max_depth:
            return

        parts = name.split("/")
        current = structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        final_key = parts[-1]
        if isinstance(obj, h5py.Dataset):
            current[final_key] = {
                "type": "dataset",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
            # Sample first element if it's image-like
            if len(obj.shape) >= 1 and obj.shape[0] > 0:
                try:
                    sample = obj[0]
                    if isinstance(sample, bytes) or (isinstance(sample, np.ndarray) and sample.dtype == np.uint8):
                        current[final_key]["note"] = "possibly encoded image"
                except:
                    pass
        elif isinstance(obj, h5py.Group):
            current[final_key] = {"type": "group", "contents": {}}

    with h5py.File(file_path, "r") as f:
        # Get attributes
        structure["_attributes"] = dict(f.attrs)

        # Visit all items
        f.visititems(lambda name, obj: visit(name, obj))

    return structure


def explore_dataset(src_dir: Path) -> None:
    """
    Explore dataset structure and print information.
    """
    print("\n" + "=" * 70)
    print("EXPLORING ROBOTWIN2.0 DATASET STRUCTURE")
    print("=" * 70)

    # Find all HDF5 files
    hdf5_files = list(src_dir.rglob("*.hdf5")) + list(src_dir.rglob("*.h5"))

    if not hdf5_files:
        print(f"\nNo HDF5 files found in {src_dir}")
        print("\nLooking for zip files...")
        zip_files = list(src_dir.rglob("*.zip"))
        if zip_files:
            print(f"Found {len(zip_files)} zip files:")
            for zf in zip_files[:10]:
                print(f"  - {zf}")
            print("\nPlease unzip them first:")
            print(f"  unzip <file>.zip -d {src_dir}/")
        return

    print(f"\nFound {len(hdf5_files)} HDF5 files")

    # Group by directory
    by_dir = defaultdict(list)
    for f in hdf5_files:
        by_dir[f.parent].append(f)

    print(f"\nDirectories containing HDF5 files:")
    for dir_path, files in sorted(by_dir.items()):
        print(f"  {dir_path}: {len(files)} files")

    # Explore first file in detail
    sample_file = hdf5_files[0]
    print(f"\n{'=' * 70}")
    print(f"SAMPLE FILE STRUCTURE: {sample_file.name}")
    print("=" * 70)

    with h5py.File(sample_file, "r") as f:
        print("\nTop-level keys:", list(f.keys()))
        print("\nAttributes:", dict(f.attrs))

        def print_structure(group, indent=0):
            prefix = "  " * indent
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    shape_str = f"shape={item.shape}, dtype={item.dtype}"
                    print(f"{prefix}- {key}: Dataset({shape_str})")

                    # Try to decode if it's an image
                    if "camera" in key.lower() or "image" in key.lower():
                        try:
                            sample = item[0]
                            decoded = decode_image(sample)
                            if decoded is not None:
                                print(f"{prefix}  -> Decoded image shape: {decoded.shape}")
                        except Exception as e:
                            print(f"{prefix}  -> Decode error: {e}")

                elif isinstance(item, h5py.Group):
                    print(f"{prefix}+ {key}/")
                    if indent < 3:
                        print_structure(item, indent + 1)

        print("\nFull structure:")
        print_structure(f)

    # Check for instruction files
    print(f"\n{'=' * 70}")
    print("INSTRUCTION FILES")
    print("=" * 70)

    instruction_files = (
        list(src_dir.rglob("*instruction*.json")) +
        list(src_dir.rglob("*lang*.json")) +
        list(src_dir.rglob("*task*.json"))
    )

    if instruction_files:
        print(f"\nFound {len(instruction_files)} instruction files:")
        for f in instruction_files[:5]:
            print(f"  - {f}")
            try:
                with open(f) as jf:
                    data = json.load(jf)
                    if isinstance(data, dict):
                        print(f"    Keys: {list(data.keys())[:5]}")
                    elif isinstance(data, list):
                        print(f"    List with {len(data)} items")
                    elif isinstance(data, str):
                        print(f"    String: {data[:100]}...")
            except Exception as e:
                print(f"    Error reading: {e}")
    else:
        print("\nNo instruction JSON files found")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total HDF5 files: {len(hdf5_files)}")
    print(f"  Total directories: {len(by_dir)}")
    print(f"\nNext step: Run with --dst-dir to process the data")


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize single frame to target size (H, W)."""
    if frame is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    if HAS_CV2:
        # cv2.resize expects (W, H)
        return cv2.resize(frame, (target_size[1], target_size[0]))
    else:
        # Nearest neighbor fallback
        h, w = frame.shape[:2]
        th, tw = target_size
        y_idx = (np.arange(th) * h / th).astype(int)
        x_idx = (np.arange(tw) * w / tw).astype(int)
        return frame[y_idx][:, x_idx]


def compose_unified_observation(
    cam_head: np.ndarray,
    cam_left: Optional[np.ndarray],
    cam_right: Optional[np.ndarray],
    output_size: Tuple[int, int] = UNIFIED_RESOLUTION,
) -> np.ndarray:
    """
    Compose camera views into unified 720×640 observation.

    Layout:
        [       cam_head (360, 640)        ]
        [ cam_left (360,320) | cam_right (360,320) ]

    For single-arm robots without left/right wrist cameras,
    the head camera is used for all views.
    """
    H, W = output_size
    h_top = H // 2      # 360
    h_bottom = H - h_top  # 360
    w_half = W // 2     # 320

    unified = np.zeros((H, W, 3), dtype=np.uint8)

    # Top: head camera
    if cam_head is not None:
        unified[:h_top, :, :] = resize_frame(cam_head, (h_top, W))

    # Bottom-left
    if cam_left is not None:
        unified[h_top:, :w_half, :] = resize_frame(cam_left, (h_bottom, w_half))
    elif cam_head is not None:
        # Fallback: use head camera
        unified[h_top:, :w_half, :] = resize_frame(cam_head, (h_bottom, w_half))

    # Bottom-right
    if cam_right is not None:
        unified[h_top:, w_half:, :] = resize_frame(cam_right, (h_bottom, W - w_half))
    elif cam_head is not None:
        # Fallback: use head camera
        unified[h_top:, w_half:, :] = resize_frame(cam_head, (h_bottom, W - w_half))

    return unified


def subsample_frames(data: np.ndarray, src_fps: float, target_fps: float = TARGET_FPS) -> np.ndarray:
    """Subsample data to target FPS."""
    if src_fps <= target_fps:
        return data

    T = data.shape[0]
    ratio = src_fps / target_fps
    indices = np.arange(0, T, ratio).astype(int)
    indices = indices[indices < T]
    return data[indices]


def load_instructions(src_dir: Path, task_name: str) -> Dict[str, str]:
    """Load instructions from JSON files."""
    instructions = {}

    # Search patterns
    patterns = [
        "**/expanded_instruction*.json",
        "**/instruction*.json",
        "**/lang*.json",
        "**/task*.json",
    ]

    for pattern in patterns:
        for json_file in src_dir.glob(pattern):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            instr = value.get("instruction") or value.get("expanded_instruction") or value.get("task")
                            if isinstance(instr, list):
                                instr = instr[0] if instr else task_name
                            instructions[key] = instr or task_name
                        elif isinstance(value, str):
                            instructions[key] = value
                        elif isinstance(value, list) and value:
                            instructions[key] = value[0]
                elif isinstance(data, str):
                    instructions["default"] = data
                elif isinstance(data, list) and data:
                    instructions["default"] = data[0] if isinstance(data[0], str) else task_name

            except Exception as e:
                logger.debug(f"Error loading {json_file}: {e}")

    if not instructions:
        instructions["default"] = task_name

    return instructions


def decode_robotwin2_image(data: bytes) -> Optional[np.ndarray]:
    """
    Decode RoboTwin2.0 image from byte string.

    RoboTwin2.0 stores images as JPEG-encoded byte strings.
    """
    if data is None:
        return None

    if not HAS_CV2:
        logger.error("OpenCV required for decoding RoboTwin2.0 images")
        return None

    try:
        # Convert to numpy array if needed
        if isinstance(data, bytes):
            nparr = np.frombuffer(data, np.uint8)
        else:
            nparr = np.frombuffer(bytes(data), np.uint8)

        # Decode JPEG
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    except Exception as e:
        logger.debug(f"Failed to decode image: {e}")

    return None


def load_robotwin2_camera(f: h5py.File, camera_path: str) -> Optional[List[np.ndarray]]:
    """
    Load camera frames from RoboTwin2.0 HDF5 file.

    Args:
        f: Open HDF5 file
        camera_path: Path to camera rgb data (e.g., 'observation/head_camera/rgb')

    Returns:
        List of decoded image frames or None
    """
    try:
        if camera_path not in f:
            return None

        data = f[camera_path]
        T = data.shape[0]

        frames = []
        for t in range(T):
            img = decode_robotwin2_image(data[t])
            frames.append(img)

        return frames

    except Exception as e:
        logger.debug(f"Failed to load camera {camera_path}: {e}")
        return None


def process_episode(
    src_path: Path,
    dst_path: Path,
    instruction: str,
    src_fps: float = 30.0,
    num_frames: Optional[int] = 81,
    is_dual_arm: bool = True,
) -> bool:
    """
    Process single HDF5 episode for Stage 2 training.

    Args:
        src_path: Source HDF5 file
        dst_path: Destination HDF5 file
        instruction: Task instruction
        src_fps: Source video FPS
        num_frames: Target number of frames (None for variable)
        is_dual_arm: Whether robot is dual-arm (affects instruction template)

    Returns:
        True if successful
    """
    try:
        with h5py.File(src_path, "r") as src:
            # Try RoboTwin2.0 specific paths first
            frames_head = load_robotwin2_camera(src, ROBOTWIN2_CAMERA_PATHS["head"])
            frames_left = load_robotwin2_camera(src, ROBOTWIN2_CAMERA_PATHS["left"])
            frames_right = load_robotwin2_camera(src, ROBOTWIN2_CAMERA_PATHS["right"])

            # Fallback: try third_view_rgb as head camera
            if frames_head is None:
                frames_head = load_robotwin2_camera(src, ROBOTWIN2_CAMERA_PATHS["third"])

            # Fallback: generic camera search
            if frames_head is None:
                obs_group = src.get("observation") or src.get("observations") or src
                head_key, head_data = find_camera_data(obs_group, "head")
                if head_data is not None:
                    frames_head = [decode_image(head_data[t]) for t in range(head_data.shape[0])]

            if frames_head is None or len(frames_head) == 0:
                logger.warning(f"No head camera found in {src_path}")
                return False

            # Get number of timesteps
            T = len(frames_head)

            # Create placeholder for missing cameras
            sample_shape = None
            for f in frames_head:
                if f is not None:
                    sample_shape = f.shape
                    break

            if sample_shape is None:
                sample_shape = (480, 640, 3)

            # Fill None frames with zeros
            frames_head = [f if f is not None else np.zeros(sample_shape, dtype=np.uint8) for f in frames_head]

            if frames_left is None:
                frames_left = [np.zeros(sample_shape, dtype=np.uint8)] * T
            else:
                frames_left = [f if f is not None else np.zeros(sample_shape, dtype=np.uint8) for f in frames_left]

            if frames_right is None:
                frames_right = [np.zeros(sample_shape, dtype=np.uint8)] * T
            else:
                frames_right = [f if f is not None else np.zeros(sample_shape, dtype=np.uint8) for f in frames_right]

            # Convert to numpy arrays
            frames_head = np.array(frames_head)
            frames_left = np.array(frames_left)
            frames_right = np.array(frames_right)

            # Subsample to target FPS
            frames_head = subsample_frames(frames_head, src_fps, TARGET_FPS)
            frames_left = subsample_frames(frames_left, src_fps, TARGET_FPS)
            frames_right = subsample_frames(frames_right, src_fps, TARGET_FPS)
            T = frames_head.shape[0]

            # Further subsample if needed
            if num_frames and T > num_frames:
                indices = np.linspace(0, T - 1, num_frames, dtype=int)
                frames_head = frames_head[indices]
                frames_left = frames_left[indices]
                frames_right = frames_right[indices]
                T = num_frames

            # Compose unified observations
            unified = np.zeros((T, *UNIFIED_RESOLUTION, 3), dtype=np.uint8)
            for t in range(T):
                unified[t] = compose_unified_observation(
                    frames_head[t],
                    frames_left[t] if not np.all(frames_left[t] == 0) else None,
                    frames_right[t] if not np.all(frames_right[t] == 0) else None,
                )

            # Load actions from RoboTwin2.0 format
            action = None
            if ROBOTWIN2_ACTION_PATH in src:
                action = src[ROBOTWIN2_ACTION_PATH][:]
                action = subsample_frames(action, src_fps, TARGET_FPS)
                if num_frames and len(action) > num_frames:
                    indices = np.linspace(0, len(action) - 1, num_frames, dtype=int)
                    action = action[indices]
            else:
                # Fallback: try generic action keys
                for action_key in ["action", "actions", "joint_action"]:
                    if action_key in src:
                        action_data = src[action_key]
                        if isinstance(action_data, h5py.Group) and "vector" in action_data:
                            action = action_data["vector"][:]
                        elif isinstance(action_data, h5py.Dataset):
                            action = action_data[:]
                        if action is not None:
                            action = subsample_frames(action, src_fps, TARGET_FPS)
                            if num_frames and len(action) > num_frames:
                                indices = np.linspace(0, len(action) - 1, num_frames, dtype=int)
                                action = action[indices]
                            break

            # Load qpos (joint positions) - use action as qpos for RoboTwin2.0
            qpos = action  # In RoboTwin2.0, joint_action/vector serves as both

            # Format instruction
            template = INSTRUCTION_TEMPLATE_DUAL if is_dual_arm else INSTRUCTION_TEMPLATE
            formatted_instruction = template.format(instruction=instruction)

            # Save output HDF5
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(dst_path, "w") as dst:
                # Observations group
                obs_grp = dst.create_group("observations")

                # Unified image (primary for training)
                obs_grp.create_dataset(
                    "unified_image",
                    data=unified,
                    compression="gzip",
                    compression_opts=4,
                )

                # Per-camera images (for debugging)
                img_grp = obs_grp.create_group("images")
                img_grp.create_dataset(
                    "cam_high",
                    data=frames_head,
                    compression="gzip",
                    compression_opts=4,
                )
                img_grp.create_dataset(
                    "cam_left_wrist",
                    data=frames_left,
                    compression="gzip",
                    compression_opts=4,
                )
                img_grp.create_dataset(
                    "cam_right_wrist",
                    data=frames_right,
                    compression="gzip",
                    compression_opts=4,
                )

                # State
                if qpos is not None:
                    obs_grp.create_dataset("qpos", data=qpos.astype(np.float32))

                # Actions
                if action is not None:
                    dst.create_dataset("action", data=action.astype(np.float32))

                # Metadata
                dst.attrs["instruction"] = formatted_instruction
                dst.attrs["instruction_raw"] = instruction
                dst.attrs["fps"] = TARGET_FPS
                dst.attrs["num_frames"] = T
                dst.attrs["source_file"] = str(src_path)

            return True

    except Exception as e:
        logger.error(f"Error processing {src_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_robotwin2_dataset(
    src_dir: Path,
    dst_dir: Path,
    task_name: str,
    instruction: Optional[str] = None,
    num_frames: int = 81,
    src_fps: float = 30.0,
    max_episodes: Optional[int] = None,
    is_dual_arm: bool = False,
) -> int:
    """
    Process RoboTwin2.0 dataset for Stage 2 training.

    Args:
        src_dir: Source directory containing HDF5 files
        dst_dir: Output directory
        task_name: Task name (used for default instruction)
        instruction: Override instruction for all episodes
        num_frames: Target frames per episode
        src_fps: Source video FPS
        max_episodes: Maximum episodes to process
        is_dual_arm: Whether robot is dual-arm

    Returns:
        Number of processed episodes
    """
    hdf5_dir = dst_dir / "hdf5"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    # Find all HDF5 files
    hdf5_files = sorted(list(src_dir.rglob("*.hdf5")) + list(src_dir.rglob("*.h5")))
    logger.info(f"Found {len(hdf5_files)} HDF5 files")

    if not hdf5_files:
        logger.error("No HDF5 files found!")
        return 0

    # Load instructions
    instructions = load_instructions(src_dir, task_name)
    default_instruction = instruction or instructions.get("default", task_name)

    episode_idx = 0
    all_instructions = {}
    processed_episodes = []

    for hdf5_file in tqdm(hdf5_files, desc="Processing episodes"):
        if max_episodes and episode_idx >= max_episodes:
            break

        # Get instruction for this episode
        ep_name = hdf5_file.stem
        ep_instruction = instruction or instructions.get(ep_name, default_instruction)

        # Process episode
        dst_path = hdf5_dir / f"episode_{episode_idx:06d}.hdf5"
        success = process_episode(
            hdf5_file,
            dst_path,
            ep_instruction,
            src_fps=src_fps,
            num_frames=num_frames,
            is_dual_arm=is_dual_arm,
        )

        if success:
            template = INSTRUCTION_TEMPLATE_DUAL if is_dual_arm else INSTRUCTION_TEMPLATE
            all_instructions[f"episode_{episode_idx:06d}"] = {
                "instruction": template.format(instruction=ep_instruction),
                "instruction_raw": ep_instruction,
                "source_file": str(hdf5_file),
            }
            processed_episodes.append(str(dst_path))
            episode_idx += 1

    # Save metadata
    config = {
        "dataset_name": f"vidarc_robotwin2_{task_name.replace(' ', '_')}",
        "dataset_type": "robotwin",
        "n_episodes": len(processed_episodes),
        "episode_len": num_frames,
        "fps": TARGET_FPS,
        "unified_resolution": list(UNIFIED_RESOLUTION),
        "stage": 2,
    }

    with open(dst_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(dst_dir / "instructions.json", "w") as f:
        json.dump(all_instructions, f, indent=2)

    episodes_list = [
        {"path": f"hdf5/episode_{i:06d}.hdf5", "name": f"episode_{i:06d}"}
        for i in range(len(processed_episodes))
    ]
    with open(dst_dir / "episodes.json", "w") as f:
        json.dump(episodes_list, f, indent=2)

    return len(processed_episodes)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RoboTwin2.0 dataset for Vidarc Stage 2 training"
    )

    parser.add_argument(
        "--src-dir", type=str, required=True,
        help="Source directory containing RoboTwin2.0 data (HDF5 files or unzipped data)"
    )
    parser.add_argument(
        "--dst-dir", type=str, default=None,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Explore dataset structure without processing"
    )
    parser.add_argument(
        "--task-name", type=str, default="manipulation task",
        help="Task name for instruction (e.g., 'stack bowls')"
    )
    parser.add_argument(
        "--instruction", type=str, default=None,
        help="Override instruction for all episodes"
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
    parser.add_argument(
        "--dual-arm", action="store_true",
        help="Robot is dual-arm (Aloha-style) - affects instruction template"
    )

    args = parser.parse_args()

    src_dir = Path(args.src_dir)

    if not src_dir.exists():
        logger.error(f"Source directory not found: {src_dir}")
        sys.exit(1)

    # Explore mode
    if args.explore:
        explore_dataset(src_dir)
        return

    # Process mode
    if args.dst_dir is None:
        logger.error("--dst-dir is required for processing")
        sys.exit(1)

    dst_dir = Path(args.dst_dir)

    logger.info("=" * 60)
    logger.info("Processing RoboTwin2.0 Dataset for Stage 2")
    logger.info("=" * 60)
    logger.info(f"Source: {src_dir}")
    logger.info(f"Output: {dst_dir}")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Frames per episode: {args.num_frames}")
    logger.info(f"Source FPS: {args.src_fps}")

    n_episodes = process_robotwin2_dataset(
        src_dir=src_dir,
        dst_dir=dst_dir,
        task_name=args.task_name,
        instruction=args.instruction,
        num_frames=args.num_frames,
        src_fps=args.src_fps,
        max_episodes=args.max_episodes,
        is_dual_arm=args.dual_arm,
    )

    logger.info("=" * 60)
    logger.info(f"Processed {n_episodes} episodes")
    logger.info(f"Output saved to: {dst_dir}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("NEXT STEPS: Stage 2 Training")
    print("=" * 60)
    print(f"""
# 1. Download checkpoints (if not already done)
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
huggingface-cli download Xiang-cd/vidar --local-dir ./checkpoints/vidar

# 2. Run Stage 2 training
torchrun --nproc_per_node=2 scripts/train_vidarc.py \\
    --config configs/vidarc_2xh200.yaml \\
    --data-dir {dst_dir} \\
    --ckpt-dir ./checkpoints/Wan2.2-TI2V-5B \\
    --pt-dir ./checkpoints/vidar/vidar.pt \\
    --max-steps 4000 \\
    --chunk-size 16
""")


if __name__ == "__main__":
    main()
