#!/usr/bin/env python3
"""
AGIBOT Dataset Preparation Script (RDT-style).

Based on:
- Vidarc paper (arXiv:2512.17661): Unified observation space 720x640
- RDT-1B (thu-ml/RoboticsDiffusionTransformer): HDF5 format + instruction templates

Unified Instruction Format:
    "The whole scene is in a realistic, industrial art style with three views:
    a fixed front camera, a movable left arm camera, and a movable right arm camera.
    The {robot_type} robot is currently performing the following task: {task_instruction}"

Camera Configuration (AGIBOT Genie-1):
    - cam_high: Fixed high/front camera (360x640 in unified space)
    - cam_left_wrist: Movable left arm camera (360x320 in unified space)
    - cam_right_wrist: Movable right arm camera (360x320 in unified space)

Output Format (HDF5):
    episode_XXXXXX.hdf5
    ├── observations/
    │   ├── images/
    │   │   ├── cam_high: (T, 480, 640, 3)
    │   │   ├── cam_left_wrist: (T, 480, 640, 3)
    │   │   └── cam_right_wrist: (T, 480, 640, 3)
    │   ├── unified_image: (T, 720, 640, 3)  # Combined view
    │   └── qpos: (T, state_dim)
    ├── action: (T, action_dim)
    └── instruction: string

Usage:
    python scripts/prepare_agibot_rdt.py \\
        --src-dir ./data/agibot_alpha \\
        --dst-dir ./data/agibot_hdf5 \\
        --robot-type aloha \\
        --num-frames 81 \\
        --fps 10
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass

import numpy as np
import h5py

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import decord
    decord.bridge.set_bridge('native')
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Instruction Templates (following RDT-style with scene descriptions)
# =============================================================================

INSTRUCTION_TEMPLATES = {
    "aloha": (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed front camera, a movable left arm camera, and a movable right arm camera. "
        "The aloha robot is currently performing the following task: {instruction}"
    ),
    "genie1": (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed high camera, a movable left arm camera, and a movable right arm camera. "
        "The genie-1 robot is currently performing the following task: {instruction}"
    ),
    "franka": (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed front camera, a fixed left camera, and a fixed right camera. "
        "The franka robot is currently performing the following task: {instruction}"
    ),
    "default": (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed front camera, a movable left arm camera, and a movable right arm camera. "
        "The robot is currently performing the following task: {instruction}"
    ),
}


# =============================================================================
# Camera Configuration
# =============================================================================

@dataclass
class CameraConfig:
    """Camera configuration for a robot type."""
    names: List[str]
    unified_layout: str  # 'standard', 'horizontal', 'vertical'
    resolution: Tuple[int, int]  # Per-camera resolution


CAMERA_CONFIGS = {
    "aloha": CameraConfig(
        names=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        unified_layout="standard",
        resolution=(480, 640),
    ),
    "genie1": CameraConfig(
        names=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        unified_layout="standard",
        resolution=(480, 640),
    ),
    "franka": CameraConfig(
        names=["cam_front", "cam_left", "cam_right"],
        unified_layout="standard",
        resolution=(480, 640),
    ),
}

# Unified observation space (from Vidarc paper)
UNIFIED_RESOLUTION = (720, 640)  # (H, W)
TARGET_FPS = 10


# =============================================================================
# Data Processing Classes
# =============================================================================

class InstructionFormatter:
    """Format instructions with scene descriptions."""

    def __init__(self, robot_type: str = "aloha"):
        self.robot_type = robot_type
        self.template = INSTRUCTION_TEMPLATES.get(
            robot_type, INSTRUCTION_TEMPLATES["default"]
        )

    def format(self, instruction: str) -> str:
        """Format instruction with scene description."""
        # Clean and normalize instruction
        instruction = self._normalize(instruction)
        return self.template.format(instruction=instruction)

    def format_variants(self, instruction: str) -> Dict[str, str]:
        """Generate instruction variants (original, simplified, expanded)."""
        normalized = self._normalize(instruction)
        return {
            "instruction": self.template.format(instruction=normalized),
            "instruction_raw": normalized,
            "instruction_simplified": self._simplify(normalized),
        }

    def _normalize(self, text: str) -> str:
        """Normalize instruction text."""
        text = text.strip()
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        # Add period if missing
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _simplify(self, text: str) -> str:
        """Create simplified version (verb + object)."""
        # Simple heuristic: take first sentence, remove articles
        text = text.split(".")[0].strip()
        for article in ["the ", "a ", "an "]:
            text = text.replace(article, "").replace(article.capitalize(), "")
        return text.strip() + "."


class UnifiedObservationComposer:
    """Compose multi-camera views into unified observation space."""

    def __init__(
        self,
        output_size: Tuple[int, int] = UNIFIED_RESOLUTION,
        camera_config: CameraConfig = None,
    ):
        self.output_size = output_size
        self.camera_config = camera_config or CAMERA_CONFIGS["aloha"]

    def compose(self, camera_images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compose camera images into unified observation.

        Args:
            camera_images: Dict mapping camera names to images (H, W, C)

        Returns:
            Unified image (720, 640, 3)
        """
        H, W = self.output_size
        unified = np.zeros((H, W, 3), dtype=np.uint8)

        # Get available cameras
        available = [c for c in self.camera_config.names if c in camera_images]

        if len(available) == 0:
            return unified

        if self.camera_config.unified_layout == "standard":
            return self._compose_standard(camera_images, available, H, W)
        else:
            return self._compose_horizontal(camera_images, available, H, W)

    def _compose_standard(
        self,
        camera_images: Dict[str, np.ndarray],
        available: List[str],
        H: int,
        W: int
    ) -> np.ndarray:
        """
        Standard layout:
            [      High/Front Camera (360, 640)      ]
            [ Left Arm (360, 320) | Right Arm (360, 320) ]
        """
        unified = np.zeros((H, W, 3), dtype=np.uint8)
        h_top = H // 2
        h_bottom = H - h_top
        w_half = W // 2

        # Find cameras by pattern matching
        high_cam = self._find_camera(available, ["high", "front", "head", "ext"])
        left_cam = self._find_camera(available, ["left"])
        right_cam = self._find_camera(available, ["right"])

        # Top: high/front camera
        if high_cam and high_cam in camera_images:
            img = camera_images[high_cam]
            resized = self._resize(img, (h_top, W))
            unified[:h_top, :, :] = resized

        # Bottom-left: left arm camera
        if left_cam and left_cam in camera_images:
            img = camera_images[left_cam]
            resized = self._resize(img, (h_bottom, w_half))
            unified[h_top:, :w_half, :] = resized

        # Bottom-right: right arm camera
        if right_cam and right_cam in camera_images:
            img = camera_images[right_cam]
            resized = self._resize(img, (h_bottom, W - w_half))
            unified[h_top:, w_half:, :] = resized

        return unified

    def _compose_horizontal(
        self,
        camera_images: Dict[str, np.ndarray],
        available: List[str],
        H: int,
        W: int
    ) -> np.ndarray:
        """Horizontal layout: [Left | Center | Right]"""
        unified = np.zeros((H, W, 3), dtype=np.uint8)
        n_cams = min(len(available), 3)
        w_each = W // n_cams

        for i, cam in enumerate(available[:3]):
            img = camera_images[cam]
            resized = self._resize(img, (H, w_each))
            unified[:, i * w_each:(i + 1) * w_each, :] = resized

        return unified

    def _find_camera(self, available: List[str], patterns: List[str]) -> Optional[str]:
        """Find camera matching any pattern."""
        for cam in available:
            cam_lower = cam.lower()
            for pattern in patterns:
                if pattern in cam_lower:
                    return cam
        return available[0] if available else None

    def _resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target (H, W)."""
        if HAS_CV2:
            return cv2.resize(img, (target_size[1], target_size[0]))
        else:
            # Nearest neighbor fallback
            h, w = img.shape[:2]
            th, tw = target_size
            y_idx = (np.arange(th) * h / th).astype(int)
            x_idx = (np.arange(tw) * w / tw).astype(int)
            return img[y_idx][:, x_idx]


class AgibotRDTProcessor:
    """Process AGIBOT dataset into RDT-style HDF5 format."""

    def __init__(
        self,
        src_dir: str,
        dst_dir: str,
        robot_type: str = "aloha",
        num_frames: int = 81,
        fps: int = 10,
    ):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.robot_type = robot_type
        self.num_frames = num_frames
        self.fps = fps

        # Initialize components
        self.instruction_formatter = InstructionFormatter(robot_type)
        self.camera_config = CAMERA_CONFIGS.get(robot_type, CAMERA_CONFIGS["aloha"])
        self.observation_composer = UnifiedObservationComposer(
            output_size=UNIFIED_RESOLUTION,
            camera_config=self.camera_config,
        )

        # Create output directories
        self.hdf5_dir = self.dst_dir / "hdf5"
        self.hdf5_dir.mkdir(parents=True, exist_ok=True)

        # Statistics accumulators
        self.stats = {
            "state_mean": None,
            "state_std": None,
            "action_mean": None,
            "action_std": None,
            "n_samples": 0,
        }

    def find_episodes(self) -> List[Dict[str, Any]]:
        """Find all episodes in source directory."""
        episodes = []

        # Look for task_info files
        task_info_dir = self.src_dir / "data" / "task_info"
        if not task_info_dir.exists():
            task_info_dir = self.src_dir / "task_info"

        if task_info_dir.exists():
            for task_file in task_info_dir.glob("task_*.json"):
                try:
                    with open(task_file, "r") as f:
                        task_data = json.load(f)

                    if isinstance(task_data, list):
                        for ep in task_data:
                            episodes.append(self._parse_episode(ep, task_file))
                    elif isinstance(task_data, dict):
                        episodes.append(self._parse_episode(task_data, task_file))
                except Exception as e:
                    logger.warning(f"Error loading {task_file}: {e}")
        else:
            # Fallback: scan observations directory
            episodes = self._find_episodes_fallback()

        logger.info(f"Found {len(episodes)} episodes")
        return episodes

    def _find_episodes_fallback(self) -> List[Dict[str, Any]]:
        """Fallback episode finding."""
        episodes = []
        obs_dir = self.src_dir / "data" / "observations"
        if not obs_dir.exists():
            obs_dir = self.src_dir / "observations"

        if obs_dir.exists():
            for task_dir in obs_dir.iterdir():
                if task_dir.is_dir():
                    for ep_dir in task_dir.iterdir():
                        if ep_dir.is_dir():
                            episodes.append({
                                "task_id": task_dir.name,
                                "episode_id": ep_dir.name,
                                "video_dir": ep_dir / "videos",
                                "instruction": f"Task {task_dir.name}",
                            })
        return episodes

    def _parse_episode(self, ep_data: Dict, task_file: Path) -> Dict[str, Any]:
        """Parse episode metadata."""
        task_id = ep_data.get("task_id", task_file.stem.replace("task_", ""))
        episode_id = ep_data.get("episode_id", "unknown")

        # Get instruction
        instruction = ""
        if "lable_info" in ep_data:
            label_info = ep_data["lable_info"]
            if "action_config" in label_info and label_info["action_config"]:
                instruction = label_info["action_config"][0].get("language", "")
        if not instruction:
            instruction = ep_data.get("task_name", f"Task {task_id}")

        # Build paths
        video_dir = self.src_dir / "data" / "observations" / str(task_id) / str(episode_id) / "videos"
        if not video_dir.exists():
            video_dir = self.src_dir / "observations" / str(task_id) / str(episode_id) / "videos"

        proprio_path = self.src_dir / "data" / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
        if not proprio_path.exists():
            proprio_path = self.src_dir / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"

        return {
            "task_id": task_id,
            "episode_id": episode_id,
            "video_dir": video_dir,
            "proprio_path": proprio_path,
            "instruction": instruction,
        }

    def process_episode(self, episode: Dict[str, Any], output_idx: int) -> Optional[str]:
        """Process single episode to HDF5."""
        try:
            video_dir = Path(episode["video_dir"])
            if not video_dir.exists():
                return None

            # Load camera videos
            camera_frames = self._load_camera_videos(video_dir)
            if not camera_frames:
                return None

            # Determine frame count
            min_frames = min(len(frames) for frames in camera_frames.values())
            if min_frames < 10:
                return None

            # Subsample frames
            if min_frames > self.num_frames:
                indices = np.linspace(0, min_frames - 1, self.num_frames, dtype=int)
            else:
                indices = list(range(min_frames))

            # Load proprioception data
            qpos, actions = self._load_proprio(episode.get("proprio_path"), indices)

            # Create unified observations
            unified_frames = []
            per_camera_frames = {cam: [] for cam in camera_frames.keys()}

            for idx in indices:
                frame_dict = {}
                for cam_name, frames in camera_frames.items():
                    if idx < len(frames):
                        frame = frames[idx]
                        frame_dict[cam_name] = frame
                        per_camera_frames[cam_name].append(frame)

                unified = self.observation_composer.compose(frame_dict)
                unified_frames.append(unified)

            # Format instruction
            instruction_variants = self.instruction_formatter.format_variants(
                episode["instruction"]
            )

            # Save to HDF5
            output_path = self.hdf5_dir / f"episode_{output_idx:06d}.hdf5"
            self._save_hdf5(
                output_path,
                unified_frames=np.array(unified_frames),
                per_camera_frames={k: np.array(v) for k, v in per_camera_frames.items()},
                qpos=qpos,
                actions=actions,
                instruction_variants=instruction_variants,
                metadata={
                    "task_id": episode["task_id"],
                    "episode_id": episode["episode_id"],
                    "robot_type": self.robot_type,
                    "fps": self.fps,
                }
            )

            return str(output_path)

        except Exception as e:
            logger.error(f"Error processing episode {episode.get('episode_id')}: {e}")
            return None

    def _load_camera_videos(self, video_dir: Path) -> Dict[str, np.ndarray]:
        """Load video frames for each camera."""
        camera_frames = {}
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        # Find video files
        video_files = {}
        for video_file in video_dir.glob("*"):
            if video_file.suffix.lower() in extensions:
                cam_name = video_file.stem.replace("_video", "")
                video_files[cam_name] = video_file

        # Load frames
        for cam_name, video_path in video_files.items():
            frames = self._load_video(video_path)
            if frames is not None:
                camera_frames[cam_name] = frames

        return camera_frames

    def _load_video(self, video_path: Path) -> Optional[np.ndarray]:
        """Load video frames."""
        try:
            if HAS_DECORD:
                vr = decord.VideoReader(str(video_path))
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()
                return frames
            elif HAS_CV2:
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                return np.array(frames) if frames else None
        except Exception as e:
            logger.warning(f"Error loading {video_path}: {e}")
            return None

    def _load_proprio(
        self,
        proprio_path: Optional[Path],
        indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load proprioception and action data."""
        if proprio_path is None or not Path(proprio_path).exists():
            n_frames = len(indices)
            return np.zeros((n_frames, 14), dtype=np.float32), \
                   np.zeros((n_frames, 14), dtype=np.float32)

        try:
            with h5py.File(proprio_path, "r") as f:
                # Load joint positions (state)
                if "/state/joint/position" in f:
                    qpos = f["/state/joint/position"][:]
                else:
                    qpos = np.zeros((len(indices), 14), dtype=np.float32)

                # Load actions
                if "/action/joint/position" in f:
                    actions = f["/action/joint/position"][:]
                else:
                    actions = qpos.copy()  # Use state as fallback

                # Subsample
                if len(qpos) > len(indices):
                    qpos = qpos[indices]
                    actions = actions[indices]

                return qpos.astype(np.float32), actions.astype(np.float32)

        except Exception as e:
            logger.warning(f"Error loading proprio: {e}")
            n_frames = len(indices)
            return np.zeros((n_frames, 14), dtype=np.float32), \
                   np.zeros((n_frames, 14), dtype=np.float32)

    def _save_hdf5(
        self,
        output_path: Path,
        unified_frames: np.ndarray,
        per_camera_frames: Dict[str, np.ndarray],
        qpos: np.ndarray,
        actions: np.ndarray,
        instruction_variants: Dict[str, str],
        metadata: Dict[str, Any],
    ):
        """Save episode to HDF5 file."""
        with h5py.File(output_path, "w") as f:
            # Observations group
            obs_group = f.create_group("observations")

            # Unified image
            obs_group.create_dataset(
                "unified_image",
                data=unified_frames,
                compression="gzip",
                compression_opts=4,
            )

            # Per-camera images
            images_group = obs_group.create_group("images")
            for cam_name, frames in per_camera_frames.items():
                images_group.create_dataset(
                    cam_name,
                    data=frames,
                    compression="gzip",
                    compression_opts=4,
                )

            # Proprioception
            obs_group.create_dataset("qpos", data=qpos)

            # Actions
            f.create_dataset("action", data=actions)

            # Instructions (with scene description)
            for key, value in instruction_variants.items():
                f.attrs[key] = value

            # Metadata
            for key, value in metadata.items():
                f.attrs[key] = str(value) if not isinstance(value, (int, float)) else value

    def create_config(self, processed_episodes: List[str]):
        """Create dataset configuration files (RDT-style)."""
        config = {
            "dataset_name": f"agibot_{self.robot_type}",
            "robot_type": self.robot_type,
            "episode_len": self.num_frames,
            "state_dim": 14,
            "action_dim": 14,
            "chunk_size": 64,
            "camera_names": self.camera_config.names,
            "unified_resolution": list(UNIFIED_RESOLUTION),
            "fps": self.fps,
            "n_episodes": len([p for p in processed_episodes if p]),
            "instruction_template": self.instruction_formatter.template,
        }

        config_path = self.dst_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Create episode list
        episodes_list = []
        for ep_path in processed_episodes:
            if ep_path:
                episodes_list.append({
                    "path": str(Path(ep_path).relative_to(self.dst_dir)),
                    "name": Path(ep_path).stem,
                })

        episodes_path = self.dst_dir / "episodes.json"
        with open(episodes_path, "w") as f:
            json.dump(episodes_list, f, indent=2)

        # Create language instructions JSON (for precomputing embeddings)
        instructions = {}
        for ep_path in processed_episodes:
            if ep_path:
                with h5py.File(ep_path, "r") as hf:
                    ep_name = Path(ep_path).stem
                    instructions[ep_name] = {
                        "instruction": hf.attrs.get("instruction", ""),
                        "instruction_raw": hf.attrs.get("instruction_raw", ""),
                        "instruction_simplified": hf.attrs.get("instruction_simplified", ""),
                    }

        instr_path = self.dst_dir / "instructions.json"
        with open(instr_path, "w") as f:
            json.dump(instructions, f, indent=2)

        logger.info(f"Created config at: {config_path}")

    def process_all(self, max_episodes: Optional[int] = None, num_workers: int = 4):
        """Process all episodes."""
        episodes = self.find_episodes()

        if max_episodes:
            episodes = episodes[:max_episodes]

        logger.info(f"Processing {len(episodes)} episodes...")

        processed = []
        if num_workers > 1 and len(episodes) > 10:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self.process_episode, ep, i): i
                    for i, ep in enumerate(episodes)
                }
                for future in tqdm(as_completed(futures), total=len(episodes)):
                    result = future.result()
                    processed.append(result)
        else:
            for i, ep in enumerate(tqdm(episodes)):
                result = self.process_episode(ep, i)
                processed.append(result)

        # Filter successful
        processed = [p for p in processed if p is not None]
        logger.info(f"Successfully processed {len(processed)} episodes")

        # Create config files
        self.create_config(processed)

        return processed


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AGIBOT dataset in RDT-style HDF5 format"
    )

    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Path to downloaded AGIBOT dataset",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="aloha",
        choices=["aloha", "genie1", "franka"],
        help="Robot type for instruction template",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Number of frames per episode (default: 81)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Target FPS (default: 10)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to process",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    if not Path(args.src_dir).exists():
        logger.error(f"Source directory not found: {args.src_dir}")
        sys.exit(1)

    if not HAS_CV2 and not HAS_DECORD:
        logger.error("Either opencv-python or decord is required")
        sys.exit(1)

    processor = AgibotRDTProcessor(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        robot_type=args.robot_type,
        num_frames=args.num_frames,
        fps=args.fps,
    )

    processor.process_all(
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
    )

    print("\n" + "=" * 60)
    print("Dataset prepared successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {args.dst_dir}")
    print(f"\nInstruction format example:")
    formatter = InstructionFormatter(args.robot_type)
    print(f'  "{formatter.format("pick up the apple")}"')
    print("\nTo train, update your config to use:")
    print(f"  data_dir: {args.dst_dir}")


if __name__ == "__main__":
    main()
