#!/usr/bin/env python3
"""
AGIBOT Dataset Preparation Script for Vidar Training.

Based on the Vidarc paper (arXiv:2512.17661):
- Dataset: AgiBot World (728,209 episodes for pretraining)
- Robot: Genie-1 Robot
- Cameras: fixed high camera + movable left/right arm cameras
- Resolution: 720x640 (unified observation space)
- FPS: 10 (downsampled from original)

Dataset structure (from HuggingFace):
    data/
    ├── task_info/
    │   └── task_XXX.json
    ├── observations/
    │   └── XXX/episode_id/videos/
    ├── parameters/
    │   └── XXX/episode_id/camera/
    └── proprio_stats/
        └── XXX/episode_id/proprio_stats.h5

Usage:
    # Download sample first
    huggingface-cli download --repo-type dataset agibot-world/AgiBotWorld-Alpha --local-dir ./data/agibot_alpha

    # Prepare dataset
    python scripts/prepare_agibot.py \
        --src-dir ./data/agibot_alpha \
        --dst-dir ./data/agibot_processed \
        --num-frames 81 \
        --fps 10 \
        --resolution 720 640

    # For subset (testing)
    python scripts/prepare_agibot.py \
        --src-dir ./data/agibot_alpha \
        --dst-dir ./data/agibot_processed \
        --max-episodes 100
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


# Camera mapping for AGIBOT (Genie-1 Robot)
# Paper: "a fixed high camera, a movable left arm camera, and a movable right arm camera"
AGIBOT_CAMERAS = {
    "high": "cam_high",      # Fixed high camera
    "left": "cam_left_wrist",  # Left arm camera
    "right": "cam_right_wrist",  # Right arm camera
}

# Target resolution from paper (unified observation space)
TARGET_RESOLUTION = (720, 640)  # (H, W)
TARGET_FPS = 10


class AgibotProcessor:
    """Process AGIBOT dataset into Vidar training format."""

    def __init__(
        self,
        src_dir: str,
        dst_dir: str,
        num_frames: int = 81,
        fps: int = 10,
        resolution: Tuple[int, int] = (720, 640),
        cameras: Optional[List[str]] = None,
    ):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.num_frames = num_frames
        self.fps = fps
        self.resolution = resolution  # (H, W)
        self.cameras = cameras or list(AGIBOT_CAMERAS.values())

        # Create output directories
        self.episodes_dir = self.dst_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def find_episodes(self) -> List[Dict[str, Any]]:
        """Find all episodes in the source directory."""
        episodes = []

        # Look for task_info files
        task_info_dir = self.src_dir / "data" / "task_info"
        if not task_info_dir.exists():
            task_info_dir = self.src_dir / "task_info"

        if not task_info_dir.exists():
            logger.warning(f"task_info directory not found in {self.src_dir}")
            # Try alternative structure
            return self._find_episodes_alternative()

        for task_file in task_info_dir.glob("task_*.json"):
            try:
                with open(task_file, "r") as f:
                    task_data = json.load(f)

                # Handle both list and dict formats
                if isinstance(task_data, list):
                    for ep in task_data:
                        episodes.append(self._parse_episode(ep, task_file))
                elif isinstance(task_data, dict):
                    episodes.append(self._parse_episode(task_data, task_file))

            except Exception as e:
                logger.warning(f"Error loading {task_file}: {e}")

        logger.info(f"Found {len(episodes)} episodes")
        return episodes

    def _find_episodes_alternative(self) -> List[Dict[str, Any]]:
        """Alternative episode finding for different directory structures."""
        episodes = []

        # Try observations directory structure
        obs_dir = self.src_dir / "data" / "observations"
        if not obs_dir.exists():
            obs_dir = self.src_dir / "observations"

        if obs_dir.exists():
            for task_dir in obs_dir.iterdir():
                if task_dir.is_dir():
                    task_id = task_dir.name
                    for ep_dir in task_dir.iterdir():
                        if ep_dir.is_dir():
                            episodes.append({
                                "task_id": task_id,
                                "episode_id": ep_dir.name,
                                "video_dir": ep_dir / "videos",
                                "instruction": f"Task {task_id}",
                            })

        logger.info(f"Found {len(episodes)} episodes (alternative method)")
        return episodes

    def _parse_episode(self, ep_data: Dict, task_file: Path) -> Dict[str, Any]:
        """Parse episode metadata."""
        task_id = ep_data.get("task_id", task_file.stem.replace("task_", ""))
        episode_id = ep_data.get("episode_id", "unknown")

        # Get instruction from action_config or task_name
        instruction = ""
        if "lable_info" in ep_data:
            label_info = ep_data["lable_info"]
            if "action_config" in label_info and label_info["action_config"]:
                instruction = label_info["action_config"][0].get("language", "")
        if not instruction:
            instruction = ep_data.get("task_name", f"Task {task_id}")

        # Build video directory path
        video_dir = self.src_dir / "data" / "observations" / str(task_id) / str(episode_id) / "videos"
        if not video_dir.exists():
            video_dir = self.src_dir / "observations" / str(task_id) / str(episode_id) / "videos"

        # Build proprio path
        proprio_path = self.src_dir / "data" / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
        if not proprio_path.exists():
            proprio_path = self.src_dir / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"

        return {
            "task_id": task_id,
            "episode_id": episode_id,
            "video_dir": video_dir,
            "proprio_path": proprio_path,
            "instruction": instruction,
            "raw_data": ep_data,
        }

    def process_episode(self, episode: Dict[str, Any], output_idx: int) -> Optional[str]:
        """Process a single episode."""
        try:
            video_dir = episode["video_dir"]
            if not video_dir.exists():
                logger.warning(f"Video dir not found: {video_dir}")
                return None

            # Find video files for each camera
            video_files = self._find_camera_videos(video_dir)
            if not video_files:
                logger.warning(f"No video files found in {video_dir}")
                return None

            # Create output directory
            ep_output_dir = self.episodes_dir / f"ep_{output_idx:06d}"
            ep_output_dir.mkdir(parents=True, exist_ok=True)

            # Process and merge videos into unified observation
            video_path = self._create_unified_video(video_files, ep_output_dir)
            if video_path is None:
                return None

            # Save instruction
            instruction_path = ep_output_dir / "instruction.txt"
            with open(instruction_path, "w") as f:
                f.write(episode["instruction"])

            # Process actions if available
            if episode.get("proprio_path") and Path(episode["proprio_path"]).exists():
                self._process_actions(episode["proprio_path"], ep_output_dir)

            return str(ep_output_dir)

        except Exception as e:
            logger.error(f"Error processing episode {episode.get('episode_id')}: {e}")
            return None

    def _find_camera_videos(self, video_dir: Path) -> Dict[str, Path]:
        """Find video files for each camera."""
        video_files = {}

        # Common video extensions
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        for cam_name in self.cameras:
            for ext in extensions:
                # Try different naming patterns
                patterns = [
                    video_dir / f"{cam_name}{ext}",
                    video_dir / f"{cam_name}_video{ext}",
                    video_dir / cam_name / f"video{ext}",
                ]
                for pattern in patterns:
                    if pattern.exists():
                        video_files[cam_name] = pattern
                        break

        # If specific cameras not found, try to find any video files
        if not video_files:
            for video_file in video_dir.glob("*"):
                if video_file.suffix.lower() in extensions:
                    cam_name = video_file.stem.replace("_video", "")
                    video_files[cam_name] = video_file

        return video_files

    def _create_unified_video(
        self,
        video_files: Dict[str, Path],
        output_dir: Path
    ) -> Optional[Path]:
        """
        Create unified observation video (720x640).

        Layout from paper: 3 cameras arranged in unified 720x640 space
        Typical layout:
            [  High Camera (top)  ]
            [ Left  |   Right     ]
        """
        if not HAS_DECORD and not HAS_CV2:
            raise ImportError("Either decord or cv2 is required for video processing")

        # Load frames from each camera
        camera_frames = {}
        min_frames = float('inf')

        for cam_name, video_path in video_files.items():
            frames = self._load_video_frames(video_path)
            if frames is not None and len(frames) > 0:
                camera_frames[cam_name] = frames
                min_frames = min(min_frames, len(frames))

        if not camera_frames:
            return None

        # Subsample to target frame count
        if min_frames > self.num_frames:
            indices = np.linspace(0, min_frames - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(min_frames))

        # Create unified frames
        unified_frames = []
        H, W = self.resolution

        for idx in indices:
            # Get frames from available cameras
            frame_dict = {}
            for cam_name, frames in camera_frames.items():
                if idx < len(frames):
                    frame_dict[cam_name] = frames[idx]

            # Create unified layout
            unified = self._compose_unified_frame(frame_dict, H, W)
            unified_frames.append(unified)

        # Save as video
        output_path = output_dir / "video.mp4"
        self._save_video(unified_frames, output_path, self.fps)

        return output_path

    def _load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
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
            logger.warning(f"Error loading video {video_path}: {e}")
            return None

    def _compose_unified_frame(
        self,
        frame_dict: Dict[str, np.ndarray],
        H: int,
        W: int
    ) -> np.ndarray:
        """
        Compose frames into unified 720x640 layout.

        Paper uses layout:
        - Top: High camera (360x640)
        - Bottom-left: Left arm camera (360x320)
        - Bottom-right: Right arm camera (360x320)
        """
        unified = np.zeros((H, W, 3), dtype=np.uint8)

        # Determine layout based on available cameras
        cam_names = list(frame_dict.keys())

        if len(cam_names) == 1:
            # Single camera: resize to full frame
            frame = frame_dict[cam_names[0]]
            unified = self._resize_frame(frame, (H, W))

        elif len(cam_names) == 2:
            # Two cameras: split horizontally
            h_half = H // 2
            for i, cam_name in enumerate(sorted(cam_names)):
                frame = frame_dict[cam_name]
                resized = self._resize_frame(frame, (h_half, W))
                unified[i * h_half:(i + 1) * h_half, :, :] = resized

        elif len(cam_names) >= 3:
            # Three cameras: standard layout
            h_top = H // 2
            h_bottom = H - h_top
            w_half = W // 2

            # Find cameras by name patterns
            high_cam = None
            left_cam = None
            right_cam = None

            for cam_name in cam_names:
                lower_name = cam_name.lower()
                if "high" in lower_name or "top" in lower_name or "head" in lower_name:
                    high_cam = cam_name
                elif "left" in lower_name:
                    left_cam = cam_name
                elif "right" in lower_name:
                    right_cam = cam_name

            # Fallback: use first three cameras
            if high_cam is None:
                high_cam = cam_names[0]
            if left_cam is None:
                left_cam = cam_names[1] if len(cam_names) > 1 else cam_names[0]
            if right_cam is None:
                right_cam = cam_names[2] if len(cam_names) > 2 else cam_names[0]

            # Top: high camera
            if high_cam in frame_dict:
                frame = frame_dict[high_cam]
                resized = self._resize_frame(frame, (h_top, W))
                unified[:h_top, :, :] = resized

            # Bottom-left: left arm camera
            if left_cam in frame_dict:
                frame = frame_dict[left_cam]
                resized = self._resize_frame(frame, (h_bottom, w_half))
                unified[h_top:, :w_half, :] = resized

            # Bottom-right: right arm camera
            if right_cam in frame_dict:
                frame = frame_dict[right_cam]
                resized = self._resize_frame(frame, (h_bottom, W - w_half))
                unified[h_top:, w_half:, :] = resized

        return unified

    def _resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame to target size (H, W)."""
        if HAS_CV2:
            return cv2.resize(frame, (target_size[1], target_size[0]))
        else:
            # Simple nearest neighbor resize using numpy
            h, w = frame.shape[:2]
            th, tw = target_size
            y_indices = (np.arange(th) * h / th).astype(int)
            x_indices = (np.arange(tw) * w / tw).astype(int)
            return frame[y_indices][:, x_indices]

    def _save_video(self, frames: List[np.ndarray], output_path: Path, fps: int):
        """Save frames as video."""
        if not HAS_CV2:
            # Fallback: save as numpy array
            np.save(str(output_path).replace(".mp4", ".npy"), np.array(frames))
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        writer.release()

    def _process_actions(self, proprio_path: Path, output_dir: Path):
        """Extract and save action data."""
        try:
            with h5py.File(proprio_path, "r") as f:
                # Extract joint positions as actions
                if "/action/joint/position" in f:
                    actions = f["/action/joint/position"][:]
                elif "/state/joint/position" in f:
                    # Use state as fallback
                    actions = f["/state/joint/position"][:]
                else:
                    return

                # Subsample to match video frames
                if len(actions) > self.num_frames:
                    indices = np.linspace(0, len(actions) - 1, self.num_frames, dtype=int)
                    actions = actions[indices]

                np.save(output_dir / "actions.npy", actions.astype(np.float32))

        except Exception as e:
            logger.warning(f"Error processing actions: {e}")

    def create_manifest(self, processed_episodes: List[str]):
        """Create manifest.json for the dataset."""
        episodes = []
        for ep_path in processed_episodes:
            if ep_path is None:
                continue
            ep_dir = Path(ep_path)
            episode = {
                "id": ep_dir.name,
                "video_path": str(ep_dir / "video.mp4"),
                "instruction_path": str(ep_dir / "instruction.txt"),
                "actions_path": str(ep_dir / "actions.npy"),
            }
            episodes.append(episode)

        manifest = {"episodes": episodes}
        manifest_path = self.dst_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created manifest with {len(episodes)} episodes")

    def process_all(self, max_episodes: Optional[int] = None, num_workers: int = 4):
        """Process all episodes."""
        episodes = self.find_episodes()

        if max_episodes:
            episodes = episodes[:max_episodes]

        logger.info(f"Processing {len(episodes)} episodes...")

        processed = []
        if num_workers > 1:
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

        # Create manifest
        self.create_manifest(processed)

        return processed


def main():
    parser = argparse.ArgumentParser(description="Prepare AGIBOT dataset for Vidar training")

    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Path to downloaded AGIBOT dataset (AgiBotWorld-Alpha or Beta)",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        required=True,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Number of frames per video (default: 81)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Target FPS (default: 10)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[720, 640],
        help="Target resolution H W (default: 720 640)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to process (for testing)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.src_dir).exists():
        logger.error(f"Source directory not found: {args.src_dir}")
        sys.exit(1)

    if not HAS_CV2 and not HAS_DECORD:
        logger.error("Either opencv-python or decord is required")
        logger.error("Install with: pip install opencv-python decord")
        sys.exit(1)

    # Process
    processor = AgibotProcessor(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        num_frames=args.num_frames,
        fps=args.fps,
        resolution=tuple(args.resolution),
    )

    processor.process_all(
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
    )

    logger.info(f"Dataset prepared at: {args.dst_dir}")
    logger.info("To train, run:")
    logger.info(f"  python scripts/train_vidar.py --config configs/vidar_finetune.yaml --data-dir {args.dst_dir}")


if __name__ == "__main__":
    main()
