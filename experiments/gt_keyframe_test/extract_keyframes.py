"""
Keyframe extraction from evaluation videos.

This module provides functions to extract keyframes from video files
for use as ground truth subgoals in policy evaluation.

Supports caching to avoid re-extracting keyframes on repeated runs.
"""

import cv2
import torch
import torchvision
import numpy as np
import json
import hashlib
import os
import sys
from base64 import b64encode, b64decode
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class KeyframeInfo:
    """Information about an extracted keyframe."""
    frame_index: int
    timestamp: float  # seconds
    image_b64: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KeyframeInfo":
        return cls(**d)


# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent / ".keyframe_cache"


def get_video_mtime(video_path: str) -> float:
    """Get modification time of video file."""
    return os.path.getmtime(video_path)


def compute_cache_key(
    video_path: str,
    strategy: str,
    **params
) -> str:
    """
    Compute a unique cache key based on video path, strategy, and parameters.

    The key includes the video's modification time to invalidate cache
    when the source video changes.
    """
    video_path = str(Path(video_path).resolve())
    mtime = get_video_mtime(video_path)

    key_data = {
        "video_path": video_path,
        "mtime": mtime,
        "strategy": strategy,
        "params": params
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class KeyframeCache:
    """
    Cache for extracted keyframes.

    Saves keyframes to disk to avoid re-extraction on repeated runs.
    Cache is automatically invalidated when source video is modified.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize keyframe cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to .keyframe_cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, List[KeyframeInfo]] = {}

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file for a given key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        video_path: str,
        strategy: str,
        **params
    ) -> Optional[List[KeyframeInfo]]:
        """
        Get cached keyframes if available.

        Args:
            video_path: Path to video file
            strategy: Extraction strategy used
            **params: Extraction parameters

        Returns:
            List of KeyframeInfo if cached, None otherwise
        """
        cache_key = compute_cache_key(video_path, strategy, **params)

        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)

                keyframes = [KeyframeInfo.from_dict(kf) for kf in data["keyframes"]]

                # Store in memory cache
                self._memory_cache[cache_key] = keyframes
                return keyframes

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)

        return None

    def set(
        self,
        video_path: str,
        strategy: str,
        keyframes: List[KeyframeInfo],
        **params
    ) -> None:
        """
        Cache extracted keyframes.

        Args:
            video_path: Path to video file
            strategy: Extraction strategy used
            keyframes: Extracted keyframes to cache
            **params: Extraction parameters
        """
        cache_key = compute_cache_key(video_path, strategy, **params)

        # Store in memory cache
        self._memory_cache[cache_key] = keyframes

        # Store on disk
        cache_path = self._get_cache_path(cache_key)
        data = {
            "video_path": str(Path(video_path).resolve()),
            "strategy": strategy,
            "params": params,
            "num_keyframes": len(keyframes),
            "keyframes": [kf.to_dict() for kf in keyframes]
        }

        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def clear(self) -> int:
        """
        Clear all cached keyframes.

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        self._memory_cache.clear()
        return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "num_entries": len(cache_files),
            "memory_entries": len(self._memory_cache),
            "total_size_mb": total_size / (1024 * 1024)
        }


# Global cache instance
_global_cache: Optional[KeyframeCache] = None


def get_cache(cache_dir: Optional[str] = None) -> KeyframeCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None or (cache_dir and Path(cache_dir) != _global_cache.cache_dir):
        _global_cache = KeyframeCache(cache_dir)
    return _global_cache


def frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Convert RGB frame to base64 JPEG string."""
    img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(torch.uint8)
    jpeg_bytes = torchvision.io.encode_jpeg(img_tensor)
    return b64encode(jpeg_bytes.numpy().tobytes()).decode('utf-8')


def extract_keyframes_uniform(
    video_path: str,
    interval: int = 8,
    max_keyframes: int = 20,
    start_frame: int = 0,
    use_cache: bool = True,
    cache: Optional[KeyframeCache] = None
) -> List[KeyframeInfo]:
    """
    Extract keyframes at uniform intervals from video.

    Args:
        video_path: Path to the video file
        interval: Frame interval between keyframes (default: 8)
        max_keyframes: Maximum number of keyframes to extract (default: 20)
        start_frame: Starting frame index (default: 0)
        use_cache: Whether to use caching (default: True)
        cache: KeyframeCache instance (uses global cache if None)

    Returns:
        List of KeyframeInfo objects
    """
    # Check cache first
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(
            video_path, "uniform",
            interval=interval, max_keyframes=max_keyframes, start_frame=start_frame
        )
        if cached is not None:
            print(f"Using cached keyframes for {video_path}")
            return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keyframes = []
    for i in range(start_frame, total_frames, interval):
        if len(keyframes) >= max_keyframes:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB, then to base64 JPEG
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_b64 = frame_to_base64(frame_rgb)

        keyframes.append(KeyframeInfo(
            frame_index=i,
            timestamp=i / fps if fps > 0 else 0,
            image_b64=image_b64
        ))

    cap.release()

    # Cache the results
    if use_cache:
        cache = cache or get_cache()
        cache.set(
            video_path, "uniform", keyframes,
            interval=interval, max_keyframes=max_keyframes, start_frame=start_frame
        )

    return keyframes


def extract_keyframes_visual_change(
    video_path: str,
    threshold: float = 0.05,
    min_interval: int = 4,
    max_keyframes: int = 20,
    use_cache: bool = True,
    cache: Optional[KeyframeCache] = None
) -> List[KeyframeInfo]:
    """
    Extract keyframes based on visual change detection.

    Frames are selected when the mean squared difference from the previous
    keyframe exceeds the threshold.

    Args:
        video_path: Path to the video file
        threshold: MSE threshold for visual change detection (default: 0.05)
        min_interval: Minimum frames between keyframes (default: 4)
        max_keyframes: Maximum number of keyframes to extract (default: 20)
        use_cache: Whether to use caching (default: True)
        cache: KeyframeCache instance (uses global cache if None)

    Returns:
        List of KeyframeInfo objects
    """
    # Check cache first
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(
            video_path, "visual_change",
            threshold=threshold, min_interval=min_interval, max_keyframes=max_keyframes
        )
        if cached is not None:
            print(f"Using cached keyframes for {video_path}")
            return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframes = []
    prev_frame = None
    last_keyframe_idx = -min_interval
    frame_idx = 0

    # Always include the first frame
    ret, first_frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        image_b64 = frame_to_base64(frame_rgb)
        keyframes.append(KeyframeInfo(
            frame_index=0,
            timestamp=0,
            image_b64=image_b64
        ))
        prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        last_keyframe_idx = 0
        frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

        if prev_frame is not None:
            diff = np.mean((frame_gray - prev_frame) ** 2)

            if diff > threshold and (frame_idx - last_keyframe_idx) >= min_interval:
                # Convert to base64
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_b64 = frame_to_base64(frame_rgb)

                keyframes.append(KeyframeInfo(
                    frame_index=frame_idx,
                    timestamp=frame_idx / fps if fps > 0 else 0,
                    image_b64=image_b64
                ))
                last_keyframe_idx = frame_idx
                prev_frame = frame_gray

                if len(keyframes) >= max_keyframes:
                    break

        frame_idx += 1

    cap.release()

    # Cache the results
    if use_cache:
        cache = cache or get_cache()
        cache.set(
            video_path, "visual_change", keyframes,
            threshold=threshold, min_interval=min_interval, max_keyframes=max_keyframes
        )

    return keyframes


def extract_keyframes_gripper_change(
    video_path: str,
    hdf5_path: Optional[str] = None,
    gripper_indices: tuple = (6, 13),
    threshold: float = 0.3,
    min_interval: int = 4,
    max_keyframes: int = 20,
    use_cache: bool = True,
    cache: Optional[KeyframeCache] = None
) -> List[KeyframeInfo]:
    """
    Extract keyframes at gripper state changes (semantic keyframes).

    Detects moments when gripper opens or closes, which typically correspond
    to important task milestones (grasping, releasing objects).

    Args:
        video_path: Path to the video file
        hdf5_path: Path to HDF5 file with action data
        gripper_indices: Indices of gripper actions in action array (default: 6, 13 for dual arm)
        threshold: Threshold for gripper change detection (default: 0.3)
        min_interval: Minimum frames between keyframes (default: 4)
        max_keyframes: Maximum number of keyframes (default: 20)
        use_cache: Whether to use caching (default: True)
        cache: KeyframeCache instance (uses global cache if None)

    Returns:
        List of KeyframeInfo objects at gripper state change moments
    """
    # Check cache first
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(
            video_path, "gripper_change",
            hdf5_path=hdf5_path, gripper_indices=gripper_indices,
            threshold=threshold, min_interval=min_interval, max_keyframes=max_keyframes
        )
        if cached is not None:
            print(f"Using cached gripper keyframes for {video_path}")
            return cached

    if hdf5_path is None or not os.path.exists(hdf5_path):
        print(f"Warning: HDF5 file not found: {hdf5_path}, falling back to visual change detection")
        return extract_keyframes_visual_change(
            video_path, threshold=0.03, min_interval=min_interval,
            max_keyframes=max_keyframes, use_cache=use_cache, cache=cache
        )

    try:
        import h5py
    except ImportError:
        print("Warning: h5py not available, falling back to visual change detection")
        return extract_keyframes_visual_change(
            video_path, threshold=0.03, min_interval=min_interval,
            max_keyframes=max_keyframes, use_cache=use_cache, cache=cache
        )

    # Load action data from HDF5
    with h5py.File(hdf5_path, 'r') as f:
        if 'action' in f:
            actions = f['action'][:]
        elif 'actions' in f:
            actions = f['actions'][:]
        else:
            print(f"Warning: No action data in {hdf5_path}, falling back to visual change")
            return extract_keyframes_visual_change(
                video_path, threshold=0.03, min_interval=min_interval,
                max_keyframes=max_keyframes, use_cache=use_cache, cache=cache
            )

    # Detect gripper state changes
    keyframe_indices = [0]  # Always include first frame
    last_keyframe_idx = 0

    for idx in range(1, len(actions)):
        if (idx - last_keyframe_idx) < min_interval:
            continue

        # Check gripper state changes for both arms
        gripper_changed = False
        for g_idx in gripper_indices:
            if g_idx < actions.shape[1]:
                prev_state = actions[idx - 1, g_idx]
                curr_state = actions[idx, g_idx]
                if abs(curr_state - prev_state) > threshold:
                    gripper_changed = True
                    break

        if gripper_changed:
            keyframe_indices.append(idx)
            last_keyframe_idx = idx

            if len(keyframe_indices) >= max_keyframes:
                break

    # Always include last frame if not already included
    if keyframe_indices[-1] != len(actions) - 1 and len(keyframe_indices) < max_keyframes:
        keyframe_indices.append(len(actions) - 1)

    print(f"Detected {len(keyframe_indices)} gripper change keyframes at indices: {keyframe_indices}")

    # Extract frames from video at detected indices
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []

    for frame_idx in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_b64 = frame_to_base64(frame_rgb)

        keyframes.append(KeyframeInfo(
            frame_index=frame_idx,
            timestamp=frame_idx / fps if fps > 0 else 0,
            image_b64=image_b64
        ))

    cap.release()

    # Cache the results
    if use_cache:
        cache = cache or get_cache()
        cache.set(
            video_path, "gripper_change", keyframes,
            hdf5_path=hdf5_path, gripper_indices=gripper_indices,
            threshold=threshold, min_interval=min_interval, max_keyframes=max_keyframes
        )

    return keyframes


def extract_keyframes_action_milestone(
    video_path: str,
    hdf5_path: str,
    action_dim: int = 14,
    velocity_threshold: float = 0.1,
    min_interval: int = 8,
    max_keyframes: int = 20,
    use_cache: bool = True,
    cache: Optional[KeyframeCache] = None
) -> List[KeyframeInfo]:
    """
    Extract keyframes at action velocity milestones (start/stop of motion).

    Detects moments when robot transitions between moving and stationary states,
    which often correspond to task phase transitions.

    Args:
        video_path: Path to the video file
        hdf5_path: Path to HDF5 file with action data
        action_dim: Dimension of action vector (default: 14)
        velocity_threshold: Threshold for motion detection
        min_interval: Minimum frames between keyframes
        max_keyframes: Maximum number of keyframes

    Returns:
        List of KeyframeInfo objects at motion milestone moments
    """
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(
            video_path, "action_milestone",
            hdf5_path=hdf5_path, velocity_threshold=velocity_threshold,
            min_interval=min_interval, max_keyframes=max_keyframes
        )
        if cached is not None:
            print(f"Using cached milestone keyframes for {video_path}")
            return cached

    if not os.path.exists(hdf5_path):
        print(f"Warning: HDF5 file not found: {hdf5_path}")
        return extract_keyframes_uniform(video_path, interval=8, max_keyframes=max_keyframes)

    try:
        import h5py
    except ImportError:
        return extract_keyframes_uniform(video_path, interval=8, max_keyframes=max_keyframes)

    with h5py.File(hdf5_path, 'r') as f:
        if 'action' in f:
            actions = f['action'][:]
        elif 'actions' in f:
            actions = f['actions'][:]
        else:
            return extract_keyframes_uniform(video_path, interval=8, max_keyframes=max_keyframes)

    # Compute action velocity (frame-to-frame difference)
    # Exclude gripper indices (6, 13) for velocity calculation
    arm_indices = [i for i in range(min(action_dim, actions.shape[1])) if i not in (6, 13)]
    velocities = np.abs(np.diff(actions[:, arm_indices], axis=0)).mean(axis=1)

    # Detect motion state changes (moving ↔ stationary)
    is_moving = velocities > velocity_threshold
    keyframe_indices = [0]  # Always include first frame
    last_keyframe_idx = 0

    for idx in range(1, len(is_moving)):
        if (idx - last_keyframe_idx) < min_interval:
            continue

        # Detect state transition
        if idx > 0 and is_moving[idx] != is_moving[idx - 1]:
            keyframe_indices.append(idx)
            last_keyframe_idx = idx

            if len(keyframe_indices) >= max_keyframes:
                break

    # Add last frame
    if keyframe_indices[-1] != len(actions) - 1 and len(keyframe_indices) < max_keyframes:
        keyframe_indices.append(len(actions) - 1)

    print(f"Detected {len(keyframe_indices)} motion milestone keyframes at indices: {keyframe_indices}")

    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []

    for frame_idx in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_b64 = frame_to_base64(frame_rgb)

        keyframes.append(KeyframeInfo(
            frame_index=frame_idx,
            timestamp=frame_idx / fps if fps > 0 else 0,
            image_b64=image_b64
        ))

    cap.release()

    if use_cache:
        cache = cache or get_cache()
        cache.set(
            video_path, "action_milestone", keyframes,
            hdf5_path=hdf5_path, velocity_threshold=velocity_threshold,
            min_interval=min_interval, max_keyframes=max_keyframes
        )

    return keyframes


def extract_keyframes_semantic(
    video_path: str,
    motion_threshold: float = 0.01,
    change_threshold: float = 0.02,
    min_interval: int = 5,
    max_keyframes: int = 20,
    use_cache: bool = True,
    cache: Optional[KeyframeCache] = None
) -> List[KeyframeInfo]:
    """
    Extract semantic keyframes from video only (no HDF5 needed).

    Detects task-relevant moments by finding:
    1. Motion stops (robot pauses → likely grasping/releasing)
    2. Significant visual changes (object state changes)

    Args:
        video_path: Path to the video file
        motion_threshold: Threshold for motion detection (lower = more sensitive)
        change_threshold: Threshold for visual change detection
        min_interval: Minimum frames between keyframes
        max_keyframes: Maximum number of keyframes

    Returns:
        List of KeyframeInfo at semantic moments
    """
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(
            video_path, "semantic",
            motion_threshold=motion_threshold, change_threshold=change_threshold,
            min_interval=min_interval, max_keyframes=max_keyframes
        )
        if cached is not None:
            print(f"Using cached semantic keyframes for {video_path}")
            return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # First pass: compute frame-to-frame motion and visual change
    motion_scores = []
    change_scores = []
    prev_gray = None
    prev_frame = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

        if prev_gray is not None:
            # Motion score (mean absolute difference)
            motion = np.mean(np.abs(gray - prev_gray))
            motion_scores.append(motion)

            # Visual change score (structural difference)
            change = np.mean((gray - prev_gray) ** 2)
            change_scores.append(change)
        else:
            motion_scores.append(0)
            change_scores.append(0)

        prev_gray = gray
        prev_frame = frame
        frame_idx += 1

    cap.release()

    motion_scores = np.array(motion_scores)
    change_scores = np.array(change_scores)

    # Detect keyframe candidates
    keyframe_indices = [0]  # Always include first frame
    last_keyframe_idx = 0

    # Smooth motion scores to reduce noise
    kernel_size = 3
    if len(motion_scores) > kernel_size:
        smoothed_motion = np.convolve(motion_scores, np.ones(kernel_size)/kernel_size, mode='same')
    else:
        smoothed_motion = motion_scores

    # Detect motion stops (low motion after high motion)
    was_moving = False
    for idx in range(1, len(smoothed_motion)):
        if (idx - last_keyframe_idx) < min_interval:
            continue

        is_moving = smoothed_motion[idx] > motion_threshold

        # Detect motion stop (transition from moving to stationary)
        if was_moving and not is_moving:
            keyframe_indices.append(idx)
            last_keyframe_idx = idx
            if len(keyframe_indices) >= max_keyframes:
                break

        # Also detect significant visual changes
        elif change_scores[idx] > change_threshold and (idx - last_keyframe_idx) >= min_interval:
            keyframe_indices.append(idx)
            last_keyframe_idx = idx
            if len(keyframe_indices) >= max_keyframes:
                break

        was_moving = is_moving

    # Add last frame if not already included
    if len(keyframe_indices) < max_keyframes and keyframe_indices[-1] != total_frames - 1:
        keyframe_indices.append(total_frames - 1)

    # Sort and deduplicate
    keyframe_indices = sorted(set(keyframe_indices))

    print(f"Detected {len(keyframe_indices)} semantic keyframes at indices: {keyframe_indices}")

    # Extract frames at detected indices
    cap = cv2.VideoCapture(video_path)
    keyframes = []

    for frame_idx in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_b64 = frame_to_base64(frame_rgb)

        keyframes.append(KeyframeInfo(
            frame_index=frame_idx,
            timestamp=frame_idx / fps if fps > 0 else 0,
            image_b64=image_b64
        ))

    cap.release()

    # Cache results
    if use_cache:
        cache = cache or get_cache()
        cache.set(
            video_path, "semantic", keyframes,
            motion_threshold=motion_threshold, change_threshold=change_threshold,
            min_interval=min_interval, max_keyframes=max_keyframes
        )

    return keyframes


def get_video_info(video_path: str) -> dict:
    """Get basic information about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "path": video_path,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()
    return info


def find_eval_videos(
    eval_result_dir: str,
    task: str,
    episode_pattern: str = "episode*.mp4"
) -> List[str]:
    """
    Find evaluation result videos for a given task.

    Args:
        eval_result_dir: Base directory containing eval results
        task: Task name
        episode_pattern: Glob pattern for episode files

    Returns:
        List of video file paths
    """
    task_dir = Path(eval_result_dir) / task
    if not task_dir.exists():
        raise ValueError(f"Task directory not found: {task_dir}")

    videos = sorted(task_dir.glob(episode_pattern))
    return [str(v) for v in videos]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract keyframes from video")
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--strategy", choices=["uniform", "visual_change"],
                        default="uniform", help="Extraction strategy")
    parser.add_argument("--interval", type=int, default=8,
                        help="Frame interval for uniform extraction")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold for visual change detection")
    parser.add_argument("--max-keyframes", type=int, default=20,
                        help="Maximum number of keyframes")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save extracted keyframes (optional)")

    # Cache options
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Custom cache directory")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cache and exit")
    parser.add_argument("--cache-stats", action="store_true",
                        help="Show cache statistics and exit")

    args = parser.parse_args()

    # Handle cache management commands
    if args.clear_cache:
        cache = get_cache(args.cache_dir)
        count = cache.clear()
        print(f"Cleared {count} cache entries from {cache.cache_dir}")
        sys.exit(0)

    if args.cache_stats:
        cache = get_cache(args.cache_dir)
        stats = cache.get_stats()
        print("Cache Statistics:")
        print(f"  Directory: {stats['cache_dir']}")
        print(f"  Disk entries: {stats['num_entries']}")
        print(f"  Memory entries: {stats['memory_entries']}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        sys.exit(0)

    # Require video_path for extraction
    if not args.video_path:
        parser.error("video_path is required for extraction")

    # Get video info
    info = get_video_info(args.video_path)
    print(f"Video info: {info}")

    # Set up cache
    use_cache = not args.no_cache
    cache = get_cache(args.cache_dir) if use_cache else None

    # Extract keyframes
    if args.strategy == "uniform":
        keyframes = extract_keyframes_uniform(
            args.video_path,
            interval=args.interval,
            max_keyframes=args.max_keyframes,
            use_cache=use_cache,
            cache=cache
        )
    else:
        keyframes = extract_keyframes_visual_change(
            args.video_path,
            threshold=args.threshold,
            max_keyframes=args.max_keyframes,
            use_cache=use_cache,
            cache=cache
        )

    print(f"Extracted {len(keyframes)} keyframes:")
    for kf in keyframes:
        print(f"  Frame {kf.frame_index} @ {kf.timestamp:.2f}s")

    # Optionally save keyframes
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for kf in keyframes:
            img_bytes = b64decode(kf.image_b64)
            img_path = output_dir / f"keyframe_{kf.frame_index:04d}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            print(f"Saved: {img_path}")
