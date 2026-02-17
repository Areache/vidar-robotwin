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


def base64_to_numpy(image_b64: str) -> np.ndarray:
    """Convert base64 JPEG string to numpy RGB array."""
    jpeg_bytes = b64decode(image_b64)
    img_tensor = torchvision.io.decode_jpeg(torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8))
    return img_tensor.permute(1, 2, 0).numpy()


def visualize_keyframes(
    keyframes: List[KeyframeInfo],
    output_path: Optional[str] = None,
    max_cols: int = 5,
    thumbnail_size: tuple = (320, 240),
    show_info: bool = True,
    title: str = "Extracted Keyframes"
) -> np.ndarray:
    """
    Visualize extracted keyframes in a single spliced image.

    Args:
        keyframes: List of KeyframeInfo objects
        output_path: Path to save the visualization (optional)
        max_cols: Maximum columns in the grid (default: 5)
        thumbnail_size: Size of each thumbnail (width, height) (default: 320x240)
        show_info: Whether to show frame index and timestamp (default: True)
        title: Title for the visualization (default: "Extracted Keyframes")

    Returns:
        Combined image as numpy array (RGB)
    """
    if not keyframes:
        raise ValueError("No keyframes to visualize")

    n_keyframes = len(keyframes)
    n_cols = min(n_keyframes, max_cols)
    n_rows = (n_keyframes + n_cols - 1) // n_cols

    thumb_w, thumb_h = thumbnail_size
    title_height = 40 if title else 0
    info_height = 25 if show_info else 0

    # Create output image
    total_height = title_height + n_rows * (thumb_h + info_height)
    total_width = n_cols * thumb_w
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    # Add title
    if title:
        cv2.putText(
            canvas, title,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )

    # Add keyframes to grid
    for idx, kf in enumerate(keyframes):
        row = idx // n_cols
        col = idx % n_cols

        # Decode image
        img = base64_to_numpy(kf.image_b64)

        # Resize to thumbnail
        img_resized = cv2.resize(img, thumbnail_size, interpolation=cv2.INTER_AREA)

        # Calculate position
        y_start = title_height + row * (thumb_h + info_height)
        x_start = col * thumb_w

        # Place thumbnail
        canvas[y_start:y_start + thumb_h, x_start:x_start + thumb_w] = img_resized

        # Add info text
        if show_info:
            info_text = f"#{kf.frame_index} @ {kf.timestamp:.2f}s"
            text_y = y_start + thumb_h + 18
            cv2.putText(
                canvas, info_text,
                (x_start + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

    # Save if output path provided
    if output_path:
        # Convert RGB to BGR for cv2.imwrite
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, canvas_bgr)
        print(f"Saved visualization to: {output_path}")

    return canvas


def visualize_keyframes_timeline(
    keyframes: List[KeyframeInfo],
    total_frames: int,
    output_path: Optional[str] = None,
    height: int = 100,
    width: int = 800
) -> np.ndarray:
    """
    Visualize keyframe positions on a timeline.

    Args:
        keyframes: List of KeyframeInfo objects
        total_frames: Total number of frames in the video
        output_path: Path to save the visualization (optional)
        height: Height of the timeline image
        width: Width of the timeline image

    Returns:
        Timeline image as numpy array (RGB)
    """
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw timeline bar
    bar_y = height // 2
    bar_height = 10
    cv2.rectangle(canvas, (20, bar_y - bar_height // 2), (width - 20, bar_y + bar_height // 2), (200, 200, 200), -1)

    # Draw keyframe markers
    for idx, kf in enumerate(keyframes):
        x = int(20 + (kf.frame_index / total_frames) * (width - 40))

        # Draw marker
        cv2.circle(canvas, (x, bar_y), 8, (0, 120, 255), -1)
        cv2.circle(canvas, (x, bar_y), 8, (0, 0, 0), 1)

        # Draw frame number (alternating above/below to avoid overlap)
        text_y = bar_y - 20 if idx % 2 == 0 else bar_y + 30
        cv2.putText(canvas, str(kf.frame_index), (x - 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw start/end labels
    cv2.putText(canvas, "0", (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(canvas, str(total_frames), (width - 50, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    if output_path:
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, canvas_bgr)
        print(f"Saved timeline to: {output_path}")

    return canvas


# =============================================================================
# HDF5 Extraction Functions
# =============================================================================

def extract_keyframes_from_hdf5(
    hdf5_path: str,
    strategy: str = "uniform",
    interval: int = 8,
    max_keyframes: int = 20,
    gripper_indices: tuple = (6, 13),
    gripper_threshold: float = 0.3,
    use_cache: bool = True,
    cache: Optional[KeyframeCache] = None
) -> List[KeyframeInfo]:
    """
    Extract keyframes directly from HDF5 file (no video needed).

    HDF5 Structure Expected:
        observations/unified_image: (T, H, W, 3) uint8 RGB images
        action: (T, 14) action array (optional, for gripper-based extraction)

    Args:
        hdf5_path: Path to HDF5 file
        strategy: "uniform", "gripper", "visual_change",
                  "action_milestone", "semantic", "composite"
        interval: Frame interval for uniform extraction
        max_keyframes: Maximum number of keyframes
        gripper_indices: Indices of gripper actions (for gripper strategy)
        gripper_threshold: Threshold for gripper state change detection
        use_cache: Whether to use caching
        cache: KeyframeCache instance

    Returns:
        List of KeyframeInfo objects
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 extraction: pip install h5py")

    # Check cache
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(
            hdf5_path, f"hdf5_{strategy}",
            interval=interval, max_keyframes=max_keyframes,
            gripper_indices=gripper_indices, gripper_threshold=gripper_threshold
        )
        if cached is not None:
            print(f"Using cached keyframes for {hdf5_path}")
            return cached

    with h5py.File(hdf5_path, 'r') as f:
        # Find image data
        if 'observations/unified_image' in f:
            images = f['observations/unified_image'][:]
        elif 'observations/images/cam_high' in f:
            # Fallback: use single camera
            images = f['observations/images/cam_high'][:]
        else:
            raise ValueError(f"No image data found in {hdf5_path}. "
                           f"Expected 'observations/unified_image' or 'observations/images/cam_high'")

        # Load actions if available (for gripper strategy)
        actions = None
        if 'action' in f:
            actions = f['action'][:]
        elif 'actions' in f:
            actions = f['actions'][:]

    total_frames = len(images)
    fps = 10  # Assume 10 FPS for HDF5 data (standard for RoboTwin)

    print(f"HDF5 loaded: {total_frames} frames, shape {images.shape}")

    # Select keyframe indices based on strategy
    if strategy == "uniform":
        keyframe_indices = list(range(0, total_frames, interval))[:max_keyframes]

    elif strategy == "gripper" and actions is not None:
        keyframe_indices = _detect_gripper_keyframes(
            actions, gripper_indices, gripper_threshold, max_keyframes
        )

    elif strategy == "visual_change":
        keyframe_indices = _detect_visual_change_keyframes(
            images, threshold=0.05, min_interval=4, max_keyframes=max_keyframes
        )

    elif strategy == "action_milestone" and actions is not None:
        keyframe_indices = _detect_action_milestone_keyframes(
            actions, velocity_threshold=0.1, min_interval=8, max_keyframes=max_keyframes
        )

    elif strategy == "semantic":
        keyframe_indices = _detect_semantic_keyframes(
            images, actions,
            motion_threshold=0.01, change_threshold=0.02,
            min_interval=5, max_keyframes=max_keyframes
        )

    elif strategy == "composite" and actions is not None:
        keyframe_indices = _detect_composite_keyframes(
            images, actions, gripper_indices, gripper_threshold,
            visual_threshold=0.05, gap_threshold=30, min_interval=4,
            max_keyframes=max_keyframes
        )

    else:
        # Default to uniform if strategy not supported
        print(f"Strategy '{strategy}' not supported or missing data, using uniform")
        keyframe_indices = list(range(0, total_frames, interval))[:max_keyframes]

    # Ensure first and last frames are included
    if 0 not in keyframe_indices:
        keyframe_indices.insert(0, 0)
    if total_frames - 1 not in keyframe_indices and len(keyframe_indices) < max_keyframes:
        keyframe_indices.append(total_frames - 1)

    keyframe_indices = sorted(set(keyframe_indices))[:max_keyframes]

    print(f"Extracting {len(keyframe_indices)} keyframes at indices: {keyframe_indices}")

    # Extract keyframes
    keyframes = []
    for frame_idx in keyframe_indices:
        img = images[frame_idx]

        # Ensure RGB format (H, W, 3)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[0] == 3:  # (C, H, W) format
            img = np.transpose(img, (1, 2, 0))

        image_b64 = frame_to_base64(img)

        keyframes.append(KeyframeInfo(
            frame_index=frame_idx,
            timestamp=frame_idx / fps,
            image_b64=image_b64
        ))

    # Cache results
    if use_cache:
        cache = cache or get_cache()
        cache.set(
            hdf5_path, f"hdf5_{strategy}", keyframes,
            interval=interval, max_keyframes=max_keyframes,
            gripper_indices=gripper_indices, gripper_threshold=gripper_threshold
        )

    return keyframes


def _detect_gripper_keyframes(
    actions: np.ndarray,
    gripper_indices: tuple,
    threshold: float,
    max_keyframes: int
) -> List[int]:
    """Detect keyframes at gripper state changes."""
    # Diagnostic: print gripper signal stats so we can calibrate threshold
    for g_idx in gripper_indices:
        if g_idx < actions.shape[1]:
            vals = actions[:, g_idx]
            deltas = np.abs(np.diff(vals))
            print(f"  [gripper diag] idx={g_idx}: val range=[{vals.min():.4f}, {vals.max():.4f}], "
                  f"delta max={deltas.max():.4f}, mean={deltas.mean():.4f}, "
                  f"p90={np.percentile(deltas, 90):.4f}, threshold={threshold:.4f}")

    keyframe_indices = [0]
    last_idx = 0
    min_interval = 4

    for idx in range(1, len(actions)):
        if idx - last_idx < min_interval:
            continue

        for g_idx in gripper_indices:
            if g_idx < actions.shape[1]:
                if abs(actions[idx, g_idx] - actions[idx - 1, g_idx]) > threshold:
                    keyframe_indices.append(idx)
                    last_idx = idx
                    break

        if len(keyframe_indices) >= max_keyframes:
            break

    return keyframe_indices


def _detect_visual_change_keyframes(
    images: np.ndarray,
    threshold: float,
    min_interval: int,
    max_keyframes: int
) -> List[int]:
    """Detect keyframes based on visual change in image sequence."""
    # Diagnostic: compute all diffs first to print stats
    all_diffs = []
    prev_g = None
    for i in range(len(images)):
        img = images[i]
        if img.ndim == 3 and img.shape[-1] == 3:
            g = np.mean(img, axis=-1) / 255.0
        else:
            g = img.astype(float) / 255.0
        if prev_g is not None:
            all_diffs.append(np.mean((g - prev_g) ** 2))
        prev_g = g
    if all_diffs:
        ad = np.array(all_diffs)
        print(f"  [visual diag] MSE range=[{ad.min():.6f}, {ad.max():.6f}], "
              f"mean={ad.mean():.6f}, p90={np.percentile(ad, 90):.6f}, "
              f"p95={np.percentile(ad, 95):.6f}, threshold={threshold:.6f}")

    keyframe_indices = [0]
    last_idx = 0

    prev_gray = None
    for idx in range(len(images)):
        img = images[idx]
        if img.ndim == 3 and img.shape[-1] == 3:
            gray = np.mean(img, axis=-1) / 255.0
        else:
            gray = img.astype(float) / 255.0

        if prev_gray is not None and idx - last_idx >= min_interval:
            diff = np.mean((gray - prev_gray) ** 2)
            if diff > threshold:
                keyframe_indices.append(idx)
                last_idx = idx

        prev_gray = gray

        if len(keyframe_indices) >= max_keyframes:
            break

    return keyframe_indices


def _detect_action_milestone_keyframes(
    actions: np.ndarray,
    velocity_threshold: float = 0.1,
    min_interval: int = 8,
    max_keyframes: int = 20
) -> List[int]:
    """Detect keyframes at motion state transitions (moving <-> stationary).

    Uses arm joint velocities (excluding gripper indices 6, 13) to find
    moments when the robot starts or stops moving — task phase boundaries.
    """
    arm_indices = [i for i in range(actions.shape[1]) if i not in (6, 13)]
    velocities = np.abs(np.diff(actions[:, arm_indices], axis=0)).mean(axis=1)

    is_moving = velocities > velocity_threshold
    keyframe_indices = [0]
    last_idx = 0

    for idx in range(1, len(is_moving)):
        if idx - last_idx < min_interval:
            continue
        # Detect state transition (moving <-> stationary)
        if is_moving[idx] != is_moving[idx - 1]:
            keyframe_indices.append(idx)
            last_idx = idx
            if len(keyframe_indices) >= max_keyframes:
                break

    return keyframe_indices


def _detect_semantic_keyframes(
    images: np.ndarray,
    actions: Optional[np.ndarray],
    motion_threshold: float = 0.01,
    change_threshold: float = 0.02,
    min_interval: int = 5,
    max_keyframes: int = 20
) -> List[int]:
    """Detect semantic keyframes via 2-pass: motion stops + visual changes.

    Pass 1: Compute frame-to-frame motion scores (grayscale MAD).
    Pass 2: Detect motion stops (high->low) and significant visual changes.
    """
    T = len(images)

    # Pass 1: compute motion and change scores from images
    motion_scores = np.zeros(T)
    change_scores = np.zeros(T)
    prev_gray = None

    for idx in range(T):
        img = images[idx]
        if img.ndim == 3 and img.shape[-1] == 3:
            gray = np.mean(img, axis=-1) / 255.0
        else:
            gray = img.astype(float) / 255.0

        if prev_gray is not None:
            motion_scores[idx] = np.mean(np.abs(gray - prev_gray))
            change_scores[idx] = np.mean((gray - prev_gray) ** 2)

        prev_gray = gray

    # Diagnostic: print signal stats for threshold calibration
    ms = motion_scores[1:]  # skip frame 0 (always 0)
    cs = change_scores[1:]
    if len(ms) > 0:
        print(f"  [semantic diag] motion MAD: range=[{ms.min():.6f}, {ms.max():.6f}], "
              f"mean={ms.mean():.6f}, p50={np.percentile(ms, 50):.6f}, "
              f"p90={np.percentile(ms, 90):.6f}, threshold={motion_threshold:.6f}")
        print(f"  [semantic diag] change MSE: range=[{cs.min():.6f}, {cs.max():.6f}], "
              f"mean={cs.mean():.6f}, p50={np.percentile(cs, 50):.6f}, "
              f"p90={np.percentile(cs, 90):.6f}, threshold={change_threshold:.6f}")

    # Smooth motion scores
    kernel_size = 3
    if T > kernel_size:
        smoothed = np.convolve(motion_scores, np.ones(kernel_size) / kernel_size, mode='same')
    else:
        smoothed = motion_scores

    # Pass 2: detect motion stops and visual changes
    keyframe_indices = [0]
    last_idx = 0
    was_moving = False

    for idx in range(1, T):
        if idx - last_idx < min_interval:
            was_moving = smoothed[idx] > motion_threshold
            continue

        is_now_moving = smoothed[idx] > motion_threshold

        # Motion stop (transition from moving to stationary)
        if was_moving and not is_now_moving:
            keyframe_indices.append(idx)
            last_idx = idx
        # Significant visual change
        elif change_scores[idx] > change_threshold:
            keyframe_indices.append(idx)
            last_idx = idx

        was_moving = is_now_moving

        if len(keyframe_indices) >= max_keyframes:
            break

    return keyframe_indices


def _detect_composite_keyframes(
    images: np.ndarray,
    actions: np.ndarray,
    gripper_indices: tuple = (6, 13),
    gripper_threshold: float = 0.3,
    visual_threshold: float = 0.05,
    gap_threshold: int = 30,
    min_interval: int = 4,
    max_keyframes: int = 20
) -> List[int]:
    """Composite strategy: gripper anchors + visual infill for large gaps.

    Step 1: Detect gripper change keyframes as primary anchors.
    Step 2: Find gaps > gap_threshold between consecutive anchors.
    Step 3: Fill gaps with visual_change keyframes.
    """
    # Step 1: gripper anchors
    anchors = _detect_gripper_keyframes(
        actions, gripper_indices, gripper_threshold, max_keyframes
    )

    # Step 2: identify large gaps
    anchors_sorted = sorted(set(anchors))
    boundary = [0] + anchors_sorted + [len(images) - 1]
    boundary = sorted(set(boundary))

    # Step 3: fill gaps with visual change keyframes
    filled = list(anchors_sorted)
    for i in range(len(boundary) - 1):
        gap_start, gap_end = boundary[i], boundary[i + 1]
        if gap_end - gap_start > gap_threshold:
            gap_images = images[gap_start:gap_end + 1]
            gap_kf = _detect_visual_change_keyframes(
                gap_images, threshold=visual_threshold,
                min_interval=min_interval, max_keyframes=5
            )
            # Offset back to global indices (skip 0 since gap_start is already an anchor)
            for k in gap_kf:
                global_idx = gap_start + k
                if global_idx not in filled:
                    filled.append(global_idx)

    filled = sorted(set(filled))[:max_keyframes]
    return filled


def get_hdf5_info(hdf5_path: str) -> dict:
    """Get information about HDF5 file structure."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    info = {"path": hdf5_path, "datasets": {}}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            info["datasets"][name] = {
                "shape": obj.shape,
                "dtype": str(obj.dtype)
            }

    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(visitor)

        # Get instruction if present
        if 'instruction' in f.attrs:
            info["instruction"] = f.attrs['instruction']

    return info


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

    parser = argparse.ArgumentParser(description="Extract keyframes from video or HDF5")
    parser.add_argument("input_path", nargs="?", help="Path to video or HDF5 file")
    parser.add_argument("--strategy", choices=["uniform", "visual_change", "gripper", "semantic"],
                        default="uniform", help="Extraction strategy")
    parser.add_argument("--interval", type=int, default=8,
                        help="Frame interval for uniform extraction")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold for visual change detection")
    parser.add_argument("--max-keyframes", type=int, default=20,
                        help="Maximum number of keyframes")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save extracted keyframes (optional)")

    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of keyframes")
    parser.add_argument("--vis-output", type=str, default=None,
                        help="Path to save visualization image")
    parser.add_argument("--vis-cols", type=int, default=5,
                        help="Number of columns in visualization grid")
    parser.add_argument("--thumb-size", type=int, nargs=2, default=[320, 240],
                        help="Thumbnail size (width height)")

    # HDF5 options
    parser.add_argument("--hdf5", action="store_true",
                        help="Input is HDF5 file (auto-detected if .hdf5 extension)")
    parser.add_argument("--hdf5-info", action="store_true",
                        help="Show HDF5 file structure and exit")

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

    # Require input_path for extraction
    if not args.input_path:
        parser.error("input_path is required for extraction")

    # Auto-detect HDF5
    is_hdf5 = args.hdf5 or args.input_path.endswith('.hdf5') or args.input_path.endswith('.h5')

    # Show HDF5 info
    if args.hdf5_info:
        if not is_hdf5:
            parser.error("--hdf5-info requires HDF5 file")
        info = get_hdf5_info(args.input_path)
        print(f"HDF5 File: {info['path']}")
        print("Datasets:")
        for name, ds_info in info['datasets'].items():
            print(f"  {name}: shape={ds_info['shape']}, dtype={ds_info['dtype']}")
        if 'instruction' in info:
            print(f"Instruction: {info['instruction']}")
        sys.exit(0)

    # Set up cache
    use_cache = not args.no_cache
    cache = get_cache(args.cache_dir) if use_cache else None

    # Extract keyframes
    if is_hdf5:
        print(f"Extracting from HDF5: {args.input_path}")
        keyframes = extract_keyframes_from_hdf5(
            args.input_path,
            strategy=args.strategy,
            interval=args.interval,
            max_keyframes=args.max_keyframes,
            use_cache=use_cache,
            cache=cache
        )
        total_frames = None  # Will be determined from HDF5 during extraction
    else:
        # Get video info
        info = get_video_info(args.input_path)
        print(f"Video info: {info}")
        total_frames = info['total_frames']

        if args.strategy == "uniform":
            keyframes = extract_keyframes_uniform(
                args.input_path,
                interval=args.interval,
                max_keyframes=args.max_keyframes,
                use_cache=use_cache,
                cache=cache
            )
        elif args.strategy == "visual_change":
            keyframes = extract_keyframes_visual_change(
                args.input_path,
                threshold=args.threshold,
                max_keyframes=args.max_keyframes,
                use_cache=use_cache,
                cache=cache
            )
        elif args.strategy == "semantic":
            keyframes = extract_keyframes_semantic(
                args.input_path,
                max_keyframes=args.max_keyframes,
                use_cache=use_cache,
                cache=cache
            )
        else:
            keyframes = extract_keyframes_uniform(
                args.input_path,
                interval=args.interval,
                max_keyframes=args.max_keyframes,
                use_cache=use_cache,
                cache=cache
            )

    print(f"\nExtracted {len(keyframes)} keyframes:")
    for kf in keyframes:
        print(f"  Frame {kf.frame_index} @ {kf.timestamp:.2f}s")

    # Generate visualization
    if args.visualize or args.vis_output:
        vis_output = args.vis_output or (str(Path(args.input_path).stem) + "_keyframes.jpg")
        print(f"\nGenerating visualization...")
        vis_img = visualize_keyframes(
            keyframes,
            output_path=vis_output,
            max_cols=args.vis_cols,
            thumbnail_size=tuple(args.thumb_size),
            title=f"Keyframes: {Path(args.input_path).name} ({args.strategy})"
        )
        print(f"Visualization saved to: {vis_output}")

    # Optionally save individual keyframes
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for kf in keyframes:
            img_bytes = b64decode(kf.image_b64)
            img_path = output_dir / f"keyframe_{kf.frame_index:04d}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            print(f"Saved: {img_path}")
