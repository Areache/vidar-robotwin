"""HDF5-based dataset for Vidar training (RDT-style format).

Based on RoboticsDiffusionTransformer data format with unified observation space.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import h5py

from .transforms import VideoTransform, get_train_transform

logger = logging.getLogger(__name__)


class HDF5VLADataset(Dataset):
    """
    HDF5-based Vision-Language-Action Dataset.

    Follows RDT-style format with unified observation space.

    Expected HDF5 structure:
        episode_XXXXXX.hdf5
        ├── observations/
        │   ├── images/
        │   │   ├── cam_high: (T, H, W, 3)
        │   │   ├── cam_left_wrist: (T, H, W, 3)
        │   │   └── cam_right_wrist: (T, H, W, 3)
        │   ├── unified_image: (T, 720, 640, 3)
        │   └── qpos: (T, state_dim)
        ├── action: (T, action_dim)
        └── instruction (attr): string

    The dataset supports:
    - Unified observation (pre-composed multi-camera view)
    - Individual camera views
    - Scene-description formatted instructions
    """

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 81,
        resolution: Tuple[int, int] = (720, 640),
        use_unified: bool = True,
        camera_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        cfg_prob: float = 0.1,
        img_history_size: int = 2,
        load_actions: bool = True,
    ):
        """
        Args:
            data_dir: Path to dataset directory containing HDF5 files
            num_frames: Number of frames to sample per episode
            resolution: Target (H, W) resolution
            use_unified: Whether to use pre-composed unified observation
            camera_names: List of camera names (if not using unified)
            transform: Optional transform to apply to videos
            cfg_prob: Probability to drop instruction for CFG
            img_history_size: Number of history frames to include
            load_actions: Whether to load action data
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.use_unified = use_unified
        self.camera_names = camera_names or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        self.transform = transform or get_train_transform(resolution)
        self.cfg_prob = cfg_prob
        self.img_history_size = img_history_size
        self.load_actions = load_actions

        # Find HDF5 files
        self.episode_files = self._find_episodes()
        logger.info(f"Found {len(self.episode_files)} episodes in {data_dir}")

        # Load config if available
        self.config = self._load_config()

        # Load precomputed instructions if available
        self.instructions = self._load_instructions()

    def _find_episodes(self) -> List[Path]:
        """Find all HDF5 episode files."""
        hdf5_dir = self.data_dir / "hdf5"
        if hdf5_dir.exists():
            search_dir = hdf5_dir
            logger.info(f"Searching for episodes in: {hdf5_dir}")
        else:
            search_dir = self.data_dir
            logger.info(f"No 'hdf5/' subdirectory found, searching in: {self.data_dir}")

        episode_files = sorted(search_dir.glob("episode_*.hdf5"))
        logger.info(f"Found {len(episode_files)} files matching 'episode_*.hdf5'")

        if not episode_files:
            # Try alternative patterns
            episode_files = sorted(search_dir.glob("*.hdf5"))
            logger.info(f"Found {len(episode_files)} files matching '*.hdf5' (fallback pattern)")

        if not episode_files:
            # Log diagnostic information
            all_files = list(search_dir.iterdir()) if search_dir.exists() else []
            logger.warning(
                f"No HDF5 files found in {search_dir}! "
                f"Directory contents: {[f.name for f in all_files[:10]]}{'...' if len(all_files) > 10 else ''}"
            )

        return episode_files

    def _load_config(self) -> Dict[str, Any]:
        """Load dataset config if available."""
        config_path = self.data_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

    def _load_instructions(self) -> Dict[str, Dict[str, str]]:
        """Load precomputed instructions if available."""
        instr_path = self.data_dir / "instructions.json"
        if instr_path.exists():
            with open(instr_path, "r") as f:
                return json.load(f)
        return {}

    def __len__(self) -> int:
        return len(self.episode_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single episode."""
        episode_path = self.episode_files[idx]

        with h5py.File(episode_path, "r") as f:
            # Load video frames
            if self.use_unified and "observations/unified_image" in f:
                video = f["observations/unified_image"][:]
            else:
                video = self._load_multicam(f)

            # Load instruction
            ep_name = episode_path.stem
            if ep_name in self.instructions:
                instruction = self.instructions[ep_name].get("instruction", "")
            else:
                instruction = f.attrs.get("instruction", "")

            # CFG: drop instruction with probability
            if torch.rand(1).item() < self.cfg_prob:
                instruction = ""

            # Subsample frames
            video = self._subsample_frames(video)

            # Convert to tensor and apply transform
            video = torch.from_numpy(video).float() / 255.0  # (T, H, W, C)
            video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

            if self.transform:
                video = self.transform(video)

            result = {
                "video": video,
                "instruction": instruction,
            }

            # Load actions if requested
            if self.load_actions and "action" in f:
                actions = f["action"][:]
                actions = self._subsample_frames(actions)
                result["actions"] = torch.from_numpy(actions).float()

            # Load state if available
            if "observations/qpos" in f:
                qpos = f["observations/qpos"][:]
                qpos = self._subsample_frames(qpos)
                result["qpos"] = torch.from_numpy(qpos).float()

            return result

    def _load_multicam(self, f: h5py.File) -> np.ndarray:
        """Load and compose multi-camera views."""
        images_group = f.get("observations/images")
        if images_group is None:
            raise ValueError("No image data found in HDF5 file")

        # Load first available camera to get dimensions
        first_cam = list(images_group.keys())[0]
        T = images_group[first_cam].shape[0]

        # Compose into unified format
        H, W = self.resolution
        unified = np.zeros((T, H, W, 3), dtype=np.uint8)

        h_top = H // 2
        w_half = W // 2

        for cam_name in images_group.keys():
            frames = images_group[cam_name][:]
            cam_lower = cam_name.lower()

            if "high" in cam_lower or "front" in cam_lower:
                # Top half
                resized = self._batch_resize(frames, (h_top, W))
                unified[:, :h_top, :, :] = resized
            elif "left" in cam_lower:
                # Bottom left
                resized = self._batch_resize(frames, (H - h_top, w_half))
                unified[:, h_top:, :w_half, :] = resized
            elif "right" in cam_lower:
                # Bottom right
                resized = self._batch_resize(frames, (H - h_top, W - w_half))
                unified[:, h_top:, w_half:, :] = resized

        return unified

    def _batch_resize(self, frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize batch of frames."""
        try:
            import cv2
            T, H, W, C = frames.shape
            th, tw = target_size
            resized = np.zeros((T, th, tw, C), dtype=frames.dtype)
            for t in range(T):
                resized[t] = cv2.resize(frames[t], (tw, th))
            return resized
        except ImportError:
            # Nearest neighbor fallback
            T, H, W, C = frames.shape
            th, tw = target_size
            y_idx = (np.arange(th) * H / th).astype(int)
            x_idx = (np.arange(tw) * W / tw).astype(int)
            return frames[:, y_idx][:, :, x_idx]

    def _subsample_frames(self, data: np.ndarray) -> np.ndarray:
        """Subsample to target frame count."""
        T = data.shape[0]
        if T == self.num_frames:
            return data
        elif T > self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            return data[indices]
        else:
            # Pad by repeating last frame
            pad_size = self.num_frames - T
            padding = np.repeat(data[-1:], pad_size, axis=0)
            return np.concatenate([data, padding], axis=0)


class HDF5VLADatasetTimestep(Dataset):
    """
    HDF5 dataset that returns single timesteps with history.

    Useful for training with temporal context like RDT.
    Returns: current frame + history frames + language instruction
    """

    def __init__(
        self,
        data_dir: str,
        resolution: Tuple[int, int] = (720, 640),
        img_history_size: int = 2,
        action_chunk_size: int = 64,
        transform: Optional[Callable] = None,
        cfg_prob: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.img_history_size = img_history_size
        self.action_chunk_size = action_chunk_size
        self.transform = transform
        self.cfg_prob = cfg_prob

        # Find episodes and build index
        self.episode_files = sorted(self.data_dir.glob("**/*.hdf5"))
        self.sample_index = self._build_sample_index()

        logger.info(f"Found {len(self.episode_files)} episodes, {len(self.sample_index)} samples")

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build index of (episode_idx, timestep) pairs."""
        index = []
        for ep_idx, ep_path in enumerate(self.episode_files):
            try:
                with h5py.File(ep_path, "r") as f:
                    if "observations/unified_image" in f:
                        T = f["observations/unified_image"].shape[0]
                    else:
                        # Get from first camera
                        imgs = f.get("observations/images")
                        if imgs:
                            first_cam = list(imgs.keys())[0]
                            T = imgs[first_cam].shape[0]
                        else:
                            continue

                    # Add samples starting from img_history_size
                    for t in range(self.img_history_size, T):
                        index.append((ep_idx, t))
            except Exception as e:
                logger.warning(f"Error reading {ep_path}: {e}")

        return index

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, timestep = self.sample_index[idx]
        ep_path = self.episode_files[ep_idx]

        with h5py.File(ep_path, "r") as f:
            # Load current + history frames
            start_t = max(0, timestep - self.img_history_size)
            end_t = timestep + 1

            if "observations/unified_image" in f:
                frames = f["observations/unified_image"][start_t:end_t]
            else:
                frames = self._load_multicam_range(f, start_t, end_t)

            # Pad history if needed
            if frames.shape[0] < self.img_history_size + 1:
                pad_size = self.img_history_size + 1 - frames.shape[0]
                padding = np.repeat(frames[:1], pad_size, axis=0)
                frames = np.concatenate([padding, frames], axis=0)

            # Load instruction
            instruction = f.attrs.get("instruction", "")
            if torch.rand(1).item() < self.cfg_prob:
                instruction = ""

            # Load action chunk
            if "action" in f:
                T_total = f["action"].shape[0]
                action_end = min(timestep + self.action_chunk_size, T_total)
                actions = f["action"][timestep:action_end]

                # Pad actions if needed
                if actions.shape[0] < self.action_chunk_size:
                    pad_size = self.action_chunk_size - actions.shape[0]
                    padding = np.repeat(actions[-1:], pad_size, axis=0)
                    actions = np.concatenate([actions, padding], axis=0)
            else:
                actions = np.zeros((self.action_chunk_size, 14), dtype=np.float32)

            # Load state
            if "observations/qpos" in f:
                state = f["observations/qpos"][timestep]
            else:
                state = np.zeros(14, dtype=np.float32)

            # Convert to tensors
            frames = torch.from_numpy(frames).float() / 255.0
            frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

            if self.transform:
                frames = self.transform(frames)

            return {
                "video": frames,  # (history+1, C, H, W)
                "instruction": instruction,
                "action": torch.from_numpy(actions).float(),
                "state": torch.from_numpy(state).float(),
                "timestep": timestep,
            }

    def _load_multicam_range(
        self,
        f: h5py.File,
        start_t: int,
        end_t: int
    ) -> np.ndarray:
        """Load and compose multi-camera range."""
        images_group = f.get("observations/images")
        if images_group is None:
            raise ValueError("No image data found")

        T = end_t - start_t
        H, W = self.resolution
        unified = np.zeros((T, H, W, 3), dtype=np.uint8)

        h_top = H // 2
        w_half = W // 2

        for cam_name in images_group.keys():
            frames = images_group[cam_name][start_t:end_t]
            cam_lower = cam_name.lower()

            if "high" in cam_lower or "front" in cam_lower:
                resized = self._batch_resize(frames, (h_top, W))
                unified[:, :h_top, :, :] = resized
            elif "left" in cam_lower:
                resized = self._batch_resize(frames, (H - h_top, w_half))
                unified[:, h_top:, :w_half, :] = resized
            elif "right" in cam_lower:
                resized = self._batch_resize(frames, (H - h_top, W - w_half))
                unified[:, h_top:, w_half:, :] = resized

        return unified

    def _batch_resize(self, frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize batch of frames."""
        try:
            import cv2
            T, H, W, C = frames.shape
            th, tw = target_size
            resized = np.zeros((T, th, tw, C), dtype=frames.dtype)
            for t in range(T):
                resized[t] = cv2.resize(frames[t], (tw, th))
            return resized
        except ImportError:
            T, H, W, C = frames.shape
            th, tw = target_size
            y_idx = (np.arange(th) * H / th).astype(int)
            x_idx = (np.arange(tw) * W / tw).astype(int)
            return frames[:, y_idx][:, :, x_idx]


def get_hdf5_dataloader(
    data_dir: str,
    batch_size: int,
    num_frames: int = 81,
    resolution: Tuple[int, int] = (720, 640),
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    drop_last: bool = True,
    cfg_prob: float = 0.1,
) -> DataLoader:
    """
    Create HDF5-based dataloader.

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size per GPU
        num_frames: Frames per episode
        resolution: Target resolution (H, W)
        num_workers: Data loading workers
        pin_memory: Pin memory for GPU transfer
        distributed: Use distributed sampler
        drop_last: Drop incomplete batches
        cfg_prob: CFG dropout probability

    Returns:
        DataLoader instance
    """
    dataset = HDF5VLADataset(
        data_dir=data_dir,
        num_frames=num_frames,
        resolution=resolution,
        cfg_prob=cfg_prob,
    )

    # Validate dataset is not empty
    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset is empty! No valid episodes found in {data_dir}. "
            f"Expected HDF5 files matching 'episode_*.hdf5' in '{data_dir}/hdf5/' directory. "
            f"Please verify your data preparation step completed successfully."
        )

    sampler = None
    shuffle = True
    world_size = 1

    if distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False

    # Auto-disable drop_last if dataset is too small to form complete batches
    effective_batch_size = batch_size * world_size
    actual_drop_last = drop_last
    if len(dataset) < effective_batch_size:
        actual_drop_last = False
        logger.warning(
            f"Dataset size ({len(dataset)}) < batch_size * world_size ({effective_batch_size}). "
            f"Disabling drop_last to allow training with incomplete batches."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=actual_drop_last,
        collate_fn=hdf5_collate_fn,
    )


def hdf5_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for HDF5 dataset."""
    videos = torch.stack([item["video"] for item in batch])
    instructions = [item["instruction"] for item in batch]

    result = {
        "video": videos,
        "instruction": instructions,
    }

    if "actions" in batch[0]:
        result["actions"] = torch.stack([item["actions"] for item in batch])

    if "qpos" in batch[0]:
        result["qpos"] = torch.stack([item["qpos"] for item in batch])

    return result
