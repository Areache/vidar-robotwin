"""Dataset implementations for Vidar training."""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Callable, Dict, List, Any, Tuple
from pathlib import Path

try:
    import decord
    decord.bridge.set_bridge('torch')
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

from .transforms import VideoTransform, get_train_transform


class VidarDataset(Dataset):
    """
    Dataset for Vidar/Vidarc training.

    Expected data structure:
        data_dir/
        ├── episodes/
        │   ├── ep_0001/
        │   │   ├── video.mp4
        │   │   ├── actions.npy      # (T, action_dim)
        │   │   └── instruction.txt
        │   └── ...
        └── manifest.json
    """

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 81,
        resolution: Tuple[int, int] = (736, 640),
        fps: int = 10,
        transform: Optional[Callable] = None,
        load_actions: bool = True,
        cfg_prob: float = 0.1
    ):
        """
        Args:
            data_dir: Path to dataset directory
            num_frames: Number of frames to sample per video
            resolution: Target (H, W) resolution
            fps: Target frames per second
            transform: Optional transform to apply
            load_actions: Whether to load action data
            cfg_prob: Probability to drop text for classifier-free guidance
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.fps = fps
        self.transform = transform or get_train_transform(resolution)
        self.load_actions = load_actions
        self.cfg_prob = cfg_prob

        # Load manifest
        self.episodes = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load dataset manifest."""
        manifest_path = self.data_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            return manifest.get("episodes", manifest)

        # Fallback: scan episodes directory
        episodes_dir = self.data_dir / "episodes"
        if not episodes_dir.exists():
            episodes_dir = self.data_dir

        episodes = []
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir.is_dir():
                episode = {
                    "id": ep_dir.name,
                    "video_path": str(ep_dir / "video.mp4"),
                    "instruction_path": str(ep_dir / "instruction.txt"),
                    "actions_path": str(ep_dir / "actions.npy"),
                }
                # Check for alternative video formats
                for ext in [".mp4", ".avi", ".mov", ".webm"]:
                    video_path = ep_dir / f"video{ext}"
                    if video_path.exists():
                        episode["video_path"] = str(video_path)
                        break

                if Path(episode["video_path"]).exists():
                    episodes.append(episode)

        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode = self.episodes[idx]

        # Load video
        video = self._load_video(episode["video_path"])

        # Load instruction
        instruction = self._load_instruction(episode.get("instruction_path"))

        # Classifier-free guidance: drop text with probability cfg_prob
        if torch.rand(1).item() < self.cfg_prob:
            instruction = ""

        # Apply transform
        if self.transform:
            video = self.transform(video)

        result = {
            "video": video,           # (T, C, H, W) or (C, T, H, W)
            "instruction": instruction,
        }

        # Load actions if needed
        if self.load_actions:
            actions = self._load_actions(episode.get("actions_path"))
            if actions is not None:
                result["actions"] = actions

        return result

    def _load_video(self, path: str) -> torch.Tensor:
        """Load video frames."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")

        if HAS_DECORD:
            return self._load_video_decord(path)
        else:
            return self._load_video_fallback(path)

    def _load_video_decord(self, path: str) -> torch.Tensor:
        """Load video using decord (fast)."""
        vr = decord.VideoReader(path)
        total_frames = len(vr)

        # Sample frame indices
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            # Pad if needed
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        # Load frames
        frames = vr.get_batch(indices)  # (T, H, W, C)

        # Convert to tensor and normalize
        if isinstance(frames, torch.Tensor):
            video = frames.float() / 255.0
        else:
            video = torch.from_numpy(frames.asnumpy()).float() / 255.0

        # (T, H, W, C) -> (T, C, H, W)
        video = video.permute(0, 3, 1, 2)

        return video

    def _load_video_fallback(self, path: str) -> torch.Tensor:
        """Fallback video loading using imageio."""
        try:
            import imageio
        except ImportError:
            raise ImportError("Please install imageio: pip install imageio[ffmpeg]")

        reader = imageio.get_reader(path)
        frames = []

        for frame in reader:
            frames.append(torch.from_numpy(frame))
            if len(frames) >= self.num_frames * 3:  # Load extra for subsampling
                break

        reader.close()

        video = torch.stack(frames)  # (T, H, W, C)

        # Subsample to target num_frames
        total = len(video)
        if total > self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            video = video[indices]

        # Normalize and permute
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

        return video

    def _load_instruction(self, path: Optional[str]) -> str:
        """Load text instruction."""
        if path is None or not Path(path).exists():
            return ""

        with open(path, "r") as f:
            return f.read().strip()

    def _load_actions(self, path: Optional[str]) -> Optional[torch.Tensor]:
        """Load action data."""
        if path is None or not Path(path).exists():
            return None

        actions = np.load(path)
        return torch.from_numpy(actions).float()


class JsonDataset(Dataset):
    """
    Dataset that loads from a JSON file with video paths and instructions.

    JSON format:
        [
            {"video": "path/to/video.mp4", "instruction": "do something"},
            ...
        ]
    """

    def __init__(
        self,
        json_path: str,
        num_frames: int = 81,
        resolution: Tuple[int, int] = (736, 640),
        transform: Optional[Callable] = None,
        cfg_prob: float = 0.1
    ):
        self.json_path = Path(json_path)
        self.num_frames = num_frames
        self.resolution = resolution
        self.transform = transform or get_train_transform(resolution)
        self.cfg_prob = cfg_prob

        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Load video
        video = self._load_video(item["video"])

        # Get instruction
        instruction = item.get("instruction", "")
        if torch.rand(1).item() < self.cfg_prob:
            instruction = ""

        # Apply transform
        if self.transform:
            video = self.transform(video)

        return {
            "video": video,
            "instruction": instruction,
        }

    def _load_video(self, path: str) -> torch.Tensor:
        """Load video frames."""
        if HAS_DECORD:
            vr = decord.VideoReader(path)
            indices = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices)
            if isinstance(frames, torch.Tensor):
                video = frames.float() / 255.0
            else:
                video = torch.from_numpy(frames.asnumpy()).float() / 255.0
            return video.permute(0, 3, 1, 2)
        else:
            raise ImportError("decord required for JsonDataset")


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    drop_last: bool = True
) -> DataLoader:
    """
    Create dataloader with optional distributed sampler.

    Args:
        dataset: Dataset instance
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle (ignored if distributed)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        distributed: Whether to use distributed sampler
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader instance
    """
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=vidar_collate_fn
    )


def vidar_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for Vidar dataset.

    Handles variable-length instructions and optional action data.
    """
    videos = torch.stack([item["video"] for item in batch])
    instructions = [item["instruction"] for item in batch]

    result = {
        "video": videos,
        "instruction": instructions,
    }

    # Collate actions if present
    if "actions" in batch[0] and batch[0]["actions"] is not None:
        actions = torch.stack([item["actions"] for item in batch])
        result["actions"] = actions

    return result
