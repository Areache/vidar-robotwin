"""Video transforms for Vidar training."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import random


class VideoTransform:
    """Preprocessing transforms for video data."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (736, 640),
        normalize: bool = True,
        random_flip: bool = False,
        flip_prob: float = 0.5
    ):
        """
        Args:
            resolution: Target (H, W) resolution
            normalize: Whether to normalize to [-1, 1]
            random_flip: Whether to apply random horizontal flip
            flip_prob: Probability of horizontal flip
        """
        self.resolution = resolution
        self.normalize = normalize
        self.random_flip = random_flip
        self.flip_prob = flip_prob

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms to video.

        Args:
            video: Video tensor (T, C, H, W) or (C, T, H, W), values in [0, 1]

        Returns:
            Transformed video tensor
        """
        # Handle different input formats
        if video.dim() == 4:
            # Assume (T, C, H, W)
            T, C, H, W = video.shape
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
            was_tchw = True
        else:
            was_tchw = False

        C, T, H, W = video.shape

        # Resize if needed
        if (H, W) != self.resolution:
            video = video.view(C * T, 1, H, W)
            video = F.interpolate(
                video,
                size=self.resolution,
                mode='bilinear',
                align_corners=False
            )
            video = video.view(C, T, *self.resolution)

        # Normalize to [-1, 1]
        if self.normalize:
            video = video * 2 - 1

        # Random horizontal flip
        if self.random_flip and random.random() < self.flip_prob:
            video = torch.flip(video, dims=[-1])

        # Convert back if needed
        if was_tchw:
            video = video.permute(1, 0, 2, 3)  # (T, C, H, W)

        return video


class UnifiedObservationTransform:
    """
    Transform multi-camera views into unified observation space.

    Per Vidarc paper, combines multiple camera views into 720x640 resolution:
    - Top: front camera (360, 640)
    - Bottom-left: left arm camera (360, 320)
    - Bottom-right: right arm camera (360, 320)
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (720, 640),
        normalize: bool = True
    ):
        self.output_size = output_size
        self.normalize = normalize

    def __call__(
        self,
        front: torch.Tensor,
        left_arm: torch.Tensor,
        right_arm: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine camera views into unified observation.

        Args:
            front: Front camera tensor (C, H, W) or (T, C, H, W)
            left_arm: Left arm camera tensor
            right_arm: Right arm camera tensor

        Returns:
            Unified observation tensor (C, 720, 640) or (T, C, 720, 640)
        """
        has_time = front.dim() == 4

        if has_time:
            T = front.shape[0]
            results = []
            for t in range(T):
                unified = self._combine_single(
                    front[t], left_arm[t], right_arm[t]
                )
                results.append(unified)
            return torch.stack(results, dim=0)
        else:
            return self._combine_single(front, left_arm, right_arm)

    def _combine_single(
        self,
        front: torch.Tensor,
        left_arm: torch.Tensor,
        right_arm: torch.Tensor
    ) -> torch.Tensor:
        """Combine single frame from each camera."""
        C = front.shape[0]
        H, W = self.output_size

        # Create output tensor
        unified = torch.zeros(C, H, W, device=front.device, dtype=front.dtype)

        # Resize and place front camera (top half)
        front_resized = F.interpolate(
            front.unsqueeze(0),
            size=(H // 2, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        unified[:, :H // 2, :] = front_resized

        # Resize and place left arm camera (bottom-left)
        left_resized = F.interpolate(
            left_arm.unsqueeze(0),
            size=(H // 2, W // 2),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        unified[:, H // 2:, :W // 2] = left_resized

        # Resize and place right arm camera (bottom-right)
        right_resized = F.interpolate(
            right_arm.unsqueeze(0),
            size=(H // 2, W // 2),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        unified[:, H // 2:, W // 2:] = right_resized

        # Normalize
        if self.normalize:
            unified = unified * 2 - 1

        return unified


class TemporalSubsample:
    """Subsample video frames to target FPS."""

    def __init__(
        self,
        target_fps: int = 10,
        source_fps: Optional[int] = None,
        num_frames: Optional[int] = None
    ):
        """
        Args:
            target_fps: Target frames per second
            source_fps: Source FPS (if known)
            num_frames: Target number of frames (overrides FPS-based sampling)
        """
        self.target_fps = target_fps
        self.source_fps = source_fps
        self.num_frames = num_frames

    def __call__(self, video: torch.Tensor, source_fps: Optional[int] = None) -> torch.Tensor:
        """
        Subsample video frames.

        Args:
            video: Video tensor (T, C, H, W)
            source_fps: Source FPS (optional, overrides init value)

        Returns:
            Subsampled video tensor
        """
        T = video.shape[0]

        if self.num_frames is not None:
            # Sample fixed number of frames
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            # Sample based on FPS ratio
            fps = source_fps or self.source_fps
            if fps is None or fps <= self.target_fps:
                return video

            step = fps // self.target_fps
            indices = torch.arange(0, T, step)

        return video[indices]


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transform(
    resolution: Tuple[int, int] = (736, 640),
    random_flip: bool = False
) -> VideoTransform:
    """Get default training transform."""
    return VideoTransform(
        resolution=resolution,
        normalize=True,
        random_flip=random_flip,
        flip_prob=0.5
    )


def get_eval_transform(
    resolution: Tuple[int, int] = (736, 640)
) -> VideoTransform:
    """Get default evaluation transform."""
    return VideoTransform(
        resolution=resolution,
        normalize=True,
        random_flip=False
    )
