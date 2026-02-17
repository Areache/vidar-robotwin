"""Trainer implementations."""

from .base import BaseTrainer
from .vidar_trainer import VidarTrainer, create_vidar_trainer
from .vidarc_trainer import VidarCausalTrainer, create_vidarc_trainer

__all__ = [
    "BaseTrainer",
    # Stage 1: Standard diffusion
    "VidarTrainer",
    "create_vidar_trainer",
    # Stage 2: Causal Self-Forcing
    "VidarCausalTrainer",
    "create_vidarc_trainer",
]
