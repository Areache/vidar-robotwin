"""Model wrappers for training."""

from .wrapper import WanModelTrainingWrapper, create_model
from .wrapper_causal import WanModelCausalTrainingWrapper, create_causal_model

__all__ = [
    # Stage 1: Standard bidirectional
    "WanModelTrainingWrapper",
    "create_model",
    # Stage 2: Causal with Self-Forcing
    "WanModelCausalTrainingWrapper",
    "create_causal_model",
]
