"""Vidar/Vidarc training package."""

from .config import (
    VidarConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LossConfig,
    SelfForcingConfig,
    DistributedConfig,
    LoggingConfig,
    OutputConfig,
    get_2xh200_config,
    get_vidar_stage1_config,
    get_vidarc_stage2_config,
)

from .losses import (
    flow_matching_loss,
    causal_flow_matching_loss,
    embodiment_aware_loss,
    idm_loss,
    VidarLoss,
    IDMLoss,
    add_noise,
    sample_timestep,
)

from .trainers import (
    BaseTrainer,
    VidarTrainer,
    create_vidar_trainer,
)

from .models import (
    WanModelTrainingWrapper,
    create_model,
)

__all__ = [
    # Config
    "VidarConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "LossConfig",
    "SelfForcingConfig",
    "DistributedConfig",
    "LoggingConfig",
    "OutputConfig",
    "get_2xh200_config",
    "get_vidar_stage1_config",
    "get_vidarc_stage2_config",
    # Losses
    "flow_matching_loss",
    "causal_flow_matching_loss",
    "embodiment_aware_loss",
    "idm_loss",
    "VidarLoss",
    "IDMLoss",
    "add_noise",
    "sample_timestep",
    # Trainers
    "BaseTrainer",
    "VidarTrainer",
    "create_vidar_trainer",
    # Models
    "WanModelTrainingWrapper",
    "create_model",
]
