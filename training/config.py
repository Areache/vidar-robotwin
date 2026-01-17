"""Training configuration for Vidar/Vidarc."""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    ckpt_dir: str = "checkpoints/Wan2.2-TI2V-5B"
    pt_dir: Optional[str] = None
    model_class: Literal["WanModel", "WanModelCausal"] = "WanModel"
    gradient_checkpointing: bool = True
    chunk_size: int = 16  # For Self-Forcing training
    same_t_across_chunks: bool = True  # Use same timestep across all chunks


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_steps: int = 14000
    batch_size: int = 128
    gradient_accumulation: int = 1
    gradient_accumulation_steps: int = 1  # Alias for gradient_accumulation

    # Optimizer
    lr: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # Scheduler
    warmup_steps: int = 200
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # Freezing
    freeze: List[str] = field(default_factory=lambda: ["t5", "vae"])

    # Embodiment-aware loss (Stage 2)
    eta: float = 3.0
    use_embodiment_loss: bool = False

    # Debug
    debug: bool = False
    max_steps: Optional[int] = None  # Override num_steps for debug


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "data/robotwin"
    num_frames: int = 81
    resolution: tuple = (736, 640)
    fps: int = 10
    num_workers: int = 4
    pin_memory: bool = True

    # Classifier-free guidance
    cfg_prob: float = 0.1


@dataclass
class LossConfig:
    """Loss configuration."""
    type: Literal["flow_matching", "causal_flow_matching"] = "flow_matching"
    embodiment_aware: bool = False
    eta: float = 3.0  # Embodiment-aware loss weight

    # IDM loss (if training IDM)
    lambda_mask: float = 3e-3


@dataclass
class SelfForcingConfig:
    """Self-Forcing training configuration (Stage 2 only)."""
    enabled: bool = False
    causal: bool = True
    chunk_size: int = 16
    kv_cache_length: int = 64
    same_step_across_blocks: bool = True


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    use_fsdp: bool = True
    sharding_strategy: Literal["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD"] = "FULL_SHARD"
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    activation_checkpointing: bool = True
    cpu_offload: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    output_dir: str = "outputs/vidar"
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 2000

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "vidar-training"
    wandb_entity: Optional[str] = None


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "outputs/vidar"
    save_path: str = "checkpoints/vidar/vidar.pt"
    save_optimizer: bool = True
    save_scheduler: bool = True
    log_interval: int = 50
    save_interval: int = 1000


@dataclass
class VidarConfig:
    """Complete Vidar training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    self_forcing: SelfForcingConfig = field(default_factory=SelfForcingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Seed
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "VidarConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "VidarConfig":
        """Load config from dictionary."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
            loss=LossConfig(**data.get("loss", {})),
            self_forcing=SelfForcingConfig(**data.get("self_forcing", {})),
            distributed=DistributedConfig(**data.get("distributed", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            output=OutputConfig(**data.get("output", {})),
            seed=data.get("seed", 42),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save_yaml(self, path: str):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save(self, path: str):
        """Alias for save_yaml."""
        self.save_yaml(path)


# Preset configs for 2xH200
def get_2xh200_config() -> VidarConfig:
    """Get config optimized for 2x H200 (120GB each)."""
    config = VidarConfig()
    config.training.batch_size = 16
    config.training.gradient_accumulation = 8
    config.distributed.sharding_strategy = "FULL_SHARD"
    config.distributed.activation_checkpointing = True
    config.self_forcing.chunk_size = 8
    return config


def get_vidar_stage1_config() -> VidarConfig:
    """Get config for Stage 1: Vidar fine-tuning."""
    config = VidarConfig()
    config.model.model_class = "WanModel"
    config.model.pt_dir = None
    config.training.num_steps = 14000
    config.loss.type = "flow_matching"
    config.loss.embodiment_aware = False
    config.self_forcing.enabled = False
    config.output.save_path = "checkpoints/vidar/vidar.pt"
    return config


def get_vidarc_stage2_config() -> VidarConfig:
    """Get config for Stage 2: Vidarc causal training."""
    config = VidarConfig()
    config.model.model_class = "WanModelCausal"
    config.model.pt_dir = "checkpoints/vidar/vidar.pt"
    config.training.num_steps = 4000
    config.loss.type = "causal_flow_matching"
    config.loss.embodiment_aware = True
    config.loss.eta = 3.0
    config.self_forcing.enabled = True
    config.self_forcing.causal = True
    config.self_forcing.chunk_size = 16
    config.output.save_path = "checkpoints/vidar/vidarc.pt"
    return config


def load_config(path: str) -> VidarConfig:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        VidarConfig instance
    """
    return VidarConfig.from_yaml(path)
