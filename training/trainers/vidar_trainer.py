"""Stage 1 Trainer: Vidar fine-tuning (standard diffusion)."""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .base import BaseTrainer
from ..config import VidarConfig
from ..models.wrapper import WanModelTrainingWrapper
from ..losses import add_noise, sample_timestep

logger = logging.getLogger(__name__)


class VidarTrainer(BaseTrainer):
    """
    Stage 1 Trainer: Vidar fine-tuning.

    Trains WanModel with standard flow matching loss:
    L = ||v_θ(x_t, t, c) - (x_0 - x_1)||²

    Components:
    - WanModel (DiT): Trainable
    - T5: Frozen text encoder
    - VAE: Frozen video encoder/decoder
    """

    def __init__(self, config: VidarConfig):
        super().__init__(config)
        self.wrapper: Optional[WanModelTrainingWrapper] = None

    def _build_model(self) -> nn.Module:
        """Build WanModel training wrapper."""
        cfg = self.config.model

        logger.info(f"Building WanModel from {cfg.ckpt_dir}")

        self.wrapper = WanModelTrainingWrapper(
            ckpt_dir=cfg.ckpt_dir,
            pt_dir=cfg.pt_dir,
            freeze_t5="t5" in self.config.training.freeze,
            freeze_vae="vae" in self.config.training.freeze,
            gradient_checkpointing=cfg.gradient_checkpointing,
            device=self.device,
        )

        # Return the DiT for FSDP wrapping (T5/VAE stay separate)
        return self.wrapper.dit

    def _get_transformer_layer_cls(self):
        """Get transformer layer classes for FSDP wrapping."""
        return WanModelTrainingWrapper.get_transformer_layer_cls()

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        """Wrap DiT with FSDP, keeping T5/VAE separate."""
        from ..distributed.fsdp_utils import wrap_model_fsdp

        cfg = self.config.distributed
        transformer_layer_cls = self._get_transformer_layer_cls()

        wrapped_dit = wrap_model_fsdp(
            model,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            transformer_layer_cls=transformer_layer_cls,
            cpu_offload=cfg.cpu_offload,
            activation_checkpointing=cfg.activation_checkpointing,
        )

        # Update wrapper's dit reference
        self.wrapper.dit = wrapped_dit

        return wrapped_dit

    def setup(self):
        """Setup training components."""
        logger.info("Setting up Vidar trainer...")

        # Build model (creates wrapper and returns dit)
        logger.info("Building model...")
        self.model = self._build_model()

        # Wrap DiT with FSDP if distributed
        if self.world_size > 1 and self.config.distributed.use_fsdp:
            logger.info("Wrapping DiT with FSDP...")
            self.model = self._wrap_fsdp(self.model)
        else:
            self.model = self.model.to(self.device)
            # Also move wrapper components to device
            self.wrapper.to(self.device)

        # Build optimizer (only DiT params)
        logger.info("Building optimizer and scheduler...")
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Build dataloader
        logger.info("Building dataloader...")
        self.dataloader = self._build_dataloader()

        # Build loss
        self.loss_fn = self._build_loss()

        # Move T5 to device for encoding (will use CPU offload if needed)
        self._setup_encoders()

        logger.info("Setup complete!")

    def _setup_encoders(self):
        """Setup T5 and VAE for encoding."""
        # For memory efficiency, we keep T5/VAE on CPU and move to GPU only when encoding
        # This is the default behavior in the wrapper

        # Optionally move VAE to GPU if we have enough memory
        if torch.cuda.is_available():
            # Check available memory
            free_mem = torch.cuda.get_device_properties(self.device).total_memory
            free_mem -= torch.cuda.memory_allocated(self.device)

            # VAE is ~2GB, move to GPU if we have > 10GB free
            if free_mem > 10 * 1024**3:
                logger.info("Moving VAE to GPU for faster encoding")
                self.wrapper.vae = self.wrapper.vae.to(self.device)

    def _build_optimizer(self):
        """Build optimizer for DiT parameters only."""
        cfg = self.config.training

        # Get trainable parameters from wrapper (DiT only)
        trainable_params = self.wrapper.get_trainable_parameters()
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        )

        return optimizer

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Single training step.

        Args:
            batch: Dict with 'video' (B, T, C, H, W) and 'instruction' (List[str])

        Returns:
            Dict with 'loss' tensor
        """
        video = batch["video"]  # (B, T, C, H, W)
        instructions = batch["instruction"]  # List[str]
        B = video.shape[0]

        # Convert video format: (B, T, C, H, W) -> (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4)

        # Normalize from [0, 1] to [-1, 1]
        video = video * 2 - 1

        # Encode video to latent space (VAE is frozen, no grad)
        with torch.no_grad():
            x1 = self._encode_video(video)  # (B, C_latent, T', H', W')

        # Encode text (T5 is frozen, no grad)
        with torch.no_grad():
            context = self._encode_text(instructions)  # (B, L, D)

        # Sample timesteps
        t = sample_timestep(B, self.device)

        # Add noise: x_t = t * x_1 + (1 - t) * x_0
        x_t, x0 = add_noise(x1, t)

        # Forward through DiT
        v_pred = self._forward_dit(x_t, t, context)

        # Compute loss: ||v_θ - (x_0 - x_1)||²
        loss = self.loss_fn(v_pred, x0, x1)

        return {"loss": loss}

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space."""
        # Handle device placement
        video_device = video.device
        vae_device = next(self.wrapper.vae.parameters()).device

        if video_device != vae_device:
            video = video.to(vae_device)

        # Encode with VAE
        latent = self.wrapper.vae.encode(video)

        # Move back to original device
        if latent.device != video_device:
            latent = latent.to(video_device)

        return latent

    def _encode_text(self, instructions: list) -> torch.Tensor:
        """Encode text instructions."""
        context = self.wrapper.encode_text(instructions)
        return context.to(self.device)

    def _forward_dit(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through DiT model."""
        return self.wrapper(x, t, context)

    def save_checkpoint(self, path: Optional[str] = None):
        """Save checkpoint (DiT weights only)."""
        from ..distributed.fsdp_utils import is_main_process

        if not is_main_process():
            return

        path = path or self.config.output.save_path
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Get DiT state dict
        dit_state = self.wrapper.get_dit_state_dict()

        checkpoint = {
            "model": dit_state,
            "step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.to_dict(),
        }

        if self.config.output.save_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        if self.config.output.save_scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location="cpu")

        # Load DiT weights
        self.wrapper.dit.load_state_dict(checkpoint["model"], strict=False)

        # Load training state
        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        logger.info(f"Loaded checkpoint at step {self.global_step}")


def create_vidar_trainer(config: VidarConfig) -> VidarTrainer:
    """Factory function to create Vidar trainer."""
    return VidarTrainer(config)
