"""Stage 2 Trainer: Vidarc causal fine-tuning with Self-Forcing."""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .base import BaseTrainer
from ..config import VidarConfig
from ..models.wrapper_causal import WanModelCausalTrainingWrapper
from ..losses import sample_timestep

logger = logging.getLogger(__name__)


class VidarCausalTrainer(BaseTrainer):
    """
    Stage 2 Trainer: Vidarc causal fine-tuning with Self-Forcing.

    Implements Self-Forcing training paradigm from the Vidarc paper:
    - Causal attention: previous frames are noise-free, attended via KV cache
    - Teacher forcing: use ground truth for KV cache during training
    - Autoregressive rollout with chunk-wise processing
    - Embodiment-aware loss with higher weight on robot regions

    Loss function (with embodiment-aware weighting):
    L = ||(1 + η·U(x_1)) ⊙ (v_θ(x_t, t, c) - (x_0 - x_1))||²

    where U(x_1) indicates robot/embodiment regions (η=3.0 default).
    """

    def __init__(self, config: VidarConfig):
        super().__init__(config)
        self.wrapper: Optional[WanModelCausalTrainingWrapper] = None

        # Self-Forcing parameters
        self.chunk_size = config.model.get("chunk_size", 16)
        self.same_t_across_chunks = config.model.get("same_t_across_chunks", True)

        # Embodiment-aware loss parameters
        self.eta = config.training.get("eta", 3.0)  # Embodiment weight
        self.use_embodiment_loss = config.training.get("use_embodiment_loss", False)

    def _build_model(self) -> nn.Module:
        """Build WanModelCausal training wrapper."""
        cfg = self.config.model

        logger.info(f"Building WanModelCausal from {cfg.ckpt_dir}")

        self.wrapper = WanModelCausalTrainingWrapper(
            ckpt_dir=cfg.ckpt_dir,
            pt_dir=cfg.pt_dir,  # Stage 1 weights (vidar.pt)
            freeze_t5="t5" in self.config.training.freeze,
            freeze_vae="vae" in self.config.training.freeze,
            gradient_checkpointing=cfg.gradient_checkpointing,
            device=self.device,
        )

        # Return the DiT for FSDP wrapping
        return self.wrapper.dit

    def _get_transformer_layer_cls(self):
        """Get transformer layer classes for FSDP wrapping."""
        return WanModelCausalTrainingWrapper.get_transformer_layer_cls()

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
        logger.info("Setting up Vidarc causal trainer (Stage 2)...")

        # Build model (creates wrapper and returns dit)
        logger.info("Building causal model...")
        self.model = self._build_model()

        # Wrap DiT with FSDP if distributed
        if self.world_size > 1 and self.config.distributed.use_fsdp:
            logger.info("Wrapping DiT with FSDP...")
            self.model = self._wrap_fsdp(self.model)
        else:
            self.model = self.model.to(self.device)
            self.wrapper.to(self.device)

        # Build optimizer
        logger.info("Building optimizer and scheduler...")
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Build dataloader
        logger.info("Building dataloader...")
        self.dataloader = self._build_dataloader()

        # Build loss
        self.loss_fn = self._build_loss()

        # Setup encoders
        self._setup_encoders()

        logger.info(f"Setup complete! Chunk size: {self.chunk_size}")

    def _setup_encoders(self):
        """Setup T5 and VAE for encoding."""
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(self.device).total_memory
            free_mem -= torch.cuda.memory_allocated(self.device)

            # Move VAE to GPU if enough memory
            if free_mem > 10 * 1024**3:
                logger.info("Moving VAE to GPU for faster encoding")
                self.wrapper.vae = self.wrapper.vae.to(self.device)

    def _build_optimizer(self):
        """Build optimizer for DiT parameters only."""
        cfg = self.config.training

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
        Single training step with Self-Forcing.

        Args:
            batch: Dict with 'video' (B, T, C, H, W) and 'instruction' (List[str])

        Returns:
            Dict with 'loss' tensor and optional metrics
        """
        video = batch["video"]  # (B, T, C, H, W)
        instructions = batch["instruction"]  # List[str]
        B = video.shape[0]

        # Convert video format: (B, T, C, H, W) -> (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4)

        # Normalize from [0, 1] to [-1, 1]
        video = video * 2 - 1

        # Encode video to latent space
        with torch.no_grad():
            x_clean = self._encode_video(video)  # (B, C_latent, T', H', W')

        # Encode text
        with torch.no_grad():
            context = self._encode_text(instructions)  # (B, L, D)

        # Sample timesteps
        t = sample_timestep(B, self.device)

        # Self-Forcing forward pass
        v_pred, v_target = self.wrapper.forward_self_forcing(
            x_clean=x_clean,
            t=t,
            context=context,
            chunk_size=self.chunk_size,
            same_t_across_chunks=self.same_t_across_chunks,
        )

        # Compute loss
        if self.use_embodiment_loss and "embodiment_mask" in batch:
            # Embodiment-aware loss
            embodiment_mask = batch["embodiment_mask"]  # (B, 1, T, H, W)
            loss = self._embodiment_aware_loss(v_pred, v_target, embodiment_mask)
        else:
            # Standard MSE loss
            loss = torch.nn.functional.mse_loss(v_pred, v_target)

        return {"loss": loss}

    def _embodiment_aware_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        embodiment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute embodiment-aware loss.

        L = ||(1 + η·U(x_1)) ⊙ (v_θ - v_target)||²

        Args:
            v_pred: Predicted velocity (B, C, T, H, W)
            v_target: Target velocity (B, C, T, H, W)
            embodiment_mask: Binary mask indicating robot regions (B, 1, T, H, W)

        Returns:
            Weighted MSE loss
        """
        # Compute weights: 1 + η * mask
        weights = 1.0 + self.eta * embodiment_mask

        # Weighted MSE
        diff = v_pred - v_target
        weighted_diff = weights * diff
        loss = (weighted_diff ** 2).mean()

        return loss

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space."""
        video_device = video.device
        vae_device = next(self.wrapper.vae.parameters()).device

        if video_device != vae_device:
            video = video.to(vae_device)

        latent = self.wrapper.vae.encode(video)

        if latent.device != video_device:
            latent = latent.to(video_device)

        return latent

    def _encode_text(self, instructions: list) -> torch.Tensor:
        """Encode text instructions."""
        context = self.wrapper.encode_text(instructions)
        return context.to(self.device)

    def train_step_standard(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Standard training step (non-causal, for comparison/warmup).

        Can be used for initial warmup before switching to Self-Forcing.
        """
        video = batch["video"]
        instructions = batch["instruction"]
        B = video.shape[0]

        # Convert video format
        video = video.permute(0, 2, 1, 3, 4)
        video = video * 2 - 1

        # Encode
        with torch.no_grad():
            x1 = self._encode_video(video)
            context = self._encode_text(instructions)

        # Sample timesteps and noise
        t = sample_timestep(B, self.device)
        x0 = torch.randn_like(x1)

        # Add noise
        t_expanded = t
        while t_expanded.dim() < x1.dim():
            t_expanded = t_expanded.unsqueeze(-1)
        x_t = t_expanded * x1 + (1 - t_expanded) * x0

        # Target velocity
        v_target = x0 - x1

        # Forward through DiT
        v_pred = self.wrapper(x_t, t, context)

        # Loss
        loss = torch.nn.functional.mse_loss(v_pred, v_target)

        return {"loss": loss}

    def save_checkpoint(self, path: Optional[str] = None):
        """Save checkpoint (DiT weights only)."""
        from ..distributed.fsdp_utils import is_main_process

        if not is_main_process():
            return

        path = path or self.config.output.save_path
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        dit_state = self.wrapper.get_dit_state_dict()

        checkpoint = {
            "model": dit_state,
            "step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.to_dict(),
            "stage": 2,  # Mark as Stage 2 checkpoint
        }

        if self.config.output.save_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        if self.config.output.save_scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved Stage 2 checkpoint to {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location="cpu")

        self.wrapper.dit.load_state_dict(checkpoint["model"], strict=False)

        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        logger.info(f"Loaded checkpoint at step {self.global_step}")


def create_vidarc_trainer(config: VidarConfig) -> VidarCausalTrainer:
    """Factory function to create Vidarc causal trainer."""
    return VidarCausalTrainer(config)
