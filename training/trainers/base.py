"""Base trainer class for Vidar/Vidarc training."""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Iterator

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..config import VidarConfig
from ..losses import VidarLoss, add_noise, sample_timestep
from ..data.dataset import VidarDataset, get_dataloader
from ..distributed.fsdp_utils import (
    init_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    wrap_model_fsdp,
    all_reduce_mean,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Base trainer class for Vidar/Vidarc.

    Handles:
    - Model loading and wrapping (FSDP)
    - Optimizer and scheduler setup
    - Data loading
    - Training loop skeleton
    - Checkpointing
    - Logging
    """

    def __init__(self, config: VidarConfig):
        """
        Args:
            config: Training configuration
        """
        self.config = config

        # Initialize distributed
        if torch.cuda.is_available():
            self.local_rank = init_distributed()
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device("cpu")

        self.rank = get_rank()
        self.world_size = get_world_size()

        # Setup logging
        self._setup_logging()

        # Initialize components (to be built)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.loss_fn = None

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _setup_logging(self):
        """Setup logging."""
        if is_main_process():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )

            # Setup wandb if enabled
            if self.config.logging.use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project=self.config.logging.wandb_project,
                        entity=self.config.logging.wandb_entity,
                        config=self.config.to_dict(),
                    )
                except ImportError:
                    logger.warning("wandb not installed, skipping")

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build and return the model. Override in subclass."""
        raise NotImplementedError

    def _build_optimizer(self) -> AdamW:
        """Build optimizer."""
        cfg = self.config.training

        # Filter frozen params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        )

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        cfg = self.config.training
        total_steps = cfg.max_steps or cfg.num_steps

        if cfg.scheduler == "cosine":
            # Warmup + cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=cfg.warmup_steps,
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - cfg.warmup_steps,
                eta_min=cfg.lr * 0.01,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[cfg.warmup_steps],
            )
        elif cfg.scheduler == "linear":
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps,
            )
        else:
            # Constant LR with warmup
            scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=cfg.warmup_steps,
            )

        return scheduler

    def _build_dataloader(self):
        """Build dataloader."""
        cfg = self.config.data
        dataset_type = getattr(cfg, 'dataset_type', 'hdf5')

        if dataset_type == "hdf5":
            from ..data.hdf5_dataset import HDF5VLADataset, get_hdf5_dataloader
            logger.info(f"Using HDF5 dataset from {cfg.data_dir}")
            return get_hdf5_dataloader(
                data_dir=cfg.data_dir,
                batch_size=self.config.training.batch_size,
                num_frames=cfg.num_frames,
                resolution=cfg.resolution,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                distributed=(self.world_size > 1),
                cfg_prob=cfg.cfg_prob,
            )
        else:
            logger.info(f"Using video dataset from {cfg.data_dir}")
            dataset = VidarDataset(
                data_dir=cfg.data_dir,
                num_frames=cfg.num_frames,
                resolution=cfg.resolution,
                fps=cfg.fps,
                cfg_prob=cfg.cfg_prob,
            )

            dataloader = get_dataloader(
                dataset,
                batch_size=self.config.training.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                distributed=(self.world_size > 1),
            )

            return dataloader

    def _build_loss(self) -> VidarLoss:
        """Build loss function."""
        cfg = self.config.loss

        return VidarLoss(
            loss_type=cfg.type,
            embodiment_aware=cfg.embodiment_aware,
            eta=cfg.eta,
            cfg_prob=self.config.data.cfg_prob,
        )

    def setup(self):
        """Setup all training components."""
        logger.info("Setting up trainer...")

        # Build model
        logger.info("Building model...")
        self.model = self._build_model()

        # Wrap with FSDP if distributed
        if self.world_size > 1 and self.config.distributed.use_fsdp:
            logger.info("Wrapping model with FSDP...")
            self.model = self._wrap_fsdp(self.model)
        else:
            self.model = self.model.to(self.device)

        # Build optimizer and scheduler
        logger.info("Building optimizer and scheduler...")
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Build dataloader
        logger.info("Building dataloader...")
        self.dataloader = self._build_dataloader()

        # Build loss
        self.loss_fn = self._build_loss()

        logger.info("Setup complete!")

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP."""
        cfg = self.config.distributed

        # Get transformer layer class for auto-wrap
        transformer_layer_cls = self._get_transformer_layer_cls()

        return wrap_model_fsdp(
            model,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            transformer_layer_cls=transformer_layer_cls,
            cpu_offload=cfg.cpu_offload,
            activation_checkpointing=cfg.activation_checkpointing,
        )

    def _get_transformer_layer_cls(self):
        """Get transformer layer classes for FSDP wrapping. Override in subclass."""
        return None

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Single training step. Override in subclass.

        Args:
            batch: Batch of data

        Returns:
            Dict with 'loss' and any other metrics
        """
        raise NotImplementedError

    def train(self):
        """Main training loop."""
        cfg = self.config.training
        total_steps = cfg.max_steps or cfg.num_steps

        logger.info(f"Starting training for {total_steps} steps...")

        # Validate dataset size before training
        dataset_size = len(self.dataloader.dataset) if hasattr(self.dataloader, 'dataset') else 0
        batch_size = self.config.training.batch_size

        if dataset_size == 0:
            raise RuntimeError(
                f"Dataset is empty! No episodes found in {self.config.data.data_dir}. "
                f"Please check that HDF5 files exist in the 'hdf5/' subdirectory "
                f"and match the pattern 'episode_*.hdf5'"
            )

        effective_batch_size = batch_size * self.world_size
        if dataset_size < effective_batch_size:
            logger.warning(
                f"Dataset size ({dataset_size}) is smaller than effective batch size "
                f"({batch_size} x {self.world_size} GPUs = {effective_batch_size}). "
                f"Training may fail or have limited batches per epoch."
            )

        self.model.train()
        data_iter = iter(self.dataloader)

        while self.global_step < total_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.dataloader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    raise RuntimeError(
                        f"DataLoader returned empty iterator after reset. "
                        f"Dataset has {dataset_size} samples but batch_size={batch_size} "
                        f"with {self.world_size} GPUs (drop_last=True). "
                        f"Try reducing batch_size or adding more data."
                    )

            # Move to device
            batch = self._to_device(batch)

            # Training step
            metrics = self._train_step_with_accumulation(batch)

            # Logging
            if self.global_step % self.config.logging.log_interval == 0:
                self._log_metrics(metrics)

            # Checkpointing
            if self.global_step % self.config.logging.save_interval == 0:
                self.save_checkpoint()

            self.global_step += 1

        # Final save
        self.save_checkpoint()
        logger.info("Training complete!")

    def _train_step_with_accumulation(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Training step with gradient accumulation."""
        accum_steps = self.config.training.gradient_accumulation

        total_loss = 0.0
        metrics = {}

        for i in range(accum_steps):
            # Forward + backward
            with torch.cuda.amp.autocast(enabled=self.config.distributed.mixed_precision != "fp32"):
                step_metrics = self.train_step(batch)

            loss = step_metrics["loss"] / accum_steps
            loss.backward()

            total_loss += step_metrics["loss"].item() / accum_steps

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Sync metrics across ranks
        if self.world_size > 1:
            total_loss = all_reduce_mean(torch.tensor(total_loss, device=self.device)).item()

        metrics["loss"] = total_loss
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        metrics["step"] = self.global_step

        return metrics

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics."""
        if not is_main_process():
            return

        # Console logging
        log_str = f"Step {self.global_step}"
        for k, v in metrics.items():
            if isinstance(v, float):
                log_str += f" | {k}: {v:.4f}"
        logger.info(log_str)

        # Wandb logging
        if self.config.logging.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=self.global_step)
            except:
                pass

    def save_checkpoint(self, path: Optional[str] = None):
        """Save checkpoint."""
        if not is_main_process():
            return

        path = path or self.config.output.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Get state dict
        if hasattr(self.model, "module"):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            "model": model_state,
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

        # Load model
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["model"], strict=False)

        # Load training state
        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        logger.info(f"Loaded checkpoint at step {self.global_step}")

    def cleanup(self):
        """Cleanup resources."""
        if self.config.logging.use_wandb and is_main_process():
            try:
                import wandb
                wandb.finish()
            except:
                pass

        cleanup_distributed()
