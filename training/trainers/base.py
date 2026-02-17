"""Base trainer class for Vidar/Vidarc training."""

import os
import re
import time
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
from ..utils.timing import time_block, print_stats, reset_stats, print_step_timings, TIME_DEBUG_ENABLED

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
        self.train_start_time = None
        self.total_steps = None
        self.wandb_run_id = None  # For resuming wandb runs

    def _setup_logging(self):
        """Setup logging."""
        if is_main_process():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )

    def _init_wandb(self):
        """
        Initialize wandb (called after checkpoint loading to support resume).
        
        Works in both online and offline modes:
        - Online mode: Resumes run on wandb cloud
        - Offline mode: Resumes local run (data saved in wandb/ directory)
        """
        if not is_main_process():
            return
        
        if not self.config.logging.use_wandb:
            return
        
        # Check if wandb is already initialized
        try:
            import wandb
            if wandb.run is not None:
                # Already initialized, just update run_id if needed
                if not self.wandb_run_id:
                    self.wandb_run_id = wandb.run.id
                return
        except:
            pass
        
        try:
            import wandb
            # Resume wandb run if run_id is available (from checkpoint)
            # This works in both online and offline modes
            init_kwargs = {
                "project": self.config.logging.wandb_project,
                "entity": self.config.logging.wandb_entity,
                "config": self.config.to_dict(),
            }
            if self.wandb_run_id:
                # In offline mode, resume parameter is ignored by wandb
                # We only set the id to reuse the same run id (creates new directory but same id)
                # This allows data continuity when syncing to cloud later
                init_kwargs["id"] = self.wandb_run_id
                # Only use resume in online mode (wandb will auto-detect mode)
                # In offline mode, wandb ignores resume and creates new directory with same id
                # This is acceptable - data will be saved and can be merged later
                logger.info(f"Using wandb run id: {self.wandb_run_id} (offline mode: will create new directory with same id)")
            wandb.init(**init_kwargs)
            # Save run id for future resume (works in offline mode)
            self.wandb_run_id = wandb.run.id if wandb.run else None
            if wandb.run:
                mode = 'offline' if wandb.run.settings.mode == 'offline' else 'online'
                logger.info(f"Wandb initialized - run id: {self.wandb_run_id}, mode: {mode}")
        except ImportError:
            logger.warning("wandb not installed, skipping")
        except Exception as e:
            # If resume fails (e.g., run not found in offline mode), log warning and continue
            logger.warning(f"Failed to resume wandb run: {e}. Starting new run.")
            try:
                import wandb
                # Try again without resume
                wandb.init(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    config=self.config.to_dict(),
                )
                self.wandb_run_id = wandb.run.id if wandb.run else None
                if wandb.run:
                    mode = 'offline' if wandb.run.settings.mode == 'offline' else 'online'
                    logger.info(f"Wandb initialized (new run) - run id: {self.wandb_run_id}, mode: {mode}")
            except:
                pass

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
            data_dir_str = cfg.data_dir if isinstance(cfg.data_dir, str) else ", ".join(str(d) for d in cfg.data_dir)
            logger.info(f"Using HDF5 dataset from {data_dir_str}")
            return get_hdf5_dataloader(
                data_dir=cfg.data_dir,
                batch_size=self.config.training.batch_size,
                num_frames=cfg.num_frames,
                resolution=cfg.resolution,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                distributed=(self.world_size > 1),
                cfg_prob=cfg.cfg_prob,
                prefetch_factor=getattr(cfg, 'prefetch_factor', None),
                persistent_workers=getattr(cfg, 'persistent_workers', False),
            )
        else:
            data_dir_str = cfg.data_dir if isinstance(cfg.data_dir, str) else ", ".join(str(d) for d in cfg.data_dir)
            logger.info(f"Using video dataset from {data_dir_str}")
            # Video dataset only supports single directory
            data_dir_to_use = cfg.data_dir if isinstance(cfg.data_dir, str) else cfg.data_dir[0]
            dataset = VidarDataset(
                data_dir=data_dir_to_use,
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
                prefetch_factor=getattr(cfg, 'prefetch_factor', None),
                persistent_workers=getattr(cfg, 'persistent_workers', False),
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
            sync_module_states=getattr(cfg, "sync_module_states", False),
            forward_prefetch=getattr(cfg, "forward_prefetch", False),
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
        self.total_steps = total_steps
        self.train_start_time = time.time()

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
            # Timing: Total step
            with time_block("Total Step", device=self.device):
                # Get batch
                with time_block("Data Loading", device=None):
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
                # Print timing stats at every log interval (for more frequent output)
                if TIME_DEBUG_ENABLED:
                    # Use available steps for window size (min of available steps or 10)
                    window_size = min(self.global_step + 1, 10)
                    print_stats(self.global_step, window_size=window_size)

            # Checkpointing (skip step 0)
            if self.global_step > 0 and self.global_step % self.config.logging.save_interval == 0:
                self.save_checkpoint()
                # Run evaluation after checkpoint save (if implemented and enabled)
                if hasattr(self, 'evaluate_checkpoint'):
                    self.evaluate_checkpoint()

            self.global_step += 1

        # Final save
        self.save_checkpoint()
        
        # Clear GPU cache before final evaluation
        if torch.cuda.is_available():
            logger.info("Clearing GPU cache before final evaluation...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(10)  # Wait longer after training completes
            logger.info("GPU cache cleared. Waiting 10 seconds for memory release...")
        
        # Run final evaluation
        if hasattr(self, 'evaluate_checkpoint'):
            self.evaluate_checkpoint()
        logger.info("Training complete!")

    def _train_step_with_accumulation(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Training step with gradient accumulation."""
        accum_steps = self.config.training.gradient_accumulation

        total_loss = 0.0
        metrics = {}

        # Timing: Gradient accumulation loop
        with time_block("Gradient Accumulation Loop", device=self.device):
            for i in range(accum_steps):
                # Timing: Autocast context
                with time_block("Autocast Context", device=self.device):
                    with torch.cuda.amp.autocast(enabled=self.config.distributed.mixed_precision != "fp32"):
                        step_metrics = self.train_step(batch)

                loss = step_metrics["loss"] / accum_steps
                
                # Timing: Backward pass
                with time_block("Backward", device=self.device):
                    loss.backward()

                # Timing: Gradient check
                if self.global_step % 10 == 0 or torch.isnan(loss) or torch.isinf(loss):
                    with time_block("Gradient Check", device=self.device):
                        total_norm = 0.0
                        param_count = 0
                        nan_grad_count = 0
                        inf_grad_count = 0
                        nan_grad_names = []
                        inf_grad_names = []
                        
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                param_count += 1
                                
                                if torch.isnan(param.grad).any():
                                    nan_grad_count += 1
                                    nan_grad_names.append(name)
                                if torch.isinf(param.grad).any():
                                    inf_grad_count += 1
                                    inf_grad_names.append(name)
                        
                        total_norm = total_norm ** (1. / 2) if total_norm > 0 else 0.0
                        
                        if nan_grad_count > 0 or inf_grad_count > 0 or torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"[DEBUG NaN] Gradient check at step {self.global_step}: "
                                        f"total_norm={total_norm:.6f}, "
                                        f"nan_grad_count={nan_grad_count}, inf_grad_count={inf_grad_count}, "
                                        f"loss={loss.item() if isinstance(loss, torch.Tensor) else loss}")
                            if nan_grad_count > 0:
                                logger.error(f"[DEBUG NaN] NaN gradients in: {nan_grad_names[:10]}")  # Show first 10
                            if inf_grad_count > 0:
                                logger.error(f"[DEBUG NaN] Inf gradients in: {inf_grad_names[:10]}")  # Show first 10

                total_loss += step_metrics["loss"].item() / accum_steps

        # Timing: Optimizer step
        with time_block("Optimizer Step", device=self.device):
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # Timing: Sync
        with time_block("Sync", device=self.device):
            # Sync metrics across ranks
            if self.world_size > 1:
                total_loss = all_reduce_mean(torch.tensor(total_loss, device=self.device)).item()
                torch.cuda.synchronize(self.device)

        metrics["loss"] = total_loss
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        metrics["step"] = self.global_step

        return metrics

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        with time_block("Data Transfer", device=self.device):
            result = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.to(self.device)
                else:
                    result[k] = v
        return result

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics with enhanced visualization."""
        if not is_main_process():
            return

        # Calculate progress
        if self.total_steps and self.total_steps > 0:
            progress_pct = (self.global_step / self.total_steps) * 100
            remaining_steps = self.total_steps - self.global_step
        else:
            progress_pct = 0.0
            remaining_steps = 0

        # Calculate time information
        time_info = ""
        if self.train_start_time:
            elapsed_time = time.time() - self.train_start_time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            elapsed_seconds = int(elapsed_time % 60)
            
            if remaining_steps > 0 and self.global_step > 0:
                # Estimate remaining time
                steps_per_sec = self.global_step / elapsed_time
                remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                remaining_hours = int(remaining_time // 3600)
                remaining_minutes = int((remaining_time % 3600) // 60)
                remaining_seconds = int(remaining_time % 60)
                time_info = f" | Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d} | ETA: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}"
            else:
                time_info = f" | Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"

        # Format key metrics
        loss_str = ""
        lr_str = ""
        other_metrics = []
        
        for k, v in metrics.items():
            if k == "loss" and isinstance(v, float):
                loss_str = f"Loss: {v:.6f}"
            elif k == "lr" and isinstance(v, float):
                # Use scientific notation for LR to display small values correctly
                # (e.g., 2e-5 = 0.00002 would show as 0.0000 with .4f format)
                lr_str = f"LR: {v:.2e}"
            elif isinstance(v, float):
                other_metrics.append(f"{k}: {v:.4f}")

        # Build enhanced log string
        if self.total_steps:
            log_str = f"[Step {self.global_step}/{self.total_steps} ({progress_pct:.2f}%)]"
        else:
            log_str = f"[Step {self.global_step}]"
        
        if loss_str:
            log_str += f" | {loss_str}"
        if lr_str:
            log_str += f" | {lr_str}"
        if other_metrics:
            log_str += " | " + " | ".join(other_metrics)
        if time_info:
            log_str += time_info

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
        
        # Auto-inject current step into filename if not already present
        if path == self.config.output.save_path:
            # Use current global step (real-time step) instead of total steps
            current_step = self.global_step
            path_obj = Path(path)
            
            # Check if step count is already in filename (e.g., vidarc_4x_1234.pt)
            # Remove any existing step number from stem first
            stem = path_obj.stem
            # Try to remove existing step pattern (e.g., _4000, _1234)
            stem_clean = re.sub(r'_\d+$', '', stem)
            
            # Insert current step count before extension: vidarc_4x.pt -> vidarc_4x_1234.pt
            new_stem = f"{stem_clean}_{current_step}"
            path = str(path_obj.parent / f"{new_stem}{path_obj.suffix}")
        
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
        
        # Save wandb run id for resuming
        if self.wandb_run_id:
            checkpoint["wandb_run_id"] = self.wandb_run_id

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
        
        # Load wandb run id for resuming
        if "wandb_run_id" in checkpoint:
            self.wandb_run_id = checkpoint["wandb_run_id"]
            logger.info(f"Found wandb run id in checkpoint: {self.wandb_run_id}")

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
