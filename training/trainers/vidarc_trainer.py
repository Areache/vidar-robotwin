"""Stage 2 Trainer: Vidarc causal fine-tuning with Self-Forcing."""

import logging
import math
import re
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .base import BaseTrainer
from ..config import VidarConfig
from ..models.wrapper_causal import WanModelCausalTrainingWrapper
from ..losses import sample_timestep, sample_timestep_for_truncation

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear layer that's compatible with FSDP.

    Instead of wrapping the original layer, this replaces it entirely
    with a frozen weight matrix + trainable low-rank adapters.
    """

    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype

        # Store original weight as frozen buffer (not a parameter, so FSDP won't shard it)
        self.register_buffer('frozen_weight', original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.register_buffer('frozen_bias', original_layer.bias.data.clone())
        else:
            self.frozen_bias = None

        # LoRA matrices - trainable parameters on same device/dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA weights (A with kaiming, B with zeros)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B already zeros

    def forward(self, x):
        # Frozen linear: x @ W^T + b
        result = nn.functional.linear(x, self.frozen_weight, self.frozen_bias)

        # LoRA: x @ A^T @ B^T * scaling
        lora_x = self.lora_dropout(x)
        lora_out = nn.functional.linear(nn.functional.linear(lora_x, self.lora_A), self.lora_B)

        return result + lora_out * self.scaling


def apply_lora_to_model(model: nn.Module, lora_config) -> nn.Module:
    """
    Apply LoRA (Low-Rank Adaptation) to the model.

    This implementation manually injects LoRA layers into the target modules,
    which is more compatible with diffusion models than PEFT's get_peft_model().

    Args:
        model: The model to apply LoRA to
        lora_config: LoRA configuration

    Returns:
        Model with LoRA applied (same model, modified in-place)
    """
    target_modules = lora_config.target_modules
    rank = lora_config.rank
    alpha = lora_config.alpha
    dropout = lora_config.dropout

    logger.info(f"Applying LoRA with rank={rank}, alpha={alpha}")
    logger.info(f"Target modules: {target_modules}")

    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Find and replace target modules with LoRA layers
    lora_layers_added = 0

    for name, module in model.named_modules():
        # Check if this module's name ends with any of the target module names
        module_name = name.split('.')[-1]
        if module_name in target_modules and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Create LoRA layer
            lora_layer = LoRALinear(module, rank, alpha, dropout)

            # Replace the original layer
            setattr(parent, attr_name, lora_layer)
            lora_layers_added += 1

    logger.info(f"Added LoRA to {lora_layers_added} layers")

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    return model


class VidarCausalTrainer(BaseTrainer):
    """
    Stage 2 Trainer: Vidarc causal fine-tuning with Self-Forcing.

    Implements Self-Forcing training paradigm from the Vidarc paper:
    - Causal attention: previous frames are noise-free, attended via KV cache
    - Teacher forcing: use ground truth for KV cache during training
    - Autoregressive rollout with chunk-wise processing
    - Embodiment-aware loss with higher weight on robot regions

    Additional optimizations:
    - Few-step diffusion: Use fewer denoising steps for efficiency
    - Stochastic gradient truncation: Only backprop through one random timestep

    Loss function (with embodiment-aware weighting):
    L = ||(1 + η·U(x_1)) ⊙ (v_θ(x_t, t, c) - (x_0 - x_1))||²

    where U(x_1) indicates robot/embodiment regions (η=3.0 default).
    """

    def __init__(self, config: VidarConfig):
        super().__init__(config)
        self.wrapper: Optional[WanModelCausalTrainingWrapper] = None

        # Self-Forcing parameters
        self.chunk_size = getattr(config.model, "chunk_size", 16)
        self.same_t_across_chunks = getattr(config.model, "same_t_across_chunks", True)

        # Embodiment-aware loss parameters
        self.eta = getattr(config.training, "eta", 3.0)  # Embodiment weight
        self.use_embodiment_loss = getattr(config.training, "use_embodiment_loss", False)

        # Training mode: Self-Forcing (True) or Standard (False)
        # Set use_self_forcing=False to enable activation_checkpointing
        self.use_self_forcing = getattr(config.self_forcing, "enabled", True)
        logger.info(f"Training mode: {'Self-Forcing' if self.use_self_forcing else 'Standard (no SF)'}")

        # LoRA parameters
        self.use_lora = getattr(config.lora, "enabled", False)
        if self.use_lora:
            logger.info(f"LoRA enabled with rank={config.lora.rank}, alpha={config.lora.alpha}")

        # Few-Step Diffusion & Stochastic Gradient Truncation parameters
        self.num_inference_steps = getattr(config.diffusion, "num_inference_steps", 10)
        self.num_train_timesteps = getattr(config.diffusion, "num_train_timesteps", 1000)
        self.stochastic_truncation = getattr(config.diffusion, "stochastic_truncation", True)
        self.truncation_strategy = getattr(config.diffusion, "truncation_strategy", "uniform")
        self.detach_kv_cache = getattr(config.diffusion, "detach_kv_cache", True)

        # Importance sampling parameters (for strategy="importance")
        self.importance_low_t = getattr(config.diffusion, "importance_low_t", 0.3)
        self.importance_high_t = getattr(config.diffusion, "importance_high_t", 0.7)
        self.importance_weight = getattr(config.diffusion, "importance_weight", 3.0)

        if self.stochastic_truncation:
            logger.info(f"Stochastic Gradient Truncation enabled: strategy={self.truncation_strategy}")
            logger.info(f"Few-Step Diffusion: num_inference_steps={self.num_inference_steps}")
            if self.truncation_strategy == "importance":
                logger.info(f"  Importance sampling: t=[{self.importance_low_t}, {self.importance_high_t}], weight={self.importance_weight}")

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

        # Disable activation checkpointing when using LoRA + Self-Forcing
        # Activation checkpointing is incompatible with stateful KV cache:
        # - Forward pass: KV cache accumulates across chunks
        # - Backward recomputation: KV cache state not preserved -> shape mismatch
        # LoRA already provides ~75% memory savings, so this is acceptable
        use_activation_checkpointing = cfg.activation_checkpointing
        if self.use_lora and self.use_self_forcing and use_activation_checkpointing:
            logger.warning("Disabling activation_checkpointing for LoRA + Self-Forcing training")
            logger.warning("(Activation checkpointing is incompatible with stateful KV cache)")
            use_activation_checkpointing = False

        wrapped_dit = wrap_model_fsdp(
            model,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            transformer_layer_cls=transformer_layer_cls,
            cpu_offload=cfg.cpu_offload,
            activation_checkpointing=use_activation_checkpointing,
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

        # Apply LoRA if enabled (before FSDP wrapping)
        if self.use_lora:
            logger.info("Applying LoRA to DiT model...")
            self.model = apply_lora_to_model(self.model, self.config.lora)
            # Update wrapper's dit reference
            self.wrapper.dit = self.model

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

        # Note: wandb will be initialized after checkpoint loading (if any)
        # This ensures we can resume the correct wandb run in offline mode

        logger.info(f"Setup complete! Chunk size: {self.chunk_size}")
        if self.use_lora:
            logger.info(f"LoRA training enabled with rank={self.config.lora.rank}")

    def _setup_encoders(self):
        """Setup T5 and VAE for encoding."""
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(self.device).total_memory
            free_mem -= torch.cuda.memory_allocated(self.device)

            # Move VAE to GPU if enough memory
            if free_mem > 10 * 1024**3:
                logger.info("Moving VAE to GPU for faster encoding")
                self.wrapper.vae.model = self.wrapper.vae.model.to(self.device)
                self.wrapper.vae.device = self.device

    def _build_optimizer(self):
        """Build optimizer for DiT parameters only."""
        cfg = self.config.training

        # Get trainable parameters
        if self.use_lora:
            # For LoRA, get parameters from the PEFT model
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            # LoRA typically needs 10-20x higher learning rate
            lr = cfg.lr
            if lr < 1e-4:
                # If using a low learning rate typical for full finetuning, scale up for LoRA
                lr = lr * 10
                logger.info(f"LoRA: Scaling learning rate from {cfg.lr} to {lr}")
        else:
            trainable_params = self.wrapper.get_trainable_parameters()
            lr = cfg.lr

        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        )

        return optimizer

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Single training step - routes to Self-Forcing or Standard based on config.

        Args:
            batch: Dict with 'video' (B, T, C, H, W) and 'instruction' (List[str])

        Returns:
            Dict with 'loss' tensor and optional metrics
        """
        # Route to appropriate training method
        if self.use_self_forcing:
            return self.train_step_self_forcing(batch)
        else:
            return self.train_step_standard(batch)

    def train_step_self_forcing(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Training step with Self-Forcing.

        Implements stochastic gradient truncation when enabled:
        - Only backpropagates through a randomly selected timestep
        - Significantly reduces computation while maintaining effectiveness

        Note: This method is NOT compatible with activation_checkpointing
        due to KV cache accumulation.
        """
        video = batch["video"]  # (B, T, C, H, W)
        instructions = batch["instruction"]  # List[str]
        B = video.shape[0]

        # Convert video format: (B, T, C, H, W) -> (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4)

        # Clamp to ensure strict [0, 1] range before normalization (matching inference)
        video = torch.clamp(video, 0.0, 1.0)

        # Normalize from [0, 1] to [-1, 1]
        video = video * 2 - 1

        # Encode video to latent space
        with torch.no_grad():
            x_clean = self._encode_video(video)  # (B, C_latent, T', H', W')

        # Encode text
        with torch.no_grad():
            context = self._encode_text(instructions)  # (B, L, D)

        # Sample timesteps using appropriate strategy
        if self.stochastic_truncation:
            # Use stochastic gradient truncation sampling
            t = sample_timestep_for_truncation(
                B, self.device,
                strategy=self.truncation_strategy,
                num_train_timesteps=self.num_train_timesteps,
                importance_low_t=self.importance_low_t,
                importance_high_t=self.importance_high_t,
                importance_weight=self.importance_weight,
                num_strata=self.num_inference_steps,  # Use same number as inference steps
            )
        else:
            # Standard uniform sampling
            num_train_timesteps = getattr(self.wrapper.dit.config, 'num_train_timesteps', 1000)
            t = sample_timestep(B, self.device, num_train_timesteps=num_train_timesteps)

        # Self-Forcing forward pass with optional gradient truncation
        if self.stochastic_truncation:
            v_pred, v_target = self._forward_self_forcing_with_truncation(
                x_clean=x_clean,
                t=t,
                context=context,
            )
        else:
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

    def _forward_self_forcing_with_truncation(
        self,
        x_clean: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple:
        """
        Self-Forcing forward pass with stochastic gradient truncation.

        Key optimization: Only backpropagates through a single randomly selected
        denoising step, reducing computational cost significantly.

        The gradient is only computed for the selected timestep s ∈ [1, T].
        KV cache updates are detached from the gradient graph.

        Args:
            x_clean: Clean latent video (B, C, T, H, W)
            t: Timestep tensor (B,) - the single timestep for gradient computation
            context: Text embeddings

        Returns:
            Tuple of (predicted velocities, target velocities)
        """
        B, C, T, H, W = x_clean.shape
        block_size = self.wrapper.get_block_size(x_clean.shape)
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size

        # Clean KV cache
        self.wrapper.dit.clean_cache()

        # Randomly select which chunk to backprop through
        # This is the "stochastic" part of stochastic gradient truncation
        grad_chunk_idx = torch.randint(0, num_chunks, (1,)).item()

        all_v_pred = []
        all_v_target = []

        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * self.chunk_size
            end_t = min(start_t + self.chunk_size, T)

            # Get current chunk
            x_chunk = x_clean[:, :, start_t:end_t, :, :]

            # Sample noise
            x0_chunk = torch.randn_like(x_chunk)

            # Use the same timestep for all chunks (as per same_t_across_chunks)
            t_chunk = t

            # Expand timestep for broadcasting
            t_expanded = t_chunk
            while t_expanded.dim() < x_chunk.dim():
                t_expanded = t_expanded.unsqueeze(-1)

            # Normalize t to [0, 1] for noising
            t_normalized = t_expanded / self.num_train_timesteps

            # Create noised chunk: x_t = (1-t) * x_1 + t * x_0
            x_noised = (1 - t_normalized) * x_chunk + t_normalized * x0_chunk

            # Target velocity
            v_target = x0_chunk - x_chunk

            # Determine if this chunk should have gradients
            requires_grad = (chunk_idx == grad_chunk_idx)

            # Build attention mask
            if chunk_idx == 0:
                attention_mask = self.wrapper._build_causal_mask(
                    x_noised.shape, block_size, self.device
                )
                prefill = True
                chunk_prefill = False
            else:
                attention_mask = self.wrapper._build_chunk_causal_mask(
                    x_noised.shape, block_size,
                    kv_len=self.wrapper.dit.kvcache_len(),
                    device=self.device
                )
                prefill = False
                chunk_prefill = False

            # Scale timestep for model input
            t_model = t_chunk

            if requires_grad:
                # This chunk gets gradients - do full forward pass
                # Step 1: Update KV cache with clean frames (detached if configured)
                t_clean = torch.zeros_like(t_model)

                if self.detach_kv_cache:
                    # Detach KV cache update from gradient graph
                    with torch.no_grad():
                        self.wrapper.forward_causal(
                            x=x_chunk,
                            t=t_clean,
                            context=context,
                            attention_mask=attention_mask,
                            use_cache=True,
                            prefill=prefill,
                            chunk_prefill=chunk_prefill,
                            block_idx=None if prefill else chunk_idx,
                        )
                else:
                    self.wrapper.forward_causal(
                        x=x_chunk,
                        t=t_clean,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=True,
                        prefill=prefill,
                        chunk_prefill=chunk_prefill,
                        block_idx=None if prefill else chunk_idx,
                    )

                # Step 2: Compute prediction with gradients
                v_pred = self.wrapper.forward_causal(
                    x=x_noised,
                    t=t_model,
                    context=context,
                    attention_mask=attention_mask,
                    use_cache=False,
                    prefill=prefill,
                    chunk_prefill=chunk_prefill,
                    block_idx=None if prefill else chunk_idx,
                )
            else:
                # This chunk is detached - no gradients
                with torch.no_grad():
                    # Update KV cache with clean frames
                    t_clean = torch.zeros_like(t_model)
                    self.wrapper.forward_causal(
                        x=x_chunk,
                        t=t_clean,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=True,
                        prefill=prefill,
                        chunk_prefill=chunk_prefill,
                        block_idx=None if prefill else chunk_idx,
                    )

                    # Compute prediction without gradients
                    v_pred = self.wrapper.forward_causal(
                        x=x_noised,
                        t=t_model,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=False,
                        prefill=prefill,
                        chunk_prefill=chunk_prefill,
                        block_idx=None if prefill else chunk_idx,
                    )

            all_v_pred.append(v_pred)
            all_v_target.append(v_target)

        # Clean up cache
        self.wrapper.dit.clean_cache()

        # Concatenate all chunks
        # Note: Only the grad_chunk_idx chunk will have gradients
        v_pred_all = torch.cat(all_v_pred, dim=2)
        v_target_all = torch.cat(all_v_target, dim=2)

        return v_pred_all, v_target_all

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
        """Encode video to latent space.

        The Wan2_2_VAE.encode() expects a list of videos (one per batch item),
        not a batched tensor. Each video should be (C, T, H, W).
        """
        video_device = video.device
        vae_device = next(self.wrapper.vae.model.parameters()).device

        if video_device != vae_device:
            video = video.to(vae_device)

        # VAE expects list of videos, not batched tensor
        # video shape: (B, C, T, H, W) -> list of (C, T, H, W)
        video_list = [video[i] for i in range(video.shape[0])]

        latent_list = self.wrapper.vae.encode(video_list)

        if latent_list is None:
            raise RuntimeError("VAE encoding failed - check video format and VAE device")

        # Stack back to batch: list of (C, T', H', W') -> (B, C, T', H', W')
        latent = torch.stack(latent_list, dim=0)

        if latent.device != video_device:
            latent = latent.to(video_device)

        return latent

    def _encode_text(self, instructions: list) -> torch.Tensor:
        """Encode text instructions."""
        context = self.wrapper.encode_text(instructions)
        # Handle list type (WanModelCausalTrainingWrapper returns list of tensors)
        if isinstance(context, list):
            # Stack list of tensors into single tensor (B, L, D)
            context = torch.stack([c.to(self.device) for c in context])
        else:
            # Already a tensor, just move to device
            context = context.to(self.device)
        return context

    def train_step_standard(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Standard training step (non-causal, for comparison/warmup).

        Supports stochastic gradient truncation for faster training.
        Can be used for initial warmup before switching to Self-Forcing.
        """
        video = batch["video"]
        instructions = batch["instruction"]
        B = video.shape[0]

        # Convert video format
        video = video.permute(0, 2, 1, 3, 4)

        # Clamp to ensure strict [0, 1] range before normalization (matching inference)
        video = torch.clamp(video, 0.0, 1.0)

        # Normalize from [0, 1] to [-1, 1] (same formula as inference)
        video = video * 2 - 1

        # Encode
        with torch.no_grad():
            x1 = self._encode_video(video)
            context = self._encode_text(instructions)

        # Sample timesteps using appropriate strategy
        if self.stochastic_truncation:
            t = sample_timestep_for_truncation(
                B, self.device,
                strategy=self.truncation_strategy,
                num_train_timesteps=self.num_train_timesteps,
                importance_low_t=self.importance_low_t,
                importance_high_t=self.importance_high_t,
                importance_weight=self.importance_weight,
                num_strata=self.num_inference_steps,
            )
        else:
            num_train_timesteps = getattr(self.wrapper.dit.config, 'num_train_timesteps', 1000)
            t = sample_timestep(B, self.device, num_train_timesteps=num_train_timesteps)

        x0 = torch.randn_like(x1)

        # Add noise - normalize t to [0, 1] for noising
        t_expanded = t / self.num_train_timesteps
        while t_expanded.dim() < x1.dim():
            t_expanded = t_expanded.unsqueeze(-1)

        # x_t = (1-t) * x1 + t * x0
        x_t = (1 - t_expanded) * x1 + t_expanded * x0

        # Target velocity
        v_target = x0 - x1

        # Forward through DiT
        v_pred = self.wrapper(x_t, t, context)

        # Loss
        loss = torch.nn.functional.mse_loss(v_pred, v_target)

        return {"loss": loss}

    def save_checkpoint(self, path: Optional[str] = None):
        """Save checkpoint (DiT weights only, or LoRA weights if using LoRA)."""
        from ..distributed.fsdp_utils import is_main_process
        # Torch 2.x exports FSDP as FullyShardedDataParallel; keep alias for compatibility
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullStateDictConfig,
                StateDictType,
            )
        except ImportError:
            # Fallback for older torch where FSDP may not be present
            FSDP = None
            FullStateDictConfig = None
            StateDictType = None

        path = path or self.config.output.save_path
        from pathlib import Path

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

        # Collect FSDP state dict properly (if FSDP is available and model is wrapped)
        if FSDP is not None and isinstance(self.model, FSDP):
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
            ):
                dit_state = self.model.state_dict()
        else:
            # Fallback: standard state_dict
            dit_state = self.model.state_dict()

        # Extract and save LoRA weights separately (after FSDP collection)
        if self.use_lora and is_main_process():
            try:
                # Extract LoRA parameters from state dict (lora_A, lora_B keys)
                lora_state_dict = {k: v for k, v in dit_state.items()
                                   if 'lora_A' in k or 'lora_B' in k}

                lora_path = Path(path).parent / f"lora_{Path(path).stem}.pt"
                torch.save({
                    "lora_state_dict": lora_state_dict,
                    "lora_config": {
                        "rank": self.config.lora.rank,
                        "alpha": self.config.lora.alpha,
                        "dropout": self.config.lora.dropout,
                        "target_modules": self.config.lora.target_modules,
                    }
                }, str(lora_path))
                logger.info(f"Saved LoRA adapter weights to {lora_path} ({len(lora_state_dict)} tensors)")
            except Exception as e:
                logger.warning(f"Failed to save LoRA adapter weights: {e}")

        if is_main_process():
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
            
            # Save wandb run id for resuming
            if hasattr(self, "wandb_run_id") and self.wandb_run_id:
                checkpoint["wandb_run_id"] = self.wandb_run_id

            torch.save(checkpoint, path)
            logger.info(f"Saved Stage 2 checkpoint to {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location="cpu")

        self.wrapper.dit.load_state_dict(checkpoint["model"], strict=False)

        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        
        # Load wandb run id for resuming (works in offline mode)
        # Priority: checkpoint > config file
        if "wandb_run_id" in checkpoint:
            self.wandb_run_id = checkpoint["wandb_run_id"]
            logger.info(f"Found wandb run id in checkpoint: {self.wandb_run_id} (will resume in offline mode)")
        elif hasattr(self.config.logging, "wandb_run_id") and self.config.logging.wandb_run_id:
            # If checkpoint doesn't have run_id, try to get from config
            self.wandb_run_id = self.config.logging.wandb_run_id
            logger.info(f"Using wandb run id from config: {self.wandb_run_id} (will resume in offline mode)")

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        logger.info(f"Loaded checkpoint at step {self.global_step}")
        
        # Initialize wandb after loading checkpoint (so we can resume if run_id exists)
        # This works in both online and offline modes
        self._init_wandb()

    def evaluate_checkpoint(self, checkpoint_path: str = None):
        """
        Run evaluation on the saved checkpoint.

        This method runs the evaluation script after checkpoint saving.
        Configured via environment variables or config.
        """
        import subprocess
        import os
        from ..distributed.fsdp_utils import is_main_process

        if not is_main_process():
            return

        # Check if evaluation is enabled
        # Priority: config file > environment variable
        if hasattr(self.config, 'evaluation') and hasattr(self.config.evaluation, 'run_after_save'):
            run_eval = self.config.evaluation.run_after_save
        else:
            run_eval = os.environ.get("RUN_EVAL_AFTER_SAVE", "false").lower() == "true"
        
        if not run_eval:
            logger.info("Post-save evaluation disabled. Set evaluation.run_after_save=true in config or RUN_EVAL_AFTER_SAVE=true to enable.")
            return

        # Clear GPU cache before evaluation to avoid OOM
        logger.info("Clearing GPU cache before evaluation...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Wait a bit for memory to be fully released
            time.sleep(5)
            logger.info("GPU cache cleared. Waiting 5 seconds for memory release...")

        checkpoint_path = checkpoint_path or self.config.output.save_path

        # Get evaluation config from environment variables
        eval_config = {
            "task_name": os.environ.get("EVAL_TASK_NAME", "adjust_bottle"),
            "task_config": os.environ.get("EVAL_TASK_CONFIG", "hd_clean"),
            "idm_path": os.environ.get("EVAL_IDM", "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/vidar_ckpts/idm.pt"),
            "prefix": os.environ.get("EVAL_PREFIX", f"step_{self.global_step}"),
            "num_new_frames": os.environ.get("EVAL_NUM_NEW_FRAMES", "16"),
            "num_sampling_step": os.environ.get("EVAL_NUM_SAMPLING_STEP", "10"),
            "cfg": os.environ.get("EVAL_CFG", "3.0"),
        }

        logger.info(f"========================================")
        logger.info(f"Running evaluation on checkpoint: {checkpoint_path}")
        logger.info(f"Task: {eval_config['task_name']}")
        logger.info(f"Config: {eval_config['task_config']}")
        logger.info(f"========================================")

        # Build evaluation command
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        eval_script = os.path.join(script_dir, "run_eval_ddp_causal.sh")

        if not os.path.exists(eval_script):
            logger.warning(f"Evaluation script not found: {eval_script}")
            return

        # Construct command
        cmd = [
            "bash", eval_script,
            eval_config["task_config"],
            checkpoint_path,
            eval_config["idm_path"],
            eval_config["prefix"],
            eval_config["num_new_frames"],
            eval_config["num_sampling_step"],
            eval_config["cfg"],
        ]

        # Set environment for evaluation
        env = os.environ.copy()
        env["TASK_NAME"] = eval_config["task_name"]

        try:
            logger.info(f"Evaluation command: TASK_NAME={eval_config['task_name']} {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                env=env,
                cwd=script_dir,
                capture_output=False,  # Show output in real-time
                text=True,
            )

            if result.returncode == 0:
                logger.info(f"Evaluation completed successfully!")
            else:
                logger.warning(f"Evaluation failed with return code: {result.returncode}")

        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")

        logger.info(f"========================================")
        logger.info(f"Resuming training...")
        logger.info(f"========================================")


def create_vidarc_trainer(config: VidarConfig) -> VidarCausalTrainer:
    """Factory function to create Vidarc causal trainer."""
    return VidarCausalTrainer(config)
