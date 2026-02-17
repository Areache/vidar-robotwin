"""Training wrapper for WanModelCausal (Stage 2: Vidarc causal training).

Implements Self-Forcing training with:
- Causal attention with KV caching
- Autoregressive rollout during training
- Re-prefilling mechanism
- Embodiment-aware loss support

IMPORTANT: PYTHONPATH must include the vidar directory with wan modules.
Set via run_train_vidarc.sh or manually:
    export PYTHONPATH=/path/to/vidar:$PYTHONPATH
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn

from ..utils.timing import time_block

# Import from wan modules (same as causal_worker.py)
# Requires PYTHONPATH to include vidar directory
from wan.modules.model_causal import WanModelCausal, WanAttentionBlock
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.block_attention import (
    get_flex_causal_block_mask_for_prefill,
    get_flex_block_mask_chunk_prefill,
)

logger = logging.getLogger(__name__)


class WanModelCausalTrainingWrapper(nn.Module):
    """
    Training wrapper for WanModelCausal (Stage 2).

    Implements Self-Forcing training paradigm from the Vidarc paper:
    - Causal attention: previous frames are noise-free, attend via KV cache
    - Teacher forcing: use ground truth for KV cache during training
    - Autoregressive rollout with re-prefilling

    Key differences from Stage 1 (WanModel):
    - Uses causal attention instead of bidirectional
    - Supports KV caching for efficient generation
    - Training simulates inference with chunk-wise generation
    """

    def __init__(
        self,
        ckpt_dir: str,
        pt_dir: Optional[str] = None,
        freeze_t5: bool = True,
        freeze_vae: bool = True,
        gradient_checkpointing: bool = True,
        device: Optional[torch.device] = None,
        debug: bool = False,
    ):
        """
        Args:
            ckpt_dir: Path to Wan2.2-TI2V-5B checkpoint directory
            pt_dir: Path to Stage 1 fine-tuned weights (vidar.pt)
            freeze_t5: Whether to freeze T5 encoder
            freeze_vae: Whether to freeze VAE
            gradient_checkpointing: Whether to enable gradient checkpointing
            device: Device to load models on
        """
        super().__init__()

        self.ckpt_dir = ckpt_dir
        self.pt_dir = pt_dir
        self.freeze_t5 = freeze_t5
        self.freeze_vae = freeze_vae
        self.gradient_checkpointing = gradient_checkpointing
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

        # Load config
        self.config = self._load_config()

        # Initialize components
        self._init_t5()
        self._init_vae()
        self._init_dit()

        # Apply freezing
        self._apply_freezing()

    def _load_config(self) -> Dict[str, Any]:
        """Load model config."""
        try:
            from wan.configs.shared_config import wan_shared_config
            return wan_shared_config
        except ImportError:
            # Fallback config matching Wan2.2-TI2V-5B directory structure
            return {
                "text_len": 512,
                "t5_dtype": torch.bfloat16,
                "t5_checkpoint": "models_t5_umt5-xxl-enc-bf16.pth",
                "t5_tokenizer": "google/umt5-xxl",
                "vae_checkpoint": "Wan2.2_VAE.pth",
                "vae_stride": (4, 8, 8),
                "patch_size": (1, 2, 2),
                "param_dtype": torch.bfloat16,
                "num_train_timesteps": 1000,
            }

    def _init_t5(self):
        """Initialize T5 text encoder."""
        logger.info("Loading T5 encoder...")

        t5_checkpoint = self.config.get("t5_checkpoint", "t5")
        t5_tokenizer = self.config.get("t5_tokenizer", "tokenizer")

        if not os.path.isabs(t5_checkpoint):
            t5_checkpoint = os.path.join(self.ckpt_dir, t5_checkpoint)
        if not os.path.isabs(t5_tokenizer):
            t5_tokenizer = os.path.join(self.ckpt_dir, t5_tokenizer)

        self.t5 = T5EncoderModel(
            text_len=self.config.get("text_len", 512),
            dtype=self.config.get("t5_dtype", torch.bfloat16),
            device=torch.device("cpu"),
            checkpoint_path=t5_checkpoint,
            tokenizer_path=t5_tokenizer,
        )

    def _init_vae(self):
        """Initialize VAE."""
        logger.info("Loading VAE...")

        vae_checkpoint = self.config.get("vae_checkpoint", "vae")
        if not os.path.isabs(vae_checkpoint):
            vae_checkpoint = os.path.join(self.ckpt_dir, vae_checkpoint)

        self.vae = Wan2_2_VAE(
            vae_pth=vae_checkpoint,
            device=torch.device("cpu"),
        )
        self.vae_stride = self.config.get("vae_stride", (4, 8, 8))

    def _init_dit(self):
        """Initialize DiT (WanModelCausal)."""
        logger.info(f"Loading WanModelCausal from {self.ckpt_dir}")

        # Load causal model
        self.dit = WanModelCausal.from_pretrained(self.ckpt_dir)
        
        # Set debug flag for all components
        self.dit.debug = self.debug
        # Propagate debug to all blocks
        for block in self.dit.blocks:
            block.debug = self.debug
            block.self_attn.debug = self.debug

        # Load Stage 1 fine-tuned weights if provided
        if self.pt_dir is not None:
            logger.info(f"Loading Stage 1 weights from {self.pt_dir}")
            state_dict = torch.load(self.pt_dir, map_location="cpu")
            # Handle checkpoint format (may be wrapped in 'model' key)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.dit.load_state_dict(state_dict, strict=False)
            # Re-set debug after loading state dict (in case it was overwritten)
            self.dit.debug = self.debug
            for block in self.dit.blocks:
                block.debug = self.debug
                block.self_attn.debug = self.debug

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for DiT blocks."""
        logger.info("Enabling gradient checkpointing for DiT blocks")

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.dit.blocks:
            block._forward = block.forward
            block.forward = lambda *args, _block=block, **kwargs: \
                torch.utils.checkpoint.checkpoint(
                    create_custom_forward(_block._forward),
                    *args,
                    use_reentrant=False,
                    **kwargs
                )

    def _apply_freezing(self):
        """Freeze T5 and VAE if requested."""
        if self.freeze_t5:
            logger.info("Freezing T5 encoder")
            # T5EncoderModel may not be a standard nn.Module
            if hasattr(self.t5, 'eval'):
                self.t5.eval()
            if hasattr(self.t5, 'model') and hasattr(self.t5.model, 'eval'):
                self.t5.model.eval()
            # Try to freeze parameters
            if hasattr(self.t5, 'parameters'):
                for param in self.t5.parameters():
                    param.requires_grad = False
            elif hasattr(self.t5, 'model') and hasattr(self.t5.model, 'parameters'):
                for param in self.t5.model.parameters():
                    param.requires_grad = False

        if self.freeze_vae:
            logger.info("Freezing VAE")
            # VAE may not be a standard nn.Module
            if hasattr(self.vae, 'eval'):
                self.vae.eval()
            if hasattr(self.vae, 'model') and hasattr(self.vae.model, 'eval'):
                self.vae.model.eval()
            # Try to freeze parameters
            if hasattr(self.vae, 'parameters'):
                for param in self.vae.parameters():
                    param.requires_grad = False
            elif hasattr(self.vae, 'model') and hasattr(self.vae.model, 'parameters'):
                for param in self.vae.model.parameters():
                    param.requires_grad = False

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.dit = self.dit.to(device)
        if not self.freeze_t5:
            self.t5.model = self.t5.model.to(device)
            self.t5.device = device
        if not self.freeze_vae:
            self.vae.model = self.vae.model.to(device)
            self.vae.device = device
        return self

    def encode_text(self, text: List[str]) -> list:
        """Encode text using T5.
        
        Returns:
            List of context tensors (one per text), as returned by T5EncoderModel.
        """
        with torch.no_grad():
            context = self.t5(text, self.device)
        return context

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video using VAE."""
        with torch.no_grad():
            video_device = video.device
            video = video.to(self.vae.device)
            latent = self.vae.encode(video)
            latent = latent.to(video_device)
        return latent

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to video using VAE."""
        with torch.no_grad():
            latent_device = latent.device
            latent = latent.to(self.vae.device)
            video = self.vae.decode(latent)
            video = video.to(latent_device)
        return video

    def get_block_size(self, latent_shape: Tuple[int, ...]) -> int:
        """
        Get block size (tokens per frame) for KV caching.

        Args:
            latent_shape: Shape of latent tensor (B, C, T, H, W)

        Returns:
            Number of tokens per frame
        """
        _, C, T, H, W = latent_shape
        patch_size = self.dit.patch_size
        h_patches = H // patch_size[1]
        w_patches = W // patch_size[2]
        return h_patches * w_patches

    def forward_causal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        x_prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        prefill: bool = False,
        chunk_prefill: bool = False,
        block_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Causal forward pass through DiT.

        Args:
            x: Current noised latent (B, C, T_cur, H, W)
            t: Timestep tensor (B,) or (B, seq_len)
            context: Text embeddings (B, L, D)
            x_prev: Previous clean frames for KV cache (optional)
            attention_mask: Causal attention mask
            use_cache: Whether to use/update KV cache
            prefill: Whether this is a prefill pass
            chunk_prefill: Whether this is a chunk prefill pass
            block_idx: Block index for positional encoding

        Returns:
            Predicted velocity tensor
        """
        B = x.shape[0]
        block_size = self.get_block_size(x.shape)

        # Add debug print for block_idx passing to model
        # logger.info(f"BLOCK_IDX DEBUG [train forward_causal:277]: "
        #            f"Passing to model: block_idx={block_idx}, prefill={prefill}, "
        #            f"chunk_prefill={chunk_prefill}, use_cache={use_cache}, "
        #            f"block_size={block_size}, x.shape={x.shape}")

        # Prepare context as list (WanModelCausal expects list)
        context_list = [context[i] for i in range(B)]

        # Forward through DiT
        output = self.dit(
            x=x,
            t=t,
            context=context_list,
            attention_mask=attention_mask,
            block_size=block_size,
            cache=use_cache,
            prefill=prefill,
            chunk_prefill=chunk_prefill,
            block_idx=block_idx,
        )

        return output

    def forward_self_forcing(
        self,
        x_clean: torch.Tensor,
        t: torch.Tensor,
        context,  # Can be torch.Tensor or List[torch.Tensor]
        chunk_size: int = 16,
        same_t_across_chunks: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-Forcing forward pass for training (single-step mode).

        This is the original teacher-forcing approach:
        1. Split video into chunks
        2. For each chunk, use previous clean frames as context (teacher forcing)
        3. Predict velocity for current noised chunk at a single timestep

        For multi-step denoising trajectory training (like guandeh17/Self-Forcing),
        use forward_self_forcing_multistep() instead.

        Args:
            x_clean: Clean latent video (B, C, T, H, W)
            t: Timestep tensor (B,) - scaled to [0, num_train_timesteps]
            context: Text embeddings (B, L, D)
            chunk_size: Number of frames per chunk
            same_t_across_chunks: Use same timestep for all chunks

        Returns:
            Tuple of (predicted velocities, target velocities)
        """
        B, C, T, H, W = x_clean.shape
        block_size = self.get_block_size(x_clean.shape)
        num_chunks = (T + chunk_size - 1) // chunk_size

        # Clean KV cache
        with time_block("Cache Clean", device=self.device):
            self.dit.clean_cache()

        all_v_pred = []
        all_v_target = []

        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, T)

            # Get current chunk
            x_chunk = x_clean[:, :, start_t:end_t, :, :]

            # Sample noise and create noised version
            x0_chunk = torch.randn_like(x_chunk)

            # Use same or different timestep per chunk
            if same_t_across_chunks:
                t_chunk = t
            else:
                t_chunk = torch.rand(B, device=x_chunk.device)

            # Expand timestep for broadcasting
            t_expanded = t_chunk
            while t_expanded.dim() < x_chunk.dim():
                t_expanded = t_expanded.unsqueeze(-1)

            # Create noised chunk: x_t = (1-t) * x_clean + t * noise
            x_noised = (1 - t_expanded) * x_chunk + t_expanded * x0_chunk
            # Target velocity: v = noise - clean
            v_target = x0_chunk - x_chunk

            # Build causal attention mask
            if chunk_idx == 0:
                attention_mask = self._build_causal_mask(
                    x_noised.shape, block_size, self.device
                )
                prefill = True
                chunk_prefill = False
            else:
                attention_mask = self._build_chunk_causal_mask(
                    x_noised.shape, block_size,
                    kv_len=self.dit.kvcache_len(),
                    device=self.device
                )
                prefill = False
                chunk_prefill = False

            # Scale timestep for model input: model expects t in [0, 1000]
            t_model = t_chunk * 1000.0

            if chunk_idx == 0:
                # Step 1: Initialize cache with clean frames
                with time_block(f"Chunk {chunk_idx} (prefill)", device=self.device, indent=1):
                    t_clean = torch.zeros_like(t_model) if isinstance(t_model, torch.Tensor) else 0.0
                    self.forward_causal(
                        x=x_chunk,
                        t=t_clean,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=False,  # Not needed, prefill handles it
                        prefill=True,     # Initializes cache
                        chunk_prefill=False,
                        block_idx=None,   # Must be None for first chunk
                    )

                    # Step 2: Predict with noised frames using the cache
                    v_pred = self.forward_causal(
                        x=x_noised,
                        t=t_model,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=False,  # Use cache without updating
                        prefill=False,    # ← KEY FIX: Don't reinitialize!
                        chunk_prefill=False,
                        block_idx=None,   # ← Must be None for first chunk
                    )
            else:
                # Step 1: Update KV cache with clean frames (t=0)
                with time_block(f"Chunk {chunk_idx} (cache)", device=self.device, indent=1):
                    t_clean = torch.zeros_like(t_model) if isinstance(t_model, torch.Tensor) else 0.0
                    self.forward_causal(
                        x=x_chunk,
                        t=t_clean,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=True,
                        prefill=prefill,
                        chunk_prefill=chunk_prefill,
                        block_idx=None if prefill else chunk_idx,
                    )

                    # Step 2: Compute prediction using noised frames
                    v_pred = self.forward_causal(
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

        # Clean up cache after training step
        self.dit.clean_cache()

        # Concatenate all chunks
        v_pred_all = torch.cat(all_v_pred, dim=2)
        v_target_all = torch.cat(all_v_target, dim=2)

        # DEBUG: Check outputs before returning
        if torch.isnan(v_pred_all).any() or torch.isinf(v_pred_all).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[DEBUG NaN] forward_self_forcing: v_pred_all contains NaN/Inf: "
                        f"nan_count={torch.isnan(v_pred_all).sum().item()}, "
                        f"inf_count={torch.isinf(v_pred_all).sum().item()}, "
                        f"shape={v_pred_all.shape}, "
                        f"min={v_pred_all.min().item():.6f}, max={v_pred_all.max().item():.6f}, "
                        f"mean={v_pred_all.mean().item():.6f}")
        
        if torch.isnan(v_target_all).any() or torch.isinf(v_target_all).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[DEBUG NaN] forward_self_forcing: v_target_all contains NaN/Inf: "
                        f"nan_count={torch.isnan(v_target_all).sum().item()}, "
                        f"inf_count={torch.isinf(v_target_all).sum().item()}, "
                        f"shape={v_target_all.shape}, "
                        f"min={v_target_all.min().item():.6f}, max={v_target_all.max().item():.6f}, "
                        f"mean={v_target_all.mean().item():.6f}")

        return v_pred_all, v_target_all

    def forward_self_forcing_multistep(
        self,
        x_clean: torch.Tensor,
        context,  # Can be torch.Tensor or List[torch.Tensor]
        chunk_size: int = 16,
        num_inference_steps: int = 10,
        num_train_timesteps: int = 1000,
        shift: float = 8.0,
        context_noise: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Self-Forcing forward pass with multi-step denoising trajectory.

        Matches guandeh17/Self-Forcing training approach:
        1. For each temporal chunk, simulate full denoising trajectory
        2. Randomly select one step to backpropagate gradients (stochastic truncation)
        3. Update KV cache with denoised result for next chunk

        This bridges the train-test gap by training on the same denoising
        trajectory used during inference.

        Args:
            x_clean: Clean latent video (B, C, T, H, W) - used only for shape/target
            context: Text embeddings (B, L, D)
            chunk_size: Number of frames per chunk
            num_inference_steps: Number of denoising steps (default 10)
            num_train_timesteps: Total timesteps for schedule (default 1000)
            shift: Timestep shift for flow matching schedule (default 8.0)
            context_noise: Optional noise level for context frames (default 0.0)

        Returns:
            Tuple of (predicted x0, target x0, exit_step_index)
        """
        B, C, T, H, W = x_clean.shape
        block_size = self.get_block_size(x_clean.shape)
        num_chunks = (T + chunk_size - 1) // chunk_size
        device = x_clean.device
        dtype = x_clean.dtype

        # Build denoising timestep schedule (high to low)
        # Using shifted schedule like Self-Forcing: t' = shift * t / (1 + (shift-1) * t)
        timesteps = self._build_timestep_schedule(
            num_inference_steps, num_train_timesteps, shift, device
        )

        # Clean KV cache
        self.dit.clean_cache()

        all_x0_pred = []
        all_x0_target = []

        # Randomly select which denoising step to backprop through (same for all chunks)
        exit_step_idx = torch.randint(0, num_inference_steps, (1,), device=device).item()

        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, T)

            # Get current chunk (target)
            x_chunk_target = x_clean[:, :, start_t:end_t, :, :]

            # Start from pure noise
            x_noisy = torch.randn_like(x_chunk_target)

            # Build attention mask
            if chunk_idx == 0:
                attention_mask = self._build_causal_mask(
                    x_noisy.shape, block_size, self.device
                )
                prefill = True
                chunk_prefill = False
            else:
                attention_mask = self._build_chunk_causal_mask(
                    x_noisy.shape, block_size,
                    kv_len=self.dit.kvcache_len(),
                    device=self.device
                )
                prefill = False
                chunk_prefill = False

            # Initialize cache with clean frames for first chunk
            if chunk_idx == 0:
                t_clean = torch.zeros(B, device=device, dtype=dtype)
                self.forward_causal(
                    x=x_chunk_target,
                    t=t_clean,
                    context=context,
                    attention_mask=attention_mask,
                    use_cache=False,  # Not needed, prefill handles it
                    prefill=True,     # Initializes cache
                    chunk_prefill=False,
                    block_idx=None,   # Must be None for first chunk
                )

            # Denoising loop
            x0_pred = None
            for step_idx, current_t in enumerate(timesteps):
                t_tensor = torch.full((B,), current_t, device=device, dtype=dtype)

                is_exit_step = (step_idx == exit_step_idx)
                is_last_step = (step_idx == num_inference_steps - 1)

                if not is_exit_step:
                    # Forward pass WITHOUT gradients
                    with torch.no_grad():
                        v_pred = self.forward_causal(
                            x=x_noisy,
                            t=t_tensor,
                            context=context,
                            attention_mask=attention_mask,
                            use_cache=False,
                            prefill=False if chunk_idx == 0 else prefill,  # ← KEY FIX: Don't reinitialize for first chunk
                            chunk_prefill=chunk_prefill,
                            block_idx=None if (chunk_idx == 0 or prefill) else chunk_idx,  # ← Must be None for first chunk
                        )

                        # Convert velocity to x0 prediction
                        # For flow matching: v = noise - x0, so x0 = noise - v
                        # But we need to use the ODE formula for proper denoising
                        x0_pred = self._velocity_to_x0(x_noisy, v_pred, current_t, num_train_timesteps)

                        if not is_last_step:
                            # Add noise at next timestep level
                            next_t = timesteps[step_idx + 1]
                            x_noisy = self._add_noise_at_timestep(x0_pred, next_t, num_train_timesteps)
                else:
                    # Forward pass WITH gradients (exit step)
                    v_pred = self.forward_causal(
                        x=x_noisy,
                        t=t_tensor,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=False,
                        prefill=False if chunk_idx == 0 else prefill,  # ← KEY FIX: Don't reinitialize for first chunk
                        chunk_prefill=chunk_prefill,
                        block_idx=None if (chunk_idx == 0 or prefill) else chunk_idx,  # ← Must be None for first chunk
                    )

                    # Convert velocity to x0 prediction
                    x0_pred = self._velocity_to_x0(x_noisy, v_pred, current_t, num_train_timesteps)
                    break  # Exit after gradient step

            # Update KV cache with denoised result (optionally with context noise)
            with torch.no_grad():
                if context_noise > 0:
                    x_for_cache = self._add_noise_at_timestep(
                        x0_pred.detach(), context_noise, num_train_timesteps
                    )
                    t_cache = torch.full((B,), context_noise, device=device, dtype=dtype)
                else:
                    x_for_cache = x0_pred.detach()
                    t_cache = torch.zeros(B, device=device, dtype=dtype)

                self.forward_causal(
                    x=x_for_cache,
                    t=t_cache,
                    context=context,
                    attention_mask=attention_mask,
                    use_cache=True,
                    prefill=False if chunk_idx == 0 else prefill,  # ← KEY FIX: Don't reinitialize for first chunk
                    chunk_prefill=chunk_prefill,
                    block_idx=None if (chunk_idx == 0 or prefill) else chunk_idx,  # ← Must be None for first chunk
                )

            all_x0_pred.append(x0_pred)
            all_x0_target.append(x_chunk_target)

        # Clean up cache
        self.dit.clean_cache()

        # Concatenate all chunks
        x0_pred_all = torch.cat(all_x0_pred, dim=2)
        x0_target_all = torch.cat(all_x0_target, dim=2)

        # DEBUG: Check outputs before returning
        if torch.isnan(x0_pred_all).any() or torch.isinf(x0_pred_all).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[DEBUG NaN] forward_self_forcing_multistep: x0_pred_all contains NaN/Inf: "
                        f"nan_count={torch.isnan(x0_pred_all).sum().item()}, "
                        f"inf_count={torch.isinf(x0_pred_all).sum().item()}, "
                        f"shape={x0_pred_all.shape}, exit_step={exit_step_idx}")
        
        if torch.isnan(x0_target_all).any() or torch.isinf(x0_target_all).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[DEBUG NaN] forward_self_forcing_multistep: x0_target_all contains NaN/Inf: "
                        f"nan_count={torch.isnan(x0_target_all).sum().item()}, "
                        f"inf_count={torch.isinf(x0_target_all).sum().item()}, "
                        f"shape={x0_target_all.shape}")

        return x0_pred_all, x0_target_all, exit_step_idx

    def _build_timestep_schedule(
        self,
        num_inference_steps: int,
        num_train_timesteps: int,
        shift: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build denoising timestep schedule with shift.

        Uses shifted schedule: t' = shift * t / (1 + (shift-1) * t)
        This is the same schedule used in Self-Forcing and flow matching models.
        """
        # Linear spacing from 1 to 0
        t_normalized = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]

        # Apply shift
        if shift != 1.0:
            t_shifted = shift * t_normalized / (1 + (shift - 1) * t_normalized)
        else:
            t_shifted = t_normalized

        # Scale to [0, num_train_timesteps]
        timesteps = t_shifted * num_train_timesteps

        return timesteps

    def _velocity_to_x0(
        self,
        x_t: torch.Tensor,
        v_pred: torch.Tensor,
        t: float,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        """
        Convert velocity prediction to x0 prediction.

        For flow matching with x_t = (1-σ)*x0 + σ*noise:
        v = noise - x0
        x0 = x_t - σ*v  (where σ = t/num_train_timesteps)
        """
        sigma = t / num_train_timesteps
        x0_pred = x_t - sigma * v_pred
        return x0_pred

    def _add_noise_at_timestep(
        self,
        x0: torch.Tensor,
        t: float,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        """
        Add noise to x0 at a specific timestep.

        x_t = (1-σ)*x0 + σ*noise  (where σ = t/num_train_timesteps)
        """
        sigma = t / num_train_timesteps
        noise = torch.randn_like(x0)
        x_t = (1 - sigma) * x0 + sigma * noise
        return x_t

    def _build_causal_mask(
        self,
        shape: Tuple[int, ...],
        block_size: int,
        device: torch.device,
    ):
        """Build causal attention BlockMask for first chunk (prefill)."""
        B, C, T, H, W = shape
        # Use proper BlockMask for flex_attention compatibility
        return get_flex_causal_block_mask_for_prefill(
            num_denoise_block=T,
            block_size=block_size,
            device=device,
        )

    def _build_chunk_causal_mask(
        self,
        shape: Tuple[int, ...],
        block_size: int,
        kv_len: int,
        device: torch.device,
    ):
        """Build attention BlockMask for subsequent chunks with KV cache."""
        B, C, T, H, W = shape
        num_kv_blocks = kv_len // block_size
        # Use proper BlockMask for flex_attention compatibility
        return get_flex_block_mask_chunk_prefill(
            num_qblock=T,
            num_kvblock=num_kv_blocks,
            block_size=block_size,
            device=device,
            debug=self.debug,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Standard forward pass for training (non-self-forcing mode).

        Uses prefill mode with causal attention mask to ensure correct
        positional encoding for each frame. This matches how the eval
        pipeline processes conditional frames.

        Key: prefill=True ensures each frame gets correct RoPE position
        (frame 0 -> pos 0, frame 1 -> pos 1, etc.) instead of all frames
        getting position 0 when prefill=False and block_idx=None.
        """
        B, C, T, H, W = x.shape
        block_size = self.get_block_size(x.shape)

        # Add debug print for standard forward (non-self-forcing mode)
        # logger.info(f"BLOCK_IDX DEBUG [train forward:481]: "
        #            f"Standard forward mode, prefill=True, block_idx=None, "
        #            f"T={T}, block_size={block_size}, x.shape={x.shape}")

        # Build causal attention mask for all frames (prefill mode)
        attention_mask = self._build_causal_mask(x.shape, block_size, x.device)

        return self.forward_causal(
            x=x,
            t=t,
            context=context,
            attention_mask=attention_mask,
            use_cache=False,
            prefill=True,  # Correct positional encoding for each frame
        )

    # =========================================================================
    # Train-Eval Alignment: New functions to match eval's cache pop behavior
    # =========================================================================

    def _simulate_cache_pop(self, keep_sink_frames: int = 1) -> bool:
        """
        Simulate eval's cache pop: keep only first frame(s) as sink.

        This matches eval behavior in textimage2video_causal_server.py:541-545:
            if kvcache_len > T * block_size:
                self.model.pop_kvcache(T * block_size)

        Args:
            keep_sink_frames: Number of frames to keep (default: 1 = first frame only)

        Returns:
            True if cache was popped and chunk_prefill is needed
        """
        block_size = self.get_block_size_default()
        sink_tokens = keep_sink_frames * block_size
        current_len = self.dit.kvcache_len()

        if current_len is None or current_len <= sink_tokens:
            return False

        pop_amount = current_len - sink_tokens
        self.dit.pop_kvcache(pop_amount)
        return True

    def get_block_size_default(self) -> int:
        """Get default block size (H*W//4 for typical resolution)."""
        # Default for 736x640 resolution: 736//8 * 640//8 // 4 = 92 * 80 // 4 = 1840
        # But actual block_size from shape is H*W//4 where H,W are latent dims
        # For 736x640 -> latent 92x80 -> block_size = 92*80//4 = 1840? No wait...
        # Looking at get_block_size: H*W//4 where H,W are from x.shape (latent)
        # For 736x640 input -> VAE stride (8,8) -> latent 92x80
        # block_size = 92 * 80 // 4 = 1840? Let me check...
        # Actually from logs: block_size=460, which is 92*80//16 or similar
        # Let's use a computed default or store it
        return 460  # Default for 736x640 resolution

    def forward_self_forcing_aligned(
        self,
        x_clean: torch.Tensor,
        t: torch.Tensor,
        context,  # Can be torch.Tensor or List[torch.Tensor]
        chunk_size: int = 1,
        same_t_across_chunks: bool = True,
        # Alignment config
        simulate_cache_pop: bool = True,
        sink_frames: int = 1,
        frames_per_round: int = 4,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-Forcing forward pass ALIGNED with eval behavior.

        Key differences from forward_self_forcing:
        1. Simulates cache pop at round boundaries (keeps only sink frame)
        2. Uses chunk_prefill=True after cache pop (matches eval)
        3. Uses relative RoPE positioning via kv_num_block (matches eval)
        4. Attention mask matches eval's sink + causal pattern

        This addresses the train-eval mismatches:
        - Mismatch #1: chunk_prefill flag (now True after pop)
        - Mismatch #2: Cache behavior (now pops to sink)
        - Mismatch #3: RoPE positioning (now relative via chunk_prefill)
        - Mismatch #4: Attention mask (now sink + causal)
        - Mismatch #5: Max context (now bounded by cache pop)

        Args:
            x_clean: Clean latent video (B, C, T, H, W)
            t: Timestep tensor (B,) in [0, 1]
            context: Text embeddings
            chunk_size: Frames per chunk (default 1 for per-frame)
            same_t_across_chunks: Use same timestep for all chunks
            simulate_cache_pop: Enable cache pop simulation
            sink_frames: Frames to keep after pop (default 1)
            frames_per_round: Frames per round before pop
            debug: Enable debug prints

        Returns:
            Tuple of (predicted velocities, target velocities)
        """
        B, C, T, H, W = x_clean.shape
        block_size = self.get_block_size(x_clean.shape)
        num_chunks = (T + chunk_size - 1) // chunk_size

        if debug:
            print(f"[TRAIN_ALIGNED] Starting: T={T}, chunk_size={chunk_size}, "
                  f"num_chunks={num_chunks}, frames_per_round={frames_per_round}, "
                  f"block_size={block_size}")

        # Clean KV cache
        with time_block("Cache Clean", device=self.device):
            self.dit.clean_cache()

        all_v_pred = []
        all_v_target = []

        chunk_idx = 0
        while chunk_idx < num_chunks:
            # Check if this is a chunk_prefill point
            is_chunk_prefill = (simulate_cache_pop and
                               chunk_idx > 0 and
                               chunk_idx % frames_per_round == 0 and
                               (num_chunks - chunk_idx) >= frames_per_round)

            # Determine how many frames to process
            if is_chunk_prefill:
                num_frames_to_process = frames_per_round  # Process multiple frames
            else:
                num_frames_to_process = 1  # Process single frame

            # Calculate time range based on num_frames_to_process
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + num_frames_to_process * chunk_size, T)

            # Get current chunk (may contain multiple frames if chunk_prefill)
            x_chunk = x_clean[:, :, start_t:end_t, :, :]

            # Sample noise and create noised version
            x0_chunk = torch.randn_like(x_chunk)

            # Use same or different timestep per chunk
            if same_t_across_chunks:
                t_chunk = t
            else:
                t_chunk = torch.rand(B, device=x_chunk.device)

            # Expand timestep for broadcasting
            t_expanded = t_chunk
            while t_expanded.dim() < x_chunk.dim():
                t_expanded = t_expanded.unsqueeze(-1)

            # Create noised chunk: x_t = (1-t) * x_clean + t * noise
            x_noised = (1 - t_expanded) * x_chunk + t_expanded * x0_chunk
            # Target velocity: v = noise - clean
            v_target = x0_chunk - x_chunk

            # Get current cache state for debug
            kv_len = self.dit.kvcache_len() if self.dit.kvcache_len() else 0

            if debug:
                print(f"[TRAIN_ALIGNED] chunk_idx={chunk_idx}, kvcache_len={kv_len}, "
                      f"kv_frames={kv_len // block_size}, processing {num_frames_to_process} frame(s)")

            # Determine mode based on chunk position
            if chunk_idx == 0:
                # First chunk: prefill mode (same as original)
                attention_mask = self._build_causal_mask(
                    x_noised.shape, block_size, self.device
                )
                prefill = True
                chunk_prefill = False

                if debug:
                    print(f"[TRAIN_ALIGNED] Mode: PREFILL (first chunk)")

            elif is_chunk_prefill:
                # Round boundary: simulate cache pop, then chunk_prefill
                kv_before = self.dit.kvcache_len()
                with time_block("Cache Pop", device=self.device):
                    needs_pop = self._simulate_cache_pop(keep_sink_frames=sink_frames)
                kv_after = self.dit.kvcache_len()

                if debug:
                    print(f"[TRAIN_ALIGNED] CACHE POP: {kv_before} -> {kv_after} tokens "
                          f"({kv_before // block_size} -> {kv_after // block_size} frames)")

                # Build chunk_prefill attention mask for multiple frames
                # After pop, cache has sink_frames blocks
                # Process num_frames_to_process frames in this iteration
                num_new_frames = num_frames_to_process
                num_sink_blocks = sink_frames

                attention_mask = get_flex_block_mask_chunk_prefill(
                    num_qblock=num_new_frames,
                    num_kvblock=num_sink_blocks,
                    block_size=block_size,
                    device=self.device,
                )

                prefill = False
                chunk_prefill = True  # KEY: Enable chunk_prefill mode

                if debug:
                    print(f"[TRAIN_ALIGNED] Mode: CHUNK_PREFILL (after pop), "
                          f"num_qblock={num_new_frames}, num_kvblock={num_sink_blocks}")

            else:
                # Normal cache mode
                attention_mask = self._build_chunk_causal_mask(
                    x_noised.shape, block_size,
                    kv_len=self.dit.kvcache_len(),
                    device=self.device
                )
                prefill = False
                chunk_prefill = False

                if debug:
                    print(f"[TRAIN_ALIGNED] Mode: CACHE (block_idx={chunk_idx})")

            # Scale timestep for model input: model expects t in [0, 1000]
            t_model = t_chunk * 1000.0

            if debug:
                print(f"[TRAIN_ALIGNED] Forward: prefill={prefill}, chunk_prefill={chunk_prefill}, "
                      f"block_idx={None if prefill or chunk_prefill else chunk_idx}")

            if chunk_idx == 0:
                # Step 1: Initialize cache with clean frames
                t_clean = torch.zeros_like(t_model) if isinstance(t_model, torch.Tensor) else 0.0
                self.forward_causal(
                    x=x_chunk,
                    t=t_clean,
                    context=context,
                    attention_mask=attention_mask,
                    use_cache=False,  # Not needed, prefill handles it
                    prefill=True,     # Initializes cache
                    chunk_prefill=False,
                    block_idx=None,   # Must be None for first chunk
                )

                # Step 2: Predict with noised frames using the cache
                v_pred = self.forward_causal(
                    x=x_noised,
                    t=t_model,
                    context=context,
                    attention_mask=attention_mask,
                    use_cache=False,  # Use cache without updating
                    prefill=False,    # ← KEY FIX: Don't reinitialize!
                    chunk_prefill=False,
                    block_idx=0,      # Use block_idx=0 for first chunk when cache exists
                )
            else:
                # 对齐评估代码：先预测，再更新 cache
                chunk_name = f"Chunk {chunk_idx} ({'chunk_prefill' if chunk_prefill else 'cache'})"
                with time_block(chunk_name, device=self.device, indent=1):
                    # Step 1: 先进行预测（使用已存在的 cache，不更新）
                    # 这与评估代码中每个 timestep 的 forward 一致
                    v_pred = self.forward_causal(
                        x=x_noised,
                        t=t_model,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=False,  # 使用已存在的 cache 但不更新
                        prefill=prefill,
                        chunk_prefill=chunk_prefill,
                        block_idx=None if prefill or chunk_prefill else chunk_idx,
                    )
                    
                    # Step 2: 预测完成后，使用 clean frame 更新 cache
                    # 这与评估代码在所有 timestep 完成后更新 cache 的逻辑一致
                    t_clean = torch.zeros_like(t_model) if isinstance(t_model, torch.Tensor) else 0.0
                    kv_len_before = self.dit.kvcache_len()
                    self.forward_causal(
                        x=x_chunk,
                        t=t_clean,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=True,  # 更新 cache
                        prefill=prefill,
                        chunk_prefill=chunk_prefill,
                        block_idx=None if prefill or chunk_prefill else chunk_idx,
                    )
                    kv_len_after = self.dit.kvcache_len()
                    if debug:
                        print(f"[TRAIN_ALIGNED] Chunk {chunk_idx} cache updated: "
                              f"{kv_len_before} -> {kv_len_after} tokens "
                              f"({kv_len_before // block_size} -> {kv_len_after // block_size} frames), "
                              f"added {kv_len_after - kv_len_before} tokens")

            all_v_pred.append(v_pred)
            all_v_target.append(v_target)

            # Increment loop counter
            chunk_idx += num_frames_to_process

        # Clean up cache after training step
        self.dit.clean_cache()

        # Concatenate all chunks
        v_pred_all = torch.cat(all_v_pred, dim=2)
        v_target_all = torch.cat(all_v_target, dim=2)

        # DEBUG: Check outputs before returning
        if torch.isnan(v_pred_all).any() or torch.isinf(v_pred_all).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[DEBUG NaN] forward_self_forcing_aligned: v_pred_all contains NaN/Inf: "
                        f"nan_count={torch.isnan(v_pred_all).sum().item()}, "
                        f"inf_count={torch.isinf(v_pred_all).sum().item()}, "
                        f"shape={v_pred_all.shape}, "
                        f"min={v_pred_all.min().item():.6f}, max={v_pred_all.max().item():.6f}, "
                        f"mean={v_pred_all.mean().item():.6f}")
        
        if torch.isnan(v_target_all).any() or torch.isinf(v_target_all).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[DEBUG NaN] forward_self_forcing_aligned: v_target_all contains NaN/Inf: "
                        f"nan_count={torch.isnan(v_target_all).sum().item()}, "
                        f"inf_count={torch.isinf(v_target_all).sum().item()}, "
                        f"shape={v_target_all.shape}, "
                        f"min={v_target_all.min().item():.6f}, max={v_target_all.max().item():.6f}, "
                        f"mean={v_target_all.mean().item():.6f}")

        if debug:
            print(f"[TRAIN_ALIGNED] Done: v_pred_all.shape={v_pred_all.shape}")

        return v_pred_all, v_target_all

    def forward_self_forcing_multistep_aligned(
        self,
        x_clean: torch.Tensor,
        context,  # Can be torch.Tensor or List[torch.Tensor]
        chunk_size: int = 1,
        num_inference_steps: int = 10,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,  # Match eval's timestep_shift
        context_noise: float = 0.0,
        # Alignment config
        simulate_cache_pop: bool = True,
        sink_frames: int = 1,
        frames_per_round: int = 4,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Self-Forcing multi-step forward pass ALIGNED with eval behavior.

        Combines multi-step denoising trajectory with train-eval alignment:
        1. Simulates full denoising trajectory per chunk
        2. Randomly selects one step to backpropagate (stochastic truncation)
        3. Simulates cache pop at round boundaries
        4. Uses chunk_prefill=True after cache pop

        Args:
            x_clean: Clean latent video (B, C, T, H, W)
            context: Text embeddings
            chunk_size: Frames per chunk
            num_inference_steps: Denoising steps (default 10)
            num_train_timesteps: Total timesteps (default 1000)
            shift: Timestep shift (default 5.0 to match eval)
            context_noise: Noise level for context frames
            simulate_cache_pop: Enable cache pop simulation
            sink_frames: Frames to keep after pop
            frames_per_round: Frames per round before pop
            debug: Enable debug prints

        Returns:
            Tuple of (predicted x0, target x0, exit_step_index)
        """
        B, C, T, H, W = x_clean.shape
        block_size = self.get_block_size(x_clean.shape)
        num_chunks = (T + chunk_size - 1) // chunk_size
        device = x_clean.device
        dtype = x_clean.dtype

        if debug:
            print(f"[TRAIN_ALIGNED_MS] Starting: T={T}, num_chunks={num_chunks}, "
                  f"frames_per_round={frames_per_round}, shift={shift}")

        # Build denoising timestep schedule
        timesteps = self._build_timestep_schedule(
            num_inference_steps, num_train_timesteps, shift, device
        )

        # Clean KV cache
        self.dit.clean_cache()

        all_x0_pred = []
        all_x0_target = []

        # Randomly select which denoising step to backprop through
        exit_step_idx = torch.randint(0, num_inference_steps, (1,), device=device).item()

        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, T)

            x_chunk_target = x_clean[:, :, start_t:end_t, :, :]
            x_noisy = torch.randn_like(x_chunk_target)

            kv_len = self.dit.kvcache_len() if self.dit.kvcache_len() else 0

            if debug:
                print(f"[TRAIN_ALIGNED_MS] chunk_idx={chunk_idx}, kvcache_len={kv_len}")

            # Determine mode based on chunk position
            if chunk_idx == 0:
                attention_mask = self._build_causal_mask(
                    x_noisy.shape, block_size, self.device
                )
                prefill = True
                chunk_prefill = False

                if debug:
                    print(f"[TRAIN_ALIGNED_MS] Mode: PREFILL")

            elif simulate_cache_pop and chunk_idx == frames_per_round:
                # Cache pop at round boundary
                kv_before = self.dit.kvcache_len()
                self._simulate_cache_pop(keep_sink_frames=sink_frames)
                kv_after = self.dit.kvcache_len()

                if debug:
                    print(f"[TRAIN_ALIGNED_MS] CACHE POP: {kv_before} -> {kv_after}")

                num_new_frames = min(frames_per_round, T - chunk_idx)
                attention_mask = get_flex_block_mask_chunk_prefill(
                    num_qblock=num_new_frames,
                    num_kvblock=sink_frames,
                    block_size=block_size,
                    device=self.device,
                )
                prefill = False
                chunk_prefill = True

                if debug:
                    print(f"[TRAIN_ALIGNED_MS] Mode: CHUNK_PREFILL")

            else:
                attention_mask = self._build_chunk_causal_mask(
                    x_noisy.shape, block_size,
                    kv_len=self.dit.kvcache_len(),
                    device=self.device
                )
                prefill = False
                chunk_prefill = False

                if debug:
                    print(f"[TRAIN_ALIGNED_MS] Mode: CACHE (block_idx={chunk_idx})")

            # Initialize cache with clean frames for first chunk
            if chunk_idx == 0:
                t_clean = torch.zeros(B, device=device, dtype=dtype)
                self.forward_causal(
                    x=x_chunk_target,
                    t=t_clean,
                    context=context,
                    attention_mask=attention_mask,
                    use_cache=False,  # Not needed, prefill handles it
                    prefill=True,     # Initializes cache
                    chunk_prefill=False,
                    block_idx=None,   # Must be None for first chunk
                )

            # Denoising loop
            x0_pred = None
            for step_idx, current_t in enumerate(timesteps):
                t_tensor = torch.full((B,), current_t, device=device, dtype=dtype)

                is_exit_step = (step_idx == exit_step_idx)
                is_last_step = (step_idx == num_inference_steps - 1)

                if not is_exit_step:
                    with torch.no_grad():
                        v_pred = self.forward_causal(
                            x=x_noisy,
                            t=t_tensor,
                            context=context,
                            attention_mask=attention_mask,
                            use_cache=False,
                            prefill=False if chunk_idx == 0 else prefill,  # ← KEY FIX: Don't reinitialize for first chunk
                            chunk_prefill=chunk_prefill,
                            block_idx=None if (chunk_idx == 0 or prefill or chunk_prefill) else chunk_idx,  # ← Must be None for first chunk
                        )
                        x0_pred = self._velocity_to_x0(x_noisy, v_pred, current_t, num_train_timesteps)

                        if not is_last_step:
                            next_t = timesteps[step_idx + 1]
                            x_noisy = self._add_noise_at_timestep(x0_pred, next_t, num_train_timesteps)
                else:
                    v_pred = self.forward_causal(
                        x=x_noisy,
                        t=t_tensor,
                        context=context,
                        attention_mask=attention_mask,
                        use_cache=False,
                        prefill=False if chunk_idx == 0 else prefill,  # ← KEY FIX: Don't reinitialize for first chunk
                        chunk_prefill=chunk_prefill,
                        block_idx=None if (chunk_idx == 0 or prefill or chunk_prefill) else chunk_idx,  # ← Must be None for first chunk
                    )
                    x0_pred = self._velocity_to_x0(x_noisy, v_pred, current_t, num_train_timesteps)
                    break

            # Update KV cache with denoised result
            with torch.no_grad():
                if context_noise > 0:
                    x_for_cache = self._add_noise_at_timestep(
                        x0_pred.detach(), context_noise, num_train_timesteps
                    )
                    t_cache = torch.full((B,), context_noise, device=device, dtype=dtype)
                else:
                    x_for_cache = x0_pred.detach()
                    t_cache = torch.zeros(B, device=device, dtype=dtype)

                self.forward_causal(
                    x=x_for_cache,
                    t=t_cache,
                    context=context,
                    attention_mask=attention_mask,
                    use_cache=True,
                    prefill=False if chunk_idx == 0 else prefill,  # ← KEY FIX: Don't reinitialize for first chunk
                    chunk_prefill=chunk_prefill,
                    block_idx=None if (chunk_idx == 0 or prefill or chunk_prefill) else chunk_idx,  # ← Must be None for first chunk
                )

            all_x0_pred.append(x0_pred)
            all_x0_target.append(x_chunk_target)

        self.dit.clean_cache()

        x0_pred_all = torch.cat(all_x0_pred, dim=2)
        x0_target_all = torch.cat(all_x0_target, dim=2)

        if debug:
            print(f"[TRAIN_ALIGNED_MS] Done: exit_step={exit_step_idx}")

        return x0_pred_all, x0_target_all, exit_step_idx

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (DiT only)."""
        return [p for p in self.dit.parameters() if p.requires_grad]

    def get_dit_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get DiT state dict for saving."""
        return self.dit.state_dict()

    @staticmethod
    def get_transformer_layer_cls():
        """Get transformer layer classes for FSDP wrapping."""
        return {WanAttentionBlock}


def create_causal_model(
    ckpt_dir: str,
    pt_dir: Optional[str] = None,
    freeze: List[str] = ["t5", "vae"],
    gradient_checkpointing: bool = True,
    device: Optional[torch.device] = None,
) -> WanModelCausalTrainingWrapper:
    """
    Create WanModelCausal training wrapper.

    Args:
        ckpt_dir: Path to Wan2.2-TI2V-5B checkpoint
        pt_dir: Path to Stage 1 fine-tuned weights (vidar.pt)
        freeze: List of components to freeze ("t5", "vae")
        gradient_checkpointing: Enable gradient checkpointing
        device: Target device

    Returns:
        WanModelCausalTrainingWrapper instance
    """
    return WanModelCausalTrainingWrapper(
        ckpt_dir=ckpt_dir,
        pt_dir=pt_dir,
        freeze_t5="t5" in freeze,
        freeze_vae="vae" in freeze,
        gradient_checkpointing=gradient_checkpointing,
        device=device,
    )
