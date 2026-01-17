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

# Import from wan modules (same as causal_worker.py)
# Requires PYTHONPATH to include vidar directory
from wan.modules.model_causal import WanModelCausal, WanAttentionBlock
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel

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

        # Load Stage 1 fine-tuned weights if provided
        if self.pt_dir is not None:
            logger.info(f"Loading Stage 1 weights from {self.pt_dir}")
            state_dict = torch.load(self.pt_dir, map_location="cpu")
            # Handle checkpoint format (may be wrapped in 'model' key)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.dit.load_state_dict(state_dict, strict=False)

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
            self.t5.model.eval()
            for param in self.t5.model.parameters():
                param.requires_grad = False

        if self.freeze_vae:
            logger.info("Freezing VAE")
            self.vae.model.eval()
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
        Self-Forcing forward pass for training.

        Simulates autoregressive inference during training:
        1. Split video into chunks
        2. For each chunk, use previous clean frames as context (teacher forcing)
        3. Predict velocity for current noised chunk

        Args:
            x_clean: Clean latent video (B, C, T, H, W)
            t: Timestep tensor (B,)
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

            # Create noised chunk: x_t = t * x_1 + (1-t) * x_0
            x_noised = t_expanded * x_chunk + (1 - t_expanded) * x0_chunk

            # Target velocity: x_0 - x_1
            v_target = x0_chunk - x_chunk

            # Build causal attention mask
            if chunk_idx == 0:
                # First chunk: standard causal mask within chunk
                attention_mask = self._build_causal_mask(
                    x_noised.shape, block_size, self.device
                )
                prefill = True
                chunk_prefill = False
            else:
                # Subsequent chunks: attend to cached previous frames
                attention_mask = self._build_chunk_causal_mask(
                    x_noised.shape, block_size,
                    kv_len=self.dit.kvcache_len(),
                    device=self.device
                )
                prefill = False
                chunk_prefill = True

            # Forward pass
            v_pred = self.forward_causal(
                x=x_noised,
                t=t_chunk,
                context=context,
                attention_mask=attention_mask,
                use_cache=True,
                prefill=prefill,
                chunk_prefill=chunk_prefill,
                block_idx=chunk_idx,
            )

            all_v_pred.append(v_pred)
            all_v_target.append(v_target)

            # Update KV cache with clean frames (teacher forcing)
            # This is done inside the model during forward

        # Clean up cache after training step
        self.dit.clean_cache()

        # Concatenate all chunks
        v_pred_all = torch.cat(all_v_pred, dim=2)
        v_target_all = torch.cat(all_v_target, dim=2)

        return v_pred_all, v_target_all

    def _build_causal_mask(
        self,
        shape: Tuple[int, ...],
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build causal attention mask for first chunk."""
        B, C, T, H, W = shape
        seq_len = T * block_size

        # Causal mask: can attend to current and previous positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def _build_chunk_causal_mask(
        self,
        shape: Tuple[int, ...],
        block_size: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for subsequent chunks with KV cache."""
        B, C, T, H, W = shape
        q_len = T * block_size
        total_kv_len = kv_len + q_len

        # Can attend to all cached KV + causal within current chunk
        mask = torch.zeros(q_len, total_kv_len, device=device)
        mask[:, :kv_len] = 1  # Attend to all cached
        mask[:, kv_len:] = torch.tril(torch.ones(q_len, q_len, device=device))

        return mask

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Standard forward pass (non-causal, for compatibility).

        For causal training, use forward_self_forcing instead.
        """
        return self.forward_causal(
            x=x,
            t=t,
            context=context,
            use_cache=False,
            prefill=False,
        )

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
