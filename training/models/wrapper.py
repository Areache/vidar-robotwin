"""Training wrapper for WanModel (Stage 1: Vidar fine-tuning).

IMPORTANT: PYTHONPATH must include the vidar directory with wan modules.
Set via run_train_vidarc.sh or manually:
    export PYTHONPATH=/path/to/vidar:$PYTHONPATH
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from functools import partial

import torch
import torch.nn as nn

# Import from wan modules (same as causal_worker.py)
# Requires PYTHONPATH to include vidar directory
from wan.modules.model import WanModel, WanAttentionBlock
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel

logger = logging.getLogger(__name__)


class WanModelTrainingWrapper(nn.Module):
    """
    Training wrapper for WanModel.

    Handles:
    - Loading pretrained weights (Wan2.2 base + optional pt_dir)
    - Freezing T5 and VAE
    - Training-friendly forward pass
    - Gradient checkpointing
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
            pt_dir: Optional path to fine-tuned weights (vidar.pt)
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
        """Load model config from checkpoint directory."""
        # Try to load config from shared_config
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

        # Get paths
        t5_checkpoint = self.config.get("t5_checkpoint", "t5")
        t5_tokenizer = self.config.get("t5_tokenizer", "tokenizer")

        if not os.path.isabs(t5_checkpoint):
            t5_checkpoint = os.path.join(self.ckpt_dir, t5_checkpoint)
        if not os.path.isabs(t5_tokenizer):
            t5_tokenizer = os.path.join(self.ckpt_dir, t5_tokenizer)

        self.t5 = T5EncoderModel(
            text_len=self.config.get("text_len", 512),
            dtype=self.config.get("t5_dtype", torch.bfloat16),
            device=torch.device("cpu"),  # Load on CPU first
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
            device=torch.device("cpu"),  # Load on CPU first
        )
        self.vae_stride = self.config.get("vae_stride", (4, 8, 8))

    def _init_dit(self):
        """Initialize DiT (WanModel)."""
        logger.info(f"Loading WanModel from {self.ckpt_dir}")

        self.dit = WanModel.from_pretrained(self.ckpt_dir)

        # Load fine-tuned weights if provided
        if self.pt_dir is not None:
            logger.info(f"Loading fine-tuned weights from {self.pt_dir}")
            state_dict = torch.load(self.pt_dir, map_location="cpu")
            self.dit.load_state_dict(state_dict, strict=False)

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for DiT blocks."""
        logger.info("Enabling gradient checkpointing for DiT blocks")

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
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
            self.t5.eval()
            for param in self.t5.parameters():
                param.requires_grad = False

        if self.freeze_vae:
            logger.info("Freezing VAE")
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.dit = self.dit.to(device)
        # Keep T5 and VAE handling flexible
        if not self.freeze_t5:
            self.t5 = self.t5.to(device)
        if not self.freeze_vae:
            self.vae = self.vae.to(device)
        return self

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode text using T5.

        Args:
            text: List of text strings

        Returns:
            Text embeddings tensor (B, L, D)
        """
        with torch.no_grad():
            # T5 handles its own device placement
            context = self.t5(text)
        return context

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video using VAE.

        Args:
            video: Video tensor (B, C, T, H, W) in range [-1, 1]

        Returns:
            Latent tensor (B, C_latent, T', H', W')
        """
        with torch.no_grad():
            # Move to VAE device temporarily
            video_device = video.device
            video = video.to(self.vae.device)
            latent = self.vae.encode(video)
            latent = latent.to(video_device)
        return latent

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video using VAE.

        Args:
            latent: Latent tensor

        Returns:
            Video tensor (B, C, T, H, W)
        """
        with torch.no_grad():
            latent_device = latent.device
            latent = latent.to(self.vae.device)
            video = self.vae.decode(latent)
            video = video.to(latent_device)
        return video

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through DiT.

        Args:
            x: Noised latent tensor (B, C, T, H, W)
            t: Timestep tensor (B,)
            context: Text embeddings (B, L, D)
            seq_len: Sequence length for positional encoding

        Returns:
            Predicted velocity tensor
        """
        # Convert to list format expected by WanModel
        B = x.shape[0]
        x_list = [x[i] for i in range(B)]
        context_list = [context[i] for i in range(B)]

        if seq_len is None:
            # Compute seq_len from input shape
            _, C, T, H, W = x.shape
            patch_size = self.dit.patch_size
            seq_len = (T // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])

        # Forward through DiT
        output_list = self.dit(
            x=x_list,
            t=t,
            context=context_list,
            seq_len=seq_len,
        )

        # Stack back to batch tensor
        output = torch.stack(output_list, dim=0)
        return output

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        context_null: torch.Tensor,
        cfg_scale: float = 7.5,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.

        Args:
            x: Noised latent tensor
            t: Timestep tensor
            context: Conditional text embeddings
            context_null: Unconditional (empty) text embeddings
            cfg_scale: Classifier-free guidance scale
            seq_len: Sequence length

        Returns:
            CFG-combined predicted velocity
        """
        # Conditional forward
        output_cond = self.forward(x, t, context, seq_len)

        # Unconditional forward
        output_uncond = self.forward(x, t, context_null, seq_len)

        # CFG combination
        output = output_uncond + cfg_scale * (output_cond - output_uncond)
        return output

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


def create_model(
    ckpt_dir: str,
    pt_dir: Optional[str] = None,
    freeze: List[str] = ["t5", "vae"],
    gradient_checkpointing: bool = True,
    device: Optional[torch.device] = None,
) -> WanModelTrainingWrapper:
    """
    Create WanModel training wrapper.

    Args:
        ckpt_dir: Path to Wan2.2-TI2V-5B checkpoint
        pt_dir: Optional path to fine-tuned weights
        freeze: List of components to freeze ("t5", "vae")
        gradient_checkpointing: Enable gradient checkpointing
        device: Target device

    Returns:
        WanModelTrainingWrapper instance
    """
    return WanModelTrainingWrapper(
        ckpt_dir=ckpt_dir,
        pt_dir=pt_dir,
        freeze_t5="t5" in freeze,
        freeze_vae="vae" in freeze,
        gradient_checkpointing=gradient_checkpointing,
        device=device,
    )
