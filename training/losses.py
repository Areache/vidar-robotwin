"""Loss functions for Vidar/Vidarc training.

Loss functions from the Vidarc paper (arXiv:2512.17661):
- Eq. 1: Flow matching diffusion loss
- Eq. 6: Causal flow matching loss
- Eq. 7: Embodiment-aware loss
- Eq. 4: IDM loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def flow_matching_loss(
    model_output: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Flow matching diffusion loss (Eq. 1).

    L = ||v_θ(x_t, t, c) - (x_0 - x_1)||²

    Where x_t = t*x_1 + (1-t)*x_0

    Args:
        model_output: Predicted velocity v_θ(x_t, t, c)
        x0: Noise tensor
        x1: Clean data tensor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    target_velocity = x0 - x1
    loss = F.mse_loss(model_output, target_velocity, reduction=reduction)
    return loss


def causal_flow_matching_loss(
    model_output: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Causal flow matching loss (Eq. 6).

    Same as flow matching, but previous frames (x_prev) are noise-free
    and attended via KV cache. The loss computation is identical.

    L = ||v_θ(x_t, t, c, x_prev) - (x_0 - x_1)||²

    Args:
        model_output: Predicted velocity with causal attention
        x0: Noise tensor
        x1: Clean data tensor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    return flow_matching_loss(model_output, x0, x1, reduction)


def embodiment_aware_loss(
    model_output: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    mask: torch.Tensor,
    eta: float = 3.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Embodiment-aware loss (Eq. 7).

    L = ||(1 + η·U(x_1)) ⊙ (v_θ - (x_0 - x_1))||²

    Where U(x_1) is the learned mask from IDM highlighting robot arm regions.

    Args:
        model_output: Predicted velocity
        x0: Noise tensor
        x1: Clean data tensor
        mask: IDM mask tensor, same spatial dims as x1, values in [0, 1]
        eta: Reweighting strength (default 3.0 from paper)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    target_velocity = x0 - x1
    error = model_output - target_velocity

    # Ensure mask has same shape as error for broadcasting
    # mask: (B, 1, H, W) or (B, C, H, W)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and error.shape[1] > 1:
        mask = mask.expand_as(error)

    # Apply mask weighting: (1 + η * mask)
    weight = 1.0 + eta * mask

    # Weighted squared error
    weighted_error = weight * error
    loss = (weighted_error ** 2)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def idm_loss(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_mask: float = 3e-3,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    IDM loss with mask regularization (Eq. 4).

    L_action = l(â - a) + λ||m||_1

    Where l(.) is the Huber loss.

    Args:
        predicted_action: Predicted action tensor (B, action_dim)
        target_action: Ground truth action tensor (B, action_dim)
        mask: Optional mask tensor for regularization
        lambda_mask: Mask regularization weight (default 3e-3)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    # Huber loss for action prediction
    action_loss = F.smooth_l1_loss(predicted_action, target_action, reduction=reduction)

    # Optional mask regularization
    if mask is not None:
        mask_loss = lambda_mask * mask.abs().mean()
        return action_loss + mask_loss

    return action_loss


class VidarLoss(nn.Module):
    """Combined loss module for Vidar/Vidarc training."""

    def __init__(
        self,
        loss_type: str = "flow_matching",
        embodiment_aware: bool = False,
        eta: float = 3.0,
        cfg_prob: float = 0.1
    ):
        """
        Args:
            loss_type: 'flow_matching' or 'causal_flow_matching'
            embodiment_aware: Whether to use embodiment-aware loss
            eta: Embodiment-aware loss weight
            cfg_prob: Classifier-free guidance probability
        """
        super().__init__()
        self.loss_type = loss_type
        self.embodiment_aware = embodiment_aware
        self.eta = eta
        self.cfg_prob = cfg_prob

    def forward(
        self,
        model_output: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            model_output: Predicted velocity from model
            x0: Noise tensor
            x1: Clean data tensor
            timestep: Diffusion timestep (unused in loss, for logging)
            mask: Optional IDM mask for embodiment-aware loss

        Returns:
            Loss value
        """
        if self.embodiment_aware and mask is not None:
            return embodiment_aware_loss(
                model_output, x0, x1, mask, self.eta
            )
        else:
            return flow_matching_loss(model_output, x0, x1)


class IDMLoss(nn.Module):
    """Loss module for IDM training."""

    def __init__(self, lambda_mask: float = 3e-3):
        super().__init__()
        self.lambda_mask = lambda_mask

    def forward(
        self,
        predicted_action: torch.Tensor,
        target_action: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return idm_loss(
            predicted_action, target_action, mask, self.lambda_mask
        )


def add_noise(x1: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to clean data for flow matching.

    x_t = t * x_1 + (1 - t) * x_0

    Args:
        x1: Clean data tensor (B, C, T, H, W) or (B, C, H, W)
        timestep: Timestep tensor (B,) with values in [0, 1]

    Returns:
        Tuple of (noised data x_t, noise x_0)
    """
    x0 = torch.randn_like(x1)

    # Expand timestep for broadcasting
    while timestep.dim() < x1.dim():
        timestep = timestep.unsqueeze(-1)

    x_t = timestep * x1 + (1 - timestep) * x0
    return x_t, x0


def sample_timestep(
    batch_size: int,
    device: torch.device,
    min_t: float = 0.0,
    max_t: float = 1.0
) -> torch.Tensor:
    """
    Sample random timesteps for training.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        min_t: Minimum timestep value
        max_t: Maximum timestep value

    Returns:
        Timestep tensor (B,)
    """
    return torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
