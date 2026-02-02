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
    #!!! 
    # x_t = timestep * x1 + (1 - timestep) * x0
    x_t = (1 - timestep) * x1 + timestep * x0
    return x_t, x0


def sample_timestep(
    batch_size: int,
    device: torch.device,
    min_t: float = 0.0,
    max_t: float = 1.0,
    num_train_timesteps: int = 1000
) -> torch.Tensor:
    """
    Sample random timesteps for training.

    IMPORTANT: Timesteps are scaled to [0, num_train_timesteps] to match inference.
    This ensures the model sees the same timestep embedding space during training
    and inference.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        min_t: Minimum timestep value (in [0,1] normalized range)
        max_t: Maximum timestep value (in [0,1] normalized range)
        num_train_timesteps: Number of training timesteps (default 1000)
                            Timesteps will be scaled to [0, num_train_timesteps]

    Returns:
        Timestep tensor (B,) scaled to [0, num_train_timesteps]
    """
    # Sample in [0, 1] normalized range
    t_normalized = torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
    # Scale to [0, num_train_timesteps] to match inference
    t_scaled = t_normalized * num_train_timesteps
    return t_scaled


# =============================================================================
# Few-Step Diffusion & Stochastic Gradient Truncation Utilities
# =============================================================================

def sample_timestep_importance(
    batch_size: int,
    device: torch.device,
    low_t: float = 0.3,
    high_t: float = 0.7,
    importance_weight: float = 3.0,
    num_train_timesteps: int = 1000
) -> torch.Tensor:
    """
    Sample timesteps with importance weighting.

    Critical timesteps (t=0.3-0.7) where structure emerges get higher
    sampling probability. This helps address the timestep imbalance issue
    in stochastic gradient truncation.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        low_t: Lower bound of high importance region (default 0.3)
        high_t: Upper bound of high importance region (default 0.7)
        importance_weight: Weight multiplier for important region (default 3.0)
        num_train_timesteps: Scale for timestep embeddings

    Returns:
        Timestep tensor (B,) scaled to [0, num_train_timesteps]
    """
    # Compute sampling probabilities
    # P(important region) = importance_weight / (importance_weight + 2 * (1 - (high_t - low_t)))
    important_range = high_t - low_t
    unimportant_range = 1.0 - important_range

    # Normalized probability of sampling from important region
    p_important = (importance_weight * important_range) / (
        importance_weight * important_range + unimportant_range
    )

    # Sample region indicator
    region = torch.rand(batch_size, device=device)
    is_important = region < p_important

    # Sample within each region
    t_normalized = torch.zeros(batch_size, device=device)

    # Important region: sample from [low_t, high_t]
    n_important = is_important.sum().item()
    if n_important > 0:
        t_normalized[is_important] = torch.rand(n_important, device=device) * important_range + low_t

    # Unimportant region: sample from [0, low_t] ∪ [high_t, 1]
    n_unimportant = batch_size - n_important
    if n_unimportant > 0:
        # Sample uniformly from [0, unimportant_range] and map to actual regions
        u = torch.rand(n_unimportant, device=device) * unimportant_range
        # Map to [0, low_t] or [high_t, 1]
        t_normalized[~is_important] = torch.where(
            u < low_t,
            u,  # [0, low_t]
            u + important_range  # [high_t, 1]
        )

    # Scale to [0, num_train_timesteps]
    return t_normalized * num_train_timesteps


def sample_timestep_stratified(
    batch_size: int,
    device: torch.device,
    num_strata: int = 10,
    num_train_timesteps: int = 1000
) -> torch.Tensor:
    """
    Sample timesteps using stratified sampling.

    Divides [0, 1] into num_strata bins and samples one timestep from each.
    This ensures coverage across all timesteps and reduces variance in
    gradient estimates.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        num_strata: Number of strata/bins (default 10 for few-step diffusion)
        num_train_timesteps: Scale for timestep embeddings

    Returns:
        Timestep tensor (B,) scaled to [0, num_train_timesteps]
    """
    # Determine which stratum each sample belongs to
    strata_width = 1.0 / num_strata
    strata_idx = torch.randint(0, num_strata, (batch_size,), device=device)

    # Sample uniformly within each stratum
    u = torch.rand(batch_size, device=device)
    t_normalized = (strata_idx.float() + u) * strata_width

    # Clamp to [0, 1] just in case
    t_normalized = torch.clamp(t_normalized, 0.0, 1.0)

    # Scale to [0, num_train_timesteps]
    return t_normalized * num_train_timesteps


def sample_timestep_for_truncation(
    batch_size: int,
    device: torch.device,
    strategy: str = "uniform",
    num_train_timesteps: int = 1000,
    **kwargs
) -> torch.Tensor:
    """
    Sample timesteps for stochastic gradient truncation.

    This is the main entry point for timestep sampling with gradient truncation.
    Only one timestep is backpropagated per batch, significantly reducing
    computation while maintaining training effectiveness.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        strategy: Sampling strategy ("uniform", "importance", "stratified")
        num_train_timesteps: Scale for timestep embeddings
        **kwargs: Additional arguments for specific strategies

    Returns:
        Timestep tensor (B,) scaled to [0, num_train_timesteps]
    """
    if strategy == "uniform":
        return sample_timestep(
            batch_size, device,
            num_train_timesteps=num_train_timesteps
        )
    elif strategy == "importance":
        return sample_timestep_importance(
            batch_size, device,
            low_t=kwargs.get("importance_low_t", 0.3),
            high_t=kwargs.get("importance_high_t", 0.7),
            importance_weight=kwargs.get("importance_weight", 3.0),
            num_train_timesteps=num_train_timesteps
        )
    elif strategy == "stratified":
        return sample_timestep_stratified(
            batch_size, device,
            num_strata=kwargs.get("num_strata", 10),
            num_train_timesteps=num_train_timesteps
        )
    else:
        raise ValueError(f"Unknown timestep sampling strategy: {strategy}")


def get_few_step_timesteps(
    num_inference_steps: int = 10,
    num_train_timesteps: int = 1000,
    device: torch.device = None
) -> torch.Tensor:
    """
    Get discrete timesteps for few-step diffusion inference.

    Returns evenly spaced timesteps for inference with few denoising steps.

    Args:
        num_inference_steps: Number of denoising steps (default 10)
        num_train_timesteps: Total training timesteps (default 1000)
        device: Device to create tensor on

    Returns:
        Timestep tensor (num_inference_steps,) with discrete steps
    """
    # Linear spacing from num_train_timesteps to 0
    step_ratio = num_train_timesteps // num_inference_steps
    timesteps = torch.arange(
        num_train_timesteps - 1,
        -1,
        -step_ratio,
        device=device
    )[:num_inference_steps]

    return timesteps
