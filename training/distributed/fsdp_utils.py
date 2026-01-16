"""FSDP utilities for distributed training."""

import os
import functools
from typing import Optional, Set, Type

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
)


def init_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is main."""
    return get_rank() == 0


def get_mixed_precision_policy(dtype: str = "bf16") -> MixedPrecision:
    """
    Get mixed precision policy for FSDP.

    Args:
        dtype: 'bf16', 'fp16', or 'fp32'

    Returns:
        MixedPrecision policy
    """
    if dtype == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif dtype == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        return MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )


def get_sharding_strategy(strategy: str) -> ShardingStrategy:
    """
    Get FSDP sharding strategy.

    Args:
        strategy: 'FULL_SHARD', 'SHARD_GRAD_OP', 'HYBRID_SHARD', or 'NO_SHARD'

    Returns:
        ShardingStrategy enum value
    """
    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    return strategies.get(strategy, ShardingStrategy.FULL_SHARD)


def get_auto_wrap_policy(
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    min_num_params: int = 1e8
):
    """
    Get auto-wrap policy for FSDP.

    Args:
        transformer_layer_cls: Set of transformer layer classes to wrap
        min_num_params: Minimum number of parameters for size-based wrapping

    Returns:
        Auto-wrap policy function
    """
    if transformer_layer_cls:
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )
    else:
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(min_num_params),
        )


def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: str = "bf16",
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    use_orig_params: bool = True,
) -> FSDP:
    """
    Wrap model with FSDP.

    Args:
        model: Model to wrap
        sharding_strategy: FSDP sharding strategy
        mixed_precision: Mixed precision dtype
        transformer_layer_cls: Transformer layer classes for auto-wrapping
        cpu_offload: Whether to offload to CPU
        activation_checkpointing: Whether to use activation checkpointing
        use_orig_params: Use original params (required for some optimizers)

    Returns:
        FSDP-wrapped model
    """
    # Get policies
    mp_policy = get_mixed_precision_policy(mixed_precision)
    shard_strategy = get_sharding_strategy(sharding_strategy)
    auto_wrap_policy = get_auto_wrap_policy(transformer_layer_cls)

    # CPU offload config
    cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None

    # Apply activation checkpointing if requested
    if activation_checkpointing and transformer_layer_cls:
        apply_activation_checkpointing(model, transformer_layer_cls)

    # Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=shard_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        use_orig_params=use_orig_params,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

    return model


def apply_activation_checkpointing(
    model: nn.Module,
    check_fn_or_cls: Optional[Set[Type[nn.Module]]] = None
):
    """
    Apply activation checkpointing to transformer layers.

    Args:
        model: Model to apply checkpointing to
        check_fn_or_cls: Set of layer classes to checkpoint
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing as _apply_ac,
    )

    if check_fn_or_cls is None:
        return

    def check_fn(module):
        return isinstance(module, tuple(check_fn_or_cls))

    _apply_ac(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn,
    )


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int = 0
):
    """
    Save FSDP checkpoint.

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        path: Save path
        rank: Rank to save from
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    # Full state dict config
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        full_state_dict_config,
    ):
        state_dict = model.state_dict()

        if rank == 0 or get_rank() == 0:
            torch.save(state_dict, path)


def load_fsdp_checkpoint(
    model: FSDP,
    path: str,
    strict: bool = False
):
    """
    Load checkpoint into FSDP model.

    Args:
        model: FSDP-wrapped model
        path: Checkpoint path
        strict: Whether to strictly enforce state dict matching
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        full_state_dict_config,
    ):
        if is_main_process():
            state_dict = torch.load(path, map_location="cpu")
        else:
            state_dict = None

        # Broadcast state dict to all ranks
        if dist.is_initialized():
            state_dict = broadcast_object(state_dict)

        model.load_state_dict(state_dict, strict=strict)


def broadcast_object(obj, src: int = 0):
    """Broadcast object from src rank to all ranks."""
    if not dist.is_initialized():
        return obj

    object_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor and compute mean."""
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def sync_gradients(model: nn.Module):
    """Synchronize gradients across all processes."""
    if not dist.is_initialized():
        return

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= get_world_size()
