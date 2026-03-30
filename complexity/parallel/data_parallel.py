"""
Data Parallel (DP) and Fully Sharded Data Parallel (FSDP).

Data Parallelism replicates the model on each GPU and splits the data:
- Each GPU processes different batches
- Gradients are synchronized across GPUs

FSDP (Fully Sharded Data Parallel) also shards model parameters:
- Reduces memory per GPU
- Enables training larger models
- ZeRO optimization stages

References:
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- ZeRO: https://arxiv.org/abs/1910.02054
"""

import os
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
)
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy
from typing import Any, Optional, Type, Set, Callable
from functools import partial
from enum import Enum


class ShardingMode(Enum):
    """FSDP sharding strategies."""
    FULL_SHARD = "full_shard"           # ZeRO-3: shard params, gradients, optimizer states
    SHARD_GRAD_OP = "shard_grad_op"     # ZeRO-2: shard gradients and optimizer states
    NO_SHARD = "no_shard"               # DDP: no sharding, just gradient sync
    HYBRID_SHARD = "hybrid_shard"       # Shard within node, replicate across nodes


class PrecisionMode(Enum):
    """Mixed precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


def get_sharding_strategy(mode: ShardingMode) -> ShardingStrategy:
    """Convert ShardingMode to FSDP ShardingStrategy."""
    mapping = {
        ShardingMode.FULL_SHARD: ShardingStrategy.FULL_SHARD,
        ShardingMode.SHARD_GRAD_OP: ShardingStrategy.SHARD_GRAD_OP,
        ShardingMode.NO_SHARD: ShardingStrategy.NO_SHARD,
        ShardingMode.HYBRID_SHARD: ShardingStrategy.HYBRID_SHARD,
    }
    return mapping[mode]


def get_mixed_precision(mode: PrecisionMode) -> Optional[MixedPrecision]:
    """Get FSDP MixedPrecision config."""
    if mode == PrecisionMode.FP32:
        return None

    dtype = torch.float16 if mode == PrecisionMode.FP16 else torch.bfloat16

    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def wrap_model_fsdp(
    model: nn.Module,
    sharding_mode: ShardingMode = ShardingMode.FULL_SHARD,
    precision: PrecisionMode = PrecisionMode.BF16,
    cpu_offload: bool = False,
    wrap_policy: Optional[Callable] = None,
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    min_num_params: int = 100_000,
    gradient_checkpointing: bool = False,
    process_group: Optional[Any] = None,
) -> nn.Module:
    """
    Wrap a model with FSDP v2 (composable fully_shard).

    Args:
        process_group: Optional process group for sharding. When using TP+DP,
            pass the DP process group so FSDP only syncs across DP replicas.

    AdamW is initialized with foreach=False so empty DTensor shards
    ([0, experts, dim] on ranks >= num_experts) don't crash _multi_tensor_adamw.
    """
    from ..models.block import TransformerBlock

    try:
        from torch.distributed._composable.fsdp import fully_shard

        if precision == PrecisionMode.BF16:
            model = model.to(torch.bfloat16)
        elif precision == PrecisionMode.FP16:
            model = model.to(torch.float16)

        if process_group is not None:
            # FSDP v2 with device_mesh for TP+DP
            from torch.distributed.device_mesh import init_device_mesh
            world_size = dist.get_world_size()
            dp_size = world_size // (world_size // dist.get_world_size(process_group))
            # Actually just count group size
            dp_size = dist.get_world_size(process_group)
            tp_size = world_size // dp_size
            mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
            dp_mesh = mesh["dp"]

            for layer in model.layers:
                fully_shard(layer, mesh=dp_mesh)
            fully_shard(model, mesh=dp_mesh)
        else:
            for layer in model.layers:
                fully_shard(layer)
            fully_shard(model)

        return model

    except ImportError:
        pass

    # Fallback: FSDP v1
    if wrap_policy is None:
        if transformer_layer_cls is not None:
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            )
        else:
            wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=min_num_params,
            )

    fsdp_config = {
        "sharding_strategy": get_sharding_strategy(sharding_mode),
        "mixed_precision": get_mixed_precision(precision),
        "auto_wrap_policy": wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }

    if cpu_offload:
        fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)

    if process_group is not None:
        fsdp_config["process_group"] = process_group

    return FSDP(model, **fsdp_config)


class DataParallelConfig:
    """Configuration for data parallelism."""

    def __init__(
        self,
        sharding_mode: ShardingMode = ShardingMode.FULL_SHARD,
        precision: PrecisionMode = PrecisionMode.BF16,
        cpu_offload: bool = False,
        gradient_accumulation_steps: int = 1,
        sync_module_states: bool = True,
    ):
        self.sharding_mode = sharding_mode
        self.precision = precision
        self.cpu_offload = cpu_offload
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.sync_module_states = sync_module_states


def simple_ddp(
    model: nn.Module,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False,
) -> nn.parallel.DistributedDataParallel:
    """
    Wrap model with simple DDP (no sharding).

    This is the simplest form of data parallelism:
    - Full model on each GPU
    - Gradient synchronization only

    Args:
        model: The model to wrap
        device_id: GPU device ID (auto-detected if None)
        find_unused_parameters: Set True for models with conditional paths
            (e.g. INL dynamics where mu/PiD params may not always participate)

    Returns:
        DDP-wrapped model
    """
    if device_id is None and torch.cuda.is_available():
        device_id = torch.cuda.current_device()

    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id] if device_id is not None else None,
        find_unused_parameters=find_unused_parameters,
    )


# =============================================================================
# Gradient Accumulation
# =============================================================================

class GradientAccumulator:
    """
    Helper for gradient accumulation.

    Gradient accumulation allows effective larger batch sizes
    by accumulating gradients over multiple forward passes.

    Usage:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for batch in dataloader:
            with accumulator.accumulate():
                loss = model(batch)
                loss.backward()

            if accumulator.should_sync():
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 1,
    ):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def accumulate(self):
        """Context manager for gradient accumulation."""
        class AccumulateContext:
            def __init__(ctx, outer):
                ctx.outer = outer

            def __enter__(ctx):
                ctx.outer.step_count += 1
                # Disable gradient sync for intermediate steps
                if hasattr(ctx.outer.model, 'no_sync'):
                    if not ctx.outer.should_sync():
                        return ctx.outer.model.no_sync()
                return ctx

            def __exit__(ctx, *args):
                pass

        return AccumulateContext(self)

    def should_sync(self) -> bool:
        """Check if gradients should be synchronized."""
        return self.step_count % self.accumulation_steps == 0

    def reset(self):
        """Reset step counter."""
        self.step_count = 0


# =============================================================================
# Utility Functions
# =============================================================================

def init_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> bool:
    """
    Initialize distributed training.

    Args:
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        init_method: Initialization method

    Returns:
        True if distributed is initialized
    """
    if dist.is_initialized():
        return True

    if not torch.cuda.is_available():
        backend = "gloo"

    rank = os.environ.get("RANK")
    if rank is None:
        return False  # single-GPU, skip distributed silently

    try:
        dist.init_process_group(backend=backend, init_method=init_method)
        if torch.cuda.is_available():
            local_rank = dist.get_rank() % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
        return True
    except Exception as e:
        print(f"Failed to initialize distributed: {e}")
        return False


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get rank of current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronization barrier across all processes."""
    if dist.is_initialized():
        dist.barrier()


def cleanup():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
