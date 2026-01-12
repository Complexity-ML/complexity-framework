"""
Tensor Parallel (TP) - Split model across GPUs.

Tensor Parallelism splits individual layers across multiple GPUs:
- Linear layers are split column-wise or row-wise
- Each GPU holds a shard of the weights
- Communication happens during forward/backward

This is essential for large models that don't fit on a single GPU.

References:
- Megatron-LM: https://arxiv.org/abs/1909.08053
- Tensor Parallelism: https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Any
from functools import wraps


def get_tensor_parallel_world_size() -> int:
    """Get the tensor parallel world size."""
    if not dist.is_initialized():
        return 1
    # Use environment variable or default group
    return dist.get_world_size()


def get_tensor_parallel_rank() -> int:
    """Get the tensor parallel rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_tensor_parallel_group() -> Optional[Any]:
    """Get the tensor parallel process group."""
    if not dist.is_initialized():
        return None
    return dist.group.WORLD


# =============================================================================
# Communication Primitives
# =============================================================================

def _reduce_scatter(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reduce-scatter operation across tensor parallel group."""
    if not dist.is_initialized():
        return tensor

    world_size = get_tensor_parallel_world_size()
    if world_size == 1:
        return tensor

    # Split tensor along dim
    chunks = tensor.chunk(world_size, dim=dim)
    output = torch.zeros_like(chunks[0])

    dist.reduce_scatter(output, list(chunks), group=get_tensor_parallel_group())
    return output


def _all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather operation across tensor parallel group."""
    if not dist.is_initialized():
        return tensor

    world_size = get_tensor_parallel_world_size()
    if world_size == 1:
        return tensor

    # Gather from all ranks
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=get_tensor_parallel_group())

    return torch.cat(gathered, dim=dim)


def _all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce operation across tensor parallel group."""
    if not dist.is_initialized():
        return tensor

    world_size = get_tensor_parallel_world_size()
    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, group=get_tensor_parallel_group())
    return tensor


class _AllReduceFunc(torch.autograd.Function):
    """Autograd function for all-reduce in forward, identity in backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return _all_reduce(tensor.clone())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _ReduceScatterFunc(torch.autograd.Function):
    """Autograd function for reduce-scatter in forward, all-gather in backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return _reduce_scatter(tensor, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return _all_gather(grad_output, ctx.dim), None


class _AllGatherFunc(torch.autograd.Function):
    """Autograd function for all-gather in forward, reduce-scatter in backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return _all_gather(tensor, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return _reduce_scatter(grad_output, ctx.dim), None


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce with gradient support."""
    return _AllReduceFunc.apply(tensor)


def reduce_scatter(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reduce-scatter with gradient support."""
    return _ReduceScatterFunc.apply(tensor, dim)


def all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather with gradient support."""
    return _AllGatherFunc.apply(tensor, dim)


# =============================================================================
# Tensor Parallel Layers
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The weight matrix is split along the output dimension (columns).
    Each GPU holds W[:, rank * cols : (rank+1) * cols]

    Y = XW where W is split column-wise

    Args:
        in_features: Input dimension
        out_features: Total output dimension (will be split across GPUs)
        bias: Whether to use bias
        gather_output: If True, all-gather the output. If False, keep split.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        world_size = get_tensor_parallel_world_size()
        assert out_features % world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"

        self.out_features_per_partition = out_features // world_size

        # Each GPU has a shard of the weight matrix
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul: [batch, seq, in] @ [in, out/world] -> [batch, seq, out/world]
        output = nn.functional.linear(x, self.weight, self.bias)

        if self.gather_output:
            # Gather from all GPUs: [batch, seq, out/world] -> [batch, seq, out]
            output = all_gather(output, dim=-1)

        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The weight matrix is split along the input dimension (rows).
    Each GPU holds W[rank * rows : (rank+1) * rows, :]

    Y = XW where W is split row-wise, requires all-reduce after.

    Args:
        in_features: Total input dimension (will be split across GPUs)
        out_features: Output dimension
        bias: Whether to use bias
        input_is_parallel: If True, input is already split across GPUs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_is_parallel: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        world_size = get_tensor_parallel_world_size()
        assert in_features % world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"

        self.in_features_per_partition = in_features // world_size

        # Each GPU has a shard of the weight matrix
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # Split input across GPUs
            world_size = get_tensor_parallel_world_size()
            rank = get_tensor_parallel_rank()
            chunk_size = x.size(-1) // world_size
            x = x[..., rank * chunk_size : (rank + 1) * chunk_size]

        # Local matmul
        output = nn.functional.linear(x, self.weight)

        # All-reduce to sum partial results
        output = all_reduce(output)

        if self.bias is not None:
            output = output + self.bias

        return output


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism.

    The embedding table is split across GPUs along the vocabulary dimension.
    Each GPU holds embeddings for vocab_size/world_size tokens.

    Args:
        num_embeddings: Total vocabulary size
        embedding_dim: Embedding dimension
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()

        assert num_embeddings % world_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by world_size ({world_size})"

        self.num_embeddings_per_partition = num_embeddings // world_size
        self.vocab_start_index = rank * self.num_embeddings_per_partition
        self.vocab_end_index = (rank + 1) * self.num_embeddings_per_partition

        self.embedding = nn.Embedding(
            self.num_embeddings_per_partition,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask tokens not on this GPU
        mask = (x >= self.vocab_start_index) & (x < self.vocab_end_index)

        # Shift indices to local range
        local_x = x - self.vocab_start_index
        local_x = local_x.clamp(0, self.num_embeddings_per_partition - 1)

        # Get local embeddings
        output = self.embedding(local_x)

        # Zero out tokens not on this GPU
        output = output * mask.unsqueeze(-1).float()

        # All-reduce to combine embeddings from all GPUs
        output = all_reduce(output)

        return output


# =============================================================================
# Utility Functions
# =============================================================================

def make_parallel(
    module: nn.Module,
    tp_size: int = 1,
) -> nn.Module:
    """
    Convert a module to use tensor parallelism.

    This function replaces Linear layers with ColumnParallel/RowParallel
    based on their position in the model.

    Args:
        module: The module to parallelize
        tp_size: Tensor parallel world size

    Returns:
        Module with tensor parallel layers
    """
    if tp_size <= 1:
        return module

    # This is a simplified version - full implementation would need
    # to understand the model architecture to properly place column vs row parallel

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Heuristic: layers ending in certain patterns get specific treatment
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                # QKV projections: column parallel
                setattr(module, name, ColumnParallelLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    gather_output=False,  # Keep split for attention
                ))
            elif 'o_proj' in name or 'down_proj' in name:
                # Output projections: row parallel
                setattr(module, name, RowParallelLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                ))
            elif 'gate_proj' in name or 'up_proj' in name:
                # MLP up projections: column parallel
                setattr(module, name, ColumnParallelLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    gather_output=False,
                ))
        else:
            # Recurse into children
            make_parallel(child, tp_size)

    return module


def initialize_tensor_parallel(
    tp_size: int,
    rank: int,
    world_size: int,
    backend: str = "nccl",
):
    """
    Initialize tensor parallel distributed environment.

    Args:
        tp_size: Tensor parallel size (GPUs per model replica)
        rank: Global rank of this process
        world_size: Total number of processes
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Create tensor parallel groups
    # For now, use the default world group
    # Full implementation would create subgroups for TP and DP

    print(f"Tensor Parallel initialized: rank={rank}, world_size={world_size}, tp_size={tp_size}")
