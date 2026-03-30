"""
GPU acceleration — patches, kernels, memory, profiling.

    from complexity.gpu import setup_gpu
    setup_gpu()  # All optimizations in one call

Modules:
    patches.py       — Triton heuristics fix, expandable segments, NCCL timeouts
    cuda_config.py   — TF32, cuDNN benchmark, matmul precision
    nccl_config.py   — P2P, InfiniBand, socket, timeout
    memory.py        — Cache clearing, memory logging, auto max batch
    compile.py       — torch.compile wrapper + cache management
    profiler.py      — StepTimer, throughput, FLOPs estimation
    fused_kernels.py — Triton: FusedRMSNorm, fused_swiglu, fused_residual_rmsnorm
    gradient.py      — Gradient compression (topk, quantize), fused clip
    precision.py     — Mixed precision config, BF16Autocast, upcast_softmax
    data_loading.py  — CUDAPrefetcher, BackgroundTokenizer

Complexity-ML — 2026
"""

from .patches import apply_gpu_patches, patch_triton_heuristics
from .cuda_config import configure_cuda_backends
from .nccl_config import configure_nccl
from .memory import clear_gpu_cache, log_memory_usage, estimate_max_batch_size
from .compile import compile_model, set_compile_env
from .profiler import StepTimer, profile_model_flops
from .fused_kernels import FusedRMSNorm, fused_swiglu, fused_residual_rmsnorm
from .gradient import GradientCompressor, clip_grad_norm_fused
from .precision import configure_mixed_precision, BF16Autocast, upcast_softmax
from .data_loading import CUDAPrefetcher, BackgroundTokenizer


def setup_gpu(**kwargs) -> None:
    """Apply all GPU optimizations in one call."""
    apply_gpu_patches(**kwargs)
    configure_cuda_backends()


__all__ = [
    "setup_gpu",
    "apply_gpu_patches",
    "patch_triton_heuristics",
    "configure_cuda_backends",
    "configure_nccl",
    "clear_gpu_cache",
    "log_memory_usage",
    "estimate_max_batch_size",
    "compile_model",
    "set_compile_env",
    "StepTimer",
    "profile_model_flops",
    "FusedRMSNorm",
    "fused_swiglu",
    "fused_residual_rmsnorm",
    "GradientCompressor",
    "clip_grad_norm_fused",
    "configure_mixed_precision",
    "BF16Autocast",
    "upcast_softmax",
    "CUDAPrefetcher",
    "BackgroundTokenizer",
]
