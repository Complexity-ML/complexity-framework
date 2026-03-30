"""
GPU runtime patches — applied once at script start.

Centralizes all monkey-patches and file modifications needed
for stable GPU training across PyTorch versions and GPU architectures.

Complexity-ML — 2026
"""

import torch
import logging

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def patch_triton_heuristics() -> bool:
    """
    Patch overly conservative XBLOCK assertions in PyTorch's triton_heuristics.py.

    On H100/H200/B200 (sm_90+), PyTorch's inductor generates Triton kernels with
    XBLOCK values that exceed the max_block assertion. This patch disables
    those assertions on disk so Triton subprocesses pick it up.

    Returns True if patched, False if already patched or not needed.
    """
    try:
        import pathlib
        import inspect
        import torch._inductor.runtime.triton_heuristics as _th

        f = pathlib.Path(inspect.getfile(_th))
        src = f.read_text()

        if "# COMPLEXITY_PATCHED" in src:
            return False

        patched = src.replace(
            "assert val <= max_block,",
            "pass  # assert val <= max_block,",
        ).replace(
            "assert max_block % block == 0,",
            "pass  # assert max_block % block == 0,",
        )

        if patched != src:
            patched += "\n# COMPLEXITY_PATCHED\n"
            f.write_text(patched)
            logger.info(f"[gpu] Patched triton XBLOCK assertions in {f}")
            return True

        return False

    except Exception as e:
        logger.debug(f"[gpu] Triton patch skipped: {e}")
        return False


def patch_expandable_segments() -> None:
    """
    Enable CUDA expandable segments to reduce memory fragmentation.

    Without this, PyTorch's caching allocator can fragment GPU memory,
    causing OOM even when total free memory is sufficient.
    """
    import os
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in current:
        new_val = "expandable_segments:True"
        if current:
            new_val = current + "," + new_val
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_val
        logger.info(f"[gpu] Set PYTORCH_CUDA_ALLOC_CONF={new_val}")


def patch_nccl_timeouts() -> None:
    """
    Increase NCCL timeouts for large clusters where synchronization takes longer.
    """
    import os
    if "NCCL_TIMEOUT" not in os.environ:
        os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes
    if "NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


def apply_gpu_patches(
    triton: bool = True,
    expandable_segments: bool = True,
    nccl: bool = True,
) -> None:
    """
    Apply all GPU patches. Call once at script start.

    Args:
        triton: Patch triton XBLOCK assertions
        expandable_segments: Enable CUDA expandable segments
        nccl: Set NCCL timeouts for large clusters

    Usage:
        from complexity.gpu import apply_gpu_patches
        apply_gpu_patches()
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _PATCHES_APPLIED = True

    if not torch.cuda.is_available():
        return

    if expandable_segments:
        patch_expandable_segments()

    if nccl:
        patch_nccl_timeouts()

    if triton:
        patch_triton_heuristics()
