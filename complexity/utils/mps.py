"""
Apple Silicon (MPS), CUDA, and ROCm helpers — device selection, memory,
autocast, seeding.

Centralizes MPS-specific concerns used across training / inference scripts.
Safe on non-Mac hosts: helpers degrade to no-ops if MPS is unavailable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from .device import (
    autocast,
    autocast_dtype,
    empty_cache,
    is_mps_available,
    is_nvidia_cuda_available as is_cuda_available,
    is_rocm_available,
    seed_all,
    select_device,
    synchronize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def set_memory_watermark(high: float = 0.0, low: Optional[float] = None) -> None:
    """
    Configure MPS memory watermarks via env vars.

    high=0.0 disables the upper limit (lets MPS use all unified memory),
    recommended on constrained machines (e.g. 24 GB M-series) to avoid
    premature OOM from PyTorch's default conservative cap.

    Must be called BEFORE any MPS tensor is allocated.
    """
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{high}"
    if low is not None:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{low}"


def enable_cpu_fallback() -> None:
    """Allow PyTorch to fall back to CPU for ops missing an MPS kernel."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


@dataclass
class MPSMemoryStats:
    allocated_mb: float
    driver_allocated_mb: float
    recommended_max_mb: float

    def __str__(self) -> str:
        return (
            f"MPS mem: alloc={self.allocated_mb:.0f}MB "
            f"driver={self.driver_allocated_mb:.0f}MB "
            f"recommended_max={self.recommended_max_mb:.0f}MB"
        )


def mps_memory_stats() -> Optional[MPSMemoryStats]:
    if not is_mps_available():
        return None
    mb = 1024 ** 2
    return MPSMemoryStats(
        allocated_mb=torch.mps.current_allocated_memory() / mb,
        driver_allocated_mb=torch.mps.driver_allocated_memory() / mb,
        recommended_max_mb=torch.mps.recommended_max_memory() / mb,
    )


# ---------------------------------------------------------------------------
# One-shot setup
# ---------------------------------------------------------------------------

def setup_mps(
    unlimited_watermark: bool = True,
    cpu_fallback: bool = True,
    seed: Optional[int] = None,
) -> torch.device:
    """Apply recommended MPS env, seed, and return the selected device."""
    if unlimited_watermark:
        set_memory_watermark(high=0.0)
    if cpu_fallback:
        enable_cpu_fallback()
    device = select_device("auto")
    if seed is not None:
        seed_all(seed)
    if device.type == "cuda" and is_rocm_available():
        logger.info(f"Device: {device} (ROCm/HIP {torch.version.hip})")
    else:
        logger.info(f"Device: {device}")
    if device.type == "mps":
        stats = mps_memory_stats()
        if stats is not None:
            logger.info(str(stats))
    return device
