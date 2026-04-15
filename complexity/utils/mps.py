"""
Apple Silicon (MPS) helpers — device selection, memory, autocast, seeding.

Centralizes MPS-specific concerns used across training / inference scripts.
Safe on non-Mac hosts: helpers degrade to no-ops if MPS is unavailable.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def select_device(preferred: str = "auto") -> torch.device:
    """Return best device. preferred in {auto, mps, cuda, cpu}."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


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


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if is_mps_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Autocast
# ---------------------------------------------------------------------------

def autocast_dtype(device: torch.device, prefer_bf16: bool = True) -> Optional[torch.dtype]:
    """
    Best mixed-precision dtype for `device`.

    MPS supports bf16 from PyTorch 2.3+. On older builds, falls back to float16.
    Returns None on CPU (autocast there is rarely worth it).
    """
    if device.type == "cpu":
        return None
    if device.type == "mps":
        return torch.bfloat16 if prefer_bf16 else torch.float16
    return torch.bfloat16 if prefer_bf16 else torch.float16


@contextmanager
def autocast(device: torch.device, dtype: Optional[torch.dtype] = None, enabled: bool = True):
    """Unified autocast ctx mgr for mps/cuda/cpu."""
    if not enabled or device.type == "cpu":
        yield
        return
    dt = dtype or autocast_dtype(device) or torch.float32
    with torch.autocast(device_type=device.type, dtype=dt):
        yield


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
    logger.info(f"Device: {device}")
    if device.type == "mps":
        stats = mps_memory_stats()
        if stats is not None:
            logger.info(str(stats))
    return device
