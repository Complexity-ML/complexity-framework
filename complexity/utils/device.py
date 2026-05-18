"""
Backend/device policy for CUDA, ROCm, MPS, and CPU.

PyTorch exposes AMD ROCm/HIP devices through the CUDA API. This module keeps
the framework-facing backend name ("rocm") separate from the actual
``torch.device("cuda")`` object used by PyTorch.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BackendName = Literal["cuda", "rocm", "mps", "cpu"]
KernelPolicy = Union[Literal["auto"], bool]


@dataclass(frozen=True)
class BackendInfo:
    name: BackendName
    device: torch.device
    device_name: str
    hip_version: Optional[str] = None
    custom_triton: bool = False
    sdpa: bool = False
    flash_attention: bool = False
    matmul: str = "cpu"
    distributed: str = "gloo"


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_rocm_available() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def is_nvidia_cuda_available() -> bool:
    return torch.cuda.is_available() and torch.version.hip is None


def get_backend(preferred: str = "auto") -> BackendName:
    if preferred == "auto":
        if is_rocm_available():
            return "rocm"
        if is_nvidia_cuda_available():
            return "cuda"
        if is_mps_available():
            return "mps"
        return "cpu"
    if preferred == "rocm":
        if not is_rocm_available():
            raise RuntimeError("ROCm requested, but this PyTorch build/device is not ROCm-enabled.")
        return "rocm"
    if preferred == "cuda":
        if is_rocm_available():
            return "rocm"
        return "cuda"
    if preferred in {"mps", "cpu"}:
        return preferred  # type: ignore[return-value]
    raise ValueError(f"Unknown device backend: {preferred}")


def select_device(preferred: str = "auto") -> torch.device:
    backend = get_backend(preferred)
    if backend in {"cuda", "rocm"}:
        return torch.device("cuda")
    return torch.device(backend)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def custom_kernels_enabled(policy: KernelPolicy = "auto") -> bool:
    """Return whether custom Triton/CUDA kernels should be used.

    ``auto`` enables custom kernels only on NVIDIA CUDA. ROCm uses PyTorch
    fallback by default; set ``COMPLEXITY_ALLOW_ROCM_TRITON=1`` or pass True to
    opt into experimental ROCm Triton paths.
    """
    if policy is True:
        return torch.cuda.is_available()
    if policy is False:
        return False
    if is_nvidia_cuda_available():
        return True
    return is_rocm_available() and os.environ.get("COMPLEXITY_ALLOW_ROCM_TRITON") == "1"


def supports_custom_triton(policy: KernelPolicy = "auto") -> bool:
    return custom_kernels_enabled(policy)


def get_backend_info(preferred: str = "auto", kernel_policy: KernelPolicy = "auto") -> BackendInfo:
    backend = get_backend(preferred)
    device = select_device(preferred)
    sdpa = hasattr(F, "scaled_dot_product_attention")
    return BackendInfo(
        name=backend,
        device=device,
        device_name=_device_name(device),
        hip_version=torch.version.hip if backend == "rocm" else None,
        custom_triton=custom_kernels_enabled(kernel_policy),
        sdpa=sdpa,
        flash_attention=sdpa and backend in {"cuda", "rocm"},
        matmul={
            "cuda": "cuBLAS/cuBLASLt",
            "rocm": "rocBLAS/hipBLASLt",
            "mps": "MPSGraph",
            "cpu": "CPU",
        }[backend],
        distributed={
            "cuda": "NCCL",
            "rocm": "RCCL via torch.distributed nccl",
            "mps": "gloo",
            "cpu": "gloo",
        }[backend],
    )


def log_backend(preferred: str = "auto", kernel_policy: KernelPolicy = "auto") -> BackendInfo:
    info = get_backend_info(preferred, kernel_policy)
    parts = [
        f"backend={info.name}",
        f"device={info.device}",
        f"name={info.device_name}",
        f"matmul={info.matmul}",
        f"distributed={info.distributed}",
        f"sdpa={str(info.sdpa).lower()}",
        f"flash_attention={str(info.flash_attention).lower()}",
        f"custom_triton={str(info.custom_triton).lower()}",
    ]
    if info.hip_version is not None:
        parts.append(f"hip={info.hip_version}")
    logger.info("[device] " + " ".join(parts))
    return info


def backend_metadata(preferred: str = "auto", kernel_policy: KernelPolicy = "auto") -> dict[str, object]:
    """Serializable backend metadata for checkpoints and run configs."""
    info = get_backend_info(preferred, kernel_policy)
    backends = sdpa_kernel_backends()
    return {
        "backend": info.name,
        "device": str(info.device),
        "device_name": info.device_name,
        "hip_version": info.hip_version,
        "custom_triton": info.custom_triton,
        "sdpa": info.sdpa,
        "flash_attention": info.flash_attention,
        "sdpa_backends": [str(backend).split(".")[-1] for backend in backends] if backends else [],
        "matmul": info.matmul,
        "distributed": info.distributed,
    }


def configure_torch_acceleration(kernel_policy: KernelPolicy = "auto", log: bool = True) -> BackendInfo:
    """Apply safe backend-level acceleration toggles and return backend info."""
    info = log_backend(kernel_policy=kernel_policy) if log else get_backend_info(kernel_policy=kernel_policy)
    if info.name == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    elif info.name == "rocm":
        torch.set_float32_matmul_precision("high")
    return info


def _parse_sdpa_backend_names() -> Optional[list[str]]:
    raw = os.environ.get("COMPLEXITY_SDPA_BACKENDS")
    if not raw:
        return None
    names = [x.strip().upper() for x in raw.split(",") if x.strip()]
    return names or None


def sdpa_kernel_backends() -> Optional[list[object]]:
    """Return requested PyTorch SDPA kernels for the active backend.

    Defaults to Flash/Efficient/Math when available. Override with
    ``COMPLEXITY_SDPA_BACKENDS=flash,efficient,math``.
    """
    try:
        from torch.nn.attention import SDPBackend
    except Exception:
        return None

    requested = _parse_sdpa_backend_names() or ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"]
    aliases = {
        "FLASH": "FLASH_ATTENTION",
        "MEM_EFFICIENT": "EFFICIENT_ATTENTION",
        "EFFICIENT": "EFFICIENT_ATTENTION",
    }
    backends = []
    for name in requested:
        attr = aliases.get(name, name)
        backend = getattr(SDPBackend, attr, None)
        if backend is not None:
            backends.append(backend)
    return backends or None


@contextmanager
def sdpa_kernel_context():
    try:
        from torch.nn.attention import sdpa_kernel
    except Exception:
        yield
        return
    backends = sdpa_kernel_backends()
    ctx = sdpa_kernel(backends) if backends else nullcontext()
    with ctx:
        yield


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if is_mps_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_dtype(device: torch.device, prefer_bf16: bool = True) -> Optional[torch.dtype]:
    if device.type == "cpu":
        return None
    if device.type == "cuda" and prefer_bf16 and hasattr(torch.cuda, "is_bf16_supported"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16 if prefer_bf16 else torch.float16


@contextmanager
def autocast(device: torch.device, dtype: Optional[torch.dtype] = None, enabled: bool = True):
    if not enabled or device.type == "cpu":
        yield
        return
    dt = dtype or autocast_dtype(device) or torch.float32
    with torch.autocast(device_type=device.type, dtype=dt):
        yield
