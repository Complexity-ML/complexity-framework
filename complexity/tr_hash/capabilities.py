"""Runtime execution policies for the unique contextual TR-HASH engine."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from .config import TRHashBackend, TRHashEngineConfig, TRHashPrecision


@dataclass(frozen=True)
class BackendDecision:
    requested: TRHashBackend
    selected: TRHashBackend
    reasons: Tuple[str, ...]
    graph_safe: bool
    quantized: bool


def _custom_cggr_available() -> bool:
    try:
        from complexity_cuda.triton_token_routed import HAS_TRITON, cggr_grouped_gemm_autograd

        return bool(HAS_TRITON and cggr_grouped_gemm_autograd is not None)
    except Exception:
        return False


def _fused_cuda_available() -> bool:
    return False


def supports_fused_cuda(config: TRHashEngineConfig) -> bool:
    """Static token-only pair kernels cannot execute the contextual route law."""
    return False


def select_backend(
    config: TRHashEngineConfig,
    *,
    device_type: str,
    cggr_available: Optional[bool] = None,
    fused_cuda_available: Optional[bool] = None,
) -> BackendDecision:
    if config.precision in {TRHashPrecision.FP8, TRHashPrecision.INT8}:
        raise NotImplementedError(
            f"{config.precision.value} requires quantized grouped-GEMM kernels"
        )
    requested = config.backend
    if requested in {TRHashBackend.FUSED_CUDA, TRHashBackend.CUDA_GRAPH}:
        raise RuntimeError(
            f"TR-HASH contextual routing does not support {requested.value}; use PyTorch, CGGR, or auto"
        )
    if requested is TRHashBackend.PYTORCH:
        return BackendDecision(requested, TRHashBackend.PYTORCH, (), False, False)
    if cggr_available is None:
        cggr_available = _custom_cggr_available()
    if requested is TRHashBackend.CGGR:
        if device_type != "cuda":
            raise RuntimeError("CGGR backend requires a CUDA device")
        if not cggr_available:
            raise RuntimeError("CGGR backend requested but Triton CGGR is unavailable")
        return BackendDecision(requested, TRHashBackend.CGGR, (), False, False)
    reasons = ["TR-HASH contextual routes require PyTorch or CGGR"]
    if device_type == "cuda" and cggr_available:
        return BackendDecision(requested, TRHashBackend.CGGR, tuple(reasons), False, False)
    reasons.append("CGGR requires CUDA" if device_type != "cuda" else "Triton CGGR is unavailable")
    return BackendDecision(requested, TRHashBackend.PYTORCH, tuple(reasons), False, False)


def precision_to_torch_dtype(precision: TRHashPrecision) -> torch.dtype:
    mapping = {
        TRHashPrecision.FP32: torch.float32,
        TRHashPrecision.BF16: torch.bfloat16,
        TRHashPrecision.FP16: torch.float16,
    }
    try:
        return mapping[precision]
    except KeyError as exc:
        raise NotImplementedError(f"{precision.value} needs the quantized TR-Hash backend") from exc
