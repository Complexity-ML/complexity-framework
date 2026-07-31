"""Backend selection and explicit TR-Hash capability reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .config import (
    TRHashBackend,
    TRHashEngineConfig,
    TRHashPhase,
    TRHashPrecision,
)


@dataclass(frozen=True)
class BackendDecision:
    requested: TRHashBackend
    selected: TRHashBackend
    reasons: Tuple[str, ...]
    graph_safe: bool
    quantized: bool


def _custom_cggr_available() -> bool:
    try:
        from complexity_cuda.triton_token_routed import (
            HAS_TRITON,
            cggr_grouped_gemm_autograd,
        )

        return bool(HAS_TRITON and cggr_grouped_gemm_autograd is not None)
    except Exception:
        return False


def select_backend(
    config: TRHashEngineConfig,
    *,
    device_type: str,
    cggr_available: Optional[bool] = None,
) -> BackendDecision:
    """Resolve a runtime backend without silently claiming unsupported modes."""

    if cggr_available is None:
        cggr_available = _custom_cggr_available()
    reasons = []

    if config.precision in {TRHashPrecision.FP8, TRHashPrecision.INT8}:
        raise NotImplementedError(
            f"{config.precision.value} is reserved by the TR-Hash API but "
            "requires the phase-2 quantized grouped-GEMM kernel"
        )

    requested = config.backend
    if requested is TRHashBackend.PYTORCH:
        return BackendDecision(
            requested=requested,
            selected=TRHashBackend.PYTORCH,
            reasons=(),
            graph_safe=False,
            quantized=False,
        )

    if requested is TRHashBackend.CUDA_GRAPH:
        if device_type != "cuda":
            raise RuntimeError("CUDA Graph backend requires a CUDA device")
        if config.phase is not TRHashPhase.INFERENCE:
            raise RuntimeError("CUDA Graph backend is inference-only")
        return BackendDecision(
            requested=requested,
            selected=TRHashBackend.CUDA_GRAPH,
            reasons=(),
            graph_safe=True,
            quantized=False,
        )

    if requested is TRHashBackend.CGGR:
        if device_type != "cuda":
            raise RuntimeError("CGGR backend requires a CUDA device")
        if not cggr_available:
            raise RuntimeError("CGGR backend requested but Triton CGGR is unavailable")
        return BackendDecision(
            requested=requested,
            selected=TRHashBackend.CGGR,
            reasons=(),
            graph_safe=False,
            quantized=False,
        )

    if device_type != "cuda":
        reasons.append("CGGR requires CUDA")
    elif not cggr_available:
        reasons.append("Triton CGGR is unavailable")
    else:
        return BackendDecision(
            requested=requested,
            selected=TRHashBackend.CGGR,
            reasons=(),
            graph_safe=False,
            quantized=False,
        )
    return BackendDecision(
        requested=requested,
        selected=TRHashBackend.PYTORCH,
        reasons=tuple(reasons),
        graph_safe=False,
        quantized=False,
    )


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
