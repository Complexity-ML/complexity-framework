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


def _fused_cuda_available() -> bool:
    try:
        from complexity_cuda.triton_token_routed import (
            HAS_TRITON,
            fused_swiglu_triton,
            hash_cggr_grouped_gemm_autograd,
            pair_hash_reduce_autograd,
            pair_hash_weighted_reduce_autograd,
            sort_pair_hash_by_expert,
        )

        return bool(
            HAS_TRITON
            and fused_swiglu_triton is not None
            and hash_cggr_grouped_gemm_autograd is not None
            and pair_hash_reduce_autograd is not None
            and pair_hash_weighted_reduce_autograd is not None
            and sort_pair_hash_by_expert is not None
        )
    except Exception:
        return False


def supports_fused_cuda(config: TRHashEngineConfig) -> bool:
    """Return whether a shape has a hash-native fused CUDA implementation."""

    return config.top_k == 2 and 2 <= config.num_experts <= 4


def select_backend(
    config: TRHashEngineConfig,
    *,
    device_type: str,
    cggr_available: Optional[bool] = None,
    fused_cuda_available: Optional[bool] = None,
) -> BackendDecision:
    """Resolve a runtime backend without silently claiming unsupported modes."""

    if cggr_available is None:
        cggr_available = _custom_cggr_available()
    if fused_cuda_available is None:
        fused_cuda_available = _fused_cuda_available()
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

    if requested is TRHashBackend.FUSED_CUDA:
        if device_type != "cuda":
            raise RuntimeError("fused CUDA backend requires a CUDA device")
        if not supports_fused_cuda(config):
            raise RuntimeError(
                "fused CUDA backend requires top_k=2 and two to four experts"
            )
        if not fused_cuda_available:
            raise RuntimeError(
                "fused CUDA backend requested but hash-native Triton kernels "
                "are unavailable"
            )
        return BackendDecision(
            requested=requested,
            selected=TRHashBackend.FUSED_CUDA,
            reasons=(),
            graph_safe=False,
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

    if (
        device_type == "cuda"
        and fused_cuda_available
        and supports_fused_cuda(config)
    ):
        return BackendDecision(
            requested=requested,
            selected=TRHashBackend.FUSED_CUDA,
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
