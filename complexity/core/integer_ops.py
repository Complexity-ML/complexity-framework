"""
Integer Operations for Complexity Framework

Core INT8 compute primitives — pure PyTorch, no Triton required.
Works on CPU (via fallback) and CUDA (via torch._int_mm).

- int8_linear: INT8xINT8->INT32 matmul
- quantize/dequantize: float <-> INT8 conversion
- LUT activations: SiLU, sigmoid, softplus as table lookups (zero FLOPs)
- Fused gate+up for SwiGLU

Ported from complexity-i64 (INL 2025).
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# =========================================================================
# Native INT8 matmul detection
# =========================================================================

_INT_MM_AVAILABLE = hasattr(torch, '_int_mm')
_INT_MM_CPU_OK = False
if _INT_MM_AVAILABLE:
    try:
        _a = torch.ones(1, 8, dtype=torch.int8)
        _b = torch.ones(8, 1, dtype=torch.int8)
        torch._int_mm(_a, _b)
        _INT_MM_CPU_OK = True
    except (RuntimeError, Exception):
        pass

_Q7 = 128  # Fixed-point scale: 7 fractional bits


# =========================================================================
# Quantize / Dequantize
# =========================================================================

def quantize_weight_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel INT8 symmetric: (out, in) float -> (out, in) int8 + (out,) scale."""
    abs_max = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / 127.0
    quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def quantize_activation_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token INT8 symmetric: (tokens, feat) -> int8 + (tokens,) scale."""
    x_f32 = x.float()
    abs_max = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / 127.0
    x_int8 = (x_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale.squeeze(-1)


# =========================================================================
# INT8 Linear — the fundamental operation
# =========================================================================

def int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    INT8 linear: y = x @ W^T

    1. Quantize activations x -> x_int8, x_scale (per-token)
    2. _int_mm(x_int8, W_int8^T) -> result_int32
    3. Rescale: result * x_scale * w_scale
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    out_features = weight_int8.shape[0]

    use_int_mm = _INT_MM_AVAILABLE and (x.is_cuda or _INT_MM_CPU_OK)

    if use_int_mm:
        x_int8, x_scale = quantize_activation_int8(x_2d)
        wt = weight_int8.t().contiguous()
        result_i32 = torch._int_mm(x_int8, wt)
        out = result_i32.float() * (x_scale.unsqueeze(1) * weight_scale.unsqueeze(0))
    else:
        # Fallback: dequant + float matmul
        w_float = weight_int8.float() * weight_scale.unsqueeze(-1)
        out = F.linear(x_2d.float(), w_float)

    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], out_features)


def int8_fused_gate_up(
    x: torch.Tensor,
    fused_int8: torch.Tensor,
    fused_scale: torch.Tensor,
    inter_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused gate+up: single quantization + single matmul, split output."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    use_int_mm = _INT_MM_AVAILABLE and (x.is_cuda or _INT_MM_CPU_OK)

    if use_int_mm:
        x_int8, x_scale = quantize_activation_int8(x_2d)
        wt = fused_int8.t().contiguous()
        result_i32 = torch._int_mm(x_int8, wt)
        result = result_i32.float() * (x_scale.unsqueeze(1) * fused_scale.unsqueeze(0))
    else:
        w_float = fused_int8.float() * fused_scale.unsqueeze(-1)
        result = F.linear(x_2d.float(), w_float)

    gate, up = result.split(inter_size, dim=-1)
    return (
        gate.reshape(*orig_shape[:-1], inter_size),
        up.reshape(*orig_shape[:-1], inter_size),
    )


# =========================================================================
# LUT Activations — zero compute, just table lookups
# =========================================================================

_SILU_LUT_MIN = -1024
_SILU_LUT_MAX = 1024


def _build_silu_lut() -> torch.Tensor:
    indices = torch.arange(_SILU_LUT_MIN, _SILU_LUT_MAX + 1, dtype=torch.float32)
    x = indices / _Q7
    return (F.silu(x) * _Q7).round().to(torch.int32)


_SILU_LUT = _build_silu_lut()


def silu_integer(x_q7: torch.Tensor) -> torch.Tensor:
    """Fixed-point SiLU via LUT. Q7 in -> Q7 out."""
    lut = _SILU_LUT.to(x_q7.device)
    clamped = x_q7.clamp(_SILU_LUT_MIN, _SILU_LUT_MAX)
    indices = (clamped - _SILU_LUT_MIN).long()
    result = lut[indices]
    result = torch.where(x_q7 > _SILU_LUT_MAX, x_q7, result)
    result = torch.where(x_q7 < _SILU_LUT_MIN, torch.zeros_like(result), result)
    return result


def _build_sigmoid_lut() -> torch.Tensor:
    indices = torch.arange(-1024, 1025, dtype=torch.float32)
    return (torch.sigmoid(indices / _Q7) * _Q7).round().to(torch.int32)


_SIGMOID_LUT = _build_sigmoid_lut()


def sigmoid_integer(x_q7: torch.Tensor) -> torch.Tensor:
    """Fixed-point sigmoid via LUT. Q7 in -> Q7 out (0=0.0, 128=1.0)."""
    lut = _SIGMOID_LUT.to(x_q7.device)
    clamped = x_q7.clamp(-1024, 1024)
    indices = (clamped + 1024).long()
    result = lut[indices]
    result = torch.where(x_q7 > 1024, torch.full_like(result, _Q7), result)
    result = torch.where(x_q7 < -1024, torch.zeros_like(result), result)
    return result


def _build_softplus_lut() -> torch.Tensor:
    indices = torch.arange(-1024, 1025, dtype=torch.float32)
    return (F.softplus(indices / _Q7) * _Q7).round().to(torch.int32)


_SOFTPLUS_LUT = _build_softplus_lut()


def softplus_integer(x_q7: torch.Tensor) -> torch.Tensor:
    """Fixed-point softplus via LUT. Q7 in -> Q7 out."""
    lut = _SOFTPLUS_LUT.to(x_q7.device)
    clamped = x_q7.clamp(-1024, 1024)
    indices = (clamped + 1024).long()
    result = lut[indices]
    result = torch.where(x_q7 > 1024, x_q7, result)
    result = torch.where(x_q7 < -1024, torch.zeros_like(result), result)
    return result


def silu_multiply_integer(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Integer SiLU*up: quantize to Q7, LUT SiLU, INT32 multiply, dequant."""
    gate_q7 = (gate.float() * _Q7).round().to(torch.int32)
    silu_q7 = silu_integer(gate_q7)
    up_q7 = (up.float() * _Q7).round().to(torch.int32)
    inter_q14 = silu_q7 * up_q7
    return inter_q14.float() / (_Q7 * _Q7)
