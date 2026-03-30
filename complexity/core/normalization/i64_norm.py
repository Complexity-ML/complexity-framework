"""
I64 Integer RMSNorm — float rsqrt (irreducible) + Q12 INT16 weight multiply.

Train in float, deploy with quantized INT16 weights.
Call quantize_weight() to convert.

Registered as "i64_rmsnorm" / "integer_rmsnorm" in the normalization registry.

INL 2025 — ported from complexity-i64.
"""

import torch
import torch.nn as nn

from ..registry import register_normalization

_Q_NORM = 128      # Q7 for normalized values
_Q_WEIGHT = 4096   # Q12 for weights


@register_normalization("i64_rmsnorm")
@register_normalization("integer_rmsnorm")
class I64RMSNorm(nn.Module):
    """Integer RMSNorm with optional fused INT8 output."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm -> float output."""
        if hasattr(self, 'weight_q12'):
            return self._forward_integer(x)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

    def _forward_integer(self, x: torch.Tensor) -> torch.Tensor:
        """Integer path: float rsqrt + Q7*Q12 -> Q19, dequant."""
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        xn = x.float() * norm
        xn_q7 = (xn * _Q_NORM).round().to(torch.int32)
        out_q19 = xn_q7 * self.weight_q12.to(torch.int32)
        return (out_q19.float() / (_Q_NORM * _Q_WEIGHT)).type_as(x)

    def quantize_weight(self):
        """Convert float weight to Q12 INT16."""
        w = self.weight.data.float()
        w_q12 = (w * _Q_WEIGHT).round().clamp(-32768, 32767).to(torch.int16)
        self.register_buffer('weight_q12', w_q12)

    def extra_repr(self) -> str:
        quantized = hasattr(self, 'weight_q12')
        return f"{self.hidden_size}, eps={self.eps}, quantized={quantized}"
