"""
Normalization layers for framework-complexity.

Available:
- RMSNorm: Efficient, used in Llama/Mistral
- LayerNorm: Standard, used in GPT/BERT
"""

import torch
import torch.nn as nn

from ..registry import register_normalization


@register_normalization("rmsnorm")
@register_normalization("rms")
@register_normalization("llama")
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm - no mean subtraction, no bias.
    Used in Llama, Mistral, Gemma, and most modern LLMs.

    Formula: x * rsqrt(mean(x^2) + eps) * weight

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"


@register_normalization("layernorm")
@register_normalization("ln")
@register_normalization("gpt")
class LayerNorm(nn.Module):
    """
    Standard Layer Normalization.

    Normalizes by subtracting mean and dividing by std.
    Used in GPT-2, BERT, original Transformer.

    Formula: (x - mean) / sqrt(var + eps) * weight + bias

    Reference: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}, bias={self.bias is not None}"


@register_normalization("identity")
@register_normalization("none")
class IdentityNorm(nn.Module):
    """
    Identity normalization (no-op).

    Useful for ablation studies or when normalization is handled elsewhere.
    """

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_norm(norm_type: str, hidden_size: int, **kwargs) -> nn.Module:
    """
    Build a normalization layer by type.

    Args:
        norm_type: Type of normalization (rmsnorm, layernorm, identity)
        hidden_size: Size of the hidden dimension
        **kwargs: Additional arguments for the norm layer

    Returns:
        Normalization module
    """
    from ..registry import NORMALIZATION_REGISTRY
    return NORMALIZATION_REGISTRY.build(norm_type, hidden_size, **kwargs)
