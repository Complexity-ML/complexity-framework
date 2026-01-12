"""
Rotary Position Embedding (RoPE) implementations.

Includes:
- Standard RoPE
- YaRN (Yet another RoPE extensioN) for extended context
- Dynamic NTK-aware scaling
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..registry import register_position


class RotaryEmbedding(nn.Module):
    """
    Standard Rotary Position Embedding (RoPE).

    Encodes position information directly into Q/K vectors via rotation.
    Benefits:
    - Relative position awareness
    - Extrapolates to longer sequences
    - No learned position embeddings

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin embeddings for given sequence length.

        Returns:
            cos: [seq_len, dim]
            sin: [seq_len, dim]
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


@register_position("rope")
@register_position("standard")
class StandardRoPE(RotaryEmbedding):
    """Standard RoPE - registered in the framework."""
    pass


@register_position("yarn")
class YaRNRoPE(RotaryEmbedding):
    """
    YaRN: Yet another RoPE extensioN.

    Allows extending context length beyond training without fine-tuning.
    Combines NTK-aware interpolation with attention scaling.

    Reference: https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scale: float = 1.0,
        original_max_seq_len: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        self.scale = scale
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        super().__init__(dim, max_seq_len, theta)

    def _build_cache(self, seq_len: int):
        """Build YaRN-scaled cos/sin cache."""
        # Compute scaling factor
        if seq_len > self.original_max_seq_len:
            scale = seq_len / self.original_max_seq_len
        else:
            scale = 1.0

        # NTK-aware frequency scaling
        dim_half = self.dim // 2
        freq_extra = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        freq_inter = 1.0 / (scale * self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))

        # Blend based on frequency
        low = max(0, int(self.beta_fast * dim_half / (2 * torch.pi)))
        high = min(dim_half - 1, int(self.beta_slow * dim_half / (2 * torch.pi)))

        inv_freq = freq_inter.clone()
        inv_freq[:low] = freq_extra[:low]

        # Smooth transition
        if high > low:
            smooth = torch.arange(low, high + 1).float() / (high - low + 1)
            inv_freq[low:high + 1] = (1 - smooth) * freq_extra[low:high + 1] + smooth * freq_inter[low:high + 1]

        inv_freq = inv_freq.to(self.inv_freq.device)

        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


@register_position("dynamic_ntk")
class DynamicNTKRoPE(RotaryEmbedding):
    """
    Dynamic NTK-aware RoPE scaling.

    Dynamically adjusts base frequency based on sequence length.
    Simpler than YaRN but effective for moderate context extension.

    Reference: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        original_max_seq_len: int = 2048,
        scaling_factor: float = 1.0,
    ):
        self.original_max_seq_len = original_max_seq_len
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_seq_len, theta)

    def _build_cache(self, seq_len: int):
        """Build dynamically scaled cos/sin cache."""
        if seq_len > self.original_max_seq_len:
            # Dynamic NTK scaling
            scale = (self.scaling_factor * seq_len / self.original_max_seq_len) - (self.scaling_factor - 1)
            theta = self.theta * (scale ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        else:
            inv_freq = self.inv_freq

        inv_freq = inv_freq.to(self.inv_freq.device)
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


# Helper functions
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K tensors.

    Args:
        q: Query tensor [batch, heads, seq, head_dim]
        k: Key tensor [batch, heads, seq, head_dim]
        cos: Cosine embeddings [seq, head_dim]
        sin: Sine embeddings [seq, head_dim]

    Returns:
        Rotated Q and K tensors
    """
    # Reshape cos/sin for broadcasting: [1, 1, seq, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# =============================================================================
# ALiBi - Attention with Linear Biases
# =============================================================================

@register_position("alibi")
class ALiBiPositionBias(nn.Module):
    """
    ALiBi: Attention with Linear Biases.

    Instead of adding position embeddings to tokens, ALiBi adds a linear
    bias to attention scores based on distance between query and key positions.

    Benefits:
    - No position embeddings to learn
    - Extrapolates to longer sequences than trained on
    - Simpler than RoPE
    - Used in BLOOM, MPT, and other models

    Reference: https://arxiv.org/abs/2108.12409

    Usage:
        alibi = ALiBiPositionBias(num_heads=32)
        # In attention: attn_scores = attn_scores + alibi(seq_len)
    """

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

        # Pre-compute bias matrix
        self._build_alibi_bias(max_seq_len)

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Get ALiBi slopes for each attention head.

        For 8 heads: 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256
        """
        def get_slopes_power_of_2(n: int) -> list:
            start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n)).floor().int().item() - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if num_heads & (num_heads - 1) == 0:  # Power of 2
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Not power of 2: interpolate
            closest_power_of_2 = 2 ** int(torch.log2(torch.tensor(num_heads)).floor().item())
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            )

        return torch.tensor(slopes, dtype=torch.float32)

    def _build_alibi_bias(self, seq_len: int):
        """Build the ALiBi bias matrix."""
        # Distance matrix: position[i] - position[j]
        positions = torch.arange(seq_len)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]

        # ALiBi bias: slope * distance (negative for causal masking effect)
        # Shape: [num_heads, seq, seq]
        alibi_bias = self.slopes.unsqueeze(1).unsqueeze(1) * distance.unsqueeze(0)

        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

    def forward(self, seq_len: int, kv_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Get ALiBi bias for given sequence length.

        Args:
            seq_len: Query sequence length
            kv_seq_len: Key/Value sequence length (for cross-attention or caching)

        Returns:
            alibi_bias: [num_heads, seq_len, kv_seq_len]
        """
        if kv_seq_len is None:
            kv_seq_len = seq_len

        # Rebuild cache if needed
        max_len = max(seq_len, kv_seq_len)
        if max_len > self.max_seq_len:
            self._build_alibi_bias(max_len)
            self.max_seq_len = max_len

        # For generation with KV cache: query is at position [kv_seq_len - seq_len : kv_seq_len]
        q_start = kv_seq_len - seq_len
        return self.alibi_bias[:, q_start:kv_seq_len, :kv_seq_len]


@register_position("learned")
class LearnedPositionEmbedding(nn.Module):
    """
    Learned absolute position embeddings.

    Classic approach used in GPT-2, BERT.
    Simple but doesn't extrapolate well beyond training length.
    """

    def __init__(self, max_seq_len: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, hidden_size)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get position embeddings.

        Args:
            seq_len: Sequence length

        Returns:
            embeddings: [seq_len, hidden_size]
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max {self.max_seq_len}. "
                "Learned position embeddings cannot extrapolate."
            )
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)
