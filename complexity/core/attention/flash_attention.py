"""
Flash Attention and Efficient Attention Variants.

Implements memory-efficient attention mechanisms:
- Flash Attention 2: O(N) memory instead of O(N²)
- Sliding Window Attention: Local attention with fixed window
- Sparse Attention: Block-sparse patterns
- Linear Attention: O(N) complexity

References:
- Flash Attention: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- Longformer: https://arxiv.org/abs/2004.05150
- Linear Attention: https://arxiv.org/abs/2006.16236
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from .base import AttentionBase, AttentionConfig
from ..registry import register_attention


# =============================================================================
# Flash Attention
# =============================================================================

@register_attention("flash")
@register_attention("flash_attention")
class FlashAttention(AttentionBase):
    """
    Flash Attention implementation.

    Uses PyTorch's native scaled_dot_product_attention with flash attention
    backend when available (requires PyTorch 2.0+).

    Benefits:
    - O(N) memory instead of O(N²)
    - 2-4x faster than standard attention
    - Supports causal masking efficiently
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.scale

        # Number of query heads per KV head (for GQA)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Check if flash attention is available
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash_available:
            import warnings
            warnings.warn(
                "Flash Attention requires PyTorch 2.0+. "
                "Falling back to standard attention."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_rotary(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        new_cache = (k, v) if use_cache else None

        # Expand KV for GQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash attention
        if self.flash_available:
            # Use PyTorch's native flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None,  # Causal if no mask provided
                scale=self.scale,
            )
        else:
            # Fallback to standard attention
            attn_output = self._standard_attention(q, k, v, attention_mask)

        # Reshape back: [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_cache

    def _apply_rotary(self, q, k, cos, sin):
        """Apply rotary position embeddings."""
        # Assuming cos, sin are [1, seq, 1, head_dim]
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _standard_attention(self, q, k, v, mask):
        """Standard attention fallback."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores + mask
        else:
            # Causal mask
            seq_len = q.size(-2)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=q.device),
                diagonal=1
            )
            scores = scores + causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)


# =============================================================================
# Sliding Window Attention
# =============================================================================

@register_attention("sliding_window")
@register_attention("local")
class SlidingWindowAttention(AttentionBase):
    """
    Sliding Window Attention (Longformer-style).

    Each token attends only to a fixed window of surrounding tokens.
    Reduces complexity from O(N²) to O(N * W) where W is window size.

    Used in:
    - Mistral (4096 window)
    - Longformer
    - BigBird (combined with global attention)
    """

    def __init__(self, config: AttentionConfig, window_size: int = 4096):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.scale
        self.window_size = window_size

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)

        # Handle KV cache (sliding window)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
            # Keep only window_size tokens
            if k.size(1) > self.window_size:
                k = k[:, -self.window_size:]
                v = v[:, -self.window_size:]

        new_cache = (k, v) if use_cache else None

        # Expand KV for GQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Create sliding window mask
        window_mask = self._create_sliding_window_mask(seq_len, k.size(1), hidden_states.device)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention with sliding window mask
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + window_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, new_cache

    def _create_sliding_window_mask(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create sliding window causal mask."""
        # Create position indices
        q_idx = torch.arange(key_len - query_len, key_len, device=device)
        k_idx = torch.arange(key_len, device=device)

        # Distance matrix
        distance = q_idx.unsqueeze(1) - k_idx.unsqueeze(0)

        # Mask: attend to past window_size tokens
        mask = torch.where(
            (distance >= 0) & (distance < self.window_size),
            torch.tensor(0.0, device=device),
            torch.tensor(float('-inf'), device=device),
        )

        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, query_len, key_len]

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# =============================================================================
# Sparse Attention
# =============================================================================

@register_attention("sparse")
@register_attention("block_sparse")
class SparseAttention(AttentionBase):
    """
    Block-Sparse Attention.

    Combines local attention with global attention on specific tokens.
    Pattern inspired by BigBird and Longformer.

    Patterns:
    - Local: Each token attends to nearby tokens
    - Global: Some tokens attend to all tokens
    - Random: Random attention for coverage
    """

    def __init__(
        self,
        config: AttentionConfig,
        block_size: int = 64,
        num_global_tokens: int = 1,  # Usually [CLS] or first token
        num_random_blocks: int = 3,
    ):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = config.scale

        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.num_random_blocks = num_random_blocks

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_rotary_to_heads(q, k, cos, sin)

        # Create sparse attention mask
        sparse_mask = self._create_sparse_mask(seq_len, hidden_states.device)

        # Compute attention with sparse mask
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + sparse_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, None

    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create block-sparse attention mask."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)

        # 1. Global tokens attend to everything (and are attended by everything)
        mask[:self.num_global_tokens, :] = 0
        mask[:, :self.num_global_tokens] = 0

        # 2. Local attention (block diagonal)
        for i in range(seq_len):
            start = max(0, i - self.block_size // 2)
            end = min(seq_len, i + self.block_size // 2 + 1)
            mask[i, start:end] = 0

        # 3. Causal masking
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(causal.bool(), float('-inf'))

        return mask.unsqueeze(0).unsqueeze(0)

    def _apply_rotary_to_heads(self, q, k, cos, sin):
        """Apply rotary embeddings to attention heads."""
        # q, k: [batch, heads, seq, head_dim]
        # cos, sin: [1, seq, 1, head_dim] or similar
        cos = cos.transpose(1, 2)  # [1, 1, seq, head_dim]
        sin = sin.transpose(1, 2)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


# =============================================================================
# Linear Attention
# =============================================================================

@register_attention("linear")
@register_attention("linear_attention")
class LinearAttention(AttentionBase):
    """
    Linear Attention with kernel feature maps.

    Replaces softmax(QK^T)V with φ(Q)(φ(K)^T V) where φ is a feature map.
    This allows computing attention in O(N) instead of O(N²).

    Reference: "Transformers are RNNs" (Katharopoulos et al., 2020)
    """

    def __init__(self, config: AttentionConfig, feature_map: str = "elu"):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.feature_map = feature_map

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Epsilon for numerical stability
        self.eps = 1e-6

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply feature map (makes Q, K non-negative)
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Linear attention: O(N * D²) instead of O(N² * D)
        # For causal: use cumulative sum trick
        if past_key_value is not None:
            # Incremental decoding
            kv_state, k_sum_state = past_key_value

            # Update states
            kv_state = kv_state + torch.einsum('bhnd,bhnm->bhdm', k, v)
            k_sum_state = k_sum_state + k.sum(dim=2, keepdim=True)

            # Compute output
            num = torch.einsum('bhnd,bhdm->bhnm', q, kv_state)
            denom = torch.einsum('bhnd,bhnd->bhn', q, k_sum_state.expand_as(q)) + self.eps
            attn_output = num / denom.unsqueeze(-1)

            new_cache = (kv_state, k_sum_state) if use_cache else None
        else:
            # Full sequence (training)
            # Causal linear attention using cumulative sum
            attn_output = self._causal_linear_attention(q, k, v)

            # Compute final states for cache
            if use_cache:
                kv_state = torch.einsum('bhnd,bhnm->bhdm', k, v)
                k_sum_state = k.sum(dim=2, keepdim=True)
                new_cache = (kv_state, k_sum_state)
            else:
                new_cache = None

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, new_cache

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map to make values non-negative."""
        if self.feature_map == "elu":
            return F.elu(x) + 1
        elif self.feature_map == "relu":
            return F.relu(x)
        elif self.feature_map == "softmax":
            return F.softmax(x, dim=-1)
        else:
            return F.elu(x) + 1

    def _causal_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Causal linear attention using cumulative sum.

        Instead of computing full N×N attention, we maintain running states.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        v_dim = v.shape[-1]

        # Initialize running states
        kv_state = torch.zeros(batch_size, num_heads, head_dim, v_dim, device=q.device, dtype=q.dtype)
        k_sum = torch.zeros(batch_size, num_heads, head_dim, device=q.device, dtype=q.dtype)

        outputs = []

        for t in range(seq_len):
            # Get current timestep
            q_t = q[:, :, t, :]  # [batch, heads, head_dim]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]  # [batch, heads, v_dim]

            # Update running states
            kv_state = kv_state + torch.einsum('bhd,bhm->bhdm', k_t, v_t)
            k_sum = k_sum + k_t

            # Compute output
            num = torch.einsum('bhd,bhdm->bhm', q_t, kv_state)
            denom = torch.einsum('bhd,bhd->bh', q_t, k_sum) + self.eps
            out_t = num / denom.unsqueeze(-1)

            outputs.append(out_t)

        return torch.stack(outputs, dim=2)


# =============================================================================
# Multi-Scale Attention
# =============================================================================

@register_attention("multiscale")
class MultiScaleAttention(AttentionBase):
    """
    Multi-Scale Attention combining local and global patterns.

    Uses different attention mechanisms for different heads:
    - Some heads do local (sliding window) attention
    - Some heads do global attention
    - Some heads do strided attention
    """

    def __init__(
        self,
        config: AttentionConfig,
        local_heads: int = None,
        window_sizes: Tuple[int, ...] = (256, 512, 1024),
    ):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = config.scale

        # Divide heads into groups
        self.local_heads = local_heads or self.num_heads // 2
        self.global_heads = self.num_heads - self.local_heads
        self.window_sizes = window_sizes

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Split heads into local and global
        q_local = q[:, :self.local_heads]
        q_global = q[:, self.local_heads:]
        k_local = k[:, :self.local_heads]
        k_global = k[:, self.local_heads:]
        v_local = v[:, :self.local_heads]
        v_global = v[:, self.local_heads:]

        # Local attention with varying window sizes
        local_outputs = []
        heads_per_window = self.local_heads // len(self.window_sizes)

        for i, window_size in enumerate(self.window_sizes):
            start = i * heads_per_window
            end = start + heads_per_window if i < len(self.window_sizes) - 1 else self.local_heads

            q_w = q_local[:, start:end]
            k_w = k_local[:, start:end]
            v_w = v_local[:, start:end]

            # Sliding window mask
            mask = self._sliding_window_mask(seq_len, window_size, hidden_states.device)

            scores = torch.matmul(q_w, k_w.transpose(-2, -1)) * self.scale
            scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_w)
            local_outputs.append(out)

        local_output = torch.cat(local_outputs, dim=1)

        # Global attention (standard causal)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device),
            diagonal=1
        )
        scores = torch.matmul(q_global, k_global.transpose(-2, -1)) * self.scale
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)
        global_output = torch.matmul(attn, v_global)

        # Combine
        attn_output = torch.cat([local_output, global_output], dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        output = self.o_proj(attn_output)
        return output, None

    def _sliding_window_mask(self, seq_len: int, window_size: int, device: torch.device):
        """Create sliding window mask."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            mask[i, start:i+1] = 0
        return mask.unsqueeze(0).unsqueeze(0)
