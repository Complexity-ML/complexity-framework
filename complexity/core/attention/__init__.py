"""
Attention implementations for framework-complexity.

Available attention types:
- gqa / grouped_query: Grouped Query Attention (Llama 2/3 style)
- mha / multi_head: Standard Multi-Head Attention
- mqa / multi_query: Multi-Query Attention (single KV head)

Usage:
    from complexity.core.attention import GroupedQueryAttention, AttentionConfig
    from complexity.core.registry import ATTENTION_REGISTRY

    # Direct instantiation
    config = AttentionConfig(hidden_size=768, num_attention_heads=12, num_key_value_heads=4)
    attn = GroupedQueryAttention(config)

    # Via registry
    attn = ATTENTION_REGISTRY.build("gqa", config)
"""

from .base import AttentionBase, AttentionConfig
from .gqa import GroupedQueryAttention, MultiHeadAttention, MultiQueryAttention

__all__ = [
    "AttentionBase",
    "AttentionConfig",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
]
