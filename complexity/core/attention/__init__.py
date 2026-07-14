"""
Sequence-mixer implementations for framework-complexity.

The historical package and registry names remain ``attention`` for backward
compatibility, but registered modules are not required to use attention.

Available sequence-mixer types:
- gqa / grouped_query: Grouped Query Attention (Llama 2/3 style)
- mha / multi_head: Standard Multi-Head Attention
- mqa / multi_query: Multi-Query Attention (single KV head)
- causal_conv / lexical_object_conv: attention-free dilated causal convolution
- causal_state_conv: attention-free causal convolution with persistent state

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
from .lexical_gqa import LexicalBiasGQA
from .lexical_key_gqa import LexicalKeyGQA
from .lexical_wrv import LexicalWRVAttention
from .routed_gqa import RoutedGQA
from .i64_attention import I64Attention
from .causal_conv import CausalConvMixer
from .causal_state_conv import CausalStateConvMixer
from .causal_fast_weight_conv import CausalFastWeightConvMixer

__all__ = [
    "AttentionBase",
    "AttentionConfig",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "LexicalBiasGQA",
    "LexicalKeyGQA",
    "LexicalWRVAttention",
    "RoutedGQA",
    "I64Attention",
    "CausalConvMixer",
    "CausalStateConvMixer",
    "CausalFastWeightConvMixer",
]
