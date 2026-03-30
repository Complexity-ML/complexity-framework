"""
Attention API - Factories pour créer des attention layers.
"""

from __future__ import annotations

from typing import Type
import torch.nn as nn

from complexity.core import (
    ATTENTION_REGISTRY,
    register_attention,
    AttentionBase,
    AttentionConfig,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)


class Attention:
    """
    Factory pour créer des attention layers.

    Usage:
        # Via factory
        attn = Attention.create("gqa", hidden_size=4096, num_heads=32, kv_heads=8)

        # Direct
        attn = Attention.gqa(hidden_size=4096, num_heads=32, kv_heads=8)
        attn = Attention.mha(hidden_size=4096, num_heads=32)
        attn = Attention.mqa(hidden_size=4096, num_heads=32)
    """

    TYPES = {
        "gqa": GroupedQueryAttention,
        "mha": MultiHeadAttention,
        "mqa": MultiQueryAttention,
    }

    @classmethod
    def create(cls, attention_type: str = "gqa", **kwargs) -> nn.Module:
        """
        Crée une attention layer.

        Args:
            attention_type: "gqa", "mha", "mqa"
            **kwargs: hidden_size, num_heads, kv_heads, dropout, ...
        """
        if attention_type in ATTENTION_REGISTRY._registry:
            attn_cls = ATTENTION_REGISTRY.get(attention_type)
            config = AttentionConfig(**kwargs)
            return attn_cls(config)

        if attention_type not in cls.TYPES:
            raise ValueError(f"Unknown attention type: {attention_type}. Use: {list(cls.TYPES.keys())}")

        attn_cls = cls.TYPES[attention_type]
        config = AttentionConfig(**kwargs)
        return attn_cls(config)

    @classmethod
    def gqa(cls, hidden_size: int, num_heads: int, kv_heads: int = None, **kwargs) -> nn.Module:
        """Grouped Query Attention."""
        return cls.create("gqa", hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=kv_heads or num_heads // 4, **kwargs)

    @classmethod
    def mha(cls, hidden_size: int, num_heads: int, **kwargs) -> nn.Module:
        """Multi-Head Attention."""
        return cls.create("mha", hidden_size=hidden_size, num_heads=num_heads, **kwargs)

    @classmethod
    def mqa(cls, hidden_size: int, num_heads: int, **kwargs) -> nn.Module:
        """Multi-Query Attention."""
        return cls.create("mqa", hidden_size=hidden_size, num_heads=num_heads, **kwargs)

    @classmethod
    def register(cls, name: str, attention_cls: Type):
        """Enregistre un nouveau type d'attention."""
        register_attention(name)(attention_cls)
        cls.TYPES[name] = attention_cls


# Aliases
GQA = GroupedQueryAttention
MHA = MultiHeadAttention
MQA = MultiQueryAttention

__all__ = [
    "Attention",
    "GQA",
    "MHA",
    "MQA",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "AttentionBase",
    "AttentionConfig",
]
