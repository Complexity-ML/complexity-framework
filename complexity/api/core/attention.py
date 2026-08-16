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


class _AttentionOutputOnly(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention

    def forward(self, *args, **kwargs):
        out = self.attention(*args, **kwargs)
        return out[0] if isinstance(out, tuple) else out


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
    def _normalize_kwargs(cls, attention_type: str, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if "num_heads" in kwargs:
            kwargs["num_attention_heads"] = kwargs.pop("num_heads")
        if "kv_heads" in kwargs:
            kwargs["num_key_value_heads"] = kwargs.pop("kv_heads")
        if "num_kv_heads" in kwargs:
            kwargs["num_key_value_heads"] = kwargs.pop("num_kv_heads")
        if "dropout" in kwargs:
            kwargs["attention_dropout"] = kwargs.pop("dropout")
        if "num_key_value_heads" not in kwargs:
            heads = kwargs.get("num_attention_heads")
            if attention_type == "mqa":
                kwargs["num_key_value_heads"] = 1
            elif heads is not None:
                kwargs["num_key_value_heads"] = heads
        return kwargs

    @classmethod
    def create(cls, attention_type: str = "gqa", **kwargs) -> nn.Module:
        """
        Crée une attention layer.

        Args:
            attention_type: "gqa", "mha", "mqa"
            **kwargs: hidden_size, num_heads, kv_heads, dropout, ...
        """
        kwargs = cls._normalize_kwargs(attention_type, kwargs)
        if attention_type in ATTENTION_REGISTRY._registry:
            attn_cls = ATTENTION_REGISTRY.get(attention_type)
            config = AttentionConfig(**kwargs)
            return _AttentionOutputOnly(attn_cls(config))

        if attention_type not in cls.TYPES:
            raise ValueError(f"Unknown attention type: {attention_type}. Use: {list(cls.TYPES.keys())}")

        attn_cls = cls.TYPES[attention_type]
        config = AttentionConfig(**kwargs)
        return _AttentionOutputOnly(attn_cls(config))

    @classmethod
    def gqa(cls, hidden_size: int, num_heads: int, kv_heads: int = None, **kwargs) -> nn.Module:
        """Grouped Query Attention."""
        if "num_kv_heads" not in kwargs and "num_key_value_heads" not in kwargs:
            kwargs["num_kv_heads"] = kv_heads or max(1, num_heads // 4)
        return cls.create("gqa", hidden_size=hidden_size, num_heads=num_heads, **kwargs)

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
