"""
Block API - Factories pour créer des transformer blocks complets.
"""

from __future__ import annotations

import torch.nn as nn

from complexity.models import TransformerBlock, ComplexityModel
from complexity.config import ModelConfig


class Block:
    """
    Factory pour créer des transformer blocks complets.

    Usage:
        block = Block.create(hidden_size=4096, num_heads=32, mlp_type="swiglu")
        block = Block.llama(hidden_size=4096, num_heads=32, kv_heads=8)
        block = Block.gpt(hidden_size=768, num_heads=12)
    """

    @classmethod
    def create(cls, **kwargs) -> nn.Module:
        """Crée un transformer block."""
        config = ModelConfig(**kwargs)
        return TransformerBlock(config, layer_idx=kwargs.get("layer_idx", 0))

    @classmethod
    def llama(cls, hidden_size: int, num_heads: int, kv_heads: int = None, intermediate_size: int = None, **kwargs) -> nn.Module:
        """Llama-style block (RMSNorm + GQA + SwiGLU)."""
        return cls.create(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_kv_heads=kv_heads or num_heads // 4,
            intermediate_size=intermediate_size or int(hidden_size * 8 / 3),
            attention_type="gqa",
            mlp_type="swiglu",
            norm_type="rmsnorm",
            position_type="rope",
            **kwargs
        )

    @classmethod
    def gpt(cls, hidden_size: int, num_heads: int, intermediate_size: int = None, **kwargs) -> nn.Module:
        """GPT-style block (LayerNorm + MHA + Standard MLP)."""
        return cls.create(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_heads,
            intermediate_size=intermediate_size or hidden_size * 4,
            attention_type="mha",
            mlp_type="standard",
            norm_type="layernorm",
            position_type="learned",
            **kwargs
        )

    @classmethod
    def moe(cls, hidden_size: int, num_heads: int, num_experts: int = 8, top_k: int = 2, **kwargs) -> nn.Module:
        """MoE block (RMSNorm + GQA + MoE)."""
        return cls.create(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            mlp_type="moe",
            num_experts=num_experts,
            moe_top_k=top_k,
            norm_type="rmsnorm",
            **kwargs
        )


__all__ = [
    "Block",
    "TransformerBlock",
    "ComplexityModel",
    "ModelConfig",
]
