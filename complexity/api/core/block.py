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

    This framework is scoped to TR-Hash MoE; dense (Llama-style SwiGLU,
    GPT-style standard MLP) block factories were removed and will return
    as explicit comparison baselines later.

    Usage:
        block = Block.create(hidden_size=4096, num_heads=32, mlp_type="token_routed")
        block = Block.moe(hidden_size=4096, num_heads=32, num_experts=8)
    """

    @classmethod
    def create(cls, **kwargs) -> nn.Module:
        """Crée un transformer block."""
        config = ModelConfig(**kwargs)
        return TransformerBlock(config, layer_idx=kwargs.get("layer_idx", 0))

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
