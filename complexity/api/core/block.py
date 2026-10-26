"""
Block API - Factories pour créer des transformer blocks complets.
"""

from __future__ import annotations

import torch.nn as nn

from complexity.config import ModelConfig
from complexity.models import ComplexityModel, TransformerBlock


class Block:
    """
    Factory pour créer des transformer blocks complets.

    TR-HASH blocks use four query heads, one key/value head, and two
    equally weighted routes among four experts beside a shared SwiGLU.

    Usage:
        block = Block.create(hidden_size=256, num_attention_heads=4)
        block = Block.moe(hidden_size=256)
    """

    @classmethod
    def create(cls, **kwargs) -> nn.Module:
        """Crée un transformer block."""
        layer_idx = kwargs.pop("layer_idx", 0)
        config = ModelConfig(**kwargs)
        return TransformerBlock(config, layer_idx=layer_idx)

    @classmethod
    def moe(
        cls,
        hidden_size: int,
        num_heads: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        **kwargs,
    ) -> nn.Module:
        """Build the canonical shared-plus-four-expert TR-HASH block."""
        options = dict(kwargs)
        mlp_type = options.pop("mlp_type", "tr_hash_engine")
        if mlp_type not in {"tr_hash_engine", "tr_hash_moe"}:
            raise ValueError("Block.moe requires the canonical TR-HASH engine")
        if options.get("num_attention_heads", num_heads) != num_heads:
            raise ValueError("num_heads and num_attention_heads must agree")
        options.update(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            mlp_type="tr_hash_engine",
            num_experts=num_experts,
            top_k=top_k,
        )
        options.setdefault("num_key_value_heads", 1)
        options.setdefault("norm_type", "rmsnorm")
        return cls.create(**options)


__all__ = [
    "Block",
    "TransformerBlock",
    "ComplexityModel",
    "ModelConfig",
]
