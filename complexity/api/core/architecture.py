"""
Architecture API - Architectures alternatives INL (O(N)).
"""

from __future__ import annotations

import torch.nn as nn

from complexity.core.architectures import (
    # Mamba (State Space Models)
    MambaBlock,
    MambaConfig,
    Mamba,
    # RWKV (Linear attention RNN)
    RWKVBlock,
    RWKVConfig,
    RWKV,
    # RetNet (Retentive Networks)
    RetNetBlock,
    RetNetConfig,
    RetNet,
    # Mixture of Depths
    MoDBlock,
    MoDConfig,
)


class Architecture:
    """
    Factory pour créer des architectures alternatives INL.

    Ces architectures offrent O(N) au lieu de O(N²) pour l'attention standard.

    Usage:
        # Mamba (State Space Models)
        model = Architecture.mamba(hidden_size=768, num_layers=12)

        # RWKV (Linear attention RNN)
        model = Architecture.rwkv(hidden_size=768, num_layers=12)

        # RetNet (Retentive Networks)
        model = Architecture.retnet(hidden_size=768, num_layers=12)

        # Blocks individuels
        block = Architecture.mamba_block(hidden_size=768)
        block = Architecture.rwkv_block(hidden_size=768)

        # Mixture of Depths
        block = Architecture.mod_block(hidden_size=768, capacity_factor=0.5)
    """

    TYPES = {
        "mamba": Mamba,
        "rwkv": RWKV,
        "retnet": RetNet,
    }

    BLOCK_TYPES = {
        "mamba": MambaBlock,
        "rwkv": RWKVBlock,
        "retnet": RetNetBlock,
        "mod": MoDBlock,
    }

    @classmethod
    def create(cls, arch_type: str = "mamba", **kwargs) -> nn.Module:
        """
        Crée un modèle complet avec architecture alternative.

        Args:
            arch_type: "mamba", "rwkv", "retnet"
            **kwargs: hidden_size, num_layers, ...
        """
        if arch_type not in cls.TYPES:
            raise ValueError(f"Unknown architecture: {arch_type}. Use: {list(cls.TYPES.keys())}")

        arch_cls = cls.TYPES[arch_type]

        config_cls = {
            "mamba": MambaConfig,
            "rwkv": RWKVConfig,
            "retnet": RetNetConfig,
        }[arch_type]

        config = config_cls(**kwargs)
        return arch_cls(config)

    @classmethod
    def block(cls, block_type: str = "mamba", **kwargs) -> nn.Module:
        """
        Crée un block individuel.

        Args:
            block_type: "mamba", "rwkv", "retnet", "mod"
            **kwargs: hidden_size, ...
        """
        if block_type not in cls.BLOCK_TYPES:
            raise ValueError(f"Unknown block type: {block_type}. Use: {list(cls.BLOCK_TYPES.keys())}")

        block_cls = cls.BLOCK_TYPES[block_type]

        config_cls = {
            "mamba": MambaConfig,
            "rwkv": RWKVConfig,
            "retnet": RetNetConfig,
            "mod": MoDConfig,
        }[block_type]

        config = config_cls(**kwargs)
        return block_cls(config)

    # ==================== Mamba ====================

    @classmethod
    def mamba(cls, hidden_size: int = 768, num_layers: int = 12, **kwargs) -> nn.Module:
        """
        Mamba - State Space Model.

        O(N) complexity, excellent pour séquences longues.
        """
        return cls.create("mamba", hidden_size=hidden_size, num_hidden_layers=num_layers, **kwargs)

    @classmethod
    def mamba_block(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """Block Mamba individuel."""
        return cls.block("mamba", hidden_size=hidden_size, **kwargs)

    # ==================== RWKV ====================

    @classmethod
    def rwkv(cls, hidden_size: int = 768, num_layers: int = 12, **kwargs) -> nn.Module:
        """
        RWKV - Linear attention RNN.

        Combine avantages des RNN et Transformers.
        """
        return cls.create("rwkv", hidden_size=hidden_size, num_hidden_layers=num_layers, **kwargs)

    @classmethod
    def rwkv_block(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """Block RWKV individuel."""
        return cls.block("rwkv", hidden_size=hidden_size, **kwargs)

    # ==================== RetNet ====================

    @classmethod
    def retnet(cls, hidden_size: int = 768, num_layers: int = 12, **kwargs) -> nn.Module:
        """
        RetNet - Retentive Networks.

        Training parallèle, inference O(1) par token.
        """
        return cls.create("retnet", hidden_size=hidden_size, num_hidden_layers=num_layers, **kwargs)

    @classmethod
    def retnet_block(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """Block RetNet individuel."""
        return cls.block("retnet", hidden_size=hidden_size, **kwargs)

    # ==================== Mixture of Depths ====================

    @classmethod
    def mod_block(cls, hidden_size: int = 768, capacity_factor: float = 0.5, **kwargs) -> nn.Module:
        """
        Mixture of Depths block.

        Sélectionne dynamiquement quels tokens passent par le block.
        """
        return cls.block("mod", hidden_size=hidden_size, capacity_factor=capacity_factor, **kwargs)


__all__ = [
    "Architecture",
    # Mamba
    "Mamba",
    "MambaBlock",
    "MambaConfig",
    # RWKV
    "RWKV",
    "RWKVBlock",
    "RWKVConfig",
    # RetNet
    "RetNet",
    "RetNetBlock",
    "RetNetConfig",
    # MoD
    "MoDBlock",
    "MoDConfig",
]
