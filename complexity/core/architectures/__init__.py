"""
Alternative Architectures beyond standard Transformers.

Implements:
- Mamba (State Space Models)
- RWKV (Linear attention RNN)
- RetNet (Retentive Networks)
- Mixture of Depths (MoD)

These architectures offer O(N) complexity vs O(N²) for standard attention.
"""

from .mamba import MambaBlock, MambaConfig, Mamba
from .rwkv import RWKVBlock, RWKVConfig, RWKV
from .retnet import RetNetBlock, RetNetConfig, RetNet
from .mixture_of_depths import MoDBlock, MoDConfig

__all__ = [
    # Mamba
    "MambaBlock",
    "MambaConfig",
    "Mamba",
    # RWKV
    "RWKVBlock",
    "RWKVConfig",
    "RWKV",
    # RetNet
    "RetNetBlock",
    "RetNetConfig",
    "RetNet",
    # Mixture of Depths
    "MoDBlock",
    "MoDConfig",
]
