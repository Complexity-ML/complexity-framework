"""
Alternative Architectures beyond standard Transformers.

Implements:
- Mamba (State Space Models)
- RWKV (Linear attention RNN)
- RetNet (Retentive Networks)
These architectures offer O(N) complexity vs O(N²) for standard attention.
"""

from .mamba import MambaBlock, MambaConfig, Mamba
from .rwkv import RWKVBlock, RWKVConfig, RWKV
from .retnet import RetNetBlock, RetNetConfig, RetNet

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
]
