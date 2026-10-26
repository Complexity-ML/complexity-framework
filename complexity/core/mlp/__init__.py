"""Canonical TR-HASH MLP and independent generic MLP components.

TRHashEngineMLP always uses its shared expert and four routed experts, with
fixed equal top-two weights. Dense and lexical classes remain standalone
components; they do not select another TR-HASH architecture. Obsolete
routing transformations are rejected explicitly.
"""

from .base import MLPBase, MLPConfig
from .dense_deterministic import DenseDeterministicMLP
from .deterministic_init import deterministic_gaussian_init_
from .lexical_channel_modulated import LexicalChannelModulatedMLP
from .lexical_modulated import LexicalModulatedMLP
from .lexical_object_micro_expert import LexicalObjectMicroExpertMLP
from .tr_hash_engine import TRHashEngineMLP

__all__ = [
    "MLPBase",
    "MLPConfig",
    "TRHashEngineMLP",
    "DenseDeterministicMLP",
    "LexicalModulatedMLP",
    "LexicalChannelModulatedMLP",
    "LexicalObjectMicroExpertMLP",
    "deterministic_gaussian_init_",
]
