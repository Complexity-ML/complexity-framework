"""
MLP implementations for framework-complexity.

This framework is scoped to TR-Hash MoE (deterministic token-ID / hash-table
routing). Standalone dense MLPs, the learned-router MoE, the i64 precision
variants, and the historical TokenRoutedMLP dispatch implementation were
removed to keep the framework TR-Hash-only. Dense/learned-router will be
reintroduced later as explicit comparison baselines. Existing
``token_routed``-format checkpoints load via
``complexity.utils.token_routed_conversion.convert_token_routed_checkpoint_dir``.

Available MLP types:
- dense_deterministic: SwiGLU with deterministic (RNG-free) init
- tr_hash_engine / tr_hash_moe: TR-Hash engine MoE (canonical, default)

Usage:
    from complexity.core.mlp import TRHashEngineMLP, MLPConfig
    from complexity.core.registry import MLP_REGISTRY

    # TR-Hash MoE
    config = MLPConfig(hidden_size=768, intermediate_size=3072, num_experts=4)
    moe = MLP_REGISTRY.build("tr_hash_engine", config)

"""

from .base import MLPBase, MLPConfig
from .tr_hash_engine import TRHashEngineMLP
from .dense_deterministic import DenseDeterministicMLP
from .lexical_modulated import LexicalModulatedMLP
from .lexical_channel_modulated import LexicalChannelModulatedMLP
from .lexical_object_micro_expert import LexicalObjectMicroExpertMLP
from .deterministic_init import deterministic_gaussian_init_

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
