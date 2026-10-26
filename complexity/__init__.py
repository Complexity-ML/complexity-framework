"""Complexity Framework: TR-Hash model research in PyTorch.

The canonical interfaces are ``ModelConfig`` and ``ComplexityModel``. Training
uses ``complexity.training.Trainer`` / ``TrainingConfig``; text generation uses
an external vLLM or SGLang endpoint via ``complexity.inference``.

    from complexity import ComplexityModel, ModelConfig
    model = ComplexityModel(ModelConfig())
    # This creates fresh weights. Load a saved model with from_pretrained(path).

Public names are loaded on demand, keeping package metadata usable before the
selected PyTorch backend and optional research dependencies are installed.
"""

from importlib import import_module as _import_module

from ._version import __version__

__author__ = "Complexity-ML"

_EXPORTS = {
    "ModelConfig": ("complexity.config", "ModelConfig"),
    "get_preset": ("complexity.config", "get_preset"),
    "PRESET_CONFIGS": ("complexity.config", "PRESET_CONFIGS"),
    "ComplexityModel": ("complexity.models", "ComplexityModel"),
    "TransformerBlock": ("complexity.models", "TransformerBlock"),
    "ATTENTION_REGISTRY": ("complexity.core.registry", "ATTENTION_REGISTRY"),
    "MLP_REGISTRY": ("complexity.core.registry", "MLP_REGISTRY"),
    "NORMALIZATION_REGISTRY": ("complexity.core.registry", "NORMALIZATION_REGISTRY"),
    "POSITION_REGISTRY": ("complexity.core.registry", "POSITION_REGISTRY"),
    "MODEL_REGISTRY": ("complexity.core.registry", "MODEL_REGISTRY"),
    "register_attention": ("complexity.core.registry", "register_attention"),
    "register_mlp": ("complexity.core.registry", "register_mlp"),
    "register_normalization": ("complexity.core.registry", "register_normalization"),
    "register_position": ("complexity.core.registry", "register_position"),
    "register_model": ("complexity.core.registry", "register_model"),
    "AttentionBase": ("complexity.core.attention", "AttentionBase"),
    "AttentionConfig": ("complexity.core.attention", "AttentionConfig"),
    "GroupedQueryAttention": ("complexity.core.attention", "GroupedQueryAttention"),
    "MultiHeadAttention": ("complexity.core.attention", "MultiHeadAttention"),
    "MultiQueryAttention": ("complexity.core.attention", "MultiQueryAttention"),
    "MLPBase": ("complexity.core.mlp", "MLPBase"),
    "MLPConfig": ("complexity.core.mlp", "MLPConfig"),
    "TRHashEngineMLP": ("complexity.core.mlp", "TRHashEngineMLP"),
    "TRHashEngine": ("complexity.tr_hash", "TRHashEngine"),
    "TRHashEngineConfig": ("complexity.tr_hash", "TRHashEngineConfig"),
    "RMSNorm": ("complexity.core.normalization", "RMSNorm"),
    "LayerNorm": ("complexity.core.normalization", "LayerNorm"),
    "RotaryEmbedding": ("complexity.core.position", "RotaryEmbedding"),
    "StandardRoPE": ("complexity.core.position", "StandardRoPE"),
    "YaRNRoPE": ("complexity.core.position", "YaRNRoPE"),
    "DynamicNTKRoPE": ("complexity.core.position", "DynamicNTKRoPE"),
    "apply_rotary_pos_emb": ("complexity.core.position", "apply_rotary_pos_emb"),
}

__all__ = [
    "__version__",
    "ModelConfig",
    "get_preset",
    "PRESET_CONFIGS",
    "ComplexityModel",
    "TransformerBlock",
    "ATTENTION_REGISTRY",
    "MLP_REGISTRY",
    "NORMALIZATION_REGISTRY",
    "POSITION_REGISTRY",
    "MODEL_REGISTRY",
    "register_attention",
    "register_mlp",
    "register_normalization",
    "register_position",
    "register_model",
    "AttentionBase",
    "AttentionConfig",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "MLPBase",
    "MLPConfig",
    "TRHashEngineMLP",
    "TRHashEngine",
    "TRHashEngineConfig",
    "RMSNorm",
    "LayerNorm",
    "RotaryEmbedding",
    "StandardRoPE",
    "YaRNRoPE",
    "DynamicNTKRoPE",
    "apply_rotary_pos_emb",
]


def __getattr__(name):
    """Load only the public feature requested by the caller."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(_import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
