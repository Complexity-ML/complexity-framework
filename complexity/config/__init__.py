"""
Configuration module for framework-complexity.

Usage:
    from complexity.config import ModelConfig, get_preset

    # Custom config
    config = ModelConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        attention_type="gqa",
        mlp_type="swiglu",
    )

    # Preset config
    config = get_preset("llama-7b")
    config = get_preset("complexity-7b")

    # Load from file
    config = ModelConfig.load("config.yaml")
"""

from .model_config import (
    ModelConfig,
    get_preset,
    PRESET_CONFIGS,
    # Preset functions
    llama_7b_config,
    llama_70b_config,
    mistral_7b_config,
    complexity_7b_config,
    gpt2_config,
)

__all__ = [
    "ModelConfig",
    "get_preset",
    "PRESET_CONFIGS",
    "llama_7b_config",
    "llama_70b_config",
    "mistral_7b_config",
    "complexity_7b_config",
    "gpt2_config",
]
