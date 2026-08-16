"""
Configuration module for framework-complexity.

Usage:
    from complexity.config import ModelConfig, get_preset

    # Custom config
    config = ModelConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
    )

    # Preset config
    config = get_preset("complexity-tiny")
    config = get_preset("complexity-7b")

    # Load from file
    config = ModelConfig.load("config.yaml")
"""

from .model_config import (
    ModelConfig,
    get_preset,
    PRESET_CONFIGS,
    # Preset functions
    complexity_tiny_config,
    complexity_small_config,
    complexity_base_config,
    complexity_large_config,
    complexity_xl_config,
    complexity_7b_config,
)

__all__ = [
    "ModelConfig",
    "get_preset",
    "PRESET_CONFIGS",
    "complexity_tiny_config",
    "complexity_small_config",
    "complexity_base_config",
    "complexity_large_config",
    "complexity_xl_config",
    "complexity_7b_config",
]
