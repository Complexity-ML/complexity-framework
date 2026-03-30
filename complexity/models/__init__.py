"""
Models module for framework-complexity.

Usage:
    from complexity.models import ComplexityModel
    from complexity.config import ModelConfig

    # Create from config
    config = ModelConfig(hidden_size=768, num_hidden_layers=12)
    model = ComplexityModel(config)

    # Create from preset
    model = ComplexityModel.from_preset("llama-7b")
    model = ComplexityModel.from_preset("complexity-7b")

    # Forward pass
    outputs = model(input_ids)
    logits = outputs["logits"]

    # Generation
    output_ids = model.generate(input_ids, max_new_tokens=100)
"""

from .block import TransformerBlock
from .builder import ComplexityModel

__all__ = [
    "TransformerBlock",
    "ComplexityModel",
]
