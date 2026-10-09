"""
Quantization module for framework-complexity.

Provides model quantization for efficient inference:
- INT8 quantization (dynamic and static)
- INT4 quantization (GPTQ, AWQ)
- Mixed precision quantization
- GGUF export for llama.cpp

Usage:
    from complexity.quantization import (
        quantize_model,
        QuantizationConfig,
        GPTQQuantizer,
        AWQQuantizer,
    )

    # Simple quantization
    quantized = quantize_model(model, bits=4, method="gptq")

    # Export to GGUF
    export_gguf(model, "model.gguf", quantization="Q4_K_M")
"""

from .awq import (
    AWQConfig,
    AWQQuantizer,
)
from .gguf_export import (
    GGUFQuantType,
    GGUFWriter,
    export_gguf,
)
from .gptq import (
    GPTQConfig,
    GPTQQuantizer,
)
from .quantizer import (
    DynamicQuantizer,
    QuantizationConfig,
    StaticQuantizer,
    quantize_model,
)

__all__ = [
    # Core
    "QuantizationConfig",
    "quantize_model",
    "DynamicQuantizer",
    "StaticQuantizer",
    # GPTQ
    "GPTQQuantizer",
    "GPTQConfig",
    # AWQ
    "AWQQuantizer",
    "AWQConfig",
    # GGUF
    "export_gguf",
    "GGUFWriter",
    "GGUFQuantType",
]
