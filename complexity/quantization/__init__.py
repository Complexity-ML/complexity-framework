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

from .quantizer import (
    QuantizationConfig,
    quantize_model,
    DynamicQuantizer,
    StaticQuantizer,
)

from .gptq import (
    GPTQQuantizer,
    GPTQConfig,
)

from .awq import (
    AWQQuantizer,
    AWQConfig,
)

from .gguf_export import (
    export_gguf,
    GGUFWriter,
    GGUFQuantType,
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
