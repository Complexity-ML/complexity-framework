"""
Core Quantization utilities.

Provides basic quantization methods:
- Dynamic quantization (quantize weights at runtime)
- Static quantization (quantize with calibration data)
- Mixed precision quantization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Set, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Quantization methods."""
    DYNAMIC = "dynamic"      # Quantize at runtime
    STATIC = "static"        # Quantize with calibration
    GPTQ = "gptq"           # Gradient-based post-training quantization
    AWQ = "awq"             # Activation-aware weight quantization
    GGUF = "gguf"           # llama.cpp format


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    bits: int = 8                          # Quantization bits (4 or 8)
    method: str = "dynamic"                # Quantization method
    group_size: int = 128                  # Group size for per-group quantization
    sym: bool = True                       # Symmetric quantization
    exclude_layers: Optional[List[str]] = None  # Layers to keep in full precision
    calibration_samples: int = 128         # Samples for calibration
    desc_act: bool = True                  # Descending activation order (for GPTQ)


def quantize_model(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    bits: int = 8,
    method: str = "dynamic",
    **kwargs,
) -> nn.Module:
    """
    Quantize a model.

    Args:
        model: Model to quantize
        config: Quantization configuration
        bits: Number of bits (4 or 8)
        method: Quantization method
        **kwargs: Additional arguments

    Returns:
        Quantized model
    """
    if config is None:
        config = QuantizationConfig(bits=bits, method=method, **kwargs)

    if config.method == "dynamic":
        quantizer = DynamicQuantizer(config)
    elif config.method == "static":
        quantizer = StaticQuantizer(config)
    else:
        raise ValueError(f"Use specific quantizer for {config.method}")

    return quantizer.quantize(model)


class DynamicQuantizer:
    """
    Dynamic quantization (weights quantized, activations at runtime).

    Pros:
    - No calibration data needed
    - Simple to use
    - Good for CPU inference

    Cons:
    - Less accurate than static
    - Limited GPU support
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        exclude = set(self.config.exclude_layers or [])

        # Modules to quantize
        modules_to_quantize = {nn.Linear}

        if self.config.bits == 8:
            # Use PyTorch native dynamic quantization
            quantized = torch.quantization.quantize_dynamic(
                model,
                modules_to_quantize,
                dtype=torch.qint8,
            )
        else:
            # For 4-bit, use custom quantization
            quantized = self._quantize_4bit(model, exclude)

        logger.info(f"Applied {self.config.bits}-bit dynamic quantization")
        return quantized

    def _quantize_4bit(self, model: nn.Module, exclude: Set[str]) -> nn.Module:
        """Apply 4-bit quantization."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name not in exclude:
                # Replace with quantized version
                parent = model
                parts = name.split('.')
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                quantized_linear = Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    group_size=self.config.group_size,
                )
                quantized_linear.quantize_weight(module.weight.data)
                if module.bias is not None:
                    quantized_linear.bias = module.bias

                setattr(parent, parts[-1], quantized_linear)

        return model


class StaticQuantizer:
    """
    Static quantization (quantize with calibration data).

    Pros:
    - More accurate than dynamic
    - Better performance
    - Good for both CPU and GPU

    Cons:
    - Requires calibration data
    - More complex setup
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
    ) -> nn.Module:
        """
        Apply static quantization.

        Args:
            model: Model to quantize
            calibration_loader: DataLoader for calibration

        Returns:
            Quantized model
        """
        model.eval()

        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Prepare
        torch.quantization.prepare(model, inplace=True)

        # Calibrate
        if calibration_loader is not None:
            with torch.no_grad():
                for batch in calibration_loader:
                    if isinstance(batch, dict):
                        model(batch['input_ids'])
                    else:
                        model(batch)

                    if self.config.calibration_samples > 0:
                        self.config.calibration_samples -= 1
                        if self.config.calibration_samples == 0:
                            break

        # Convert
        torch.quantization.convert(model, inplace=True)

        logger.info("Applied static quantization")
        return model


# =============================================================================
# Low-bit Linear Layers
# =============================================================================

class Linear4bit(nn.Module):
    """
    4-bit quantized linear layer.

    Uses per-group quantization for accuracy.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Number of groups
        self.num_groups = (in_features + group_size - 1) // group_size

        # Quantized weights stored as int8 (2 int4 values packed)
        self.register_buffer(
            'weight_packed',
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8)
        )

        # Scales per group
        self.register_buffer(
            'scales',
            torch.ones(out_features, self.num_groups, dtype=torch.float16)
        )

        # Zero points per group
        self.register_buffer(
            'zeros',
            torch.zeros(out_features, self.num_groups, dtype=torch.float16)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize_weight(self, weight: torch.Tensor):
        """
        Quantize weight tensor to 4-bit.

        Args:
            weight: [out_features, in_features] float tensor
        """
        out_features, in_features = weight.shape

        # Pad to group boundary
        if in_features % self.group_size != 0:
            pad_size = self.group_size - (in_features % self.group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))

        # Reshape for per-group quantization
        weight = weight.view(out_features, -1, self.group_size)

        # Compute scales and zeros per group
        w_min = weight.min(dim=-1, keepdim=True).values
        w_max = weight.max(dim=-1, keepdim=True).values

        scales = (w_max - w_min) / 15  # 4-bit: 0-15
        zeros = -w_min / (scales + 1e-8)

        # Quantize
        weight_quant = torch.clamp(
            torch.round((weight - w_min) / (scales + 1e-8)),
            0, 15
        ).to(torch.uint8)

        # Pack two 4-bit values into one uint8
        weight_quant = weight_quant.view(out_features, -1)
        weight_packed = (weight_quant[:, ::2] << 4) | weight_quant[:, 1::2]

        # Store
        self.weight_packed.copy_(weight_packed[:, :self.weight_packed.size(1)])
        self.scales.copy_(scales.squeeze(-1).half())
        self.zeros.copy_(zeros.squeeze(-1).half())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        # Dequantize weights
        weight = self._dequantize()

        # Linear operation
        output = torch.nn.functional.linear(x, weight[:, :self.in_features], self.bias)

        return output

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights to float."""
        # Unpack 4-bit values
        weight_low = self.weight_packed & 0x0F
        weight_high = self.weight_packed >> 4

        # Interleave
        weight = torch.stack([weight_high, weight_low], dim=-1).view(
            self.out_features, -1
        ).float()

        # Reshape for groups
        weight = weight.view(self.out_features, -1, self.group_size)

        # Dequantize
        scales = self.scales.unsqueeze(-1).float()
        zeros = self.zeros.unsqueeze(-1).float()

        weight = (weight - zeros) * scales

        return weight.view(self.out_features, -1)


class Linear8bit(nn.Module):
    """
    8-bit quantized linear layer.

    More accurate than 4-bit, still significant memory savings.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Quantized weights
        self.register_buffer(
            'weight_quant',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Per-channel scale
        self.register_buffer(
            'scale',
            torch.ones(out_features, dtype=torch.float32)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight tensor to 8-bit."""
        # Per-channel scale
        scale = weight.abs().max(dim=1).values / 127
        scale = scale.clamp(min=1e-8)

        # Quantize
        weight_quant = torch.clamp(
            torch.round(weight / scale.unsqueeze(1)),
            -128, 127
        ).to(torch.int8)

        self.weight_quant.copy_(weight_quant)
        self.scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        # Dequantize
        weight = self.weight_quant.float() * self.scale.unsqueeze(1)

        return torch.nn.functional.linear(x, weight, self.bias)


# =============================================================================
# Utility Functions
# =============================================================================

def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size in MB.

    Returns dict with total, parameters, and buffers sizes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "total_mb": (param_size + buffer_size) / (1024 ** 2),
        "params_mb": param_size / (1024 ** 2),
        "buffers_mb": buffer_size / (1024 ** 2),
    }


def compare_outputs(
    original: nn.Module,
    quantized: nn.Module,
    sample_input: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare outputs between original and quantized models.

    Returns metrics like MSE, max error, etc.
    """
    original.eval()
    quantized.eval()

    with torch.no_grad():
        out_orig = original(sample_input)
        out_quant = quantized(sample_input)

        if isinstance(out_orig, dict):
            out_orig = out_orig['logits']
            out_quant = out_quant['logits']

        mse = ((out_orig - out_quant) ** 2).mean().item()
        max_err = (out_orig - out_quant).abs().max().item()
        cosine = torch.nn.functional.cosine_similarity(
            out_orig.flatten().unsqueeze(0),
            out_quant.flatten().unsqueeze(0),
        ).item()

    return {
        "mse": mse,
        "max_error": max_err,
        "cosine_similarity": cosine,
    }
