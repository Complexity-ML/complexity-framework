"""
AWQ: Activation-aware Weight Quantization.

AWQ observes that not all weights are equally important.
It protects salient weights by scaling them before quantization.

Key insight: 1% of salient weights can cause significant accuracy loss.
Solution: Scale weights based on activation magnitude.

Reference: https://arxiv.org/abs/2306.00978
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AWQConfig:
    """Configuration for AWQ quantization."""
    bits: int = 4                    # Quantization bits
    group_size: int = 128            # Group size
    zero_point: bool = True          # Use zero-point quantization
    version: str = "gemm"            # "gemm" or "gemv" kernel version


class AWQQuantizer:
    """
    AWQ Quantizer.

    Performs activation-aware weight quantization:
    1. Collect activation statistics
    2. Identify salient weights (high activation magnitude)
    3. Scale salient weights to protect them
    4. Quantize scaled weights
    5. Absorb scaling into adjacent layers

    Usage:
        quantizer = AWQQuantizer(config)
        quantized_model = quantizer.quantize(model, calibration_data)
    """

    def __init__(self, config: AWQConfig):
        self.config = config
        self.act_scales = {}

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Any,
    ) -> nn.Module:
        """
        Quantize model using AWQ.

        Args:
            model: Model to quantize
            calibration_loader: DataLoader with calibration data

        Returns:
            Quantized model
        """
        model.eval()
        device = next(model.parameters()).device

        # Step 1: Collect activation scales
        logger.info("Collecting activation statistics...")
        self._collect_activation_scales(model, calibration_loader, device)

        # Step 2: Compute optimal scaling factors
        logger.info("Computing optimal scales...")
        scales = self._compute_optimal_scales(model)

        # Step 3: Apply scaling and quantize
        logger.info("Quantizing weights...")
        self._apply_scaling_and_quantize(model, scales)

        return model

    def _collect_activation_scales(
        self,
        model: nn.Module,
        calibration_loader: Any,
        device: torch.device,
    ):
        """Collect activation magnitude statistics."""
        hooks = []
        act_sums = {}
        act_counts = {}

        def hook_fn(name):
            def fn(module, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]

                # Compute mean absolute activation per channel
                if len(inp.shape) == 3:  # [batch, seq, hidden]
                    act_mag = inp.abs().mean(dim=(0, 1))
                else:
                    act_mag = inp.abs().mean(dim=0)

                if name not in act_sums:
                    act_sums[name] = act_mag.detach()
                    act_counts[name] = 1
                else:
                    act_sums[name] += act_mag.detach()
                    act_counts[name] += 1

            return fn

        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Run calibration
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device)
                else:
                    input_ids = batch.to(device)

                model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute average scales
        for name in act_sums:
            self.act_scales[name] = act_sums[name] / act_counts[name]

    def _compute_optimal_scales(
        self,
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute optimal scaling factors for each layer.

        The scale protects salient weights (those with high activation).
        """
        scales = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.act_scales:
                act_scale = self.act_scales[name]
                weight = module.weight.data

                # Compute weight scale per input channel
                weight_scale = weight.abs().max(dim=0).values

                # AWQ scaling: s = act^alpha * w^(1-alpha), alpha=0.5
                alpha = 0.5
                s = (act_scale ** alpha) * (weight_scale ** (1 - alpha))

                # Normalize scale
                s = s / s.mean()

                scales[name] = s.clamp(min=1e-4)

        return scales

    def _apply_scaling_and_quantize(
        self,
        model: nn.Module,
        scales: Dict[str, torch.Tensor],
    ):
        """Apply scaling and quantize weights."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in scales:
                scale = scales[name]
                weight = module.weight.data

                # Scale weights: W_scaled = W / s
                # This makes salient weights smaller (easier to quantize)
                weight_scaled = weight / scale.unsqueeze(0)

                # Quantize
                quant_weight, w_scale, w_zeros = self._quantize_weight(weight_scaled)

                # Replace layer
                self._replace_with_quantized(
                    model, name, module, quant_weight, w_scale, w_zeros, scale
                )

    def _quantize_weight(self, weight: torch.Tensor):
        """Quantize weight tensor per group."""
        out_features, in_features = weight.shape
        group_size = self.config.group_size

        # Reshape for groups
        num_groups = (in_features + group_size - 1) // group_size
        if in_features % group_size != 0:
            pad = group_size - (in_features % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad))

        weight = weight.view(out_features, num_groups, group_size)

        # Compute scale and zero per group
        if self.config.zero_point:
            w_min = weight.min(dim=-1, keepdim=True).values
            w_max = weight.max(dim=-1, keepdim=True).values
            scale = (w_max - w_min) / (2 ** self.config.bits - 1)
            zeros = -w_min / (scale + 1e-8)
        else:
            w_max = weight.abs().max(dim=-1, keepdim=True).values
            scale = w_max / (2 ** (self.config.bits - 1) - 1)
            zeros = torch.zeros_like(scale)

        # Quantize
        if self.config.zero_point:
            quant = torch.clamp(
                torch.round(weight / (scale + 1e-8) + zeros),
                0, 2 ** self.config.bits - 1
            )
        else:
            quant = torch.clamp(
                torch.round(weight / (scale + 1e-8)),
                -(2 ** (self.config.bits - 1)),
                2 ** (self.config.bits - 1) - 1
            )

        return (
            quant.to(torch.int8).view(out_features, -1)[:, :in_features],
            scale.squeeze(-1),
            zeros.squeeze(-1),
        )

    def _replace_with_quantized(
        self,
        model: nn.Module,
        name: str,
        layer: nn.Linear,
        quant_weight: torch.Tensor,
        scale: torch.Tensor,
        zeros: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        """Replace original layer with AWQ quantized version."""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Create AWQ layer
        awq_layer = AWQLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            bits=self.config.bits,
            group_size=self.config.group_size,
        )

        awq_layer.set_weights(quant_weight, scale, zeros, input_scale)
        if layer.bias is not None:
            awq_layer.bias = layer.bias

        setattr(parent, parts[-1], awq_layer)


class AWQLinear(nn.Module):
    """
    AWQ-quantized linear layer.

    Stores quantized weights and scales for efficient inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        num_groups = (in_features + group_size - 1) // group_size

        # Quantized weights (packed if 4-bit)
        if bits == 4:
            self.register_buffer(
                'qweight',
                torch.zeros(out_features, in_features // 2, dtype=torch.int8)
            )
        else:
            self.register_buffer(
                'qweight',
                torch.zeros(out_features, in_features, dtype=torch.int8)
            )

        # Quantization parameters
        self.register_buffer('scales', torch.zeros(out_features, num_groups))
        self.register_buffer('zeros', torch.zeros(out_features, num_groups))

        # Input scaling factor (from AWQ)
        self.register_buffer('input_scale', torch.ones(in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def set_weights(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        """Set quantized weights and scales."""
        if self.bits == 4:
            # Pack to int8
            packed = (qweight[:, ::2] << 4) | (qweight[:, 1::2] & 0x0F)
            self.qweight.copy_(packed)
        else:
            self.qweight.copy_(qweight)

        self.scales.copy_(scales)
        self.zeros.copy_(zeros)
        self.input_scale.copy_(input_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        # Scale input (reverse the AWQ scaling)
        x = x * self.input_scale

        # Dequantize weights
        weight = self._dequantize()

        return torch.nn.functional.linear(x, weight, self.bias)

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights."""
        if self.bits == 4:
            # Unpack
            low = self.qweight & 0x0F
            high = (self.qweight >> 4) & 0x0F
            q = torch.stack([high, low], dim=-1).view(self.out_features, -1).float()
        else:
            q = self.qweight.float()

        # Dequantize per group
        q = q.view(self.out_features, -1, self.group_size)
        scales = self.scales.unsqueeze(-1)
        zeros = self.zeros.unsqueeze(-1)

        weight = (q - zeros) * scales

        return weight.view(self.out_features, -1)[:, :self.in_features]


def awq_quantize(
    model: nn.Module,
    calibration_loader: Any,
    bits: int = 4,
    group_size: int = 128,
) -> nn.Module:
    """
    Convenience function for AWQ quantization.

    Args:
        model: Model to quantize
        calibration_loader: Calibration data
        bits: Quantization bits
        group_size: Group size

    Returns:
        Quantized model
    """
    config = AWQConfig(bits=bits, group_size=group_size)
    quantizer = AWQQuantizer(config)
    return quantizer.quantize(model, calibration_loader)
