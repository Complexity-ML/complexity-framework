"""
GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

GPTQ uses approximate second-order information to minimize quantization error.
It quantizes weights one column at a time, updating remaining weights to
compensate for the error.

Reference: https://arxiv.org/abs/2210.17323
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization."""
    bits: int = 4                    # Quantization bits
    group_size: int = 128            # Group size for per-group quantization
    damp_percent: float = 0.01       # Dampening percentage for Hessian
    desc_act: bool = True            # Descending activation order
    sym: bool = True                 # Symmetric quantization
    true_sequential: bool = True     # Quantize layers sequentially
    static_groups: bool = False      # Static group assignment


class GPTQQuantizer:
    """
    GPTQ Quantizer.

    Quantizes model weights to 4-bit using second-order optimization
    to minimize output error.

    Usage:
        quantizer = GPTQQuantizer(config)
        quantized_model = quantizer.quantize(model, calibration_data)
    """

    def __init__(self, config: GPTQConfig):
        self.config = config
        self.quantizers = {}

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Any,
        layer_names: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Quantize model using GPTQ.

        Args:
            model: Model to quantize
            calibration_loader: DataLoader with calibration data
            layer_names: Specific layers to quantize (None = all linear)

        Returns:
            Quantized model
        """
        model.eval()
        device = next(model.parameters()).device

        # Find all linear layers
        layers_to_quantize = self._find_layers(model, layer_names)

        logger.info(f"Quantizing {len(layers_to_quantize)} layers")

        # Collect calibration data
        inps, attention_mask = self._collect_inputs(
            model, calibration_loader, device
        )

        # Quantize each layer
        if self.config.true_sequential:
            self._quantize_sequential(model, layers_to_quantize, inps, attention_mask)
        else:
            self._quantize_parallel(model, layers_to_quantize, inps, attention_mask)

        return model

    def _find_layers(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, nn.Linear]:
        """Find layers to quantize."""
        layers = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if layer_names is None or name in layer_names:
                    layers[name] = module

        return layers

    def _collect_inputs(
        self,
        model: nn.Module,
        calibration_loader: Any,
        device: torch.device,
    ):
        """Collect input activations for calibration."""
        inps = []
        attention_masks = []

        # Hook to capture inputs
        captured = []

        def hook_fn(module, inp, out):
            captured.append(inp[0].detach().cpu())

        # Register hook on first layer
        first_module = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_module = module
                break

        if first_module is None:
            raise ValueError("No linear layers found")

        handle = first_module.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if isinstance(batch, dict):
                        input_ids = batch['input_ids'].to(device)
                        mask = batch.get('attention_mask')
                    else:
                        input_ids = batch.to(device)
                        mask = None

                    model(input_ids)
                    inps.append(captured[-1])

                    if mask is not None:
                        attention_masks.append(mask.cpu())

                    if len(inps) >= self.config.bits * 32:  # Enough samples
                        break

        finally:
            handle.remove()

        inps = torch.cat(inps, dim=0)
        if attention_masks:
            attention_masks = torch.cat(attention_masks, dim=0)
        else:
            attention_masks = None

        return inps, attention_masks

    def _quantize_sequential(
        self,
        model: nn.Module,
        layers: Dict[str, nn.Linear],
        inps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        """Quantize layers sequentially (more accurate)."""
        for name, layer in layers.items():
            logger.info(f"Quantizing layer: {name}")

            # Create GPTQ solver for this layer
            solver = GPTQSolver(
                layer,
                self.config.bits,
                self.config.group_size,
                self.config.damp_percent,
                self.config.sym,
            )

            # Add calibration data
            solver.add_batch(inps)

            # Quantize
            quantized_weight, scale, zeros = solver.quantize()

            # Replace layer with quantized version
            self._replace_layer(model, name, layer, quantized_weight, scale, zeros)

    def _quantize_parallel(
        self,
        model: nn.Module,
        layers: Dict[str, nn.Linear],
        inps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        """Quantize layers in parallel (faster but less accurate)."""
        for name, layer in layers.items():
            solver = GPTQSolver(
                layer,
                self.config.bits,
                self.config.group_size,
                self.config.damp_percent,
                self.config.sym,
            )
            solver.add_batch(inps)
            quantized_weight, scale, zeros = solver.quantize()
            self._replace_layer(model, name, layer, quantized_weight, scale, zeros)

    def _replace_layer(
        self,
        model: nn.Module,
        name: str,
        layer: nn.Linear,
        quantized_weight: torch.Tensor,
        scale: torch.Tensor,
        zeros: torch.Tensor,
    ):
        """Replace original layer with quantized version."""
        # Navigate to parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Create quantized layer
        quant_layer = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            bits=self.config.bits,
            group_size=self.config.group_size,
        )

        # Set quantized weights
        quant_layer.set_weights(quantized_weight, scale, zeros)
        if layer.bias is not None:
            quant_layer.bias = layer.bias

        # Replace
        setattr(parent, parts[-1], quant_layer)


class GPTQSolver:
    """
    GPTQ optimization solver for a single layer.

    Uses Cholesky decomposition to efficiently solve the
    column-wise quantization problem.
    """

    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        damp_percent: float = 0.01,
        sym: bool = True,
    ):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.sym = sym

        self.device = layer.weight.device
        self.rows = layer.out_features
        self.cols = layer.in_features

        # Initialize Hessian
        self.H = torch.zeros((self.cols, self.cols), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """
        Add batch to Hessian computation.

        H = X^T X (input covariance)
        """
        inp = inp.to(self.device)

        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])

        self.H += inp.T @ inp
        self.nsamples += inp.shape[0]

    def quantize(self):
        """
        Perform GPTQ quantization.

        Returns:
            quantized_weight: Quantized weights
            scale: Quantization scales
            zeros: Zero points
        """
        W = self.layer.weight.data.clone()
        H = self.H / self.nsamples

        # Dampening
        damp = self.damp_percent * torch.diag(H).mean()
        H += damp * torch.eye(self.cols, device=self.device)

        # Cholesky decomposition
        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
        except:
            # Fallback if Cholesky fails
            H_inv = torch.linalg.inv(H)

        H_inv_diag = torch.diag(H_inv)

        # Quantization parameters
        num_groups = (self.cols + self.group_size - 1) // self.group_size

        scale = torch.zeros(self.rows, num_groups, device=self.device)
        zeros = torch.zeros(self.rows, num_groups, device=self.device)

        # Quantize column by column
        Q = torch.zeros_like(W)

        for i in range(self.cols):
            group_idx = i // self.group_size

            # Get column
            w = W[:, i]

            # Compute scale and zero for this group (first column of group)
            if i % self.group_size == 0:
                group_w = W[:, i:min(i + self.group_size, self.cols)]

                if self.sym:
                    # Symmetric: scale only
                    w_max = group_w.abs().max(dim=1).values
                    scale[:, group_idx] = w_max / (2 ** (self.bits - 1) - 1)
                    zeros[:, group_idx] = 0
                else:
                    # Asymmetric: scale and zero
                    w_min = group_w.min(dim=1).values
                    w_max = group_w.max(dim=1).values
                    scale[:, group_idx] = (w_max - w_min) / (2 ** self.bits - 1)
                    zeros[:, group_idx] = -w_min / (scale[:, group_idx] + 1e-8)

            # Quantize column
            s = scale[:, group_idx]
            z = zeros[:, group_idx]

            if self.sym:
                q = torch.clamp(
                    torch.round(w / (s + 1e-8)),
                    -(2 ** (self.bits - 1)),
                    2 ** (self.bits - 1) - 1
                )
            else:
                q = torch.clamp(
                    torch.round(w / (s + 1e-8) + z),
                    0,
                    2 ** self.bits - 1
                )

            Q[:, i] = q

            # Update remaining weights to compensate for error
            # This is the key GPTQ innovation
            error = w - (q - z) * s

            if i < self.cols - 1:
                # Update remaining columns
                update = error.unsqueeze(1) @ H_inv[i, i+1:].unsqueeze(0) / H_inv_diag[i]
                W[:, i+1:] += update

        return Q, scale, zeros


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer for inference.

    Stores weights in low-bit format and dequantizes on-the-fly.
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

        # Store quantized weights
        if bits == 4:
            # Pack two 4-bit values into one int8
            self.register_buffer(
                'weight_packed',
                torch.zeros(out_features, in_features // 2, dtype=torch.int8)
            )
        else:
            self.register_buffer(
                'weight_quant',
                torch.zeros(out_features, in_features, dtype=torch.int8)
            )

        self.register_buffer('scale', torch.zeros(out_features, num_groups))
        self.register_buffer('zeros', torch.zeros(out_features, num_groups))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def set_weights(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zeros: torch.Tensor,
    ):
        """Set quantized weights."""
        if self.bits == 4:
            # Pack to int8
            q = quantized.to(torch.int8)
            packed = (q[:, ::2] << 4) | (q[:, 1::2] & 0x0F)
            self.weight_packed.copy_(packed)
        else:
            self.weight_quant.copy_(quantized.to(torch.int8))

        self.scale.copy_(scale)
        self.zeros.copy_(zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dequantization."""
        weight = self._dequantize()
        return torch.nn.functional.linear(x, weight, self.bias)

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights to float."""
        if self.bits == 4:
            # Unpack
            low = self.weight_packed & 0x0F
            high = (self.weight_packed >> 4) & 0x0F
            q = torch.stack([high, low], dim=-1).view(self.out_features, -1).float()
        else:
            q = self.weight_quant.float()

        # Dequantize per group
        q = q.view(self.out_features, -1, self.group_size)
        scale = self.scale.unsqueeze(-1)
        zeros = self.zeros.unsqueeze(-1)

        weight = (q - zeros) * scale

        return weight.view(self.out_features, -1)[:, :self.in_features]
