"""
FLOPS and MACs counting for neural networks.

Computes:
- FLOPs (Floating Point Operations)
- MACs (Multiply-Accumulate operations)
- Parameter counts
- Model complexity analysis
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from functools import reduce
import operator


@dataclass
class ModelComplexity:
    """Model complexity metrics."""
    total_flops: int = 0
    total_macs: int = 0
    total_params: int = 0
    trainable_params: int = 0
    per_layer: Optional[Dict[str, Dict[str, int]]] = None

    def __str__(self) -> str:
        return (
            f"Model Complexity:\n"
            f"  FLOPs: {self.total_flops / 1e9:.2f} G\n"
            f"  MACs: {self.total_macs / 1e9:.2f} G\n"
            f"  Params: {self.total_params / 1e6:.2f} M\n"
            f"  Trainable: {self.trainable_params / 1e6:.2f} M"
        )


def count_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> int:
    """
    Count FLOPs for a model given input shape.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, seq_len, ...)
        device: Device for computation

    Returns:
        Total FLOPs
    """
    counter = FLOPSCounter()
    return counter.count(model, input_shape, device)


def estimate_macs(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> int:
    """
    Estimate MACs (Multiply-Accumulate operations).

    MACs ≈ FLOPs / 2 for most operations.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape

    Returns:
        Estimated MACs
    """
    flops = count_flops(model, input_shape)
    return flops // 2


def get_model_complexity(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> ModelComplexity:
    """
    Get comprehensive model complexity metrics.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device for computation

    Returns:
        ModelComplexity with all metrics
    """
    counter = FLOPSCounter()
    counter.count(model, input_shape, device)

    return counter.get_complexity()


class FLOPSCounter:
    """
    Count FLOPs for neural network models.

    Supports common layers:
    - Linear
    - Conv1d, Conv2d, Conv3d
    - BatchNorm
    - LayerNorm
    - Attention mechanisms
    - Embedding

    Example:
        counter = FLOPSCounter()
        flops = counter.count(model, (1, 512))
        print(f"Model FLOPs: {flops / 1e9:.2f} G")
    """

    def __init__(self):
        """Initialize FLOPS counter."""
        self.total_flops = 0
        self.total_macs = 0
        self.layer_flops: Dict[str, int] = {}
        self.hooks = []

    def reset(self):
        """Reset counters."""
        self.total_flops = 0
        self.total_macs = 0
        self.layer_flops = {}

    def count(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Count FLOPs for model.

        Args:
            model: Model to analyze
            input_shape: Input tensor shape
            device: Computation device

        Returns:
            Total FLOPs
        """
        self.reset()

        if device is None:
            device = next(model.parameters()).device

        # Register hooks
        self._register_hooks(model)

        # Create dummy input
        dummy_input = torch.zeros(input_shape, device=device, dtype=torch.long)

        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                model(dummy_input)
            except Exception:
                # Try with float input
                dummy_input = torch.zeros(input_shape, device=device)
                model(dummy_input)

        # Remove hooks
        self._remove_hooks()

        return self.total_flops

    def _register_hooks(self, model: nn.Module):
        """Register forward hooks on all layers."""
        for name, module in model.named_modules():
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: self._hook_fn(m, i, o, n)
            )
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _hook_fn(self, module: nn.Module, input: tuple, output: Any, name: str):
        """Hook function to count FLOPs for a module."""
        flops = self._compute_flops(module, input, output)
        self.total_flops += flops
        self.total_macs += flops // 2

        if flops > 0:
            self.layer_flops[name] = flops

    def _compute_flops(self, module: nn.Module, input: tuple, output: Any) -> int:
        """Compute FLOPs for a single module."""
        if isinstance(module, nn.Linear):
            return self._linear_flops(module, input, output)
        elif isinstance(module, nn.Conv1d):
            return self._conv1d_flops(module, input, output)
        elif isinstance(module, nn.Conv2d):
            return self._conv2d_flops(module, input, output)
        elif isinstance(module, nn.Conv3d):
            return self._conv3d_flops(module, input, output)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return self._batchnorm_flops(module, input, output)
        elif isinstance(module, nn.LayerNorm):
            return self._layernorm_flops(module, input, output)
        elif isinstance(module, nn.Embedding):
            return self._embedding_flops(module, input, output)
        elif isinstance(module, nn.MultiheadAttention):
            return self._attention_flops(module, input, output)
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
            return self._activation_flops(module, input, output)
        elif isinstance(module, nn.Softmax):
            return self._softmax_flops(module, input, output)
        elif isinstance(module, nn.Dropout):
            return 0  # Dropout doesn't contribute FLOPs
        else:
            return 0

    def _linear_flops(self, module: nn.Linear, input: tuple, output: Any) -> int:
        """FLOPs for linear layer: 2 * batch * in * out (mul + add)."""
        batch_size = input[0].numel() // input[0].size(-1)
        return 2 * batch_size * module.in_features * module.out_features

    def _conv1d_flops(self, module: nn.Conv1d, input: tuple, output: Any) -> int:
        """FLOPs for Conv1d."""
        batch_size = input[0].size(0)
        output_length = output.size(2)

        kernel_flops = (
            module.in_channels //
            module.groups *
            module.kernel_size[0] *
            2 - 1  # mul + add
        )
        return batch_size * output_length * module.out_channels * kernel_flops

    def _conv2d_flops(self, module: nn.Conv2d, input: tuple, output: Any) -> int:
        """FLOPs for Conv2d."""
        batch_size = input[0].size(0)
        output_height = output.size(2)
        output_width = output.size(3)

        kernel_flops = (
            module.in_channels //
            module.groups *
            module.kernel_size[0] *
            module.kernel_size[1] *
            2 - 1
        )
        return batch_size * output_height * output_width * module.out_channels * kernel_flops

    def _conv3d_flops(self, module: nn.Conv3d, input: tuple, output: Any) -> int:
        """FLOPs for Conv3d."""
        batch_size = input[0].size(0)
        output_depth = output.size(2)
        output_height = output.size(3)
        output_width = output.size(4)

        kernel_flops = (
            module.in_channels //
            module.groups *
            reduce(operator.mul, module.kernel_size) *
            2 - 1
        )
        return batch_size * output_depth * output_height * output_width * module.out_channels * kernel_flops

    def _batchnorm_flops(self, module: nn.Module, input: tuple, output: Any) -> int:
        """FLOPs for BatchNorm: 4 ops per element (sub mean, div std, scale, shift)."""
        return input[0].numel() * 4

    def _layernorm_flops(self, module: nn.LayerNorm, input: tuple, output: Any) -> int:
        """FLOPs for LayerNorm."""
        # Mean: N ops, Var: 2N ops, Normalize: 2N ops, Scale+Shift: 2N ops
        return input[0].numel() * 7

    def _embedding_flops(self, module: nn.Embedding, input: tuple, output: Any) -> int:
        """FLOPs for Embedding: essentially 0 (just lookup)."""
        return 0

    def _attention_flops(self, module: nn.MultiheadAttention, input: tuple, output: Any) -> int:
        """FLOPs for MultiheadAttention."""
        # Q, K, V projections: 3 * 2 * B * S * d * d
        # Attention scores: 2 * B * h * S * S * (d/h)
        # Attention @ V: 2 * B * h * S * (d/h) * S
        # Output projection: 2 * B * S * d * d

        q = input[0]
        batch_size = q.size(1)  # (S, B, d)
        seq_len = q.size(0)
        embed_dim = module.embed_dim

        projection_flops = 3 * 2 * batch_size * seq_len * embed_dim * embed_dim
        attention_flops = 2 * batch_size * seq_len * seq_len * embed_dim
        output_flops = 2 * batch_size * seq_len * embed_dim * embed_dim

        return projection_flops + attention_flops + output_flops

    def _activation_flops(self, module: nn.Module, input: tuple, output: Any) -> int:
        """FLOPs for activation functions."""
        # ReLU: 1 op per element (comparison)
        # GELU, SiLU, Sigmoid, Tanh: ~5-10 ops per element
        if isinstance(module, nn.ReLU):
            return input[0].numel()
        else:
            return input[0].numel() * 8  # Approximate

    def _softmax_flops(self, module: nn.Softmax, input: tuple, output: Any) -> int:
        """FLOPs for Softmax: exp + sum + div."""
        return input[0].numel() * 5

    def get_complexity(self) -> ModelComplexity:
        """Get complexity metrics."""
        return ModelComplexity(
            total_flops=self.total_flops,
            total_macs=self.total_macs,
            per_layer=self.layer_flops.copy(),
        )

    def report(self) -> str:
        """Generate FLOPs report."""
        lines = ["=" * 50, "FLOPs Analysis Report", "=" * 50]

        lines.append(f"\nTotal FLOPs: {self.total_flops / 1e9:.2f} G")
        lines.append(f"Total MACs: {self.total_macs / 1e9:.2f} G")

        if self.layer_flops:
            lines.append("\nTop 10 Layers by FLOPs:")
            sorted_layers = sorted(
                self.layer_flops.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for name, flops in sorted_layers:
                pct = 100 * flops / self.total_flops if self.total_flops > 0 else 0
                lines.append(f"  {name}: {flops / 1e9:.4f} G ({pct:.1f}%)")

        lines.append("=" * 50)
        return "\n".join(lines)


def estimate_transformer_flops(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    sequence_length: int,
    batch_size: int = 1,
    vocab_size: int = 32000,
) -> Dict[str, int]:
    """
    Estimate FLOPs for a transformer model.

    This provides a quick estimate without running the model.

    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        intermediate_size: MLP intermediate dimension
        num_heads: Number of attention heads
        sequence_length: Input sequence length
        batch_size: Batch size
        vocab_size: Vocabulary size

    Returns:
        Dictionary with FLOPs breakdown
    """
    B, S, H, I = batch_size, sequence_length, hidden_size, intermediate_size

    # Embedding (lookup, essentially free)
    embedding_flops = 0

    # Per-layer attention
    # Q, K, V projections: 3 * 2BSH²
    # Attention scores: 2BS²H
    # Softmax: 5BS²h
    # Attention @ V: 2BS²H
    # Output projection: 2BSH²
    attention_flops = (
        3 * 2 * B * S * H * H +  # Q, K, V
        2 * B * S * S * H +       # QK^T
        5 * B * S * S * num_heads +  # Softmax
        2 * B * S * S * H +       # Attn @ V
        2 * B * S * H * H         # Output
    )

    # Per-layer MLP (assuming SwiGLU or similar)
    # Up projection: 2BSHI
    # Gate projection: 2BSHI
    # Down projection: 2BSIH
    mlp_flops = 2 * B * S * H * I * 3

    # LayerNorms: 2 per layer
    layernorm_flops = 2 * 7 * B * S * H

    # Total per layer
    per_layer_flops = attention_flops + mlp_flops + layernorm_flops

    # Total
    total_layer_flops = per_layer_flops * num_layers

    # Final layernorm + output projection
    output_flops = 7 * B * S * H + 2 * B * S * H * vocab_size

    total_flops = total_layer_flops + output_flops + embedding_flops

    return {
        "embedding": embedding_flops,
        "attention_per_layer": attention_flops,
        "mlp_per_layer": mlp_flops,
        "layernorm_per_layer": layernorm_flops,
        "total_per_layer": per_layer_flops,
        "all_layers": total_layer_flops,
        "output": output_flops,
        "total": total_flops,
        "total_gflops": total_flops / 1e9,
    }


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    Get parameter count per named parameter.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to counts
    """
    return {name: param.numel() for name, param in model.named_parameters()}
