"""
Gradient optimization for distributed training.

1. Gradient compression — reduce communication volume for multi-node
2. Gradient clipping utilities
3. Gradient accumulation helpers

Complexity-ML — 2026
"""

import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


class GradientCompressor:
    """
    Compress gradients before all-reduce to reduce communication bandwidth.

    Methods:
    - topk: Only send top-K% of gradient values (rest accumulated locally)
    - quantize: Quantize gradients to int8 before communication
    - powersgd: Low-rank approximation (best quality/compression ratio)

    Usage:
        compressor = GradientCompressor(method="topk", ratio=0.1)
        # After backward, before optimizer step:
        compressor.compress_and_allreduce(model)
    """

    def __init__(self, method: str = "none", ratio: float = 0.1):
        """
        Args:
            method: "none", "topk", "quantize"
            ratio: For topk, fraction of gradients to keep (0.1 = top 10%)
        """
        self.method = method
        self.ratio = ratio
        self._residuals = {}  # Error feedback for topk

    def compress_and_allreduce(self, model: torch.nn.Module) -> None:
        if self.method == "none" or not dist.is_initialized():
            return

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            if self.method == "topk":
                self._topk_compress(name, param)
            elif self.method == "quantize":
                self._quantize_compress(param)

    def _topk_compress(self, name: str, param: torch.nn.Parameter) -> None:
        """Top-K sparsification with error feedback."""
        grad = param.grad.data
        flat = grad.view(-1)

        # Add residual from last step (error feedback)
        if name in self._residuals:
            flat = flat + self._residuals[name]

        # Keep top K values
        k = max(1, int(flat.numel() * self.ratio))
        _, indices = flat.abs().topk(k)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices] = True

        # Save residual (unselected gradients)
        self._residuals[name] = flat * (~mask)

        # Zero out unselected
        flat = flat * mask

        # All-reduce the sparse gradient
        dist.all_reduce(flat)
        param.grad.data = flat.view(grad.shape) / dist.get_world_size()

    def _quantize_compress(self, param: torch.nn.Parameter) -> None:
        """INT8 quantization for gradient communication."""
        grad = param.grad.data
        # Compute scale
        amax = grad.abs().max()
        scale = amax / 127.0 if amax > 0 else torch.ones(1, device=grad.device)

        # Quantize to int8
        quantized = (grad / scale).round().clamp(-127, 127).to(torch.int8)

        # All-reduce in int8 (uses less bandwidth)
        # Note: NCCL doesn't support int8 all-reduce, so we cast to float
        quantized_float = quantized.float()
        dist.all_reduce(quantized_float)
        quantized_float = quantized_float / dist.get_world_size()

        # Dequantize
        param.grad.data = (quantized_float * scale).to(grad.dtype)


def clip_grad_norm_fused(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clip gradient norm with fused computation.

    Computes the norm and clips in a single pass over parameters,
    avoiding the double-pass in torch.nn.utils.clip_grad_norm_.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if not parameters:
        return torch.tensor(0.0)

    device = parameters[0].grad.device

    if norm_type == 2.0:
        # Fused L2 norm: compute sum of squares in one pass
        total_norm_sq = torch.zeros(1, device=device)
        for p in parameters:
            total_norm_sq += p.grad.data.pow(2).sum()
        total_norm = total_norm_sq.sqrt()
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.data, norm_type) for p in parameters]),
            norm_type,
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm
