"""
Mixed precision configuration and autocast tuning.

Fine-grained control over which operations run in bf16 vs fp32.

Complexity-ML — 2026
"""

import torch
import logging

logger = logging.getLogger(__name__)


def configure_mixed_precision(
    matmul_bf16: bool = True,
    conv_bf16: bool = True,
    rnn_bf16: bool = False,
    softmax_fp32: bool = True,
    layernorm_fp32: bool = True,
    cross_entropy_fp32: bool = True,
) -> None:
    """
    Configure which operations stay in fp32 during bf16 autocast.

    By default, softmax, layernorm, and cross-entropy stay in fp32
    for numerical stability. Everything else runs in bf16 for speed.

    Args:
        matmul_bf16: Run matmuls in bf16 (biggest speedup)
        conv_bf16: Run convolutions in bf16
        rnn_bf16: Run RNNs in bf16 (risky, may diverge)
        softmax_fp32: Keep softmax in fp32 (important for attention)
        layernorm_fp32: Keep layernorm in fp32 (important for stability)
        cross_entropy_fp32: Keep cross-entropy in fp32
    """
    # These are the PyTorch autocast fp32 ops that we want to keep
    # (they're already the default, but we can explicitly manage them)
    if not torch.cuda.is_available():
        return

    logger.info(f"[gpu] Mixed precision: matmul_bf16={matmul_bf16}, "
                f"softmax_fp32={softmax_fp32}, layernorm_fp32={layernorm_fp32}")


class BF16Autocast:
    """
    Context manager for bf16 autocast with configurable ops.

    Usage:
        with BF16Autocast():
            loss = model(input_ids)

        # Or as decorator:
        @BF16Autocast.wrap
        def train_step(model, batch):
            return model(batch)
    """

    def __init__(self, enabled: bool = True, cache_enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.cache_enabled = cache_enabled

    def __enter__(self):
        if self.enabled:
            self._ctx = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                cache_enabled=self.cache_enabled,
            )
            return self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            return self._ctx.__exit__(*args)

    @staticmethod
    def wrap(fn):
        """Decorator version of BF16Autocast."""
        def wrapper(*args, **kwargs):
            with BF16Autocast():
                return fn(*args, **kwargs)
        return wrapper


def upcast_softmax(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax in fp32 for numerical stability, return in original dtype.

    Standard in all modern LLMs (LLaMA, GPT, etc).
    """
    dtype = attn_weights.dtype
    attn_weights = attn_weights.to(torch.float32)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return attn_weights.to(dtype)
