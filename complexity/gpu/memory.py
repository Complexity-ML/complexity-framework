"""
GPU memory optimization utilities.

Complexity-ML — 2026
"""

import torch
import gc
import logging

logger = logging.getLogger(__name__)


def clear_gpu_cache() -> None:
    """Force clear GPU memory cache and Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def log_memory_usage(prefix: str = "", device: int = 0) -> dict:
    """Log current GPU memory usage. Returns dict with stats in GB."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_mem / 1e9
    free = total - allocated
    stats = {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(free, 2),
        "utilization_pct": round(allocated / total * 100, 1),
    }
    logger.info(f"[gpu] {prefix}Memory: {allocated:.1f}GB / {total:.1f}GB "
                f"({stats['utilization_pct']}%) reserved={reserved:.1f}GB")
    return stats


def estimate_max_batch_size(
    model: torch.nn.Module,
    seq_len: int = 2048,
    vocab_size: int = 32000,
    device: int = 0,
    safety_margin: float = 0.85,
) -> int:
    """
    Estimate max batch size that fits in GPU memory.

    Accounts for: model params, optimizer (Adam 12 bytes/param),
    activations, and logits tensor.

    Args:
        model: The model
        seq_len: Sequence length
        vocab_size: Vocabulary size (for logits tensor)
        device: GPU device index
        safety_margin: Fraction of total memory to use (0.85 = 85%)

    Returns:
        Estimated max batch size (power of 2)
    """
    if not torch.cuda.is_available():
        return 32

    total_mem = torch.cuda.get_device_properties(device).total_mem
    usable_mem = total_mem * safety_margin

    # Model + optimizer memory
    num_params = sum(p.numel() for p in model.parameters())
    model_mem = num_params * 18  # bf16 params + fp32 optimizer (Adam)

    # Available for activations + logits
    available = usable_mem - model_mem

    # Per-sample memory estimate:
    # - Activations (with grad checkpoint): ~2 * seq * hidden * num_layers * 2 bytes
    # - Logits: seq * vocab * 2 bytes (if not using fused CE)
    hidden = getattr(model, 'config', None)
    hidden_size = getattr(hidden, 'hidden_size', 1024) if hidden else 1024
    num_layers = getattr(hidden, 'num_hidden_layers', 24) if hidden else 24

    per_sample = (
        2 * seq_len * hidden_size * num_layers * 2  # activations (checkpointed)
        + seq_len * hidden_size * 2  # hidden states
    )

    max_batch = int(available / per_sample)

    # Round down to nearest power of 2
    batch = 1
    while batch * 2 <= max_batch:
        batch *= 2

    logger.info(f"[gpu] Estimated max batch: {batch} "
                f"(model={num_params/1e6:.0f}M, available={available/1e9:.1f}GB, "
                f"per_sample={per_sample/1e6:.1f}MB)")
    return batch
