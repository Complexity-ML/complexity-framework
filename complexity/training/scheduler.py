"""Learning rate schedulers."""

import torch
import torch.nn as nn
import logging

from .config import TrainingConfig

logger = logging.getLogger(__name__)


def resolve_scheduler_name(config: TrainingConfig, model: nn.Module = None) -> str:
    """
    Resolve 'auto' scheduler to concrete scheduler name.

    Auto always resolves to cosine (standard for all model types).
    """
    if config.lr_scheduler != "auto":
        return config.lr_scheduler
    return "cosine"


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig = None,
    num_training_steps: int = None,
    model: nn.Module = None,
    scheduler_type: str = None,
    num_warmup_steps: int = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR, ConstantLR, SequentialLR,
    )

    if config is None:
        if scheduler_type is None:
            scheduler_type = "cosine"
        if num_training_steps is None:
            raise ValueError("num_training_steps is required when config is not provided")
        config = TrainingConfig(
            lr_scheduler=scheduler_type,
            warmup_steps=0 if num_warmup_steps is None else num_warmup_steps,
            learning_rate=optimizer.param_groups[0].get("lr", 1e-4),
        )
    elif scheduler_type is not None:
        config.lr_scheduler = scheduler_type
    if num_warmup_steps is not None:
        config.warmup_steps = num_warmup_steps
    if num_training_steps is None:
        num_training_steps = config.max_steps

    warmup_steps = min(config.warmup_steps, max(num_training_steps - 1, 0))
    scheduler_name = resolve_scheduler_name(config, model)

    # Warmup (shared by all schedulers)
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(warmup_steps, 1),
    )

    if scheduler_name == "cosine":
        decay = CosineAnnealingLR(
            optimizer,
            T_max=max(num_training_steps - warmup_steps, 1),
            eta_min=config.learning_rate * config.min_lr_ratio,
        )

    elif scheduler_name == "wsd":
        # Warmup-Stable-Decay (LLaMA 3 style)
        # 3 phases: warmup → stable (peak LR) → cosine decay
        # Default: 80% stable, 20% decay (after warmup)
        remaining = max(num_training_steps - warmup_steps, 1)
        decay_ratio = getattr(config, 'wsd_decay_ratio', 0.2)
        stable_steps = int(remaining * (1 - decay_ratio))
        decay_steps = remaining - stable_steps

        stable = ConstantLR(optimizer, factor=1.0, total_iters=stable_steps)
        final_decay = CosineAnnealingLR(
            optimizer,
            T_max=max(decay_steps, 1),
            eta_min=config.learning_rate * config.min_lr_ratio,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, stable, final_decay],
            milestones=[warmup_steps, warmup_steps + stable_steps],
        )

    elif scheduler_name == "linear":
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.min_lr_ratio,
            total_iters=max(num_training_steps - warmup_steps, 1),
        )

    else:  # constant
        decay = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=max(num_training_steps - warmup_steps, 1),
        )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )
