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
    config: TrainingConfig,
    num_training_steps: int,
    model: nn.Module = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR, ConstantLR, SequentialLR,
    )

    warmup_steps = config.warmup_steps
    scheduler_name = resolve_scheduler_name(config, model)

    # Warmup (shared by all schedulers)
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    if scheduler_name == "cosine":
        decay = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=config.learning_rate * config.min_lr_ratio,
        )

    elif scheduler_name == "wsd":
        # Warmup-Stable-Decay (LLaMA 3 style)
        # 3 phases: warmup → stable (peak LR) → cosine decay
        # Default: 80% stable, 20% decay (after warmup)
        remaining = num_training_steps - warmup_steps
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
            total_iters=num_training_steps - warmup_steps,
        )

    else:  # constant
        decay = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=num_training_steps - warmup_steps,
        )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )
