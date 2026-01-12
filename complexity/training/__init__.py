"""
Training module for framework-complexity.

Provides a complete training solution:
- Distributed training with FSDP
- Mixed precision (FP16, BF16)
- Gradient accumulation
- Checkpointing
- Learning rate scheduling
- Logging and metrics

Usage:
    from complexity.training import Trainer, TrainingConfig

    config = TrainingConfig(
        max_steps=100000,
        batch_size=32,
        learning_rate=1e-4,
        precision="bf16",
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
    )

    trainer.train()
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    MetricsTracker,
    get_lr_scheduler,
    EarlyStoppingCallback,
    WandBCallback,
    TensorBoardCallback,
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "MetricsTracker",
    "get_lr_scheduler",
    "EarlyStoppingCallback",
    "WandBCallback",
    "TensorBoardCallback",
]
