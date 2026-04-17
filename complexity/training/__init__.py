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

from .config import TrainingConfig
from .trainer import Trainer
from .scheduler import get_lr_scheduler, resolve_scheduler_name
from .metrics import MetricsTracker
from .callbacks import EarlyStoppingCallback, WandBCallback, TensorBoardCallback, TqdmCallback
from .moe_telemetry import global_expert_shares, detect_num_experts

__all__ = [
    "Trainer",
    "TrainingConfig",
    "MetricsTracker",
    "get_lr_scheduler",
    "resolve_scheduler_name",
    "EarlyStoppingCallback",
    "WandBCallback",
    "TensorBoardCallback",
    "TqdmCallback",
    "global_expert_shares",
    "detect_num_experts",
]
