"""
Trainer API - Re-exports from complexity.training.

For full trainer functionality, use:
    from complexity.training import Trainer, TrainingConfig
"""

from complexity.training.trainer import (
    Trainer as IntegratedTrainer,
    TrainingConfig,
    EvaluationConfig,
)
from complexity.training.optimizers import (
    get_optimizer,
    get_scheduler,
    OptimizerConfig,
)

# Simple Trainer wrapper for API convenience
from dataclasses import dataclass, field
from typing import Dict, Any
import torch


@dataclass
class TrainerConfig:
    """Simple trainer config - maps to TrainingConfig."""
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 10
    batch_size: int = 32
    gradient_accumulation: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    precision: str = "bf16"
    output_dir: str = "outputs"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_training_config(self) -> TrainingConfig:
        """Convert to internal TrainingConfig."""
        return TrainingConfig(
            max_steps=self.max_steps,
            eval_every_n_steps=self.eval_steps,
            save_every_n_steps=self.save_steps,
            log_every_n_steps=self.log_steps,
            gradient_accumulation_steps=self.gradient_accumulation,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_grad_norm=self.gradient_clip,
            output_dir=self.output_dir,
        )


class Trainer:
    """Simple trainer API wrapping IntegratedTrainer."""

    def __init__(self, model, config: TrainerConfig = None, **kwargs):
        self._model = model
        self._config = config or TrainerConfig(**kwargs)
        self._trainer = None

    def train(self, train_data, val_data=None, **kwargs):
        """Train the model."""
        training_config = self._config.to_training_config()
        for k, v in kwargs.items():
            if hasattr(training_config, k):
                setattr(training_config, k, v)

        self._trainer = IntegratedTrainer(
            model=self._model,
            config=training_config,
            train_dataloader=train_data.get_dataloader() if hasattr(train_data, 'get_dataloader') else train_data,
            eval_dataloader=val_data.get_dataloader() if val_data and hasattr(val_data, 'get_dataloader') else val_data,
        )
        return self._trainer.train()

    def save(self, path: str):
        """Save model."""
        if self._trainer:
            self._trainer.save_checkpoint(path)
        else:
            torch.save(self._model.state_dict(), path)


__all__ = [
    "Trainer",
    "TrainerConfig",
    "IntegratedTrainer",
    "TrainingConfig",
    "EvaluationConfig",
    "get_optimizer",
    "get_scheduler",
    "OptimizerConfig",
]
