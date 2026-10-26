"""Convenience training API backed by the canonical integrated trainer."""

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from complexity.models import ComplexityModel
from complexity.training.config import TrainingConfig
from complexity.training.optimizers import get_optimizer
from complexity.training.trainer import Trainer as IntegratedTrainer

from .model import Model


@dataclass
class TrainerConfig:
    """Short aliases for TrainingConfig; ``extra`` accepts its full field names.

    Existing alias defaults are preserved. Use TrainingConfig directly when
    configuring distributed runs or reproducing an established training recipe.
    """

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
    num_workers: int = 0
    resume_from: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_training_config(self) -> TrainingConfig:
        values = {
            "max_steps": self.max_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "log_steps": self.log_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler": self.scheduler,
            "grad_clip": self.gradient_clip,
            "precision": self.precision,
            "checkpoint_dir": self.output_dir,
            "num_workers": self.num_workers,
            "resume_from": self.resume_from,
        }
        known = {item.name for item in fields(TrainingConfig)}
        unknown = self.extra.keys() - known
        if unknown:
            raise ValueError(f"Unknown TrainingConfig fields: {', '.join(sorted(unknown))}")
        values.update(deepcopy(self.extra))
        return TrainingConfig(**values)


class Trainer:
    """Train a native nn.Module or convenience Model with a dataset/loader.

        trainer = Trainer(model, TrainerConfig(max_steps=10, precision="fp32"))
        trainer.train(dataset)

    ``save`` exports model weights. ``save_checkpoint`` writes resumable run
    state to the configured checkpoint directory. Nothing starts until train
    is explicitly called.
    """

    def __init__(self, model, config=None, **kwargs):
        if isinstance(model, Model):
            model = model.module
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or complexity.api.Model")
        if config is not None and kwargs:
            raise ValueError("Pass either config or TrainerConfig keyword fields")
        if config is not None and not isinstance(config, (TrainerConfig, TrainingConfig)):
            raise TypeError("config must be a TrainerConfig or TrainingConfig")
        self._model = model
        self._config = deepcopy(config) if config is not None else TrainerConfig(**kwargs)
        self._trainer = None

    @property
    def backend(self):
        """Integrated trainer after train() is called, otherwise None."""
        return self._trainer

    def train(
        self,
        train_data,
        val_data=None,
        *,
        optimizer=None,
        scheduler=None,
        compute_loss=None,
        callbacks=None,
        **kwargs,
    ):
        """Run explicit training; kwargs override validated TrainingConfig fields.

        Existing DataLoaders are used as supplied. Convenience Dataset loaders
        use this trainer's batch size/worker count, with evaluation unshuffled.
        ``resume_from`` restores the backend's optimizer, scheduler and state.
        """
        training_config = (
            self._config.to_training_config()
            if isinstance(self._config, TrainerConfig)
            else deepcopy(self._config)
        )
        known = {item.name for item in fields(TrainingConfig)}
        unknown = kwargs.keys() - known
        if unknown:
            raise ValueError(f"Unknown TrainingConfig fields: {', '.join(sorted(unknown))}")
        training_config = replace(training_config, **kwargs)

        def loader(data, *, evaluation=False):
            if data is None:
                return None
            if hasattr(data, "get_dataloader"):
                options = {
                    "batch_size": training_config.batch_size,
                    "num_workers": training_config.num_workers,
                }
                if evaluation:
                    options["shuffle"] = False
                return data.get_dataloader(**options)
            return data

        self._trainer = IntegratedTrainer(
            model=self._model,
            config=training_config,
            train_dataloader=loader(train_data),
            eval_dataloader=loader(val_data, evaluation=True),
            optimizer=optimizer,
            scheduler=scheduler,
            compute_loss=compute_loss,
            callbacks=callbacks,
        )
        return self._trainer.train()

    def save(self, path: str, *, safe_serialization: bool = True):
        """Export model weights; native models include config.json in a directory.

        Generic nn.Modules retain the historical state_dict file behavior.
        For resumable state use save_checkpoint after training has initialized.
        """
        if isinstance(self._model, ComplexityModel):
            self._model.save_pretrained(path, safe_serialization=safe_serialization)
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path)

    def save_checkpoint(self, *, tag: str = "manual") -> str:
        """Save optimizer/scheduler/model/run state using the backend manager."""
        if self._trainer is None:
            raise RuntimeError("Call train() before saving a resumable training checkpoint")
        if not tag or any(char in tag for char in ("/", "\\")) or tag in {".", ".."}:
            raise ValueError("checkpoint tag must be a non-empty filename component")
        self._trainer.state.step = self._trainer.global_step
        self._trainer.state.epoch = self._trainer.epoch
        self._trainer.state.learning_rate = self._trainer.scheduler.get_last_lr()[0]
        return self._trainer.checkpoint_manager.save(
            step=self._trainer.global_step,
            training_state=self._trainer.state,
            tag=tag,
        )


__all__ = ["Trainer", "TrainerConfig", "IntegratedTrainer", "TrainingConfig", "get_optimizer"]
