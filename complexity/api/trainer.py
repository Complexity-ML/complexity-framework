"""
Trainer API - Flexible avec defaults.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Callable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@dataclass
class TrainerConfig:
    """Config trainer - defaults sensibles, tout overridable."""
    # Steps
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 10
    # Batch
    batch_size: int = 32
    gradient_accumulation: int = 1
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    # Scheduler
    warmup_steps: int = 1000
    scheduler: str = "cosine"  # cosine, linear, constant
    min_lr_ratio: float = 0.1
    # Gradient
    gradient_clip: float = 1.0
    # Precision
    precision: str = "bf16"  # fp32, fp16, bf16
    # Output
    output_dir: str = "outputs"
    # Extra - tout ce qu'on veut
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.extra.items():
            setattr(self, k, v)


class Trainer:
    """
    Trainer flexible.

    Defaults sensibles, TOUT overridable via **kwargs.

    Examples:
        # Simple
        trainer = Trainer(model, dataset)
        trainer.train()

        # Override ce qu'on veut
        trainer = Trainer(model, dataset, lr=1e-5, batch_size=64, scheduler="linear")

        # Passer une config
        trainer = Trainer(model, dataset, config=TrainerConfig(max_steps=50000))
    """

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        config: TrainerConfig = None,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Config = fournie ou créée, puis kwargs override
        if config:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            self.config = config
        else:
            self.config = TrainerConfig(**kwargs)

        # State
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Setup
        self._setup()

    def _setup(self):
        """Setup optimizer, scheduler, scaler."""
        # Output dirs
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(exist_ok=True)

        # Optimizer
        decay, no_decay = [], []
        for name, p in self.model.module.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        self.optimizer = AdamW([
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.config.lr, betas=self.config.betas)

        # Scheduler
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.config.warmup_steps)
        if self.config.scheduler == "cosine":
            main = CosineAnnealingLR(self.optimizer, T_max=self.config.max_steps - self.config.warmup_steps, eta_min=self.config.lr * self.config.min_lr_ratio)
        else:
            main = LinearLR(self.optimizer, start_factor=1.0, end_factor=1.0, total_iters=self.config.max_steps)
        self.scheduler = SequentialLR(self.optimizer, [warmup, main], milestones=[self.config.warmup_steps])

        # Scaler
        if self.config.precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast_dtype = torch.float16
        elif self.config.precision == "bf16":
            self.scaler = None
            self.autocast_dtype = torch.bfloat16
        else:
            self.scaler = None
            self.autocast_dtype = torch.float32

    def train(self, steps: int = None, **kwargs) -> Dict[str, Any]:
        """
        Entraîne. kwargs override config pour ce run.

        Args:
            steps: Override max_steps pour ce run
            **kwargs: Override n'importe quoi (lr, batch_size, ...)
        """
        # Override temporaire
        max_steps = steps or kwargs.get("max_steps", self.config.max_steps)
        batch_size = kwargs.get("batch_size", self.config.batch_size)

        target = self.step + max_steps

        print(f"\n{'='*50}")
        print(f"Training")
        print(f"{'='*50}")
        print(f"  Model: {self.model.num_parameters:,} params")
        print(f"  Dataset: {len(self.train_dataset):,} examples")
        print(f"  Batch: {batch_size} x {self.config.gradient_accumulation}")
        print(f"  LR: {self.config.lr}")
        print(f"  Steps: {max_steps}")
        print(f"{'='*50}\n")

        loader = self.train_dataset.get_dataloader(batch_size=batch_size, shuffle=True)
        self.model.train_mode()

        total_loss = 0.0
        start = time.time()

        while self.step < target:
            for batch in loader:
                if self.step >= target:
                    break

                loss = self._step(batch)
                total_loss += loss

                if (self.step + 1) % self.config.gradient_accumulation == 0:
                    self._update()

                self.step += 1

                # Log
                if self.step % self.config.log_steps == 0:
                    avg = total_loss / self.config.log_steps
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {self.step:>6} | Loss: {avg:.4f} | LR: {lr:.2e}")
                    total_loss = 0.0

                # Eval
                if self.eval_dataset and self.step % self.config.eval_steps == 0:
                    m = self.evaluate()
                    print(f"  -> Eval: {m['loss']:.4f}")

                # Save
                if self.step % self.config.save_steps == 0:
                    self.save()

            self.epoch += 1

        self.save(final=True)

        elapsed = time.time() - start
        print(f"\nDone! {self.step} steps in {elapsed/60:.1f}min")

        return {"steps": self.step, "time": elapsed}

    def _step(self, batch: Dict) -> float:
        """Un step de training."""
        input_ids = batch["input_ids"].to(self.model.device)
        labels = input_ids.clone()

        with torch.cuda.amp.autocast(enabled=self.autocast_dtype != torch.float32, dtype=self.autocast_dtype):
            out = self.model(input_ids)
            logits = out["logits"]

            # Loss
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            loss = loss / self.config.gradient_accumulation

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation

    def _update(self):
        """Update weights."""
        if self.config.gradient_clip > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.config.gradient_clip)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()

    def evaluate(self, **kwargs) -> Dict[str, float]:
        """Évalue. kwargs override (batch_size, ...)."""
        if not self.eval_dataset:
            return {}

        batch_size = kwargs.get("batch_size", self.config.batch_size)
        loader = self.eval_dataset.get_dataloader(batch_size=batch_size, shuffle=False)

        self.model.eval()
        total, n = 0.0, 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.model.device)
                out = self.model(input_ids)
                logits = out["logits"]

                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = input_ids[..., 1:].contiguous().view(-1)
                loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
                total += loss.item()
                n += 1

        self.model.train_mode()

        avg = total / max(1, n)
        if avg < self.best_loss:
            self.best_loss = avg
            self.save(best=True)

        return {"loss": avg}

    def resume(self, path: Union[str, Path], **kwargs):
        """Resume. kwargs override ce qu'on veut."""
        path = Path(path)

        # State
        state_file = path / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self.step = state.get("step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            print(f"[Trainer] Resumed from step {self.step}")

        # Model
        model_file = path / "model.pt"
        if model_file.exists():
            self.model.module.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))

        # Optimizer
        opt_file = path / "optimizer.pt"
        if opt_file.exists():
            self.optimizer.load_state_dict(torch.load(opt_file, map_location="cpu", weights_only=True))

    def save(self, path: Union[str, Path] = None, final: bool = False, best: bool = False):
        """Sauvegarde."""
        if path is None:
            base = Path(self.config.output_dir) / "checkpoints"
            if best:
                path = base / "best"
            elif final:
                path = base / "final"
            else:
                path = base / f"step-{self.step}"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.module.state_dict(), path / "model.pt")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        with open(path / "state.json", "w") as f:
            json.dump({"step": self.step, "epoch": self.epoch, "best_loss": self.best_loss}, f)

        with open(path / "config.json", "w") as f:
            json.dump({k: v for k, v in self.config.__dict__.items() if k != "extra"}, f, indent=2, default=str)

        print(f"[Trainer] Saved to {path}")

    def get_config_yaml(self) -> str:
        """Export YAML pour utiliser avec CLI."""
        import yaml
        cfg = {
            "training": {
                "max_steps": self.config.max_steps,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "warmup_steps": self.config.warmup_steps,
                "gradient_clip": self.config.gradient_clip,
                "mixed_precision": self.config.precision,
            }
        }
        return yaml.dump(cfg, default_flow_style=False)

    def __repr__(self):
        return f"Trainer(step={self.step}, lr={self.config.lr}, batch={self.config.batch_size})"
