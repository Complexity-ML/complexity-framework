"""
Integrated Trainer for framework-complexity.

A complete training solution that combines:
- Distributed training (FSDP, TP, PP)
- Mixed precision (FP16, BF16)
- Gradient accumulation
- Checkpointing
- Logging and metrics
- Learning rate scheduling

References:
- HuggingFace Trainer
- PyTorch Lightning
- Nanotron DistributedTrainer
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json
from datetime import datetime

from ..parallel.data_parallel import (
    wrap_model_fsdp,
    ShardingMode,
    PrecisionMode,
    init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    cleanup,
)
from ..utils.checkpointing import CheckpointManager, TrainingState
from ..utils.security import AuditLogger, SecureTrainingContext

logger = logging.getLogger(__name__)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Learning rate
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    min_lr_ratio: float = 0.1

    # Precision
    precision: str = "bf16"  # fp32, fp16, bf16
    grad_clip: float = 1.0

    # Distributed
    use_fsdp: bool = True
    sharding_mode: str = "full_shard"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from: Optional[str] = None

    # Logging
    log_steps: int = 10
    eval_steps: int = 500
    log_dir: str = "logs"

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler": self.lr_scheduler,
            "min_lr_ratio": self.min_lr_ratio,
            "precision": self.precision,
            "grad_clip": self.grad_clip,
            "use_fsdp": self.use_fsdp,
            "sharding_mode": self.sharding_mode,
            "checkpoint_dir": self.checkpoint_dir,
            "save_steps": self.save_steps,
        }


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""

    warmup_steps = config.warmup_steps

    if config.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup + Cosine decay
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=config.learning_rate * config.min_lr_ratio,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    elif config.lr_scheduler == "linear":
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.min_lr_ratio,
            total_iters=num_training_steps - warmup_steps,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )

    else:  # constant
        from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        constant = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=num_training_steps - warmup_steps,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, constant],
            milestones=[warmup_steps],
        )


# =============================================================================
# Metrics Tracker
# =============================================================================

class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[Dict[str, Any]] = []
        self.current_metrics: Dict[str, float] = {}
        self.smoothed_metrics: Dict[str, float] = {}
        self.smoothing = 0.99

        # Timing
        self.start_time = time.time()
        self.step_times: List[float] = []

    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics."""
        self.current_metrics = metrics

        # Exponential moving average
        for key, value in metrics.items():
            if key in self.smoothed_metrics:
                self.smoothed_metrics[key] = (
                    self.smoothing * self.smoothed_metrics[key] +
                    (1 - self.smoothing) * value
                )
            else:
                self.smoothed_metrics[key] = value

        # Record
        record = {
            "step": step,
            "timestamp": time.time() - self.start_time,
            **metrics,
        }
        self.metrics_history.append(record)

    def log_step_time(self, step_time: float):
        """Log time for a step."""
        self.step_times.append(step_time)
        if len(self.step_times) > 100:
            self.step_times.pop(0)

    def get_throughput(self, batch_size: int, seq_len: int) -> float:
        """Get tokens per second."""
        if not self.step_times:
            return 0.0
        avg_time = sum(self.step_times) / len(self.step_times)
        if avg_time == 0:
            return 0.0
        return (batch_size * seq_len) / avg_time

    def save(self, filename: str = "metrics.json"):
        """Save metrics to file."""
        with open(self.log_dir / filename, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "total_steps": len(self.metrics_history),
            "total_time": time.time() - self.start_time,
            "final_loss": self.smoothed_metrics.get("loss", None),
            "average_throughput": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
        }


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Integrated trainer for framework-complexity models.

    Features:
    - Distributed training with FSDP
    - Mixed precision training
    - Gradient accumulation
    - Automatic checkpointing
    - Learning rate scheduling
    - Comprehensive logging

    Usage:
        from complexity.training import Trainer, TrainingConfig
        from complexity import ComplexityModel, get_preset

        config = get_preset("llama-7b")
        model = ComplexityModel(config)

        train_config = TrainingConfig(
            max_steps=100000,
            batch_size=32,
            learning_rate=1e-4,
        )

        trainer = Trainer(
            model=model,
            config=train_config,
            train_dataloader=train_loader,
        )

        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compute_loss: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []

        # Initialize distributed
        self.distributed = init_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()

        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")

        # Wrap model with FSDP if enabled
        if config.use_fsdp and self.distributed:
            precision = PrecisionMode.BF16 if config.precision == "bf16" else (
                PrecisionMode.FP16 if config.precision == "fp16" else PrecisionMode.FP32
            )
            sharding = ShardingMode(config.sharding_mode)

            self.model = wrap_model_fsdp(
                model,
                sharding_mode=sharding,
                precision=precision,
            )
        else:
            self.model = model.to(self.device)

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            self.optimizer = optimizer

        # Scheduler
        num_training_steps = config.max_steps
        if scheduler is None:
            self.scheduler = get_lr_scheduler(
                self.optimizer,
                config,
                num_training_steps,
            )
        else:
            self.scheduler = scheduler

        # Loss function
        if compute_loss is None:
            self.compute_loss = self._default_loss
        else:
            self.compute_loss = compute_loss

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            max_checkpoints=config.save_total_limit,
        )

        # Metrics
        self.metrics = MetricsTracker(config.log_dir)

        # Audit logging for security
        self.audit = AuditLogger(
            log_path=str(Path(config.log_dir) / "audit.log")
        )

        # Training state
        self.state = TrainingState()
        self.global_step = 0
        self.epoch = 0

        # Mixed precision scaler
        self.scaler = None
        if config.precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()

        # Log training start
        if self.is_main:
            self.audit.log_training_start(config.to_dict())
            logger.info(f"Trainer initialized on {self.world_size} devices")
            logger.info(f"Config: {config}")

    def _default_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Default loss computation (cross-entropy for LM)."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids[:, 1:]).to(self.device)

        # Forward pass
        outputs = model(input_ids)

        # Shift logits for next-token prediction
        if outputs.dim() == 3:  # [batch, seq, vocab]
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, :shift_logits.size(1)].contiguous()
        else:
            shift_logits = outputs
            shift_labels = labels

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training summary dictionary
        """
        self.model.train()

        # Resume from checkpoint if specified
        if self.config.resume_from:
            self.state = self.checkpoint_manager.load(self.config.resume_from)
            self.global_step = self.state.step
            self.epoch = self.state.epoch
            logger.info(f"Resumed from step {self.global_step}")

        # Training loop
        accumulation_steps = self.config.gradient_accumulation_steps

        try:
            while self.global_step < self.config.max_steps:
                self.epoch += 1

                for batch_idx, batch in enumerate(self.train_dataloader):
                    step_start = time.time()

                    # Forward pass
                    loss = self._training_step(batch)

                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps

                    # Backward pass
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Optimizer step (after accumulation)
                    if (batch_idx + 1) % accumulation_steps == 0:
                        self._optimizer_step()
                        self.global_step += 1

                        # Logging
                        step_time = time.time() - step_start
                        self.metrics.log_step_time(step_time)

                        if self.global_step % self.config.log_steps == 0:
                            self._log_step(loss.item() * accumulation_steps, step_time)

                        # Evaluation
                        if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                            eval_loss = self.evaluate()
                            if self.is_main:
                                logger.info(f"Step {self.global_step} - Eval Loss: {eval_loss:.4f}")

                        # Checkpointing
                        if self.global_step % self.config.save_steps == 0:
                            self._save_checkpoint()

                        # Callbacks
                        for callback in self.callbacks:
                            callback(self, self.global_step, loss.item())

                        # Check if done
                        if self.global_step >= self.config.max_steps:
                            break

                # Check max epochs
                if self.config.max_epochs and self.epoch >= self.config.max_epochs:
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(tag="interrupted")

        # Final save
        self._save_checkpoint(tag="final")

        # Log training end
        summary = self.metrics.get_summary()
        if self.is_main:
            self.audit.log_training_end(summary)
            self.metrics.save()

        return summary

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute single training step."""
        # Mixed precision context
        if self.config.precision in ["fp16", "bf16"]:
            dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = self.compute_loss(self.model, batch)
        else:
            loss = self.compute_loss(self.model, batch)

        return loss

    def _optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        # Unscale gradients
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad()

    def _log_step(self, loss: float, step_time: float):
        """Log training step."""
        lr = self.scheduler.get_last_lr()[0]

        metrics = {
            "loss": loss,
            "lr": lr,
            "step_time": step_time,
        }

        self.metrics.update(metrics, self.global_step)

        if self.is_main:
            throughput = self.metrics.get_throughput(
                self.config.batch_size,
                512,  # Assume seq_len
            )
            logger.info(
                f"Step {self.global_step} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Throughput: {throughput:.0f} tok/s"
            )

    def _save_checkpoint(self, tag: str = "step"):
        """Save checkpoint."""
        self.state.step = self.global_step
        self.state.epoch = self.epoch
        self.state.learning_rate = self.scheduler.get_last_lr()[0]

        self.checkpoint_manager.save(
            step=self.global_step,
            training_state=self.state,
            tag=tag,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation."""
        if self.eval_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            loss = self._training_step(batch)
            total_loss += loss.item()
            num_batches += 1

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)

        # Sync across processes
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / self.world_size

        return avg_loss


# =============================================================================
# Callbacks
# =============================================================================

class EarlyStoppingCallback:
    """Early stopping callback."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, trainer: Trainer, step: int, loss: float):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping at step {step}")
                trainer.global_step = trainer.config.max_steps  # Stop training


class WandBCallback:
    """Weights & Biases logging callback."""

    def __init__(self, project: str, name: Optional[str] = None):
        try:
            import wandb
            self.wandb = wandb
            if is_main_process():
                wandb.init(project=project, name=name)
        except ImportError:
            self.wandb = None
            logger.warning("wandb not installed, skipping W&B logging")

    def __call__(self, trainer: Trainer, step: int, loss: float):
        if self.wandb and is_main_process():
            self.wandb.log({
                "loss": loss,
                "lr": trainer.scheduler.get_last_lr()[0],
                "step": step,
            })


class TensorBoardCallback:
    """TensorBoard logging callback."""

    def __init__(self, log_dir: str = "runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            if is_main_process():
                self.writer = SummaryWriter(log_dir)
            else:
                self.writer = None
        except ImportError:
            self.writer = None
            logger.warning("tensorboard not installed")

    def __call__(self, trainer: Trainer, step: int, loss: float):
        if self.writer:
            self.writer.add_scalar("Loss/train", loss, step)
            self.writer.add_scalar("LR", trainer.scheduler.get_last_lr()[0], step)
