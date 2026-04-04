"""Training configuration."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer_type: str = "adamw"  # adamw, adamw_mup, muon, muon_tr
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    muon_lr: float = 0.02         # Muon LR for 2D weights (only used when optimizer_type="muon" or "muon_tr")
    mup_base_width: int = 256     # muP reference width (only used when optimizer_type="adamw_mup")
    expert_lr_scale: float = 1.5  # LR multiplier for routed experts (muon_tr only)
    expert_weight_decay: float = 0.005  # Weight decay for experts (muon_tr only)
    adaptive_ns: bool = True      # Adaptive Newton-Schulz iterations per expert (muon_tr only)
    warmup_steps: int = 1000
    lr_scheduler: str = "auto"    # auto, cosine, wsd, linear, constant
    min_lr_ratio: float = 0.1
    wsd_decay_ratio: float = 0.2  # WSD: fraction of post-warmup steps for decay (default: 20%)

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
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "muon_lr": self.muon_lr,
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
