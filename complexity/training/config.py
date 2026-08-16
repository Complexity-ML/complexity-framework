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
    optimizer_type: str = "adamw"  # adamw, adamw_mup, muon, muon_tr, adam_tr
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

    # Diagnostic: assert all canary params are updated after first step.
    # Catches silent zero-grad bugs from forward-only custom kernels.
    # Set True only when intentionally freezing some params.
    skip_param_update_check: bool = False

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

    def __post_init__(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive when set")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not 0.0 <= self.wsd_decay_ratio <= 1.0:
            raise ValueError("wsd_decay_ratio must be in [0, 1]")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive")
        if self.log_steps <= 0:
            raise ValueError("log_steps must be positive")
        if self.eval_steps < 0:
            raise ValueError("eval_steps must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

    @property
    def eval_every_n_steps(self) -> int:
        """Backward-compatible alias for eval_steps."""
        return self.eval_steps

    @eval_every_n_steps.setter
    def eval_every_n_steps(self, value: int) -> None:
        self.eval_steps = value

    @property
    def save_every_n_steps(self) -> int:
        """Backward-compatible alias for save_steps."""
        return self.save_steps

    @save_every_n_steps.setter
    def save_every_n_steps(self, value: int) -> None:
        self.save_steps = value

    @property
    def log_every_n_steps(self) -> int:
        """Backward-compatible alias for log_steps."""
        return self.log_steps

    @log_every_n_steps.setter
    def log_every_n_steps(self, value: int) -> None:
        self.log_steps = value

    @property
    def max_grad_norm(self) -> float:
        """Backward-compatible alias for grad_clip."""
        return self.grad_clip

    @max_grad_norm.setter
    def max_grad_norm(self, value: float) -> None:
        self.grad_clip = value

    @property
    def output_dir(self) -> str:
        """Backward-compatible alias for checkpoint_dir."""
        return self.checkpoint_dir

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        self.checkpoint_dir = value

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
