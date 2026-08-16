"""
Distributed Checkpointing for framework-complexity.

Handles saving and loading of model checkpoints in distributed settings:
- Sharded checkpoints for FSDP
- Full checkpoints for single GPU
- Optimizer state saving/loading
- Training state (step, epoch, etc.)

References:
- PyTorch distributed checkpointing: https://pytorch.org/docs/stable/distributed.checkpoint.html
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.checkpoint import (
    save_state_dict,
    load_state_dict,
    FileSystemReader,
    FileSystemWriter,
)
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


def _is_dtensor(obj) -> bool:
    """Check if an object is a DTensor (without importing DTensor directly)."""
    return type(obj).__name__ == "DTensor"


def _has_dtensors(obj) -> bool:
    """Recursively check if a nested structure contains any DTensor values."""
    if _is_dtensor(obj):
        return True
    if isinstance(obj, dict):
        return any(_has_dtensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_dtensors(v) for v in obj)
    return False


def _detensor_state_dict(state_dict: dict) -> dict:
    """Recursively convert DTensor values to plain tensors.

    Uses to_local() instead of full_tensor() to avoid all_gather collectives
    that require an active NCCL backend. The checkpoint was saved via
    _save_regular (rank 0 only) with FSDP v2's full state_dict, so DTensors
    carry Replicate() placement — to_local() returns the complete tensor.
    """
    result = {}
    for k, v in state_dict.items():
        if _is_dtensor(v):
            result[k] = v.to_local().contiguous()
        elif isinstance(v, dict):
            result[k] = _detensor_state_dict(v)
        else:
            result[k] = v
    return result


def _rewrap_optimizer_state_as_dtensors(optimizer: torch.optim.Optimizer) -> None:
    """Convert plain tensor optimizer moments back to DTensors matching each param.

    After loading per-rank optimizer files (saved with to_local()), the moments
    are plain tensors. FSDP v2 params are DTensors, so AdamW's lerp_ fails with
    'mixed torch.Tensor and DTensor'. This converts exp_avg / exp_avg_sq back to
    DTensors with the same device_mesh and placements as the corresponding param.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return

    for group in optimizer.param_groups:
        for p in group["params"]:
            if not isinstance(p, DTensor):
                continue
            if p not in optimizer.state:
                continue
            state = optimizer.state[p]
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sqs"):
                if key not in state:
                    continue
                val = state[key]
                if isinstance(val, DTensor):
                    continue  # already wrapped
                if not isinstance(val, torch.Tensor):
                    continue
                try:
                    device = torch.device(f"cuda:{dist.get_rank() if dist.is_initialized() else 0}")
                    state[key] = DTensor.from_local(
                        val.to(device),
                        p.device_mesh,
                        p.placements,
                        run_check=False,
                    )
                except Exception:
                    pass  # leave as plain tensor if anything goes wrong


@dataclass
class TrainingState:
    """Training state to save/restore."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    total_tokens: int = 0
    learning_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        return cls(**data)


class CheckpointManager:
    """
    Manages model checkpoints for distributed training.

    Features:
    - Save/load model weights
    - Save/load optimizer state
    - Save/load training state
    - Support for FSDP sharded checkpoints
    - Automatic checkpoint rotation

    Usage:
        manager = CheckpointManager(
            checkpoint_dir="checkpoints",
            model=model,
            optimizer=optimizer,
        )

        # Save
        manager.save(step=1000, training_state=state)

        # Load
        manager.load(checkpoint_path="checkpoints/step_1000")

        # Load latest
        manager.load_latest()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        max_checkpoints: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_checkpoints = max_checkpoints

        self.is_fsdp = isinstance(model, FSDP)
        self.is_main = not dist.is_initialized() or dist.get_rank() == 0

    def save(
        self,
        step: int,
        training_state: Optional[TrainingState] = None,
        tag: str = "step",
    ) -> str:
        """
        Save a checkpoint.

        Args:
            step: Current training step
            training_state: Training state to save
            tag: Checkpoint tag (e.g., "step", "best", "final")

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"{tag}_{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if self.is_fsdp:
            self._save_fsdp(checkpoint_path)
        else:
            self._save_regular(checkpoint_path)

        # Save training state (only on main process)
        if self.is_main and training_state is not None:
            state_path = checkpoint_path / "training_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, 'w') as f:
                json.dump(training_state.to_dict(), f, indent=2)

        # Rotate old checkpoints
        self._rotate_checkpoints(tag)

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    def _save_fsdp(self, checkpoint_path: Path):
        """Save FSDP sharded checkpoint."""
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Option 1: Sharded checkpoint (recommended for large models)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            state_dict = {"model": self.model.state_dict()}

            if self.optimizer is not None:
                state_dict["optimizer"] = FSDP.optim_state_dict(
                    self.model,
                    self.optimizer,
                )

            save_state_dict(
                state_dict=state_dict,
                storage_writer=FileSystemWriter(str(checkpoint_path)),
            )

    def _save_regular(self, checkpoint_path: Path):
        """Save checkpoint using torch.distributed.checkpoint.state_dict API.

        Uses get_model_state_dict (full, all-gather on all ranks) and
        get_optimizer_state_dict (per-rank shards) for correct FSDP v2 handling.
        """
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Model weights: all ranks participate in all-gather, rank 0 saves
        try:
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
            )
            model_sd = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if self.is_main:
                save_dict = {"model": model_sd}
                if self.scheduler is not None:
                    save_dict["scheduler"] = self.scheduler.state_dict()
                torch.save(save_dict, checkpoint_path / "checkpoint.pt")

                # Also save as safetensors (HF-compatible, full weights)
                try:
                    from safetensors.torch import save_file
                    # Convert DTensors to plain tensors for safetensors
                    clean_sd = {}
                    for k, v in model_sd.items():
                        if _is_dtensor(v):
                            clean_sd[k] = v.to_local().contiguous()
                        elif isinstance(v, torch.Tensor):
                            clean_sd[k] = v.contiguous()
                    save_file(clean_sd, checkpoint_path / "model.safetensors")
                except Exception as e:
                    logger.debug(f"safetensors save skipped: {e}")

            # Optimizer: ensure all params have state initialized before saving
            if self.optimizer is not None:
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p not in self.optimizer.state:
                            self.optimizer.state[p]["step"] = torch.tensor(0, dtype=torch.float32)
                            self.optimizer.state[p]["exp_avg"] = torch.zeros_like(p)
                            self.optimizer.state[p]["exp_avg_sq"] = torch.zeros_like(p)
                opt_sd = get_optimizer_state_dict(
                    self.model, self.optimizer,
                    options=StateDictOptions(full_state_dict=False, cpu_offload=True),
                )
                torch.save(opt_sd, checkpoint_path / f"optimizer_rank{rank}.pt")

        except Exception as e:
            # The new torch.distributed.checkpoint.state_dict API requires
            # optimizer.state to be a flat {param: state_dict} mapping, which
            # MuonTRWithAdamW does not expose (it forwards to two underlying
            # optimizers). Fall back to per-rank legacy save which handles our
            # wrapper correctly. Logged at debug level since this is expected.
            logger.debug(f"get_model_state_dict not compatible ({e}), using legacy save")
            if self.is_main:
                model_sd = {"model": self.model.state_dict()}
                if self.scheduler is not None:
                    model_sd["scheduler"] = self.scheduler.state_dict()
                torch.save(model_sd, checkpoint_path / "checkpoint.pt")

                # Also save as safetensors for HF-compatible loading
                try:
                    from safetensors.torch import save_file
                    raw_sd = self.model.state_dict()
                    clean_sd = {}
                    for k, v in raw_sd.items():
                        if hasattr(v, "full_tensor"):
                            try:
                                v = v.full_tensor()
                            except Exception:
                                pass
                        if _is_dtensor(v):
                            v = v.to_local()
                        if isinstance(v, torch.Tensor):
                            clean_sd[k] = v.detach().cpu().contiguous()
                    save_file(clean_sd, checkpoint_path / "model.safetensors")
                except Exception as ex:
                    logger.debug(f"safetensors save skipped: {ex}")

            if self.optimizer is not None:
                opt_sd = self.optimizer.state_dict()
                opt_sd_local = _detensor_state_dict(opt_sd)
                torch.save(opt_sd_local, checkpoint_path / f"optimizer_rank{rank}.pt")

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        load_optimizer: bool = True,
    ) -> Optional[TrainingState]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (None = load latest)
            load_optimizer: Whether to load optimizer state

        Returns:
            Training state if available
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path is None:
                logger.warning("No checkpoint found")
                return None

        checkpoint_path = Path(checkpoint_path)

        if self.is_fsdp:
            self._load_fsdp(checkpoint_path, load_optimizer)
        else:
            self._load_regular(checkpoint_path, load_optimizer)

        # Load training state
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                return TrainingState.from_dict(json.load(f))

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return None

    def _load_fsdp(self, checkpoint_path: Path, load_optimizer: bool):
        """Load FSDP sharded checkpoint."""
        # Fall back to regular load if no .metadata (checkpoint saved via _save_regular)
        if not (checkpoint_path / ".metadata").exists():
            logger.info("No .metadata found, falling back to regular checkpoint load")
            self._load_regular(checkpoint_path, load_optimizer)
            return

        with FSDP.state_dict_type(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            state_dict = {"model": self.model.state_dict()}

            if load_optimizer and self.optimizer is not None:
                state_dict["optimizer"] = FSDP.optim_state_dict(
                    self.model,
                    self.optimizer,
                )

            load_state_dict(
                state_dict=state_dict,
                storage_reader=FileSystemReader(str(checkpoint_path)),
            )

            self.model.load_state_dict(state_dict["model"])

            if load_optimizer and self.optimizer is not None:
                FSDP.optim_state_dict_to_load(
                    self.model,
                    self.optimizer,
                    state_dict["optimizer"],
                )

    def _load_regular(self, checkpoint_path: Path, load_optimizer: bool):
        """Load checkpoint using torch.distributed.checkpoint.state_dict API."""
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

        # Model weights
        try:
            from torch.distributed.checkpoint.state_dict import (
                set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
            )
            set_model_state_dict(
                self.model, state_dict["model"],
                options=StateDictOptions(full_state_dict=True, strict=True),
            )
        except Exception as e:
            logger.warning(f"set_model_state_dict failed ({e}), using load_state_dict")
            self.model.load_state_dict(state_dict["model"])

        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

        # Optimizer state
        if load_optimizer and self.optimizer is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            rank_opt_file = checkpoint_path / f"optimizer_rank{rank}.pt"
            if rank_opt_file.exists():
                try:
                    from torch.distributed.checkpoint.state_dict import (
                        set_optimizer_state_dict, StateDictOptions,
                    )
                    opt_sd = torch.load(rank_opt_file, map_location="cpu", weights_only=False)
                    set_optimizer_state_dict(
                        self.model, self.optimizer,
                        optim_state_dict=opt_sd,
                        options=StateDictOptions(full_state_dict=False),
                    )
                    logger.info(f"Loaded optimizer state from {rank_opt_file.name}")
                except Exception as e:
                    logger.warning(f"Skipping optimizer state ({rank_opt_file.name}): {e} — moments restart from zero")

    def load_latest(self, load_optimizer: bool = True) -> Optional[TrainingState]:
        """Load the most recent checkpoint."""
        return self.load(checkpoint_path=None, load_optimizer=load_optimizer)

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("step_*"))
        if not checkpoints:
            return None

        # Sort by step number
        def get_step(p):
            try:
                return int(p.name.split("_")[1])
            except:
                return 0

        checkpoints.sort(key=get_step, reverse=True)
        return checkpoints[0]

    def _rotate_checkpoints(self, tag: str):
        """Remove old checkpoints to save disk space."""
        if not self.is_main:
            return

        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{tag}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep only max_checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            if checkpoint.is_dir():
                import shutil
                shutil.rmtree(checkpoint)
            else:
                checkpoint.unlink()

    def save_best(
        self,
        step: int,
        loss: float,
        training_state: Optional[TrainingState] = None,
    ) -> bool:
        """
        Save checkpoint only if it's the best so far.

        Args:
            step: Current training step
            loss: Current loss value
            training_state: Training state

        Returns:
            True if checkpoint was saved (new best)
        """
        if training_state is None:
            training_state = TrainingState(step=step)

        if loss < training_state.best_loss:
            training_state.best_loss = loss
            self.save(step, training_state, tag="best")
            return True

        return False


# =============================================================================
# Activation Checkpointing (Gradient Checkpointing)
# =============================================================================

def enable_activation_checkpointing(model: nn.Module, checkpoint_layers: bool = True):
    """
    Enable activation checkpointing to reduce memory usage.

    Activation checkpointing recomputes activations during backward
    instead of storing them, trading compute for memory.

    Args:
        model: The model to enable checkpointing for
        checkpoint_layers: Whether to checkpoint transformer layers
    """
    from torch.utils.checkpoint import checkpoint

    if hasattr(model, 'layers'):
        for layer in model.layers:
            layer.use_checkpoint = True

    # For FSDP models
    if isinstance(model, FSDP):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        # Apply to transformer blocks
        apply_activation_checkpointing(
            model,
            check_fn=lambda module: hasattr(module, 'self_attn'),
            checkpoint_impl=CheckpointImpl.REENTRANT,
        )


def checkpoint_sequential(
    functions: list,
    input: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Apply gradient checkpointing to sequential functions.

    Args:
        functions: List of functions to apply
        input: Input tensor
        **kwargs: Additional arguments

    Returns:
        Output tensor
    """
    from torch.utils.checkpoint import checkpoint

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end):
                input = functions[j](input, **kwargs)
            return input
        return forward

    # Checkpoint every function
    for i, fn in enumerate(functions):
        input = checkpoint(run_function(i, i + 1, functions), input, use_reentrant=False)

    return input
