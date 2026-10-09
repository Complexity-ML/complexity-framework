"""Explicit evaluation coverage and validation tracking for SFT workflows.

These inspection and state primitives do not change an evaluator, start a
runner or activate early stopping. Minimum coverage is an authored policy,
not a universal statistical guarantee or a requirement of the 200M ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


def validate_evaluation_coverage(
    train_examples: int,
    eval_examples: int,
    *,
    minimum_fraction: float,
    partition_name: str,
) -> dict:
    """Measure declared coverage and reject an explicitly undersized panel."""
    if type(train_examples) is not int or train_examples < 1:
        raise ValueError("train_examples must be a positive integer")
    if type(eval_examples) is not int or eval_examples < 0:
        raise ValueError("eval_examples must be a nonnegative integer")
    if (
        type(minimum_fraction) not in {int, float}
        or not math.isfinite(minimum_fraction)
        or not 0 <= minimum_fraction <= 1
    ):
        raise ValueError("minimum evaluation fraction must be between 0 and 1")
    if not isinstance(partition_name, str) or not partition_name.strip():
        raise ValueError("evaluation partition requires a name")
    fraction = eval_examples / train_examples
    if fraction < minimum_fraction:
        raise ValueError(
            f"{partition_name} evaluation is only {fraction:.2%} of training "
            f"({eval_examples:,}/{train_examples:,}); required minimum is "
            f"{minimum_fraction:.2%}. Use a qualitative regression panel instead."
        )
    return {
        "partition": partition_name,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "fraction": fraction,
        "minimum_fraction": minimum_fraction,
        "coverage_passed": True,
    }


@dataclass(frozen=True)
class EvaluationPartitions:
    selection: Path | None
    natural: Path | None


def discover_sft_evaluation_partitions(root: str | Path) -> EvaluationPartitions:
    """Inspect local partition names without silently selecting runner inputs."""
    root = Path(root)
    if (root / "sft.idx.json").is_file():
        selection = root if root.name in {"diagnostic", "eval"} else None
        return EvaluationPartitions(selection, None)
    diagnostic, natural = root / "diagnostic", root / "eval"
    if (diagnostic / "sft.idx.json").is_file():
        return EvaluationPartitions(
            diagnostic, natural if (natural / "sft.idx.json").is_file() else None
        )
    return EvaluationPartitions(natural if (natural / "sft.idx.json").is_file() else None, None)


def complete_epoch_eligible(step: int, *, steps_per_epoch: int, minimum_epochs: int) -> bool:
    """Gate selection/stopping until the declared number of epochs is covered."""
    for name, value, minimum in (
        ("step", step, 0),
        ("steps_per_epoch", steps_per_epoch, 1),
        ("minimum_epochs", minimum_epochs, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    return step >= steps_per_epoch * minimum_epochs


@dataclass(frozen=True)
class ValidationState:
    """A measured entry checkpoint is the baseline, never an implicit win."""

    best_loss: float = math.inf
    misses: int = 0

    def __post_init__(self):
        if (
            type(self.best_loss) not in {int, float}
            or math.isnan(self.best_loss)
            or self.best_loss == -math.inf
        ):
            raise ValueError("best_loss must be finite or positive infinity")
        if type(self.misses) is not int or self.misses < 0:
            raise ValueError("validation misses must be a nonnegative integer")

    @classmethod
    def from_baseline(cls, loss: float | None) -> ValidationState:
        if loss is not None and (type(loss) not in {int, float} or not math.isfinite(loss)):
            raise ValueError("validation baseline must be finite when provided")
        return cls(math.inf if loss is None else float(loss))

    def observe(self, loss: float, *, min_delta: float) -> tuple[bool, ValidationState]:
        if type(loss) not in {int, float} or not math.isfinite(loss):
            raise ValueError("validation loss must be finite")
        if type(min_delta) not in {int, float} or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        improved = loss < self.best_loss - min_delta
        return improved, (
            ValidationState(float(loss), 0)
            if improved
            else ValidationState(self.best_loss, self.misses + 1)
        )

    def should_stop(
        self, *, patience: int, step: int, steps_per_epoch: int, minimum_epochs: int = 1
    ) -> bool:
        if type(patience) is not int or patience < 1:
            raise ValueError("early-stopping patience must be a positive integer")
        return (
            complete_epoch_eligible(
                step, steps_per_epoch=steps_per_epoch, minimum_epochs=minimum_epochs
            )
            and self.misses >= patience
        )
