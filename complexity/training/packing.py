"""Task-agnostic packed-epoch scheduling contracts.

Packing changes the number of optimizer inputs needed to expose the same
number of source items.  Task adapters provide an exposure factor per epoch;
this module owns the shared step accounting and its invariants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PackingKind(str, Enum):
    """Packing semantics used by a training input pipeline."""

    TOKEN_STREAM = "token-stream"
    SEQUENCE = "sequence"
    SOURCE_EXPOSURE = "source-exposure"
    FIXED_SHAPE = "fixed-shape"


@dataclass(frozen=True)
class PipelinePackingContract:
    """Declare whether a training family must pack its source inputs."""

    pipeline: str
    kind: PackingKind
    required: bool
    reason: str

    def validate(self) -> None:
        if not self.pipeline.strip():
            raise ValueError("packing contract pipeline name cannot be empty")
        if not self.reason.strip():
            raise ValueError(f"{self.pipeline} packing contract requires a reason")
        if self.kind is PackingKind.FIXED_SHAPE and self.required:
            raise ValueError("fixed-shape pipelines cannot require packing")
        if self.kind is not PackingKind.FIXED_SHAPE and not self.required:
            raise ValueError(
                f"{self.pipeline} supports {self.kind.value} packing and must enable it"
            )


# Every core training family must be present here. Variable-length/source-
# composition pipelines pack by contract; naturally fixed-shape pipelines
# explicitly declare why packing is not applicable instead of silently using 0.
FRAMEWORK_PACKING_CONTRACTS: dict[str, PipelinePackingContract] = {
    "text-pretraining": PipelinePackingContract(
        "text-pretraining",
        PackingKind.TOKEN_STREAM,
        True,
        "concatenate documents into full causal-token windows",
    ),
    "supervised-finetuning": PipelinePackingContract(
        "supervised-finetuning",
        PackingKind.SEQUENCE,
        True,
        "pack complete supervised examples into full token windows",
    ),
    "vision-supervised-finetuning": PipelinePackingContract(
        "vision-supervised-finetuning",
        PackingKind.FIXED_SHAPE,
        False,
        "detector examples are already fixed-shape image and target tensors",
    ),
    "detector-pretraining": PipelinePackingContract(
        "detector-pretraining",
        PackingKind.SOURCE_EXPOSURE,
        True,
        "count all source images composed into each Mosaic canvas",
    ),
    "vision-pretraining": PipelinePackingContract(
        "vision-pretraining",
        PackingKind.FIXED_SHAPE,
        False,
        "each sample is already one fixed-shape image tensor",
    ),
    "audio-pretraining": PipelinePackingContract(
        "audio-pretraining",
        PackingKind.FIXED_SHAPE,
        False,
        "the audio collator already batches padded feature tensors",
    ),
    "video-pretraining": PipelinePackingContract(
        "video-pretraining",
        PackingKind.FIXED_SHAPE,
        False,
        "each sample is a fixed-length spatiotemporal clip",
    ),
    "vision-language-training": PipelinePackingContract(
        "vision-language-training",
        PackingKind.FIXED_SHAPE,
        False,
        "multimodal examples contain indivisible image-token alignments",
    ),
    "sensor-fusion-training": PipelinePackingContract(
        "sensor-fusion-training",
        PackingKind.FIXED_SHAPE,
        False,
        "each example is an aligned fixed-window sensor observation",
    ),
}

# Token packs are checkpoint/progress partitions of an actual causal-token
# budget. They are meaningful only for token-budget text pretraining. Other
# families keep their own sample/sequence/source-exposure scheduling.
TOKEN_PACK_PIPELINES = frozenset({"text-pretraining"})


def validate_token_pack_pipeline(pipeline: str) -> None:
    """Reject accidental reuse of token-pack scheduling outside pretraining."""

    if pipeline not in TOKEN_PACK_PIPELINES:
        raise ValueError(
            f"token packs are restricted to text-pretraining, got {pipeline!r}"
        )


def validate_framework_packing_contracts() -> None:
    """Fail fast when a training family has an ambiguous packing policy."""

    for key, contract in FRAMEWORK_PACKING_CONTRACTS.items():
        if key != contract.pipeline:
            raise ValueError(f"packing contract key mismatch: {key} != {contract.pipeline}")
        contract.validate()


@dataclass(frozen=True)
class PackedEpochSchedule:
    """Resolved packed steps with source-exposure conservation metadata."""

    full_steps: int
    exposure_factors: tuple[float, ...]
    steps: tuple[int, ...]
    enabled: bool

    @property
    def total_steps(self) -> int:
        return sum(self.steps)

    @property
    def unpacked_total_steps(self) -> int:
        return self.full_steps * len(self.steps)

    def assert_source_exposure(self) -> None:
        """Ensure packing never underexposes an epoch's source-item budget."""

        for steps, factor in zip(self.steps, self.exposure_factors, strict=True):
            exposure = steps * factor
            if exposure < self.full_steps:
                raise ValueError("packed epoch underexposes source items")
            if self.enabled and exposure >= self.full_steps + factor:
                raise ValueError("packed epoch rounding exceeds one packed step")


def resolve_packed_epoch_schedule(
    *,
    full_steps: int,
    exposure_factors: Iterable[float],
    enabled: bool = True,
) -> PackedEpochSchedule:
    """Convert per-step source exposure into a task-independent epoch schedule."""

    full_steps = int(full_steps)
    if full_steps < 1:
        raise ValueError("full_steps must be positive")
    factors = tuple(float(factor) for factor in exposure_factors)
    if not factors:
        raise ValueError("packed epoch schedule requires at least one epoch")
    if any(not math.isfinite(factor) or factor < 1.0 for factor in factors):
        raise ValueError("packed epoch exposure factors must be finite and at least one")
    steps = tuple(
        math.ceil(full_steps / factor) if enabled else full_steps
        for factor in factors
    )
    schedule = PackedEpochSchedule(
        full_steps=full_steps,
        exposure_factors=factors if enabled else (1.0,) * len(factors),
        steps=steps,
        enabled=bool(enabled),
    )
    schedule.assert_source_exposure()
    return schedule


@dataclass(frozen=True)
class TokenPackSchedule:
    """Split a token-budget pretrain into checkpoint packs without epochs."""

    target_tokens: int
    tokens_per_step: int
    token_packs: int
    total_steps: int
    boundaries: tuple[int, ...]

    @property
    def actual_tokens(self) -> int:
        return self.total_steps * self.tokens_per_step

    @property
    def pack_step_counts(self) -> tuple[int, ...]:
        previous = 0
        counts: list[int] = []
        for boundary in self.boundaries:
            counts.append(boundary - previous)
            previous = boundary
        return tuple(counts)

    def assert_token_budget(self) -> None:
        if len(self.boundaries) != self.token_packs:
            raise ValueError("token-pack boundary count mismatch")
        if self.boundaries[-1] != self.total_steps:
            raise ValueError("final token pack must end at total_steps")
        if any(count < 1 for count in self.pack_step_counts):
            raise ValueError("every token pack must contain at least one optimizer step")
        if self.actual_tokens < self.target_tokens:
            raise ValueError("token packs undertrain the requested token budget")
        if self.actual_tokens >= self.target_tokens + self.tokens_per_step:
            raise ValueError("token-pack rounding exceeds one optimizer step")


def resolve_token_pack_schedule(
    *,
    target_tokens: int,
    tokens_per_step: int,
    token_packs: int,
    pipeline: str = "text-pretraining",
) -> TokenPackSchedule:
    """Resolve exact optimizer-step boundaries for a token-only schedule."""

    validate_token_pack_pipeline(pipeline)
    target_tokens = int(target_tokens)
    tokens_per_step = int(tokens_per_step)
    token_packs = int(token_packs)
    if target_tokens < 1 or tokens_per_step < 1 or token_packs < 1:
        raise ValueError("token-pack schedule values must all be positive")
    total_steps = math.ceil(target_tokens / tokens_per_step)
    if token_packs > total_steps:
        raise ValueError("token packs cannot outnumber optimizer steps")
    boundaries = tuple(
        math.ceil(pack * total_steps / token_packs)
        for pack in range(1, token_packs + 1)
    )
    schedule = TokenPackSchedule(
        target_tokens=target_tokens,
        tokens_per_step=tokens_per_step,
        token_packs=token_packs,
        total_steps=total_steps,
        boundaries=boundaries,
    )
    schedule.assert_token_budget()
    return schedule
