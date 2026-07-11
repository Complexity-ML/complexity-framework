"""Synthetic diagnostics for associative recall in causal language models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RecallBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor


@dataclass(frozen=True)
class RecallMetrics:
    accuracy: float
    mean_rank: float
    mean_margin: float
    nll: float


def _sample_distinct_pair(
    batch_size: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vocab_size < 32:
        raise ValueError("vocab_size must be at least 32")
    low = 16
    span = vocab_size - low
    first = torch.randint(low, vocab_size, (batch_size,), generator=generator)
    offset = torch.randint(1, span, (batch_size,), generator=generator)
    second = low + (first - low + offset) % span
    return first, second


def build_associative_recall_batch(
    batch_size: int,
    distance: int,
    vocab_size: int,
    seed: int,
) -> RecallBatch:
    """Create ``key, value, fillers, marker, key -> value`` examples."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if distance < 0:
        raise ValueError("distance must be non-negative")
    generator = torch.Generator().manual_seed(seed)
    keys, values = _sample_distinct_pair(batch_size, vocab_size, generator)
    fillers = torch.randint(
        16,
        vocab_size,
        (batch_size, distance),
        generator=generator,
    )
    marker = torch.full((batch_size, 1), 1, dtype=torch.long)
    input_ids = torch.cat(
        (keys[:, None], values[:, None], fillers, marker, keys[:, None]), dim=1
    )
    return RecallBatch(input_ids=input_ids, target_ids=values)


def build_induction_batch(
    batch_size: int,
    distance: int,
    vocab_size: int,
    seed: int,
) -> RecallBatch:
    """Create ``a, b, value, fillers, a, b -> value`` induction examples."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if distance <= 0:
        raise ValueError("distance must be positive")
    generator = torch.Generator().manual_seed(seed)
    first, target = _sample_distinct_pair(batch_size, vocab_size, generator)
    second, _ = _sample_distinct_pair(batch_size, vocab_size, generator)
    fillers = torch.randint(
        16,
        vocab_size,
        (batch_size, distance - 1),
        generator=generator,
    )
    input_ids = torch.cat(
        (
            first[:, None],
            second[:, None],
            target[:, None],
            fillers,
            first[:, None],
            second[:, None],
        ),
        dim=1,
    )
    return RecallBatch(input_ids=input_ids, target_ids=target)


def score_target_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> RecallMetrics:
    """Score next-token logits against one target per batch item."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if target_ids.shape != (logits.shape[0],):
        raise ValueError("target_ids must have shape [batch]")
    target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
    predictions = logits.argmax(dim=-1)
    ranks = 1 + (logits > target_logits[:, None]).sum(dim=-1)
    non_target = logits.clone()
    non_target.scatter_(1, target_ids[:, None], float("-inf"))
    margins = target_logits - non_target.max(dim=-1).values
    return RecallMetrics(
        accuracy=float((predictions == target_ids).float().mean().item()),
        mean_rank=float(ranks.float().mean().item()),
        mean_margin=float(margins.mean().item()),
        nll=float(F.cross_entropy(logits, target_ids).item()),
    )


__all__ = [
    "RecallBatch",
    "RecallMetrics",
    "build_associative_recall_batch",
    "build_induction_batch",
    "score_target_logits",
]
