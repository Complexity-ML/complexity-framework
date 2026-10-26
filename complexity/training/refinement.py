"""Auditable zero-copy text refinement and balanced replay plans.

These functions select rows from manifest metadata. They neither tokenize,
download, allocate a model nor start a training process.
"""

from __future__ import annotations

import copy
import math
import re
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .finetuning import validate_refinement_plan

REFINEMENT_PEAK_LR = 1e-4
REFINEMENT_TOKENS_PER_STEP = 3_932_160
REFINEMENT_WARMUP_TOKENS = 500_000_000
REFINEMENT_WEIGHT_DECAY = 0.1
REFINEMENT_LR_SCHEDULER = "cosine"


def parse_token_count(value: str) -> int:
    """Parse a positive token budget, with optional K/M/B/T suffix."""
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT]?)\s*", value.upper())
    if match is None:
        raise ValueError(f"invalid token count: {value!r}")
    scale = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
    count = int(Decimal(match.group(1)) * scale[match.group(2)])
    if count < 1:
        raise ValueError("token count must be positive")
    return count


def build_refinement_plan(pretrain_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Derive one clean pass over exactly the pretraining unique core."""
    phases = pretrain_plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("pretraining plan must contain a phases list")
    matches = [phase for phase in phases if phase.get("name") == "unique_core"]
    if len(matches) != 1:
        raise ValueError("pretraining plan must contain exactly one unique_core phase")
    core = copy.deepcopy(matches[0])
    core["passes"] = 1
    if not core.get("sources"):
        raise ValueError("pretraining unique_core must contain sources")
    sequence_length = _integer(pretrain_plan.get("seq_len"), "plan sequence length")
    source_tokens = {}
    for name, selections in core["sources"].items():
        if not isinstance(selections, list) or not selections:
            raise ValueError(f"unique core {name} must contain shard selections")
        if len({selection["file"] for selection in selections}) != len(selections):
            raise ValueError(f"unique core {name} contains duplicate shard files")
        source_tokens[name] = (
            sum(
                _integer(selection["rows"], f"selected rows for {name}") for selection in selections
            )
            * sequence_length
        )
    if source_tokens != pretrain_plan.get("source_unique_tokens") or sum(
        source_tokens.values()
    ) != pretrain_plan.get("unique_tokens"):
        raise ValueError("pretraining unique_tokens must describe the exact unique_core coverage")
    refinement = {
        key: copy.deepcopy(value)
        for key, value in pretrain_plan.items()
        if key not in {"phases", "trained_tokens", "source_passes"}
    }
    refinement.update(
        trained_tokens=int(pretrain_plan["unique_tokens"]),
        source_passes={name: 1 for name in core["sources"]},
        phases=[core],
    )
    if "requested_trained_tokens" in refinement:
        refinement["requested_trained_tokens"] = refinement["trained_tokens"]
    if "source_replay_tokens" in refinement:
        refinement["source_replay_tokens"] = {name: 0 for name in source_tokens}
    if "unused_trained_tokens" in refinement:
        refinement["unused_trained_tokens"] = 0
    validate_refinement_plan(refinement, pretrain_plan)
    return refinement


def _integer(value, name, *, positive=True):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def allocate_aligned_rows(
    requested_tokens: int,
    *,
    seq_len: int,
    row_alignment: int,
    weights: Mapping[str, int],
    capacity_rows: Mapping[str, int],
) -> dict[str, int]:
    """Allocate capacity-bounded row blocks with deterministic remainders."""
    _integer(requested_tokens, "requested token count", positive=False)
    _integer(seq_len, "sequence length")
    _integer(row_alignment, "row alignment")
    if not weights or set(weights) != set(capacity_rows):
        raise ValueError("weights and capacities must name the same nonempty sources")
    for name, value in weights.items():
        _integer(value, f"source weight for {name}")
    capacities = {
        name: _integer(rows, f"capacity for {name}", positive=False) // row_alignment
        for name, rows in capacity_rows.items()
    }
    remaining = requested_tokens // (seq_len * row_alignment)
    if requested_tokens and not remaining:
        raise ValueError("requested token budget is smaller than one aligned row block")
    if remaining > sum(capacities.values()):
        raise ValueError("requested token budget exceeds available aligned rows")
    allocation = {name: 0 for name in weights}
    while remaining:
        active = [name for name in sorted(weights) if allocation[name] < capacities[name]]
        total_weight = sum(weights[name] for name in active)
        quotas = {name: divmod(remaining * weights[name], total_weight) for name in active}
        granted = 0
        for name in active:
            blocks = min(quotas[name][0], capacities[name] - allocation[name])
            allocation[name] += blocks
            granted += blocks
        remaining -= granted
        for name in sorted(active, key=lambda name: (-quotas[name][1], name)):
            if not remaining:
                break
            if allocation[name] < capacities[name]:
                allocation[name] += 1
                remaining -= 1
    return {name: blocks * row_alignment for name, blocks in allocation.items()}


def _manifest_prefix(shards: Sequence[Mapping[str, Any]], target_rows: int) -> list[dict]:
    remaining, selections = target_rows, []
    for shard in shards:
        if not remaining:
            break
        count = min(_integer(shard["rows"], "manifest shard rows"), remaining)
        selections.append({"file": str(shard["file"]), "rows": count})
        remaining -= count
    if remaining:
        raise ValueError(f"manifest is missing {remaining:,} requested rows")
    return selections


def build_agentic_plans(
    dataset,
    source_weights: Mapping[str, int],
    *,
    dataset_uri: str,
    revision: str,
    requested_unique_tokens: int,
    requested_pretrain_tokens: int,
    row_alignment: int,
) -> tuple[dict, dict]:
    """Select a weighted core and at most one proportional replay subset.

    Inputs are an already-inspected corpus object and explicit provenance.
    Alignment tails are reported in the plans and never silently overdrawn.
    Zero-replay plans are supported without an empty replay phase.
    """
    from .corpus_mixture import REPLAY_PLAN_FORMAT

    _integer(requested_unique_tokens, "unique token count")
    _integer(requested_pretrain_tokens, "pretraining token count")
    if requested_pretrain_tokens < requested_unique_tokens:
        raise ValueError("pretraining tokens must be at least the unique-token budget")
    if (
        not isinstance(dataset_uri, str)
        or not dataset_uri
        or not isinstance(revision, str)
        or not revision
    ):
        raise ValueError("dataset URI and revision must be explicitly declared")
    known_sources = {source.name for source in dataset.sources}
    if set(source_weights) != known_sources:
        raise ValueError(
            f"source config mismatch: missing={sorted(known_sources - set(source_weights))}, unknown={sorted(set(source_weights) - known_sources)}"
        )
    core = allocate_aligned_rows(
        requested_unique_tokens,
        seq_len=dataset.seq_len,
        row_alignment=row_alignment,
        weights=source_weights,
        capacity_rows=dataset._rows_by_source,
    )
    selected = {
        name: _manifest_prefix(dataset._source_manifests[name]["shards"], core[name])
        for name in sorted(core)
        if core[name]
    }
    core = {name: rows for name, rows in core.items() if rows}
    replay = allocate_aligned_rows(
        requested_pretrain_tokens - requested_unique_tokens,
        seq_len=dataset.seq_len,
        row_alignment=row_alignment,
        weights=core,
        capacity_rows=core,
    )
    replay_sources = {
        name: _manifest_prefix(selected[name], replay[name])
        for name in sorted(replay)
        if replay[name]
    }
    actual_unique = sum(core.values()) * dataset.seq_len
    actual_replay = sum(replay.values()) * dataset.seq_len
    phases = [{"name": "unique_core", "passes": 1, "sources": selected}]
    if replay_sources:
        phases.append({"name": "balanced_replay_2", "passes": 1, "sources": replay_sources})
    pretrain = {
        "format": REPLAY_PLAN_FORMAT,
        "seq_len": dataset.seq_len,
        "selection_mode": "balanced_manifest_order",
        "row_alignment": row_alignment,
        "requested_unique_tokens": requested_unique_tokens,
        "requested_trained_tokens": requested_pretrain_tokens,
        "unique_tokens": actual_unique,
        "trained_tokens": actual_unique + actual_replay,
        "source_unique_tokens": {name: rows * dataset.seq_len for name, rows in core.items()},
        "source_replay_tokens": {name: rows * dataset.seq_len for name, rows in replay.items()},
        "source_passes": {name: 2 if replay[name] else 1 for name in core},
        "unused_unique_tokens": requested_unique_tokens - actual_unique,
        "unused_trained_tokens": requested_pretrain_tokens - actual_unique - actual_replay,
        "phases": phases,
        "dataset": dataset_uri,
        "revision": revision,
        "recipe": {
            "model_preset": "complexity-100m",
            "unique_core": "proportional immutable manifest selection",
            "pretraining": "unique core followed by proportional replay",
            "refinement": "one clean pass over the exact unique core",
        },
    }
    return pretrain, build_refinement_plan(pretrain)


def validate_refinement_recipe(
    *,
    stage: str,
    model_preset: str,
    learning_rate: float,
    lr_scheduler: str,
    warmup_tokens: int | None,
    warmup_steps: int | None,
    weight_decay: float,
    tokens_per_step: int,
) -> None:
    """Preserve the historical, measured Agentic 100M optimizer contract."""
    if stage != "refinement" or model_preset != "complexity-100m":
        return
    mismatches = []
    if not math.isclose(learning_rate, REFINEMENT_PEAK_LR, rel_tol=0, abs_tol=1e-12):
        mismatches.append(f"lr={learning_rate!r} (expected {REFINEMENT_PEAK_LR})")
    if lr_scheduler != REFINEMENT_LR_SCHEDULER:
        mismatches.append(f"lr_scheduler={lr_scheduler!r} (expected {REFINEMENT_LR_SCHEDULER!r})")
    if warmup_steps is not None or warmup_tokens != REFINEMENT_WARMUP_TOKENS:
        mismatches.append(
            f"warmup must use {REFINEMENT_WARMUP_TOKENS} tokens (got tokens={warmup_tokens!r}, steps={warmup_steps!r})"
        )
    if not math.isclose(weight_decay, REFINEMENT_WEIGHT_DECAY, rel_tol=0, abs_tol=1e-12):
        mismatches.append(f"weight_decay={weight_decay!r} (expected {REFINEMENT_WEIGHT_DECAY})")
    if tokens_per_step != REFINEMENT_TOKENS_PER_STEP:
        mismatches.append(
            f"tokens_per_step={tokens_per_step:,} (expected {REFINEMENT_TOKENS_PER_STEP:,})"
        )
    if mismatches:
        raise ValueError(
            "Agentic 100M canonical refinement must reproduce the observed 200M optimizer regime: "
            + "; ".join(mismatches)
        )
