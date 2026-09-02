#!/usr/bin/env python3
"""Build the audited 100M Agentic pretraining and refinement plans.

The production recipe uses a 70B-token unique core, replays a proportional
55B-token subset during pretraining (125B total token exposures), then derives
an exact one-pass 70B lexical-refinement plan from the same unique core.
Nothing is copied or retokenized: plans only reference immutable Hub shards.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from complexity.training import PretokenizedCorpusMixtureDataset, validate_refinement_plan
from complexity.training.corpus_mixture import REPLAY_PLAN_FORMAT
from scripts.build_tr_hash_refinement_plan import build_refinement_plan

DEFAULT_DATASET = "hf://datasets/AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K"
DEFAULT_REVISION = "fc738b3a10c5c093e3b34b48bcf1cb7066184706"
DEFAULT_SOURCE_CONFIG = Path("configs/agentic_pretraining/tr_hash_pretraining_125b.json")
DEFAULT_PRETRAIN_PLAN = Path(
    "configs/replay_plans/tr_hash_agentic_100m_70b_unique_125b_pretrain.json"
)
DEFAULT_REFINEMENT_PLAN = Path(
    "configs/replay_plans/tr_hash_agentic_100m_70b_refinement.json"
)


def parse_token_count(value: str) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT]?)\s*", value.upper())
    if match is None:
        raise ValueError(f"invalid token count: {value!r}")
    scale = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
    return int(float(match.group(1)) * scale[match.group(2)])


def _allocate_aligned_rows(
    requested_tokens: int,
    *,
    seq_len: int,
    row_alignment: int,
    weights: Mapping[str, int],
    capacity_rows: Mapping[str, int],
) -> dict[str, int]:
    """Allocate proportional, capacity-bounded row blocks deterministically."""

    if requested_tokens <= 0:
        raise ValueError("requested token count must be positive")
    if row_alignment < 1:
        raise ValueError("row alignment must be positive")
    if set(weights) != set(capacity_rows):
        raise ValueError("weights and capacities must name the same sources")
    if any(int(weight) <= 0 for weight in weights.values()):
        raise ValueError("source weights must be positive")

    target_blocks = requested_tokens // (seq_len * row_alignment)
    capacity_blocks = {
        name: int(rows) // row_alignment for name, rows in capacity_rows.items()
    }
    if target_blocks > sum(capacity_blocks.values()):
        raise ValueError("requested token budget exceeds available aligned rows")

    allocation = {name: 0 for name in weights}
    remaining = target_blocks
    while remaining:
        active = [
            name for name in sorted(weights) if allocation[name] < capacity_blocks[name]
        ]
        if not active:
            raise ValueError("unable to satisfy aligned token allocation")
        total_weight = sum(int(weights[name]) for name in active)
        quotas = {
            name: remaining * int(weights[name]) / total_weight for name in active
        }
        granted = 0
        for name in active:
            available = capacity_blocks[name] - allocation[name]
            blocks = min(int(quotas[name]), available)
            allocation[name] += blocks
            granted += blocks
        remaining -= granted
        if remaining == 0:
            break
        # Largest-remainder assignment also guarantees progress when every
        # proportional quota is below one block.
        ranked = sorted(
            active,
            key=lambda name: (-(quotas[name] - int(quotas[name])), name),
        )
        for name in ranked:
            if remaining == 0:
                break
            if allocation[name] < capacity_blocks[name]:
                allocation[name] += 1
                remaining -= 1

    return {name: blocks * row_alignment for name, blocks in allocation.items()}


def _select_manifest_prefix(
    shards: Sequence[Mapping[str, Any]], target_rows: int
) -> list[dict[str, Any]]:
    remaining = int(target_rows)
    selections: list[dict[str, Any]] = []
    for shard in shards:
        if remaining == 0:
            break
        rows = min(int(shard["rows"]), remaining)
        selections.append({"file": str(shard["file"]), "rows": rows})
        remaining -= rows
    if remaining:
        raise ValueError(f"manifest is missing {remaining:,} requested rows")
    return selections


def _select_core_prefix(
    selections: Sequence[Mapping[str, Any]], target_rows: int
) -> list[dict[str, Any]]:
    return _select_manifest_prefix(selections, target_rows)


def build_plans(
    dataset: PretokenizedCorpusMixtureDataset,
    source_weights: Mapping[str, int],
    *,
    dataset_uri: str,
    revision: str,
    requested_unique_tokens: int,
    requested_pretrain_tokens: int,
    row_alignment: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if requested_pretrain_tokens < requested_unique_tokens:
        raise ValueError("pretraining tokens must be at least the unique-token budget")
    known_sources = {source.name for source in dataset.sources}
    if set(source_weights) != known_sources:
        missing = sorted(known_sources - set(source_weights))
        unknown = sorted(set(source_weights) - known_sources)
        raise ValueError(f"source config mismatch: missing={missing}, unknown={unknown}")

    core_rows = _allocate_aligned_rows(
        requested_unique_tokens,
        seq_len=dataset.seq_len,
        row_alignment=row_alignment,
        weights=source_weights,
        capacity_rows=dataset._rows_by_source,
    )
    replay_rows = _allocate_aligned_rows(
        requested_pretrain_tokens - requested_unique_tokens,
        seq_len=dataset.seq_len,
        row_alignment=row_alignment,
        weights=core_rows,
        capacity_rows=core_rows,
    )

    core_sources = {
        name: _select_manifest_prefix(
            dataset._source_manifests[name]["shards"], core_rows[name]
        )
        for name in sorted(core_rows)
    }
    replay_sources = {
        name: _select_core_prefix(core_sources[name], replay_rows[name])
        for name in sorted(replay_rows)
        if replay_rows[name]
    }
    actual_unique_tokens = sum(core_rows.values()) * dataset.seq_len
    actual_replay_tokens = sum(replay_rows.values()) * dataset.seq_len
    pretrain_plan: dict[str, Any] = {
        "format": REPLAY_PLAN_FORMAT,
        "seq_len": dataset.seq_len,
        "selection_mode": "balanced_manifest_order",
        "row_alignment": row_alignment,
        "requested_unique_tokens": requested_unique_tokens,
        "requested_trained_tokens": requested_pretrain_tokens,
        "unique_tokens": actual_unique_tokens,
        "trained_tokens": actual_unique_tokens + actual_replay_tokens,
        "source_unique_tokens": {
            name: rows * dataset.seq_len for name, rows in core_rows.items()
        },
        "source_replay_tokens": {
            name: rows * dataset.seq_len for name, rows in replay_rows.items()
        },
        "source_passes": {
            name: 2 if replay_rows[name] else 1 for name in core_rows
        },
        "phases": [
            {"name": "unique_core", "passes": 1, "sources": core_sources},
            {"name": "balanced_replay_2", "passes": 1, "sources": replay_sources},
        ],
        "dataset": dataset_uri,
        "revision": revision,
        "recipe": {
            "model_preset": "complexity-100m",
            "unique_core": "70B proportional selection from the immutable 125B corpus",
            "pretraining": "70B unique + 55B proportional replay",
            "refinement": "one clean pass over the exact 70B unique core",
        },
    }
    refinement_plan = build_refinement_plan(pretrain_plan)
    validate_refinement_plan(refinement_plan, pretrain_plan)
    return pretrain_plan, refinement_plan


def _source_weights(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source config must contain a non-empty sources list")
    return {str(source["name"]): int(source["target_tokens"]) for source in sources}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-data", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--pretrain-output", type=Path, default=DEFAULT_PRETRAIN_PLAN)
    parser.add_argument("--refinement-output", type=Path, default=DEFAULT_REFINEMENT_PLAN)
    parser.add_argument("--unique-tokens", type=parse_token_count, default="70B")
    parser.add_argument("--pretrain-tokens", type=parse_token_count, default="125B")
    parser.add_argument("--row-alignment", type=int, default=512)
    parser.add_argument("--cache-dir", default="artifacts/tr_hash_agentic_100m_plan_cache")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = PretokenizedCorpusMixtureDataset(
        args.tokenized_data,
        cache_dir=args.cache_dir,
        revision=args.revision,
        prefetch_shards=0,
    )
    pretrain, refinement = build_plans(
        dataset,
        _source_weights(args.source_config),
        dataset_uri=args.tokenized_data,
        revision=args.revision,
        requested_unique_tokens=args.unique_tokens,
        requested_pretrain_tokens=args.pretrain_tokens,
        row_alignment=args.row_alignment,
    )
    for path, payload in (
        (args.pretrain_output, pretrain),
        (args.refinement_output, refinement),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Pretraining plan: {args.pretrain_output}")
    print(
        f"  unique={pretrain['unique_tokens']:,} trained={pretrain['trained_tokens']:,}"
    )
    print(f"Refinement plan: {args.refinement_output}")
    print(f"  unique/trained={refinement['trained_tokens']:,}")


if __name__ == "__main__":
    main()
