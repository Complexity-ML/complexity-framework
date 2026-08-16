#!/usr/bin/env python3
"""Build a zero-copy 70B-token selection and quality-replay plan.

The plan references rows in an existing ``tr-hash-token-mixture-v1`` dataset.
It never copies or retokenizes token shards. Optional per-shard quality scores
rank shards within each corpus; without scores, stable manifest order is used
and the plan does not claim shard-level quality ranking.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

from complexity.training import PretokenizedCorpusMixtureDataset
from complexity.training.corpus_mixture import REPLAY_PLAN_FORMAT

DEFAULT_UNIQUE_BUDGETS = {
    "dclm": 20_000_000_000,
    "fineweb_edu_dedup": 25_000_000_000,
    "stack_edu": 8_000_000_000,
    "finemath": 5_000_000_000,
    "infiwebmath": 5_000_000_000,
    "cosmopedia_v2": 7_000_000_000,
}
DEFAULT_REPLAY_PASSES = {
    "dclm": 1,
    "fineweb_edu_dedup": 2,
    "stack_edu": 2,
    "finemath": 3,
    "infiwebmath": 3,
    "cosmopedia_v2": 2,
}


def _parse_token_count(value: str) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT]?)\s*", value.upper())
    if match is None:
        raise ValueError(f"invalid token count: {value!r}")
    scale = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
    return int(float(match.group(1)) * scale[match.group(2)])


def _parse_mapping(value: str, *, token_counts: bool) -> dict[str, int]:
    result = {}
    for item in value.split(","):
        name, separator, raw = item.partition("=")
        if not separator or not name.strip() or not raw.strip():
            raise ValueError(f"invalid name=value entry: {item!r}")
        result[name.strip()] = _parse_token_count(raw) if token_counts else int(raw)
    return result


def _format_mapping(values: Mapping[str, int], *, suffix: str = "") -> str:
    return ",".join(f"{name}={value}{suffix}" for name, value in values.items())


def build_replay_plan(
    dataset: PretokenizedCorpusMixtureDataset,
    *,
    unique_token_budgets: Mapping[str, int],
    replay_passes: Mapping[str, int],
    row_alignment: int,
    quality_scores: Mapping[str, float] | None = None,
) -> dict:
    if row_alignment < 1:
        raise ValueError("row_alignment must be positive")
    known_sources = {source.name for source in dataset.sources}
    if set(unique_token_budgets) != set(replay_passes):
        raise ValueError("unique budgets and replay passes must name the same sources")
    unknown = set(unique_token_budgets) - known_sources
    if unknown:
        raise ValueError(f"unknown replay-plan sources: {sorted(unknown)}")
    selected_by_source = {}
    actual_budget_tokens = {}
    for source_name, token_budget in unique_token_budgets.items():
        if token_budget < dataset.seq_len * row_alignment:
            raise ValueError(f"token budget for {source_name} is too small")
        passes = int(replay_passes[source_name])
        if passes < 1:
            raise ValueError(f"replay passes for {source_name} must be positive")
        target_rows = token_budget // dataset.seq_len
        target_rows -= target_rows % row_alignment
        shards = list(dataset._source_manifests[source_name]["shards"])
        if quality_scores is not None:
            keys = [f"corpora/{source_name}/{shard['file']}" for shard in shards]
            missing = [key for key in keys if key not in quality_scores]
            if missing:
                raise ValueError(
                    f"quality scores missing {len(missing)} shard(s) for {source_name}"
                )
            shards.sort(
                key=lambda shard: (
                    -float(quality_scores[f"corpora/{source_name}/{shard['file']}"]),
                    str(shard["file"]),
                )
            )
        remaining = target_rows
        selections = []
        for shard in shards:
            if remaining == 0:
                break
            rows = min(int(shard["rows"]), remaining)
            selections.append({"file": str(shard["file"]), "rows": rows})
            remaining -= rows
        if remaining:
            raise ValueError(
                f"source {source_name} has {remaining:,} fewer rows than requested"
            )
        selected_by_source[source_name] = selections
        actual_budget_tokens[source_name] = target_rows * dataset.seq_len

    phases = [
        {
            "name": "unique_core",
            "passes": 1,
            "sources": selected_by_source,
        }
    ]
    for pass_number in range(2, max(replay_passes.values()) + 1):
        sources = {
            name: selected_by_source[name]
            for name, passes in replay_passes.items()
            if passes >= pass_number
        }
        if sources:
            phases.append(
                {
                    "name": f"quality_replay_{pass_number}",
                    "passes": 1,
                    "sources": sources,
                }
            )
    unique_tokens = sum(actual_budget_tokens.values())
    trained_tokens = sum(
        actual_budget_tokens[name] * int(replay_passes[name])
        for name in actual_budget_tokens
    )
    return {
        "format": REPLAY_PLAN_FORMAT,
        "seq_len": dataset.seq_len,
        "selection_mode": "quality_score" if quality_scores is not None else "manifest_order",
        "row_alignment": row_alignment,
        "requested_unique_tokens": sum(unique_token_budgets.values()),
        "unique_tokens": unique_tokens,
        "trained_tokens": trained_tokens,
        "source_unique_tokens": actual_budget_tokens,
        "source_passes": dict(replay_passes),
        "phases": phases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-data", required=True)
    parser.add_argument("--output", default="plans/tr_hash_70b_quality_replay.json")
    parser.add_argument(
        "--unique-budgets",
        default=_format_mapping(
            {name: tokens // 10**9 for name, tokens in DEFAULT_UNIQUE_BUDGETS.items()},
            suffix="B",
        ),
    )
    parser.add_argument(
        "--replay-passes",
        default=_format_mapping(DEFAULT_REPLAY_PASSES),
    )
    parser.add_argument("--row-alignment", type=int, default=512)
    parser.add_argument(
        "--quality-scores",
        default=None,
        help="Optional JSON mapping corpora/<source>/<shard>.bin to a numeric score.",
    )
    parser.add_argument("--cache-dir", default="artifacts/tr_hash_token_cache")
    parser.add_argument("--revision", default="main")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = PretokenizedCorpusMixtureDataset(
        args.tokenized_data,
        cache_dir=args.cache_dir,
        revision=args.revision,
        prefetch_shards=0,
    )
    quality_scores = (
        json.loads(Path(args.quality_scores).read_text(encoding="utf-8"))
        if args.quality_scores
        else None
    )
    plan = build_replay_plan(
        dataset,
        unique_token_budgets=_parse_mapping(args.unique_budgets, token_counts=True),
        replay_passes=_parse_mapping(args.replay_passes, token_counts=False),
        row_alignment=args.row_alignment,
        quality_scores=quality_scores,
    )
    plan["dataset"] = args.tokenized_data
    plan["revision"] = args.revision
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Replay plan: {output}")
    print(f"Unique tokens: {plan['unique_tokens']:,}")
    print(f"Trained token exposures: {plan['trained_tokens']:,}")
    print(f"Selection mode: {plan['selection_mode']}")


if __name__ == "__main__":
    main()
