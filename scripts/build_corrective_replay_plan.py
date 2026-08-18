#!/usr/bin/env python3
"""Build a replay plan that corrects for an unplanned mid-training data reset.

A process restart with no persisted dataset position (fixed going forward by
resume_skip_rows, see corpus_mixture.py and PR that added it) makes
__iter__() start phase 1 over from shard 0 -- so the first N shards of every
source get an unplanned *extra* pass before the run catches back up to fresh
material. Left alone, sources with a scheduled 2nd/3rd replay pass would
replay those same already-double-exposed shards yet again later, stacking
even more exposure on top of what they already got by accident, while the
rest of the corpus is under-represented in comparison.

This script does NOT touch phase 1 (unique_core) -- that already happened,
correctly or not, and can't be undone. It only rewrites the LATER replay
phases: for each source with already_double_exposed_shards > 0, it drops
that many shards (the ones already over-exposed) from that source's later
replay passes, and backfills the same row count with shards the run has
never shown the model at all -- shards beyond what phase 1 already selected,
pulled from the same per-source token pool. Net effect: every source's
total trained-token exposure and phase-2/3 row counts stay exactly as
planned; only *which* rows fill that budget changes, away from the
accidentally-doubled range and onto genuinely fresh material.

Usage:
    python -m scripts.build_corrective_replay_plan \
        --tokenized-data hf://datasets/Pacific-i64/data-32k-200b-tokens \
        --already-double-exposed dclm=0,fineweb_edu_dedup=10,stack_edu=4,finemath=2,infiwebmath=2,cosmopedia_v2=3 \
        --output artifacts/tr_hash_70b_quality_replay_corrected.json

already-double-exposed is a shard COUNT (not a token count): how many of a
source's phase-1 shards (counting from the first) already got an unplanned
extra pass from the reset. DCLM has no scheduled replay pass at all, so it
never appears in a later phase and needs no correction regardless of what
happened to it during the reset -- pass 0 for it (or omit it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from complexity.training import PretokenizedCorpusMixtureDataset

from scripts.build_tr_hash_70b_replay_plan import (
    DEFAULT_REPLAY_PASSES,
    DEFAULT_UNIQUE_BUDGETS,
    _format_mapping,
    _parse_mapping,
    build_replay_plan,
)


def build_corrective_replay_plan(
    dataset: PretokenizedCorpusMixtureDataset,
    *,
    unique_token_budgets: Mapping[str, int],
    replay_passes: Mapping[str, int],
    already_double_exposed_shards: Mapping[str, int],
    row_alignment: int,
) -> dict:
    unknown = set(already_double_exposed_shards) - set(unique_token_budgets)
    if unknown:
        raise ValueError(f"unknown sources in already_double_exposed_shards: {sorted(unknown)}")

    # Phase 1 is untouched -- it already happened. Reuse its exact selection
    # logic so the corrected plan's unique_core phase is bit-for-bit what an
    # uncorrected plan would have produced.
    base_plan = build_replay_plan(
        dataset,
        unique_token_budgets=unique_token_budgets,
        replay_passes=replay_passes,
        row_alignment=row_alignment,
    )
    unique_core_selection = base_plan["phases"][0]["sources"]

    corrected_selection: dict[str, list[dict]] = {}
    for source_name, selection in unique_core_selection.items():
        burned = already_double_exposed_shards.get(source_name, 0)
        if burned <= 0 or replay_passes[source_name] < 2:
            corrected_selection[source_name] = selection
            continue
        if burned > len(selection):
            raise ValueError(
                f"{source_name}: already_double_exposed_shards={burned} exceeds "
                f"its {len(selection)} phase-1 shards"
            )
        keep = selection[burned:]
        recovered_rows = sum(s["rows"] for s in selection[:burned])

        already_used_files = {s["file"] for s in selection}
        all_shards = dataset._source_manifests[source_name]["shards"]
        fresh_candidates = [s for s in all_shards if s["file"] not in already_used_files]

        backfill: list[dict] = []
        remaining = recovered_rows
        for shard in fresh_candidates:
            if remaining <= 0:
                break
            take = min(int(shard["rows"]), remaining)
            backfill.append({"file": str(shard["file"]), "rows": take})
            remaining -= take
        if remaining:
            raise ValueError(
                f"{source_name}: not enough fresh shards to backfill "
                f"{recovered_rows:,} rows recovered from {burned} burned shards "
                f"({remaining:,} rows short)"
            )
        corrected_selection[source_name] = keep + backfill

    # Rebuild every phase from corrected_selection instead of the original
    # unique_core selection, mirroring build_replay_plan's phase-construction
    # loop exactly (same "passes >= pass_number" gating, same naming).
    phases = [{"name": "unique_core", "passes": 1, "sources": unique_core_selection}]
    for pass_number in range(2, max(replay_passes.values()) + 1):
        sources = {
            name: corrected_selection[name]
            for name, passes in replay_passes.items()
            if passes >= pass_number
        }
        if sources:
            phases.append(
                {"name": f"quality_replay_{pass_number}_corrected", "passes": 1, "sources": sources}
            )

    plan = dict(base_plan)
    plan["phases"] = phases
    plan["correction"] = {
        "already_double_exposed_shards": dict(already_double_exposed_shards),
        "note": (
            "phase 1 unchanged; later replay phases swap already-double-exposed "
            "shards for previously-unused ones from the same source, same row "
            "counts, so trained_tokens/source_unique_tokens/source_passes are "
            "unchanged from the uncorrected plan"
        ),
    }
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-data", required=True)
    parser.add_argument("--output", default="plans/tr_hash_70b_quality_replay_corrected.json")
    parser.add_argument(
        "--unique-budgets",
        default=_format_mapping(
            {name: tokens // 10**9 for name, tokens in DEFAULT_UNIQUE_BUDGETS.items()},
            suffix="B",
        ),
    )
    parser.add_argument("--replay-passes", default=_format_mapping(DEFAULT_REPLAY_PASSES))
    parser.add_argument(
        "--already-double-exposed",
        required=True,
        help="name=shard_count,... -- how many of each source's phase-1 shards "
        "already got an unplanned extra pass from the reset",
    )
    parser.add_argument("--row-alignment", type=int, default=512)
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
    plan = build_corrective_replay_plan(
        dataset,
        unique_token_budgets=_parse_mapping(args.unique_budgets, token_counts=True),
        replay_passes=_parse_mapping(args.replay_passes, token_counts=False),
        already_double_exposed_shards=_parse_mapping(args.already_double_exposed, token_counts=False),
        row_alignment=args.row_alignment,
    )
    plan["dataset"] = args.tokenized_data
    plan["revision"] = args.revision
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Corrective replay plan: {output}")
    print(f"Trained token exposures (unchanged from original): {plan['trained_tokens']:,}")
    print(f"Correction: {json.dumps(plan['correction']['already_double_exposed_shards'])}")


if __name__ == "__main__":
    main()
