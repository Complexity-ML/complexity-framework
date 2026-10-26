"""Plan zero-copy corpus selection, replay and exposure correction.

The reference maps describe the historical 200M pretraining mixture; they do
not alter or supply defaults for the separate paused 1B recipe. All planner
inputs are explicit and no datasets are downloaded or modified by this module.
"""

from __future__ import annotations

import math
from typing import Mapping

from .corpus_mixture import REPLAY_PLAN_FORMAT, PretokenizedCorpusMixtureDataset
from .corpus_preparation import _positive_integer

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


def build_replay_plan(
    dataset: PretokenizedCorpusMixtureDataset,
    *,
    unique_token_budgets: Mapping[str, int],
    replay_passes: Mapping[str, int],
    row_alignment: int,
    quality_scores: Mapping[str, float] | None = None,
) -> dict:
    row_alignment = _positive_integer(row_alignment, "row_alignment")
    if not unique_token_budgets:
        raise ValueError("replay plan must include at least one source")
    known_sources = {source.name for source in dataset.sources}
    if set(unique_token_budgets) != set(replay_passes):
        raise ValueError("unique budgets and replay passes must name the same sources")
    unknown = set(unique_token_budgets) - known_sources
    if unknown:
        raise ValueError(f"unknown replay-plan sources: {sorted(unknown)}")
    selected_by_source = {}
    actual_budget_tokens = {}
    for source_name, token_budget in unique_token_budgets.items():
        token_budget = _positive_integer(token_budget, f"token budget for {source_name}")
        if token_budget < dataset.seq_len * row_alignment:
            raise ValueError(f"token budget for {source_name} is too small")
        _positive_integer(replay_passes[source_name], f"replay passes for {source_name}")
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
            if any(not math.isfinite(float(quality_scores[key])) for key in keys):
                raise ValueError(f"quality scores for {source_name} must be finite")
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
            raise ValueError(f"source {source_name} has {remaining:,} fewer rows than requested")
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
        actual_budget_tokens[name] * int(replay_passes[name]) for name in actual_budget_tokens
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
    for name, count in already_double_exposed_shards.items():
        if isinstance(count, bool):
            raise ValueError(f"double-exposed shard count for {name} must be an integer")
        _positive_integer(count + 1, f"double-exposed shard count for {name}")

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

    # unique_tokens here means every DISTINCT (source, shard) touched ANYWHERE
    # in the plan, not just phase 1's budget -- corpus_mixture.py's loader
    # validates against exactly that broader definition. In an uncorrected
    # plan the two happen to coincide, because every later replay phase reuses
    # phase 1's own shards verbatim; this plan deliberately breaks that by
    # pointing later phases at previously-unused shards, so it must be
    # recomputed rather than copied from base_plan. trained_tokens is
    # unaffected: it's a straight sum of rows-per-phase x passes, and every
    # phase's row count is unchanged -- only which shards supply those rows.
    unique_rows: dict[tuple[str, str], int] = {}
    trained_rows = 0
    for phase in phases:
        phase_rows = 0
        for source_name, selections in phase["sources"].items():
            for selection in selections:
                key = (source_name, selection["file"])
                unique_rows[key] = max(unique_rows.get(key, 0), selection["rows"])
                phase_rows += selection["rows"]
        trained_rows += phase_rows * phase["passes"]

    plan = dict(base_plan)
    plan["phases"] = phases
    plan["unique_tokens"] = sum(unique_rows.values()) * dataset.seq_len
    plan["trained_tokens"] = trained_rows * dataset.seq_len
    plan["correction"] = {
        "already_double_exposed_shards": dict(already_double_exposed_shards),
        "note": (
            "phase 1 unchanged; later replay phases swap already-double-exposed "
            "shards for previously-unused ones from the same source, same row "
            "counts per phase. trained_tokens (total rows trained on) is "
            "unchanged from the uncorrected plan; unique_tokens is larger, "
            "since the backfill shards are genuinely new distinct material "
            "the uncorrected plan never touched at all"
        ),
    }
    return plan
