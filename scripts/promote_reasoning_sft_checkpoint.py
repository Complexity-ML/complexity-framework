#!/usr/bin/env python3
"""Promote the best reasoning checkpoint subject to retention guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _checkpoint_directory(value: str) -> Path:
    path = Path(value).resolve()
    return path.parent if path.name == "checkpoint.pt" else path


def promote(
    summary: dict[str, Any],
    reasoning_reports: list[dict[str, Any]],
    zero_shot_reports: list[dict[str, Any]],
    source_zero_shot: dict[str, Any],
    *,
    piqa_tolerance: float = 0.01,
    arc_tolerance: float = 0.02,
    arc_norm_tolerance: float = 0.01,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = {
        _checkpoint_directory(candidate["checkpoint"]): dict(candidate)
        for candidate in summary["candidates"]
    }
    reasoning = {_checkpoint_directory(report["model"]): report for report in reasoning_reports}
    zero_shot = {
        _checkpoint_directory(report["checkpoint"]): report for report in zero_shot_reports
    }
    paths = set(reasoning)
    if paths != set(zero_shot) or not paths.issubset(candidates):
        raise ValueError("reasoning, zero-shot and PIQA candidate sets do not match")
    best_piqa = max(candidates[path]["piqa_acc_norm"] for path in paths)
    source_arc = float(source_zero_shot["combined"]["acc"])
    source_arc_norm = float(source_zero_shot["combined"]["acc_norm"])
    audited = []
    for path in sorted(paths):
        candidate = candidates[path]
        reasoning_report = reasoning[path]
        zero_report = zero_shot[path]
        candidate["arc_reasoning"] = reasoning_report["combined"]
        candidate["arc_zero_shot"] = zero_report["combined"]
        candidate["retention_eligible"] = candidate[
            "piqa_acc_norm"
        ] >= best_piqa - piqa_tolerance and (
            float(zero_report["combined"]["acc"]) >= source_arc - arc_tolerance
            or float(zero_report["combined"]["acc_norm"]) >= source_arc_norm - arc_norm_tolerance
        )
        audited.append(candidate)
    eligible = [candidate for candidate in audited if candidate["retention_eligible"]]
    if not eligible:
        raise ValueError("no reasoning candidate passed the PIQA and ARC retention guards")
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["arc_reasoning"]["native_accuracy"],
            candidate["arc_reasoning"]["flexible_accuracy"],
            candidate["arc_reasoning"]["strict_format_rate"],
            -candidate["matched_eval_loss"],
            candidate["piqa_acc_norm"],
        ),
    )
    selected_path = _checkpoint_directory(selected["checkpoint"])
    summary["initial_piqa_selected"] = summary["selected"]
    summary["selected"] = selected
    summary["reasoning_candidates"] = audited
    summary["selection_policy"] = (
        "maximum matched-probe ARC native reasoning accuracy among screened early, "
        f"PIQA-best and final checkpoints, subject to PIQA acc_norm within {piqa_tolerance:.2f} and "
        f"full Combined ARC raw accuracy within {arc_tolerance:.2f} or normalized "
        f"accuracy within {arc_norm_tolerance:.2f} of the source; "
        "then flexible reasoning accuracy, strict format rate, held-out SFT loss and PIQA"
    )
    return summary, reasoning[selected_path], zero_shot[selected_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reasoning-report", type=Path, action="append", required=True)
    parser.add_argument("--zero-shot-report", type=Path, action="append", required=True)
    parser.add_argument("--source-zero-shot", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-reasoning-report", type=Path, required=True)
    parser.add_argument("--selected-zero-shot-report", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    reasoning_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.reasoning_report
    ]
    zero_shot_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.zero_shot_report
    ]
    source = json.loads(args.source_zero_shot.read_text(encoding="utf-8"))
    summary, reasoning, zero_shot = promote(summary, reasoning_reports, zero_shot_reports, source)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.selected_checkpoint.write_text(summary["selected"]["checkpoint"] + "\n")
    args.selected_reasoning_report.write_text(json.dumps(reasoning, indent=2) + "\n")
    args.selected_zero_shot_report.write_text(json.dumps(zero_shot, indent=2) + "\n")
    source_trace = Path(reasoning["traces"])
    target_trace = args.selected_reasoning_report.with_suffix(".jsonl")
    target_trace.write_bytes(source_trace.read_bytes())
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
