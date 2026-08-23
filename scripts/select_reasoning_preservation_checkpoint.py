#!/usr/bin/env python3
"""Select a reasoning-preservation checkpoint without sacrificing the SFT base."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["step"]): row for row in csv.DictReader(handle)}


def behavior_passes(report: dict[str, Any], prompt_ids: set[str]) -> int:
    failed = set(report.get("failures", {})) & prompt_ids
    return len(prompt_ids - failed)


def build_summary(
    evaluation_root: Path,
    metrics_path: Path,
    checkpoint_root: Path,
    panel_path: Path,
    *,
    piqa_tolerance: float = 0.005,
    arc_tolerance: float = 0.01,
    behavior_tolerance: int = 1,
) -> dict[str, Any]:
    rows = metric_rows(metrics_path)
    panel = read_json(panel_path)
    prompt_ids = {str(item["id"]) for item in panel["prompts"]}
    source_dir = evaluation_root / "source_sft_v2"
    source_promotion = read_json(source_dir / "promotion.json")
    source_piqa_report = read_json(source_dir / "piqa.json")
    source_arc = read_json(source_dir / "arc_zero_shot.json")
    source_reasoning = read_json(source_dir / "arc_reasoning_64.json")
    source_piqa = float(source_piqa_report["benchmarks"]["piqa"]["acc_norm"])
    source_behavior = behavior_passes(source_promotion, prompt_ids)
    source_arc_raw = float(source_arc["combined"]["acc"])
    source_arc_norm = float(source_arc["combined"]["acc_norm"])

    candidates: list[dict[str, Any]] = []
    for report_dir in sorted(evaluation_root.glob("step_*")):
        if not report_dir.is_dir():
            continue
        piqa_report = read_json(report_dir / "piqa.json")
        promotion = read_json(report_dir / "promotion.json")
        arc = read_json(report_dir / "arc_zero_shot.json")
        reasoning = read_json(report_dir / "arc_reasoning_64.json")
        step = int(piqa_report["checkpoint_step"])
        checkpoint = checkpoint_root / f"step_{step:06d}"
        if not (checkpoint / "checkpoint.pt").is_file():
            raise FileNotFoundError(checkpoint / "checkpoint.pt")
        metric = rows.get(step)
        if metric is None or not metric.get("matched_eval_loss"):
            raise ValueError(f"missing matched evaluation metrics for step {step}")
        piqa = piqa_report["benchmarks"]["piqa"]
        behavior = behavior_passes(promotion, prompt_ids)
        arc_raw = float(arc["combined"]["acc"])
        arc_norm = float(arc["combined"]["acc_norm"])
        eligible = (
            float(piqa["acc_norm"]) >= source_piqa - piqa_tolerance
            and behavior >= source_behavior - behavior_tolerance
            and (
                arc_raw >= source_arc_raw - arc_tolerance
                or arc_norm >= source_arc_norm - arc_tolerance
            )
        )
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "eligible": eligible,
                "behavior_passes": behavior,
                "behavior_total": len(prompt_ids),
                "behavior_failures": promotion.get("failures", {}),
                "piqa_acc": float(piqa["acc"]),
                "piqa_acc_norm": float(piqa["acc_norm"]),
                "arc_acc": arc_raw,
                "arc_acc_norm": arc_norm,
                "arc_reasoning_native_accuracy": float(reasoning["combined"]["native_accuracy"]),
                "arc_reasoning_native_parse_rate": float(
                    reasoning["combined"]["native_parse_rate"]
                ),
                "matched_eval_loss": float(metric["matched_eval_loss"]),
                "matched_eval_ppl": float(metric["matched_eval_ppl"]),
                "report_directory": str(report_dir.resolve()),
            }
        )
    if not candidates:
        raise ValueError("no fully evaluated preservation checkpoints found")
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda item: (
                item["arc_reasoning_native_accuracy"],
                item["behavior_passes"],
                item["piqa_acc_norm"],
                item["arc_acc_norm"],
                -item["matched_eval_loss"],
                -item["step"],
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": 1,
        "selection_policy": (
            "retain source SFT PIQA within 0.005, behavior panel within one pass, "
            "and full ARC raw or normalized accuracy within 0.01; then maximize "
            "native ARC-64 generative accuracy, behavior passes, PIQA, ARC retention, "
            "held-out loss and prefer the earlier step"
        ),
        "release_ready": selected is not None,
        "source": {
            "checkpoint": source_arc["checkpoint"],
            "behavior_passes": source_behavior,
            "behavior_total": len(prompt_ids),
            "piqa_acc_norm": source_piqa,
            "arc_acc": source_arc_raw,
            "arc_acc_norm": source_arc_norm,
            "arc_reasoning_native_accuracy": float(source_reasoning["combined"]["native_accuracy"]),
        },
        "selected": selected,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(
        args.evaluation_root,
        args.metrics,
        args.checkpoint_root,
        args.panel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if summary["selected"] is None:
        args.selected_checkpoint.unlink(missing_ok=True)
        raise SystemExit(3)
    args.selected_checkpoint.write_text(summary["selected"]["checkpoint"] + "\n")


if __name__ == "__main__":
    main()
