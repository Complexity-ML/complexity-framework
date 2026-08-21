#!/usr/bin/env python3
"""Rank the three clean-SFT epochs after PIQA and behavior evaluation."""

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
        rows = {int(row["step"]): row for row in csv.DictReader(handle)}
    return rows


def build_summary(
    evaluation_root: Path,
    metrics_path: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    rows = metric_rows(metrics_path)
    report_dirs = sorted(path for path in evaluation_root.glob("epoch_*_step_*") if path.is_dir())
    if len(report_dirs) != 3:
        raise ValueError(f"Expected exactly three evaluated epochs, found {len(report_dirs)}")

    candidates = []
    seen_steps: set[int] = set()
    for epoch, report_dir in enumerate(report_dirs, start=1):
        piqa_report = read_json(report_dir / "piqa.json")
        promotion = read_json(report_dir / "promotion.json")
        step = int(piqa_report["checkpoint_step"])
        if step in seen_steps:
            raise ValueError(f"Duplicate evaluated checkpoint step: {step}")
        seen_steps.add(step)
        checkpoint = checkpoint_root / f"step_{step:06d}"
        if not (checkpoint / "checkpoint.pt").is_file():
            raise FileNotFoundError(checkpoint / "checkpoint.pt")
        row = rows.get(step)
        if row is None or not row.get("matched_eval_loss"):
            raise ValueError(f"Missing matched evaluation metrics for step {step}")
        piqa = piqa_report["benchmarks"]["piqa"]
        candidates.append(
            {
                "epoch": epoch,
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "promotion_passed": bool(promotion["passed"]),
                "promotion_failures": promotion.get("failures", {}),
                "piqa_acc": float(piqa["acc"]),
                "piqa_acc_norm": float(piqa["acc_norm"]),
                "piqa_correct": int(piqa["correct"]),
                "piqa_correct_norm": int(piqa["correct_norm"]),
                "matched_eval_loss": float(row["matched_eval_loss"]),
                "matched_eval_ppl": float(row["matched_eval_ppl"]),
                "report_directory": str(report_dir.resolve()),
            }
        )

    eligible = [candidate for candidate in candidates if candidate["promotion_passed"]]
    selected = (
        max(
            eligible,
            key=lambda candidate: (
                candidate["piqa_acc_norm"],
                candidate["piqa_acc"],
                -candidate["matched_eval_loss"],
                -candidate["epoch"],
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": 1,
        "selection_policy": (
            "promotion gate pass, then maximum PIQA acc_norm, maximum PIQA acc, "
            "minimum matched SFT eval loss, earliest epoch"
        ),
        "release_ready": selected is not None,
        "selected": selected,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.evaluation_root, args.metrics, args.checkpoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["release_ready"]:
        if args.selected_checkpoint.exists():
            args.selected_checkpoint.unlink()
        raise SystemExit(3)
    args.selected_checkpoint.write_text(
        str(summary["selected"]["checkpoint"]) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
