#!/usr/bin/env python3
"""Select the strongest reasoning-SFT checkpoint from complete evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["step"]): row for row in csv.DictReader(handle)}


def build_summary(
    evaluation_root: Path,
    metrics_path: Path,
    checkpoint_root: Path,
    *,
    minimum_piqa_acc_norm: float = 0.0,
) -> dict[str, Any]:
    metric_rows = _metric_rows(metrics_path)
    candidates = []
    for report_dir in sorted(evaluation_root.glob("step_*")):
        if not report_dir.is_dir() or not (report_dir / "piqa.json").is_file():
            continue
        report = _read_json(report_dir / "piqa.json")
        step = int(report["checkpoint_step"])
        checkpoint = checkpoint_root / f"step_{step:06d}"
        if not (checkpoint / "checkpoint.pt").is_file():
            raise FileNotFoundError(checkpoint / "checkpoint.pt")
        metrics = metric_rows.get(step)
        if metrics is None or not metrics.get("matched_eval_loss"):
            raise ValueError(f"missing matched SFT evaluation metrics for step {step}")
        piqa = report["benchmarks"]["piqa"]
        candidate = {
            "step": step,
            "checkpoint": str(checkpoint.resolve()),
            "piqa_acc": float(piqa["acc"]),
            "piqa_acc_norm": float(piqa["acc_norm"]),
            "piqa_correct": int(piqa["correct"]),
            "piqa_correct_norm": int(piqa["correct_norm"]),
            "matched_eval_loss": float(metrics["matched_eval_loss"]),
            "matched_eval_ppl": float(metrics["matched_eval_ppl"]),
            "eligible": float(piqa["acc_norm"]) >= minimum_piqa_acc_norm,
            "report_directory": str(report_dir.resolve()),
        }
        candidates.append(candidate)
    if not candidates:
        raise ValueError("no completely evaluated reasoning-SFT checkpoints found")
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda candidate: (
                candidate["piqa_acc_norm"],
                candidate["piqa_acc"],
                -candidate["matched_eval_loss"],
                candidate["step"],
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": 1,
        "selection_policy": (
            "maximum PIQA acc_norm, maximum PIQA acc, minimum matched "
            "reasoning-SFT eval loss, latest step"
            if minimum_piqa_acc_norm <= 0
            else (
                f"PIQA acc_norm >= {minimum_piqa_acc_norm:.2f}; then maximum "
                "PIQA acc_norm, maximum PIQA acc, minimum matched reasoning-SFT "
                "eval loss, latest step"
            )
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
    parser.add_argument("--minimum-piqa-acc-norm", type=float, default=0.0)
    args = parser.parse_args()
    summary = build_summary(
        args.evaluation_root,
        args.metrics,
        args.checkpoint_root,
        minimum_piqa_acc_norm=args.minimum_piqa_acc_norm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["release_ready"]:
        args.selected_checkpoint.unlink(missing_ok=True)
        raise SystemExit(3)
    args.selected_checkpoint.write_text(
        str(summary["selected"]["checkpoint"]) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
