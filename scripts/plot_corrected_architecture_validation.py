#!/usr/bin/env python3
"""Plot corrected architecture validation directly from measured JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def synthetic_result(path: Path, architecture: str, task: str) -> dict:
    payload = json.loads(path.read_text())
    return next(
        result
        for result in payload["results"]
        if result["architecture"] == architecture and result["task"] == task
    )


def metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def eval_curve(rows: list[dict[str, str]]) -> tuple[list[int], list[float]]:
    points = [
        (int(row["step"]), float(row["eval_loss"]))
        for row in rows
        if row["eval_loss"] not in ("", "nan")
        and math.isfinite(float(row["eval_loss"]))
    ]
    return [point[0] for point in points], [point[1] for point in points]


def throughput_curve(rows: list[dict[str, str]]) -> tuple[list[int], list[float]]:
    points = [
        (int(row["step"]), float(row["tok_s"]))
        for row in rows
        if int(row["step"]) > 50
        and row["tok_s"] not in ("", "nan")
        and math.isfinite(float(row["tok_s"]))
    ]
    return [point[0] for point in points], [point[1] for point in points]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa-synthetic", type=Path, required=True)
    parser.add_argument("--context-synthetic", type=Path, required=True)
    parser.add_argument("--gqa-metrics", type=Path, required=True)
    parser.add_argument("--context-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    colors = {"gqa": "#2563eb", "context": "#059669"}
    labels = {"gqa": "GQA", "context": "Corrected shared context"}
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)

    for axis, task, title in (
        (axes[0, 0], "associative_recall", "Associative recall"),
        (axes[0, 1], "induction", "Induction"),
    ):
        for name, path, architecture in (
            ("gqa", args.gqa_synthetic, "gqa"),
            ("context", args.context_synthetic, "causal_fast_weight_conv"),
        ):
            final = synthetic_result(path, architecture, task)["history"][-1]["evaluation"]
            axis.plot(
                [row["distance"] for row in final],
                [row["accuracy"] for row in final],
                marker="o",
                linewidth=2.2,
                color=colors[name],
                label=labels[name],
            )
        axis.axhline(1 / 256, color="#6b7280", linestyle=":", label="Chance")
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.set_xlabel("Dependency distance")
        axis.set_ylabel("Exact accuracy")
        axis.grid(alpha=0.25)

    gqa_rows = metric_rows(args.gqa_metrics)
    context_rows = metric_rows(args.context_metrics)
    for name, rows in (("gqa", gqa_rows), ("context", context_rows)):
        steps, values = eval_curve(rows)
        axes[1, 0].plot(
            steps, values, marker="o", markersize=3, linewidth=2,
            color=colors[name], label=labels[name],
        )
        steps, values = throughput_curve(rows)
        axes[1, 1].plot(
            steps, values, linewidth=1.7, color=colors[name], label=labels[name],
        )
    axes[1, 0].set_title("FineWeb held-out loss")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 0].set_ylabel("Loss (lower is better)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].set_title("H100 training throughput")
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].set_ylabel("Tokens/s (higher is better)")
    axes[1, 1].grid(alpha=0.25)

    axes[0, 0].legend(loc="lower left", frameon=False)
    figure.suptitle(
        "Corrected attention-free architecture validation",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
