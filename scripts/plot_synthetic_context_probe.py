#!/usr/bin/env python3
"""Plot matched synthetic context-learning controls from measured JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {"gqa": "#2563eb", "causal_conv": "#dc2626"}
LABELS = {"gqa": "GQA", "causal_conv": "Causal convolution"}
TASK_LABELS = {
    "associative_recall": "Associative recall",
    "induction": "Induction",
}


def metric_at_distance(checkpoint: dict, distance: int, metric: str) -> float:
    return next(row[metric] for row in checkpoint["evaluation"] if row["distance"] == distance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.json_path.read_text())
    indexed = {
        (result["architecture"], result["task"]): result
        for result in payload["results"]
    }

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for column, task in enumerate(("associative_recall", "induction")):
        learning_axis = axes[0, column]
        distance_axis = axes[1, column]
        for architecture in ("gqa", "causal_conv"):
            result = indexed[(architecture, task)]
            history = result["history"]
            learning_axis.plot(
                [checkpoint["step"] for checkpoint in history],
                [metric_at_distance(checkpoint, 16, "accuracy") for checkpoint in history],
                color=COLORS[architecture],
                linewidth=2,
                marker="o",
                markersize=3,
                label=LABELS[architecture],
            )
            final = history[-1]["evaluation"]
            distance_axis.plot(
                [row["distance"] for row in final],
                [row["accuracy"] for row in final],
                color=COLORS[architecture],
                linewidth=2,
                marker="o",
                label=LABELS[architecture],
            )
        learning_axis.axhline(1 / 256, color="#6b7280", linestyle=":", label="Chance")
        learning_axis.set_title(f"{TASK_LABELS[task]} — learning at distance 16")
        learning_axis.set_xlabel("Training step")
        learning_axis.set_ylabel("Exact accuracy")
        learning_axis.set_ylim(-0.03, 1.03)
        learning_axis.grid(alpha=0.25)

        distance_axis.axhline(1 / 256, color="#6b7280", linestyle=":", label="Chance")
        distance_axis.axvline(46, color="#111827", linestyle="--", label="Conv receptive field")
        distance_axis.set_xscale("log", base=2)
        distance_axis.set_title(f"{TASK_LABELS[task]} — final distance sweep")
        distance_axis.set_xlabel("Dependency distance")
        distance_axis.set_ylabel("Exact accuracy")
        distance_axis.set_ylim(-0.03, 1.03)
        distance_axis.grid(alpha=0.25)

    axes[0, 0].legend(loc="upper left", frameon=False)
    axes[1, 0].legend(loc="lower left", frameon=False)
    gqa_params = indexed[("gqa", "associative_recall")]["parameters"]
    conv_params = indexed[("causal_conv", "associative_recall")]["parameters"]
    figure.suptitle(
        "Matched synthetic contextual retrieval — "
        f"GQA {gqa_params:,} vs causal convolution {conv_params:,} parameters",
        fontsize=14,
        fontweight="bold",
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
