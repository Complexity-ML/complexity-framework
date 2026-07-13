#!/usr/bin/env python3
"""Render the measured attention-free checkpoint profile from JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {"associative_recall": "#2563eb", "induction": "#dc2626"}
LABELS = {"associative_recall": "Associative recall", "induction": "Induction"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    data = json.loads(args.json_path.read_text())
    receptive_field = data["architecture"]["receptive_field_tokens"]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    metrics = [
        ("accuracy", "Target accuracy", False),
        ("mean_rank", "Mean target rank", True),
        ("nll", "Target NLL", False),
    ]
    for axis, (metric, title, log_y) in zip(axes.flat[:3], metrics):
        for task in ("associative_recall", "induction"):
            rows = data[task]
            axis.plot(
                [row["distance"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=2,
                markersize=4,
                color=COLORS[task],
                label=LABELS[task],
            )
        axis.axvline(receptive_field, color="#111827", linestyle="--", linewidth=1.3, label="Receptive field")
        axis.set_xscale("log", base=2)
        if log_y:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Dependency distance (tokens)")
        axis.grid(True, alpha=0.25)

    equivalence = data["incremental_equivalence"]["comparisons"]
    axis = axes.flat[3]
    axis.plot(
        [row["length"] for row in equivalence],
        [max(row["max_abs_difference"], 1e-12) for row in equivalence],
        marker="o",
        linewidth=2,
        color="#059669",
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_title("Full vs incremental hidden-state error")
    axis.set_xlabel("Decoded context length")
    axis.set_ylabel("Maximum absolute error")
    axis.grid(True, alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    checkpoint = data["checkpoint"]
    figure.suptitle(
        "Attention-free checkpoint behavioral profile\n"
        f"{checkpoint['parameters']:,} parameters · {checkpoint['registered_total_tokens']:,} training tokens",
        fontsize=14,
        fontweight="bold",
        y=1.025,
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
