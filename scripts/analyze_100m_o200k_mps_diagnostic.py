#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


RUNS = {
    "Dense GQA": "diagnostic-mps-dense-gqa-o200k-s42",
    "TR-GQA modulo cyclic": "diagnostic-mps-tr-gqa-modulo-cyclic-o200k-s42",
}


def read_eval_rows(path: Path) -> dict[int, float]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["step"]): float(row["eval_loss"])
            for row in rows
            if row["eval_loss"].lower() != "nan"
        }


def main() -> None:
    root = Path("runs")
    values = {
        label: read_eval_rows(root / run_name / "metrics.csv")
        for label, run_name in RUNS.items()
    }
    common_steps = sorted(set.intersection(*(set(run) for run in values.values())))
    print("NLL — lower is better")
    print("step,dense,modulo_cyclic,cyclic_minus_dense")
    for step in common_steps:
        dense = values["Dense GQA"][step]
        cyclic = values["TR-GQA modulo cyclic"][step]
        print(
            f"{step},{dense:.6f},{cyclic:.6f},{cyclic - dense:+.6f}"
        )


if __name__ == "__main__":
    main()
