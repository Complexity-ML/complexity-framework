from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.select_reasoning_sft_checkpoint import build_summary


def _candidate(root: Path, checkpoint_root: Path, step: int, acc_norm: float) -> None:
    report = root / f"step_{step:06d}"
    report.mkdir(parents=True)
    (report / "piqa.json").write_text(
        json.dumps(
            {
                "checkpoint_step": step,
                "benchmarks": {
                    "piqa": {
                        "acc": acc_norm - 0.01,
                        "acc_norm": acc_norm,
                        "correct": 1200,
                        "correct_norm": 1210,
                    }
                },
            }
        )
    )
    checkpoint = checkpoint_root / f"step_{step:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.pt").write_bytes(b"complete")


def test_selects_maximum_piqa_without_hiding_lower_results(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    checkpoints = tmp_path / "checkpoints"
    _candidate(evaluations, checkpoints, 250, 0.68)
    _candidate(evaluations, checkpoints, 500, 0.70)
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "matched_eval_loss", "matched_eval_ppl"),
        )
        writer.writeheader()
        writer.writerow({"step": 250, "matched_eval_loss": 1.2, "matched_eval_ppl": 3.3})
        writer.writerow({"step": 500, "matched_eval_loss": 1.1, "matched_eval_ppl": 3.0})

    summary = build_summary(evaluations, metrics, checkpoints)

    assert summary["release_ready"] is True
    assert summary["selected"]["step"] == 500
    assert len(summary["candidates"]) == 2
    assert ">=" not in summary["selection_policy"]
