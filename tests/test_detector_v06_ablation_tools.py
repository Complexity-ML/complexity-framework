from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch

from scripts.collect_detector_v06_ablations import METRICS, collect, write_reports
from scripts.detector_checkpoint_status import (
    COMPLETE,
    INCOMPATIBLE,
    INCOMPLETE,
    NOT_FOUND,
    checkpoint_status,
    latest_resumable_checkpoint,
)

PROJECT_ROOT = Path(__file__).parents[1]


def write_validation(path: Path, value: float) -> None:
    path.mkdir(parents=True)
    (path / "validation.json").write_text(json.dumps({name: value for name in METRICS}))


def test_collects_classic_and_nms_free_results(tmp_path: Path) -> None:
    root = tmp_path / "ablations"
    reference = tmp_path / "reference"
    write_validation(reference / "best", 0.5)
    write_validation(reference / "best_nms_free", 0.4)
    write_validation(root / "no-stal" / "best", 0.3)
    write_validation(root / "no-stal" / "best_nms_free", 0.2)

    rows = collect(root, reference)
    assert [row["arm"] for row in rows] == [
        "full",
        "full:nms-free",
        "no-stal",
        "no-stal:nms-free",
    ]

    write_reports(root, rows)
    assert "no-stal:nms-free" in (root / "summary.md").read_text()
    assert (root / "summary.csv").read_text().count("\n") == 5


def test_collects_reference_before_ablation_root_exists(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    write_validation(reference / "best", 0.5)

    rows = collect(tmp_path / "not-created-yet", reference)

    assert [row["arm"] for row in rows] == ["full"]


def write_training_state(
    root: Path, step: int, *, epoch: int, total_epochs: int, batch: int = 0
) -> Path:
    checkpoint = root / f"step_{step:06d}"
    checkpoint.mkdir(parents=True)
    torch.save(
        {
            "epoch": epoch,
            "batch_in_epoch": batch,
            "total_epochs": total_epochs,
        },
        checkpoint / "training_state.pt",
    )
    return checkpoint


def test_checkpoint_status_only_completes_at_final_epoch(tmp_path: Path) -> None:
    root = tmp_path / "arm"
    write_training_state(root, 100, epoch=2, total_epochs=5)
    latest = write_training_state(root, 200, epoch=5, total_epochs=5)

    assert latest_resumable_checkpoint(root) == latest
    assert checkpoint_status(root, 5) == (COMPLETE, latest)


def test_checkpoint_status_resumes_partial_run_even_with_best_metrics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "arm"
    write_validation(root / "best", 0.5)
    latest = write_training_state(root, 100, epoch=2, total_epochs=5, batch=7)

    assert checkpoint_status(root, 5) == (INCOMPLETE, latest)


def test_checkpoint_status_rejects_different_budget(tmp_path: Path) -> None:
    root = tmp_path / "arm"
    latest = write_training_state(root, 100, epoch=2, total_epochs=5)

    assert checkpoint_status(root, 10) == (INCOMPATIBLE, latest)
    assert checkpoint_status(tmp_path / "missing", 5) == (NOT_FOUND, None)


def dry_run_ablation(dataset: str, **environment: str) -> str:
    result = subprocess.run(
        ["bash", "scripts/vast_ablate_detector_v06.sh", "no-stal"],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "DATASET": dataset,
            "DRY_RUN": "1",
            "VENV_ACTIVATE": "/dev/null",
            "WORKSPACE": "/workspace",
            "REPO_ROOT": str(PROJECT_ROOT),
            **environment,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_coco_ablation_matches_reference_training_and_validation_settings() -> None:
    command = dry_run_ablation("coco")

    assert "--backbone-lr-multiplier 1.0" in command
    assert "--eval-confidence 0.20" in command
    assert "--eval-max-detections 100" in command


def test_voc_ablation_preserves_reference_settings_and_supports_resume() -> None:
    command = dry_run_ablation("voc", RESUME_CHECKPOINT="artifacts/run/step_001000")

    assert "--backbone-lr-multiplier 0.1" in command
    assert "--eval-confidence 0.05" in command
    assert "--eval-max-detections 300" in command
    assert "--resume artifacts/run/step_001000" in command
    assert "--backbone-checkpoint" not in command
