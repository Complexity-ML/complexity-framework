from __future__ import annotations

import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from complexity.generative.detection import TRHashDetectorConfig
from complexity.generative.detection.training import (
    NATIVE_DETECTOR_IMPLEMENTATION,
    resolve_initialization_provenance,
)
from scripts.collect_detector_specialization_ablations import (
    ARMS,
    EXPECTED_FEATURES,
    METRICS,
    collect,
    write_reports,
)
from scripts.detector_checkpoint_status import (
    COMPLETE,
    INCOMPATIBLE,
    INCOMPLETE,
    checkpoint_status,
    latest_resumable_checkpoint,
)

PROJECT_ROOT = Path(__file__).parents[1]
COCO_CHECKPOINT = "artifacts/detector_coco_v06_native/best"


def _dry_run(script: str, **environment: str) -> str:
    result = subprocess.run(
        ["bash", script],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "VENV_ACTIVATE": "/dev/null",
            "REPO_ROOT": str(PROJECT_ROOT),
            **environment,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_native_coco_stage_starts_complete_detector_from_scratch() -> None:
    command = _dry_run("scripts/vast_train_detector_specialized_coco.sh")

    assert "--annotations artifacts/COCO/annotations/instances_train2017.json" in command
    assert "--validation-annotations artifacts/COCO/annotations/instances_val2017.json" in command
    assert "--output artifacts/detector_coco_v06_native" in command
    assert "--epochs 245" in command
    assert "--batch-size 16" in command
    assert "--image-size 640" in command
    assert "--vision-num-experts 4" in command
    assert "--vision-top-k 2" in command
    assert "--optimizer musgd" in command
    assert "--musgd-muon-weight 0.528" in command
    assert "--musgd-sgd-weight 0.674" in command
    assert "--backbone-lr-multiplier 1.0" in command
    assert "--ema-decay 0.9999" in command
    assert "--require-triton" in command
    assert "--require-random-init" in command
    assert "--provenance-dataset coco-2017" in command
    assert "--detector-checkpoint" not in command
    assert "--backbone-checkpoint" not in command
    assert "--resume" not in command
    assert "Objects365" not in command
    assert "ultralytics" not in command.lower()


def test_native_coco_stage_requires_mosaic_packed_epochs_by_default() -> None:
    command = _dry_run("scripts/vast_train_detector_specialized_coco.sh")

    assert "--mosaic 0.909" in command
    assert "--mosaic-packed-epoch" in command


@pytest.mark.parametrize("mosaic", ("0", "0.0", "00.000"))
def test_native_coco_stage_rejects_zero_mosaic(mosaic: str) -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _dry_run(
            "scripts/vast_train_detector_specialized_coco.sh",
            MOSAIC=mosaic,
        )

    assert error.value.returncode == 64
    assert "Mosaic-free detector pretraining is forbidden" in error.value.stderr


def test_native_coco_stage_rejects_packed_epoch_zero() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _dry_run(
            "scripts/vast_train_detector_specialized_coco.sh",
            MOSAIC_PACKED_EPOCH="0",
        )

    assert error.value.returncode == 64
    assert "unpacked detector pretraining is forbidden" in error.value.stderr


def test_native_coco_wrapper_resumes_instead_of_transferring() -> None:
    wrapper = (PROJECT_ROOT / "scripts/vast_run_detector_coco_native.sh").read_text()

    assert "detector_checkpoint_status.py" in wrapper
    assert "RESUME_CHECKPOINT" in wrapper
    assert "random initialization" in wrapper
    assert "detector-checkpoint" not in wrapper
    assert "backbone-checkpoint" not in wrapper


def test_coco_xet_bootstrap_verifies_official_archives_and_layout() -> None:
    bootstrap = (PROJECT_ROOT / "scripts/vast_download_coco2017_xet.sh").read_text()

    assert "HF_XET_HIGH_PERFORMANCE=1" in bootstrap
    assert "cced6f7f71b7629ddf16f17bbcfab6b2" in bootstrap
    assert "442b8da7639aecaf257c1dceb8ba8c80" in bootstrap
    assert "f4bbac642086de4f52a3fdda2de5fa2c" in bootstrap
    assert "expected_train=118287" in bootstrap
    assert "expected_validation=5000" in bootstrap
    assert "instances_train2017.json" in bootstrap
    assert "instances_val2017.json" in bootstrap


def test_detector_checkpoint_status_can_enforce_exact_step_budget(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_000004"
    checkpoint.mkdir()
    torch.save(
        {
            "epoch": 1,
            "batch_in_epoch": 0,
            "total_epochs": 1,
            "step": 4,
        },
        checkpoint / "training_state.pt",
    )

    assert checkpoint_status(tmp_path, 1, expected_steps=4) == (COMPLETE, checkpoint)
    assert checkpoint_status(tmp_path, 1, expected_steps=3) == (
        INCOMPATIBLE,
        checkpoint,
    )


def test_detector_checkpoint_status_prefers_newer_best_over_named_step(
    tmp_path: Path,
) -> None:
    numbered = tmp_path / "step_000100"
    numbered.mkdir()
    torch.save(
        {
            "epoch": 1,
            "batch_in_epoch": 0,
            "total_epochs": 3,
            "step": 100,
        },
        numbered / "training_state.pt",
    )
    best = tmp_path / "best"
    best.mkdir()
    torch.save(
        {
            "epoch": 2,
            "batch_in_epoch": 0,
            "total_epochs": 3,
            "step": 120,
        },
        best / "training_state.pt",
    )

    assert latest_resumable_checkpoint(tmp_path) == best
    assert checkpoint_status(tmp_path, 3) == (INCOMPLETE, best)


def test_native_random_init_policy_rejects_external_weights() -> None:
    args = Namespace(
        resume=None,
        backbone_checkpoint=Path("tower"),
        detector_checkpoint=None,
        class_map=None,
        require_random_init=True,
        provenance_dataset="coco-2017",
    )
    with pytest.raises(ValueError, match="forbids"):
        resolve_initialization_provenance(args)


def test_native_random_init_policy_accepts_matching_exact_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_000100"
    checkpoint.mkdir()
    provenance = {
        "format_version": 1,
        "implementation": NATIVE_DETECTOR_IMPLEMENTATION,
        "initialization": "random",
        "external_checkpoint": None,
        "dataset": "coco-2017",
    }
    (checkpoint / "provenance.json").write_text(json.dumps(provenance))
    args = Namespace(
        resume=checkpoint,
        backbone_checkpoint=None,
        detector_checkpoint=None,
        class_map=None,
        require_random_init=True,
        provenance_dataset="coco-2017",
    )

    assert resolve_initialization_provenance(args) == provenance


def test_specialization_arms_are_strictly_cumulative() -> None:
    commands = {
        arm: _dry_run(
            "scripts/vast_train_detector_specialized_coco.sh",
            ABLATION=arm,
        )
        for arm in ARMS
    }

    assert "--level-adapters" not in commands["baseline"]
    assert "--class-level-hash-gate" not in commands["baseline"]
    assert "--level-adapters" in commands["adapters"]
    assert "--class-level-hash-gate" not in commands["adapters"]
    assert "--class-level-hash-gate" in commands["hash-gate"]
    assert "--object-weighting" not in commands["hash-gate"]
    assert "--object-weighting" in commands["weighting"]
    assert "--level-aux-loss-weight 0.10" in commands["auxiliary"]
    assert "--gate-calibration-loss-weight 0.10" in commands["auxiliary"]
    assert "--object-contrastive-loss-weight" not in commands["auxiliary"]
    assert "--object-contrastive-loss-weight 0.05" in commands["full"]


def test_video_stage_also_transfers_the_task_aligned_detector() -> None:
    launcher = (PROJECT_ROOT / "scripts/vast_train_detector_specialized_video.sh").read_text()

    assert f'INTERMEDIATE="${{INTERMEDIATE:-{COCO_CHECKPOINT}}}"' in launcher
    assert '--detector-checkpoint "$INTERMEDIATE"' in launcher
    assert "--backbone-checkpoint" not in launcher


def test_framework_has_no_image_classification_pretraining_dependency() -> None:
    roots = ("complexity", "scripts", "deploy", "docs", "tests")
    offenders: list[str] = []
    forbidden = "image" + "net"
    for root_name in roots:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text().lower()
            except UnicodeDecodeError:
                continue
            if forbidden in content or forbidden in path.name.lower():
                offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def _write_ablation_arm(
    root: Path,
    arm: str,
    *,
    expected_epochs: int,
    score: float,
    config_override: dict[str, object] | None = None,
    options_override: dict[str, object] | None = None,
) -> None:
    config = TRHashDetectorConfig().to_dict()
    config.update(EXPECTED_FEATURES[arm])
    if config_override:
        config.update(config_override)
    best = root / arm / "best"
    best.mkdir(parents=True)
    (best / "config.json").write_text(json.dumps(config))
    (best / "validation.json").write_text(json.dumps({name: score for name in METRICS}))
    final = root / arm / "step_000100"
    final.mkdir()
    options = {"seed": 3, "dataset_size": 118_287, "world_size": 8}
    if options_override:
        options.update(options_override)
    torch.save(
        {
            "epoch": expected_epochs,
            "batch_in_epoch": 0,
            "total_epochs": expected_epochs,
            "training_options": options,
        },
        final / "training_state.pt",
    )


def test_specialization_report_requires_complete_controlled_runs(tmp_path: Path) -> None:
    for index, arm in enumerate(ARMS):
        _write_ablation_arm(
            tmp_path,
            arm,
            expected_epochs=5,
            score=0.10 + index * 0.01,
        )

    rows, manifest = collect(tmp_path, expected_epochs=5)
    assert [row["arm"] for row in rows] == list(ARMS)
    assert rows[0]["delta_map50_95"] == pytest.approx(0.0)
    assert rows[-1]["delta_map50_95"] == pytest.approx(0.05)
    assert manifest["protocol"].startswith("random -> COCO 2017")

    write_reports(tmp_path, rows, manifest)
    assert "full" in (tmp_path / "summary.md").read_text()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "protocol.json").is_file()


def test_specialization_report_rejects_uncontrolled_budget_drift(tmp_path: Path) -> None:
    for arm in ARMS:
        _write_ablation_arm(
            tmp_path,
            arm,
            expected_epochs=5,
            score=0.1,
            options_override={"seed": 9} if arm == "full" else None,
        )

    with pytest.raises(ValueError, match="uncontrolled training-budget drift"):
        collect(tmp_path, expected_epochs=5)
