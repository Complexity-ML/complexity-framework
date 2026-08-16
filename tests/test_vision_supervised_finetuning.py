from __future__ import annotations

import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from complexity.generative.detection import TRHashObjectDetector, coco_v8_nano_config
from complexity.generative.detection.training import resolve_initialization_provenance
from complexity.training.finetuning import (
    FULL_PARAMETER_FINETUNING_PIPELINES,
    TEXT_SUPERVISED_FINETUNING,
    VISION_SUPERVISED_FINETUNING,
    validate_full_parameter_finetuning,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_only_vision_sft_is_exempt_from_full_parameter_ban() -> None:
    assert FULL_PARAMETER_FINETUNING_PIPELINES == {VISION_SUPERVISED_FINETUNING}
    validate_full_parameter_finetuning(VISION_SUPERVISED_FINETUNING)

    with pytest.raises(ValueError, match="restricted to"):
        validate_full_parameter_finetuning(TEXT_SUPERVISED_FINETUNING)
    with pytest.raises(ValueError, match="restricted to"):
        validate_full_parameter_finetuning("unknown-finetuning")


def test_v8_vision_sft_updates_every_model_parameter() -> None:
    model = TRHashObjectDetector(coco_v8_nano_config())

    assert all(parameter.requires_grad for parameter in model.parameters())


def _vision_sft_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "training_purpose": VISION_SUPERVISED_FINETUNING,
        "resume": None,
        "backbone_checkpoint": None,
        "detector_checkpoint": Path("checkpoint/best"),
        "class_map": None,
        "require_random_init": False,
        "provenance_dataset": "coco-2017",
    }
    values.update(overrides)
    return Namespace(**values)


def test_vision_sft_is_weight_transfer_not_exact_resume() -> None:
    provenance = resolve_initialization_provenance(_vision_sft_args())

    assert provenance["initialization"] == "detector-transfer"
    assert provenance["external_checkpoint"] == "checkpoint/best"


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"detector_checkpoint": None}, "requires --detector-checkpoint"),
        ({"resume": Path("resume")}, "forbids --resume"),
        ({"backbone_checkpoint": Path("tower")}, "forbids --resume"),
        ({"require_random_init": True}, "cannot be combined"),
    ),
)
def test_vision_sft_rejects_ambiguous_initialization(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_initialization_provenance(_vision_sft_args(**overrides))


def test_v8_vision_sft_launcher_resets_training_with_a_clean_recipe() -> None:
    result = subprocess.run(
        ["bash", "scripts/vast_finetune_detector_coco_v08_nano.sh"],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "VENV_ACTIVATE": "/dev/null",
            "REPO_ROOT": str(PROJECT_ROOT),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    command = result.stdout

    assert "--training-purpose vision-supervised-finetuning" in command
    assert "--detector-checkpoint artifacts/detector_coco_v08_nano_o2m/best" in command
    assert "--no-end-to-end" in command
    assert "--resume" not in command
    assert "--backbone-checkpoint" not in command
    assert "--require-random-init" not in command
    assert "--architecture-version 8" in command
    assert "--vision-num-experts 8" in command
    assert "--vision-shared-width 216" in command
    assert "--vision-expert-width 27" in command
    assert "--epochs 30" in command
    assert "--lr 5.4e-4" in command
    assert "--augmentation light" in command
    assert "--mosaic 0.0" in command
    assert "--mixup 0.0" in command
    assert "--copy-paste 0.0" in command
    assert "--random-erasing 0.0" in command
    assert "--packed-epochs 1" in command
    assert "--eval-every 1" in command
    assert "--mosaic-packed-epoch" not in command
    assert "lora" not in command.lower()


def test_text_sft_launchers_remain_lora_only() -> None:
    for path in (
        Path("scripts/sft_500m_32k_tr.py"),
        Path("scripts/run_sft_curriculum.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "TEXT_SUPERVISED_FINETUNING" in source
        assert "--lora-rank" in source
        assert "validate_full_parameter_finetuning" in source
