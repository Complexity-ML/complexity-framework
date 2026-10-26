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
    IMAGE_EDITING_SUPERVISED_FINETUNING,
    IMAGE_GENERATION_SUPERVISED_FINETUNING,
    IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
    REFINEMENT_STAGE,
    TEXT_REFINEMENT,
    TEXT_SUPERVISED_FINETUNING,
    VISION_SUPERVISED_FINETUNING,
    validate_full_parameter_finetuning,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_full_parameter_pipeline_allowlist_is_explicit() -> None:
    """Regression guard: every full-parameter pipeline is named explicitly."""

    assert FULL_PARAMETER_FINETUNING_PIPELINES == {
        VISION_SUPERVISED_FINETUNING,
        IMAGE_EDITING_SUPERVISED_FINETUNING,
        IMAGE_GENERATION_SUPERVISED_FINETUNING,
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
        TEXT_SUPERVISED_FINETUNING,
        TEXT_REFINEMENT,
    }
    validate_full_parameter_finetuning(VISION_SUPERVISED_FINETUNING)
    validate_full_parameter_finetuning(
        IMAGE_GENERATION_SUPERVISED_FINETUNING,
        source_stage=REFINEMENT_STAGE,
    )
    validate_full_parameter_finetuning(
        IMAGE_EDITING_SUPERVISED_FINETUNING,
        source_stage=REFINEMENT_STAGE,
    )
    validate_full_parameter_finetuning(
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
        source_stage=REFINEMENT_STAGE,
    )
    validate_full_parameter_finetuning(
        TEXT_SUPERVISED_FINETUNING,
        source_stage=REFINEMENT_STAGE,
    )
    validate_full_parameter_finetuning(
        TEXT_REFINEMENT,
        unique_tokens=70_000_000_000,
        pretrain_unique_tokens=70_000_000_000,
    )

    with pytest.raises(ValueError, match="restricted to"):
        validate_full_parameter_finetuning("unknown-finetuning")


def test_text_refinement_requires_the_unique_token_totals_to_match() -> None:
    """The text-refinement exemption isn't a bare pipeline-name
    check: a mismatched unique_tokens (a different, smaller corpus smuggled
    in under this pipeline's name) must still be rejected, and both totals
    must be supplied at all -- the exemption cannot be claimed silently."""

    with pytest.raises(ValueError, match="requires unique_tokens"):
        validate_full_parameter_finetuning(TEXT_REFINEMENT)

    with pytest.raises(ValueError, match="exactly match"):
        validate_full_parameter_finetuning(
            TEXT_REFINEMENT,
            unique_tokens=19_000_000,
            pretrain_unique_tokens=70_000_000_000,
        )


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


def test_packaged_text_sft_requires_a_declared_adaptation_mode(tmp_path) -> None:
    from complexity.training.sft import build_sft_plan
    from complexity.training.stack_sft.config import StackSFTConfig

    # The direct full-parameter exception must be declared before any files
    # are read. Normal release stage gates remain covered above.
    with pytest.raises(ValueError, match="requires explicit experiment"):
        build_sft_plan(
            tmp_path / "missing-checkpoint",
            tmp_path / "missing-data",
            experiment="",
            batch_size=1,
            world_size=1,
            expert_lr_scale=1.0,
        )
    # The staged workflow accepts adapters only; rank zero cannot silently
    # turn the deleted curriculum launcher's replacement into a full SFT.
    with pytest.raises(ValueError, match="positive LoRA rank"):
        StackSFTConfig.from_dict(
            {
                "name": "test",
                "base_checkpoint": "missing",
                "lora_rank": 0,
                "stages": [
                    {
                        "name": "stage",
                        "steps": 1,
                        "lr": 1e-4,
                        "mix": {"sources": [{"name": "local", "weight": 1, "path": "missing"}]},
                    }
                ],
            }
        )
