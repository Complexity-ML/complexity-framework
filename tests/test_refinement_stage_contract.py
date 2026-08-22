from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from complexity.training.finetuning import (
    IMAGE_EDITING_SUPERVISED_FINETUNING,
    IMAGE_GENERATION_MODEL_FAMILY,
    IMAGE_GENERATION_SUPERVISED_FINETUNING,
    IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
    PRETRAINING_STAGE,
    REFINEMENT_STAGE,
    SUPERVISED_FINETUNING_STAGE,
    TEXT_MODEL_FAMILY,
    VISION_MODEL_FAMILY,
    refinement_corpus_fingerprint,
    validate_full_parameter_finetuning,
    validate_refinement_plan,
    validate_training_stage_transition,
)

PROJECT_ROOT = Path(__file__).parents[1]
PRETRAIN_PLAN = PROJECT_ROOT / "configs/replay_plans/tr_hash_70b_quality_replay.json"
REFINEMENT_PLAN = (
    PROJECT_ROOT / "configs/replay_plans/tr_hash_70b_unique_only_phase2.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_non_vision_lineage_requires_refinement_before_sft() -> None:
    with pytest.raises(ValueError, match="direct pretraining -> SFT is forbidden"):
        validate_training_stage_transition(
            TEXT_MODEL_FAMILY,
            PRETRAINING_STAGE,
            SUPERVISED_FINETUNING_STAGE,
        )
    with pytest.raises(ValueError, match="requires a refinement checkpoint"):
        validate_training_stage_transition(
            IMAGE_GENERATION_MODEL_FAMILY,
            PRETRAINING_STAGE,
            SUPERVISED_FINETUNING_STAGE,
        )

    validate_training_stage_transition(
        TEXT_MODEL_FAMILY,
        REFINEMENT_STAGE,
        SUPERVISED_FINETUNING_STAGE,
    )
    validate_training_stage_transition(
        TEXT_MODEL_FAMILY,
        SUPERVISED_FINETUNING_STAGE,
        SUPERVISED_FINETUNING_STAGE,
    )


def test_vision_is_the_integrated_refinement_exception() -> None:
    validate_training_stage_transition(
        VISION_MODEL_FAMILY,
        PRETRAINING_STAGE,
        SUPERVISED_FINETUNING_STAGE,
    )


@pytest.mark.parametrize(
    "pipeline",
    (
        IMAGE_GENERATION_SUPERVISED_FINETUNING,
        IMAGE_EDITING_SUPERVISED_FINETUNING,
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
    ),
)
def test_every_non_vision_full_parameter_sft_requires_refinement(
    pipeline: str,
) -> None:
    with pytest.raises(ValueError, match="source_stage='refinement'"):
        validate_full_parameter_finetuning(pipeline)

    validate_full_parameter_finetuning(
        pipeline,
        source_stage=REFINEMENT_STAGE,
    )


@pytest.mark.parametrize(
    ("same_corpus", "fresh_optimizer", "message"),
    (
        (False, True, "exact pretraining corpus"),
        (True, False, "fresh optimizer"),
        (None, True, "exact pretraining corpus"),
        (True, None, "fresh optimizer"),
    ),
)
def test_refinement_requires_same_corpus_and_fresh_optimization(
    same_corpus: bool | None,
    fresh_optimizer: bool | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_training_stage_transition(
            TEXT_MODEL_FAMILY,
            PRETRAINING_STAGE,
            REFINEMENT_STAGE,
            same_corpus=same_corpus,
            fresh_optimizer=fresh_optimizer,
        )


def test_committed_refinement_plan_matches_exact_pretrain_unique_core() -> None:
    pretrain = _load(PRETRAIN_PLAN)
    refinement = _load(REFINEMENT_PLAN)

    fingerprint = validate_refinement_plan(refinement, pretrain)

    assert fingerprint == refinement_corpus_fingerprint(pretrain)
    assert fingerprint == refinement_corpus_fingerprint(refinement)
    assert len(fingerprint) == 64


def test_same_token_count_with_different_shard_rows_is_not_refinement() -> None:
    pretrain = _load(PRETRAIN_PLAN)
    refinement = _load(REFINEMENT_PLAN)
    altered = copy.deepcopy(refinement)
    first_source = next(iter(altered["phases"][0]["sources"].values()))
    first_source[0]["rows"] -= 1

    # This is the regression: totals still match, but corpus identity does not.
    assert altered["unique_tokens"] == pretrain["unique_tokens"]
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_refinement_plan(altered, pretrain)


def test_refinement_plan_rejects_replay_even_when_unique_core_matches() -> None:
    pretrain = _load(PRETRAIN_PLAN)
    refinement = _load(REFINEMENT_PLAN)
    altered = copy.deepcopy(refinement)
    altered["phases"].append(copy.deepcopy(altered["phases"][0]))
    altered["phases"][1]["name"] = "replay"
    altered["trained_tokens"] *= 2

    with pytest.raises(ValueError, match="only the unique_core"):
        validate_refinement_plan(altered, pretrain)


def test_every_text_sft_launcher_declares_checkpoint_source_stage() -> None:
    launchers = [
        path
        for path in (PROJECT_ROOT / "scripts").glob("*.sh")
        if "-m scripts.sft_500m_32k_tr" in path.read_text(encoding="utf-8")
        or "-m scripts.sft_tr" in path.read_text(encoding="utf-8")
    ]
    assert launchers
    for launcher in launchers:
        source = launcher.read_text(encoding="utf-8")
        assert "--source-stage" in source, launcher


def test_non_vision_modality_trainers_wire_source_stage_into_the_guard() -> None:
    trainers = (
        PROJECT_ROOT / "complexity/generative/image/training.py",
        PROJECT_ROOT / "complexity/generative/image/edit_training.py",
        PROJECT_ROOT / "complexity/generative/vision_language/training.py",
    )
    for trainer in trainers:
        source = trainer.read_text(encoding="utf-8")
        assert '"--source-stage"' in source, trainer
        assert "source_stage=args.source_stage" in source, trainer


def test_vision_launcher_does_not_add_a_duplicate_refinement_stage() -> None:
    launcher = (
        PROJECT_ROOT / "scripts/vast_finetune_detector_coco_v08_nano.sh"
    ).read_text(encoding="utf-8")

    assert "TRAINING_PURPOSE=vision-supervised-finetuning" in launcher
    assert "--source-stage refinement" not in launcher
