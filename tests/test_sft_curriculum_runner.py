from __future__ import annotations

import pytest

from complexity.training.sft_curriculum import CurriculumStage
from scripts.run_sft_curriculum import selected_checkpoint, stage_plan


def test_stage_plan_covers_every_example_before_early_stopping() -> None:
    stage = CurriculumStage(
        name="direct",
        max_examples=10_000,
        epochs=3,
        lr=1e-6,
        batch_size=32,
        seq_len=192,
    )

    plan = stage_plan(stage, examples=10_000, world_size=1)

    assert plan["steps_per_epoch"] == 313
    assert plan["eval_steps"] == 313
    assert plan["save_steps"] == 313
    assert plan["total_steps"] == 939


def test_stage_plan_scales_steps_by_world_size() -> None:
    stage = CurriculumStage(
        name="balanced",
        max_examples=100_000,
        epochs=3,
        lr=2e-7,
        batch_size=32,
    )

    plan = stage_plan(stage, examples=100_000, world_size=4)

    assert plan["steps_per_epoch"] == 782
    assert plan["total_steps"] == 2346


def test_stage_plan_allows_runtime_batch_override() -> None:
    stage = CurriculumStage(name="reasoning", max_examples=8_000, epochs=1, lr=1e-6)

    plan = stage_plan(stage, examples=8_000, world_size=2, batch_size_override=24)

    assert plan["batch_size"] == 24
    assert plan["steps_per_epoch"] == 167
    assert plan["total_steps"] == 167


def test_stage_plan_preserves_schedule_with_lora_lr_multiplier() -> None:
    stage = CurriculumStage(name="reasoning", max_examples=8_000, epochs=1, lr=1e-6)

    plan = stage_plan(stage, examples=8_000, world_size=2, lr_multiplier=20.0)

    assert plan["lr"] == pytest.approx(2e-5)


def test_stage_without_validation_improvement_keeps_its_source(tmp_path) -> None:
    stage_root = tmp_path / "stage"
    source = tmp_path / "source"
    stage_root.mkdir()
    source.mkdir()
    periodic = stage_root / "step_000938"
    periodic.mkdir()

    assert selected_checkpoint(
        stage_root,
        source_checkpoint=source,
    ) == source


def test_stage_with_validation_improvement_selects_trained_checkpoint(
    tmp_path,
) -> None:
    stage_root = tmp_path / "stage"
    source = tmp_path / "source"
    best = stage_root / "best" / "step_000313"
    best.mkdir(parents=True)
    source.mkdir()
    (stage_root / "best.json").write_text(
        '{"checkpoint": "' + str(best) + '"}\n',
        encoding="utf-8",
    )

    assert selected_checkpoint(
        stage_root,
        source_checkpoint=source,
    ) == best
