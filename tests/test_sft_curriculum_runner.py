from __future__ import annotations

from complexity.training.sft_curriculum import CurriculumStage
from scripts.run_sft_curriculum import stage_plan


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
