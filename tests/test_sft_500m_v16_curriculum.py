from __future__ import annotations

from pathlib import Path

from complexity.training.sft_curriculum import (
    audit_planned_exposures,
    load_curriculum,
)

CONFIG = Path("configs/sft_500m_32k_v16.yaml")
TASKS = (
    "brainstorming_creativity",
    "casual_conversation",
    "context_clarification",
    "conversation_empathy",
    "critique_revision",
    "explanation_learning",
    "extraction_classification",
    "grounded_qa",
    "planning_comparison",
    "practical_action",
    "reasoning_verification",
    "safety_uncertainty",
    "summarization_synthesis",
    "troubleshooting",
    "writing_transformation",
)


def _examples() -> list[dict[str, object]]:
    return [
        {
            "example_id": f"{task}-{index:05d}",
            "task": task,
            "num_tokens": 256,
            "supervised_tokens": 96,
        }
        for task in TASKS
        for index in range(400 if task == "casual_conversation" else 2_000)
    ]


def test_500m_v16_profile_has_complete_non_overlapping_exposure_groups() -> None:
    curriculum = load_curriculum(CONFIG)

    assigned = [task for group in curriculum.exposure_groups for task in group.tasks]
    targets = {group.name: group.target_share for group in curriculum.exposure_groups}

    assert sorted(assigned) == sorted(TASKS)
    assert len(assigned) == len(set(assigned))
    assert targets == {
        "distilled_reasoning": 0.20,
        "natural_conversation": 0.25,
        "instruction_and_structured": 0.55,
    }
    for stage in curriculum.stages:
        if stage.balance_by == "weighted_task":
            assert abs(sum(dict(stage.task_weights).values()) - 1.0) < 1e-9


def test_500m_v16_planned_exposures_hit_the_training_contract() -> None:
    audit = audit_planned_exposures(_examples(), load_curriculum(CONFIG))

    assert audit["passed"] is True
    assert audit["total_exposures"] == 12_000
    assert audit["groups"]["distilled_reasoning"]["exposures"] == 2_400
    assert audit["groups"]["natural_conversation"]["exposures"] == 3_000
    assert audit["groups"]["instruction_and_structured"]["exposures"] == 6_600
    assert audit["stage_exposures"]["casual-adaptation"] == {
        "selected_examples": 400,
        "epochs": 5,
        "total_exposures": 2_000,
        "task_exposures": {"casual_conversation": 2_000},
    }


def test_500m_v16_reasoning_responses_are_length_bounded() -> None:
    curriculum = load_curriculum(CONFIG)
    reasoning_tasks = set(curriculum.exposure_groups[0].tasks)

    for stage in curriculum.stages:
        if reasoning_tasks.intersection(dict(stage.task_weights)):
            assert stage.filters.max_supervised_tokens is not None
            assert stage.filters.max_supervised_tokens <= 192
            assert "chain of thought" in stage.filters.exclude_response_substrings
