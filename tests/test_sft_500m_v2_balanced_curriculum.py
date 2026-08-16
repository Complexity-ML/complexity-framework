from __future__ import annotations

from pathlib import Path

import pytest

from complexity.training.sft_curriculum import (
    audit_stage_loss_targets,
    load_curriculum,
)


CONFIG = Path("configs/sft_500m_32k_v2_balanced.yaml")
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
LOSS_KEYS = (set(TASKS) - {"casual_conversation"}) | {
    "casual_social",
    "casual_reasoning",
    "casual_instruction",
}
REASONING_TASKS = (
    "reasoning_verification",
    "planning_comparison",
    "explanation_learning",
    "critique_revision",
    "troubleshooting",
)


def _examples() -> list[dict[str, object]]:
    # Aggregate visible-label counts measured from the audited 229,026-example
    # release. One metadata row per loss cell keeps this regression fast while
    # preserving the exact coefficient derivation used by the full shard.
    supervised_tokens = {
        "reasoning_verification": 8_778_371,
        "planning_comparison": 15_113,
        "explanation_learning": 7_395_622,
        "critique_revision": 1_805,
        "troubleshooting": 19_399,
        "conversation_empathy": 152_125,
        "practical_action": 29_274,
        "grounded_qa": 127_419,
        "context_clarification": 293_079,
        "brainstorming_creativity": 25_329,
        "writing_transformation": 183_315,
        "summarization_synthesis": 244_072,
        "extraction_classification": 163_899,
        "safety_uncertainty": 207_259,
    }
    ordinary = [
        {
            "example_id": f"{task}-aggregate",
            "task": task,
            "num_tokens": supervised_tokens[task],
            "supervised_tokens": supervised_tokens[task],
        }
        for task in TASKS
        if task != "casual_conversation"
    ]
    casual_domains = (
        ("social_greeting", 109_205),
        ("addition", 610_482),
        ("instruction_formatting", 182_359),
    )
    casual = [
        {
            "example_id": f"casual-{domain}-aggregate",
            "task": "casual_conversation",
            "domain": domain,
            "num_tokens": tokens,
            "supervised_tokens": tokens,
        }
        for domain, tokens in casual_domains
    ]
    return ordinary + casual


def test_v2_balanced_profile_assigns_every_family_once() -> None:
    curriculum = load_curriculum(CONFIG)
    stage = curriculum.stages[0]
    assigned = [task for group in stage.loss_groups for task in group.tasks]

    assert stage.name == "full-shard-weighted"
    assert stage.max_examples is None
    assert stage.balance_by == "none"
    assert stage.filters == stage.filters.__class__()
    assert set(assigned) == LOSS_KEYS
    assert len(assigned) == len(set(assigned))
    assert sum(group.target_share for group in stage.loss_groups) == pytest.approx(1.0)
    assert stage.max_task_loss_weight == pytest.approx(30.0)
    assert all(group.task_target_shares for group in stage.loss_groups)
    assert all(
        sum(share for _, share in group.task_target_shares) == pytest.approx(1.0)
        for group in stage.loss_groups
    )
    assert {cell.name for cell in stage.loss_cells} == {
        "casual_social",
        "casual_reasoning",
        "casual_instruction",
    }


def test_v2_balanced_profile_keeps_full_shard_and_weights_token_loss() -> None:
    examples = _examples()
    audit = audit_stage_loss_targets(
        examples,
        load_curriculum(CONFIG),
        "full-shard-weighted",
    )

    assert audit["passed"] is True
    assert audit["selected_examples"] == len(examples)
    assert audit["weighted_group_shares"] == pytest.approx(
        {
            "distilled_reasoning": 0.20,
            "natural_conversation": 0.20,
            "instruction_and_structured": 0.60,
        }
    )
    assert audit["weighted_task_shares"] == pytest.approx(
        {
            "reasoning_verification": 0.0925,
            "explanation_learning": 0.0800,
            "casual_reasoning": 0.0150,
            "planning_comparison": 0.0050,
            "troubleshooting": 0.0050,
            "critique_revision": 0.0025,
            "casual_social": 0.1400,
            "conversation_empathy": 0.0600,
            "context_clarification": 0.1000,
            "extraction_classification": 0.0800,
            "grounded_qa": 0.0800,
            "practical_action": 0.0400,
            "brainstorming_creativity": 0.0350,
            "safety_uncertainty": 0.0700,
            "summarization_synthesis": 0.0700,
            "writing_transformation": 0.0600,
            "casual_instruction": 0.0650,
        }
    )
    assert audit["weights_within_cap"] is True
    assert max(audit["task_loss_weights"].values()) <= 30.0


def test_v2_balanced_profile_maps_new_capability_domains() -> None:
    stage = load_curriculum(CONFIG).stages[0]
    cells = {cell.name: set(cell.domains) for cell in stage.loss_cells}

    assert {
        "concept_definition",
        "bullet_constraints",
        "length_constraints",
        "sentence_constraints",
        "structured_constraints",
    } <= cells["casual_instruction"]


def test_v2_balanced_profile_does_not_reintroduce_10k_sampling() -> None:
    source = CONFIG.read_text(encoding="utf-8")

    assert "max_examples: all" in source
    assert "max_examples: 10000" not in source
    assert "balance_by: weighted_task" not in source
