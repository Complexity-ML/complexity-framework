from __future__ import annotations

from pathlib import Path

from complexity.training.sft_curriculum import (
    CurriculumFilters,
    CurriculumStage,
    SFTCurriculum,
    audit_stage_loss_targets,
    load_curriculum,
    loss_weight_key,
    select_stage_examples,
)


def _examples(count: int = 120) -> list[dict]:
    return [
        {
            "example_id": f"example-{index:04d}",
            "task": ("qa", "planning", "writing")[index % 3],
            "training_representation": (
                "natural_instruction" if index % 4 else "natural_multi_turn"
            ),
            "num_tokens": 80 + index,
            "supervised_tokens": 20 + index % 70,
        }
        for index in range(count)
    ]


def _curriculum() -> SFTCurriculum:
    return SFTCurriculum(
        seed=42,
        stages=(
            CurriculumStage(
                name="direct",
                max_examples=12,
                epochs=2,
                lr=1e-6,
                filters=CurriculumFilters(
                    training_representations=("natural_instruction",),
                    max_num_tokens=160,
                ),
            ),
            CurriculumStage(
                name="expanded",
                max_examples=30,
                epochs=1,
                lr=5e-7,
            ),
            CurriculumStage(
                name="full",
                max_examples=None,
                epochs=1,
                lr=1e-7,
                balance_by="none",
            ),
        ),
    )


def test_curriculum_selection_is_deterministic_and_balanced() -> None:
    examples = _examples()
    curriculum = _curriculum()

    first = select_stage_examples(examples, curriculum, "direct")
    second = select_stage_examples(reversed(examples), curriculum, "direct")

    assert [row["example_id"] for row in first] == [
        row["example_id"] for row in second
    ]
    assert len(first) == 12
    counts = {
        task: sum(row["task"] == task for row in first)
        for task in {row["task"] for row in first}
    }
    assert max(counts.values()) - min(counts.values()) <= 1


def test_later_stage_retains_every_previous_example() -> None:
    examples = _examples()
    curriculum = _curriculum()
    direct = select_stage_examples(examples, curriculum, "direct")
    expanded = select_stage_examples(examples, curriculum, "expanded")

    assert len(expanded) == 30
    assert {row["example_id"] for row in direct}.issubset(
        {row["example_id"] for row in expanded}
    )


def test_full_stage_is_dynamic_instead_of_using_a_fixed_cap() -> None:
    curriculum = _curriculum()
    assert len(select_stage_examples(_examples(73), curriculum, "full")) == 73
    assert len(select_stage_examples(_examples(119), curriculum, "full")) == 119


def test_weighted_task_stage_makes_casual_conversation_majority() -> None:
    examples = [
        {
            "example_id": f"{task}-{index:04d}",
            "task": task,
            "mode": "chat",
            "num_tokens": 96,
            "supervised_tokens": 32,
        }
        for task in (
            "casual_conversation",
            "conversation_empathy",
            "practical_action",
        )
        for index in range(200)
    ]
    curriculum = SFTCurriculum(
        seed=42,
        stages=(
            CurriculumStage(
                name="conversation-blend",
                max_examples=100,
                epochs=1,
                lr=1e-7,
                balance_by="weighted_task",
                task_weights=(
                    ("casual_conversation", 0.70),
                    ("conversation_empathy", 0.20),
                    ("practical_action", 0.10),
                ),
                filters=CurriculumFilters(modes=("chat",)),
            ),
        ),
    )

    selected = select_stage_examples(examples, curriculum, "conversation-blend")
    counts = {
        task: sum(row["task"] == task for row in selected)
        for task in (
            "casual_conversation",
            "conversation_empathy",
            "practical_action",
        )
    }

    assert counts == {
        "casual_conversation": 70,
        "conversation_empathy": 20,
        "practical_action": 10,
    }


def test_weighted_task_stage_preserves_target_after_retaining_casual_stage() -> None:
    examples = [
        {
            "example_id": f"{task}-{index:04d}",
            "task": task,
            "mode": "chat",
            "num_tokens": 96,
            "supervised_tokens": 32,
        }
        for task in (
            "casual_conversation",
            "conversation_empathy",
            "practical_action",
        )
        for index in range(300)
    ]
    curriculum = SFTCurriculum(
        seed=42,
        stages=(
            CurriculumStage(
                name="casual-only",
                max_examples=50,
                epochs=1,
                lr=2e-7,
                filters=CurriculumFilters(tasks=("casual_conversation",)),
            ),
            CurriculumStage(
                name="conversation-blend",
                max_examples=100,
                epochs=1,
                lr=1e-7,
                balance_by="weighted_task",
                task_weights=(
                    ("casual_conversation", 0.70),
                    ("conversation_empathy", 0.20),
                    ("practical_action", 0.10),
                ),
                filters=CurriculumFilters(modes=("chat",)),
            ),
        ),
    )

    selected = select_stage_examples(examples, curriculum, "conversation-blend")

    assert sum(row["task"] == "casual_conversation" for row in selected) == 70
    assert sum(row["task"] == "conversation_empathy" for row in selected) == 20
    assert sum(row["task"] == "practical_action" for row in selected) == 10


def test_weighted_task_stage_keeps_weights_when_retained_group_is_exhausted() -> None:
    examples = [
        {
            "example_id": f"casual-{index:04d}",
            "task": "casual_conversation",
            "mode": "chat",
        }
        for index in range(70)
    ] + [
        {
            "example_id": f"{task}-{index:04d}",
            "task": task,
            "mode": "chat",
        }
        for task in ("conversation_empathy", "practical_action")
        for index in range(100)
    ]
    curriculum = SFTCurriculum(
        seed=42,
        stages=(
            CurriculumStage(
                name="casual-only",
                max_examples=70,
                epochs=1,
                lr=2e-7,
                filters=CurriculumFilters(tasks=("casual_conversation",)),
            ),
            CurriculumStage(
                name="conversation-blend",
                max_examples=100,
                epochs=1,
                lr=1e-7,
                balance_by="weighted_task",
                task_weights=(
                    ("casual_conversation", 0.70),
                    ("conversation_empathy", 0.20),
                    ("practical_action", 0.10),
                ),
                filters=CurriculumFilters(modes=("chat",)),
            ),
        ),
    )

    selected = select_stage_examples(examples, curriculum, "conversation-blend")

    assert sum(row["task"] == "casual_conversation" for row in selected) == 70
    assert sum(row["task"] == "conversation_empathy" for row in selected) == 20
    assert sum(row["task"] == "practical_action" for row in selected) == 10


def test_curriculum_uses_projected_metadata_for_difficulty() -> None:
    examples = _examples(30)
    metadata = {
        row["example_id"]: {
            "difficulty": "low" if index % 2 == 0 else "high"
        }
        for index, row in enumerate(examples)
    }
    curriculum = SFTCurriculum(
        seed=1,
        stages=(
            CurriculumStage(
                name="low",
                max_examples=None,
                epochs=1,
                lr=1e-6,
                filters=CurriculumFilters(difficulties=("low",)),
            ),
        ),
    )

    selected = select_stage_examples(examples, curriculum, "low", metadata)

    assert len(selected) == 15
    assert all(metadata[row["example_id"]]["difficulty"] == "low" for row in selected)


def test_curriculum_caps_repeated_surfaces_per_task() -> None:
    examples = [
        {
            "example_id": f"example-{index:03d}",
            "task": "qa" if index < 20 else "writing",
            "training_representation": "natural_instruction",
            "num_tokens": 64,
            "supervised_tokens": 16,
        }
        for index in range(40)
    ]
    metadata = {
        row["example_id"]: {
            "response": "Common opening words repeat here " + row["example_id"],
            "structure_signature": f"signature-{index % 3}",
        }
        for index, row in enumerate(examples)
    }
    curriculum = SFTCurriculum(
        seed=9,
        stages=(
            CurriculumStage(
                name="curated",
                max_examples=None,
                epochs=1,
                lr=1e-6,
                filters=CurriculumFilters(
                    max_structure_occurrences_per_task=2,
                    max_opening_occurrences_per_task=4,
                    opening_words=3,
                ),
            ),
        ),
    )

    selected = select_stage_examples(examples, curriculum, "curated", metadata)

    for task in ("qa", "writing"):
        task_rows = [row for row in selected if row["task"] == task]
        assert len(task_rows) == 4
        signature_counts = {
            signature: sum(
                row["structure_signature"] == signature for row in task_rows
            )
            for signature in {row["structure_signature"] for row in task_rows}
        }
        assert max(signature_counts.values()) <= 2


def test_curriculum_rejects_configured_boilerplate() -> None:
    examples = _examples(12)
    metadata = {
        row["example_id"]: {
            "response": (
                "The answer stays useful only while the record is current."
                if index % 2 == 0
                else "Use the current record and state its date."
            )
        }
        for index, row in enumerate(examples)
    }
    curriculum = SFTCurriculum(
        seed=3,
        stages=(
            CurriculumStage(
                name="curated",
                max_examples=None,
                epochs=1,
                lr=1e-6,
                filters=CurriculumFilters(
                    exclude_response_substrings=("answer stays useful only",),
                ),
            ),
        ),
    )

    selected = select_stage_examples(examples, curriculum, "curated", metadata)

    assert len(selected) == 6
    assert all("stays useful" not in row["response"] for row in selected)


def test_curriculum_yaml_loads_all_stages(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.yaml"
    path.write_text(
        """\
version: 1
seed: 7
stages:
  - name: first
    max_examples: 10
    epochs: 2
    lr: 1.0e-6
    filters:
      tasks: [qa]
  - name: full
    max_examples: all
    epochs: 1
    lr: 1.0e-7
""",
        encoding="utf-8",
    )

    curriculum = load_curriculum(path)

    assert curriculum.seed == 7
    assert curriculum.stages[0].filters.tasks == ("qa",)
    assert curriculum.stages[1].max_examples is None


def test_loss_cells_route_one_task_by_domain_without_dropping_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "curriculum.yaml"
    path.write_text(
        """\
version: 1
seed: 7
stages:
  - name: full
    max_examples: all
    epochs: 1
    lr: 1.0e-6
    balance_by: none
    loss_cells:
      casual_social:
        task: casual
        domains: [greeting, gratitude]
      casual_math:
        task: casual
        domains: [addition]
    loss_groups:
      conversation:
        target_share: 0.5
        tasks: [casual_social]
        task_target_shares: {casual_social: 1.0}
      tasks:
        target_share: 0.5
        tasks: [casual_math, qa]
        task_target_shares: {casual_math: 0.5, qa: 0.5}
""",
        encoding="utf-8",
    )
    examples = [
        {
            "example_id": "hello",
            "task": "casual",
            "domain": "greeting",
            "supervised_tokens": 10,
        },
        {
            "example_id": "sum",
            "task": "casual",
            "domain": "addition",
            "supervised_tokens": 20,
        },
        {
            "example_id": "fact",
            "task": "qa",
            "domain": "general",
            "supervised_tokens": 10,
        },
    ]
    curriculum = load_curriculum(path)
    stage = curriculum.stage("full")

    assert loss_weight_key(stage, examples[0]) == "casual_social"
    assert loss_weight_key(stage, examples[1]) == "casual_math"
    assert loss_weight_key(stage, examples[2]) == "qa"
    audit = audit_stage_loss_targets(examples, curriculum, "full")
    assert audit["passed"] is True
    assert audit["selected_examples"] == len(examples)
    assert audit["weighted_group_shares"] == {
        "conversation": 0.5,
        "tasks": 0.5,
    }


def test_curriculum_yaml_loads_weighted_task_profile(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.yaml"
    path.write_text(
        """\
version: 1
seed: 7
stages:
  - name: conversational
    max_examples: 100
    epochs: 1
    lr: 1.0e-7
    balance_by: weighted_task
    task_weights:
      casual_conversation: 0.7
      conversation_empathy: 0.2
      practical_action: 0.1
""",
        encoding="utf-8",
    )

    curriculum = load_curriculum(path)

    assert curriculum.stages[0].task_weights == (
        ("casual_conversation", 0.7),
        ("conversation_empathy", 0.2),
        ("practical_action", 0.1),
    )
