from __future__ import annotations

from pathlib import Path

from complexity.training.sft_curriculum import (
    CurriculumFilters,
    CurriculumStage,
    SFTCurriculum,
    load_curriculum,
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
