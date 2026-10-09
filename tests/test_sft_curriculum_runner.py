from __future__ import annotations

from pathlib import Path

import pytest

from complexity.training.curriculum_planning import (
    CurriculumPlanningConfig,
    load_example_index,
    plan_curriculum,
    selected_checkpoint,
    stage_plan,
)
from complexity.training.sft_curriculum import CurriculumStage


def test_curriculum_runner_requires_a_release_ready_sft_shard() -> None:
    data = _planning_config()
    config = CurriculumPlanningConfig.from_dict(data)
    assert config.require_release_ready is True
    assert config.pack_sequences is True
    with pytest.raises(ValueError, match="release-ready data and sequence packing"):
        CurriculumPlanningConfig.from_dict({**data, "require_release_ready": False})
    with pytest.raises(ValueError, match="release-ready data and sequence packing"):
        CurriculumPlanningConfig.from_dict({**data, "pack_sequences": False})


def _planning_config():
    return {
        "checkpoint": "checkpoint",
        "sft_bin": "dataset",
        "curriculum_config": "curriculum.yaml",
        "through_stage": "final",
        "output_root": "output",
    }


def test_curriculum_defaults_to_preservation_first_lora() -> None:
    args = CurriculumPlanningConfig.from_dict(_planning_config())

    assert args.lora_rank == 16
    assert args.lora_alpha == 16
    assert args.expert_lr_multiplier == pytest.approx(0.25)


def test_curriculum_planner_merges_semantic_loss_sidecar(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "examples.jsonl").write_text(
        '{"example_id":"one","task":"casual_conversation"}\n',
        encoding="utf-8",
    )
    (train / "loss_metadata.jsonl").write_text(
        '{"example_id":"one","domain":"social_greeting","mode":"chat"}\n',
        encoding="utf-8",
    )

    examples = load_example_index(tmp_path)

    assert examples == [
        {
            "example_id": "one",
            "task": "casual_conversation",
            "domain": "social_greeting",
            "mode": "chat",
        }
    ]


@pytest.mark.parametrize("rank", [0, -1])
def test_curriculum_rejects_full_parameter_training(rank: int) -> None:
    with pytest.raises(ValueError, match="positive LoRA rank"):
        CurriculumPlanningConfig.from_dict({**_planning_config(), "lora_rank": rank})


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


def test_stage_plan_uses_packed_items_for_epoch_schedule() -> None:
    stage = CurriculumStage(
        name="packed",
        max_examples=4,
        epochs=3,
        lr=1e-6,
        batch_size=1,
        seq_len=8,
    )

    plan = stage_plan(
        stage,
        examples=4,
        world_size=1,
        example_lengths=[3, 3, 3, 3],
    )

    assert plan["pack_sequences"] is True
    assert plan["training_items"] == 2
    assert plan["steps_per_epoch"] == 2
    assert plan["total_steps"] == 6


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


def test_curriculum_preflight_validates_release_and_packing_without_writes(tmp_path):
    import json

    data = tmp_path / "data"
    train = data / "train"
    train.mkdir(parents=True)
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps({"quality_status": "passed", "release_quality": {"ready": True}}))
    (train / "examples.jsonl").write_text(
        "".join(json.dumps({"example_id": str(i), "task": "tiny", "num_tokens": 3}) + "\n" for i in range(4))
    )
    curriculum = tmp_path / "curriculum.yaml"
    curriculum.write_text("version: 1\nseed: 42\nstages:\n  - name: final\n    max_examples: all\n    epochs: 3\n    lr: 0.000001\n    batch_size: 1\n    seq_len: 8\n")
    config = CurriculumPlanningConfig.from_dict({
        **_planning_config(), "sft_bin": str(data), "curriculum_config": str(curriculum),
        "output_root": str(tmp_path / "not_created"),
    })
    before = {str(file): file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    report = plan_curriculum(config)
    assert report["stages"][0]["training_items"] == 2
    assert report["stages"][0]["total_steps"] == 6
    assert report["training_started"] is False
    assert {str(file): file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()} == before
    manifest.write_text(json.dumps({"quality_status": "failed"}))
    with pytest.raises(ValueError, match="quality_status must be passed"):
        plan_curriculum(config)
    assert not (tmp_path / "not_created").exists()


def test_curriculum_planning_does_not_truncate_oversized_examples():
    stage = CurriculumStage(name="packed", max_examples=1, epochs=1, lr=1e-6, seq_len=8)
    with pytest.raises(ValueError, match="without truncation"):
        stage_plan(stage, examples=1, world_size=1, example_lengths=[9])


def test_sidecar_must_match_index_without_duplicate_ids(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "examples.jsonl").write_text('{"example_id":"one","num_tokens":3}\n')
    sidecar = train / "loss_metadata.jsonl"
    sidecar.write_text('{"example_id":"one"}\n{"example_id":"one"}\n')
    with pytest.raises(ValueError, match="duplicate example IDs"):
        load_example_index(tmp_path)
    sidecar.write_text('{"example_id":"other"}\n')
    with pytest.raises(ValueError, match="exactly match"):
        load_example_index(tmp_path)
