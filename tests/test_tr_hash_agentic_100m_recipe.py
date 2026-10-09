from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from complexity.models import ComplexityModel
from complexity.training.lineage import (
    make_tr_hash_config,
    resolve_lineage_schedule,
    validate_lineage_plans,
)
from complexity.training.lineage_runner import TRHashTextLineageRunner
from complexity.training.refinement import build_agentic_plans as build_plans
from complexity.training.refinement import parse_token_count


def _fake_dataset():
    return SimpleNamespace(
        seq_len=4,
        sources=(SimpleNamespace(name="agentic"), SimpleNamespace(name="foundation")),
        _rows_by_source={"agentic": 5_000, "foundation": 5_000},
        _source_manifests={
            "agentic": {
                "shards": [
                    {"file": "tokens-00000.bin", "rows": 3_000},
                    {"file": "tokens-00001.bin", "rows": 2_000},
                ]
            },
            "foundation": {
                "shards": [
                    {"file": "tokens-00000.bin", "rows": 2_500},
                    {"file": "tokens-00001.bin", "rows": 2_500},
                ]
            },
        },
    )


def test_agentic_100m_preset_has_expected_parameter_budget() -> None:
    model = ComplexityModel(make_tr_hash_config("complexity-100m"))

    assert sum(parameter.numel() for parameter in model.parameters()) == 100_366_720
    assert model.config.vocab_size == 32_000
    assert model.config.routing_strategy == "token_id_multi_hash"
    assert model.config.route_hash_count == 2


def test_agentic_100m_plan_keeps_exact_unique_core_for_refinement(tmp_path) -> None:
    pretrain, refinement = build_plans(
        _fake_dataset(),
        {"agentic": 40, "foundation": 60},
        dataset_uri="hf://datasets/example/agentic",
        revision="pinned-revision",
        requested_unique_tokens=24_000,
        requested_pretrain_tokens=36_000,
        row_alignment=10,
    )

    assert pretrain["unique_tokens"] == 24_000
    assert pretrain["trained_tokens"] == 36_000
    assert refinement["unique_tokens"] == 24_000
    assert refinement["trained_tokens"] == 24_000
    assert len(refinement["phases"]) == 1
    assert refinement["phases"][0] == pretrain["phases"][0]

    pretrain_path = tmp_path / "pretrain.json"
    refinement_path = tmp_path / "refinement.json"
    pretrain_path.write_text(__import__("json").dumps(pretrain), encoding="utf-8")
    refinement_path.write_text(__import__("json").dumps(refinement), encoding="utf-8")
    fingerprint = validate_lineage_plans(
        stage="refinement",
        tokenized_plan=str(refinement_path),
        pretrain_plan=str(pretrain_path),
    )
    assert fingerprint is not None


def test_agentic_100m_token_count_parser() -> None:
    assert parse_token_count("70B") == 70_000_000_000
    assert parse_token_count("125B") == 125_000_000_000


def test_direct_refinement_entrypoint_rejects_low_peak_before_loading_data() -> None:
    runner = object.__new__(TRHashTextLineageRunner)
    args = SimpleNamespace(
        stage="refinement",
        model_preset="complexity-100m",
        lr=3e-5,
        lr_scheduler="cosine",
        warmup_tokens=500_000_000,
        warmup_steps=None,
        weight_decay=0.1,
        batch_size=16,
        gradient_accumulation=30,
        seq_len=2048,
    )
    with pytest.raises(ValueError, match="lr=3e-05"):
        runner.build_dataset(None, args, rank=0, world_size=4)


def test_agentic_100m_schedule_bounds_steps_to_audited_plan(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"trained_tokens": 1_000_003}), encoding="utf-8")
    report = resolve_lineage_schedule(
        json.loads(plan.read_text()),
        world_size=4,
        batch_size=2,
        gradient_accumulation=1,
        sequence_length=8,
    )

    assert report["max_steps"] == 15625
    assert report["unused_tail"] == 3
    assert report["scheduled_tokens"] == 1_000_000
    assert report["tokens_per_step"] == 64
    with pytest.raises(ValueError, match="exceeds audited plan"):
        resolve_lineage_schedule(
            json.loads(plan.read_text()),
            world_size=4,
            batch_size=2,
            gradient_accumulation=1,
            sequence_length=8,
            max_steps=15626,
        )
