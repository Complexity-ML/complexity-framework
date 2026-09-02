from __future__ import annotations

from types import SimpleNamespace

from complexity.models import ComplexityModel
from scripts.build_tr_hash_agentic_100m_plans import build_plans, parse_token_count
from scripts.train_tr_hash_text_lineage import make_tr_hash_config, validate_lineage_plans


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
