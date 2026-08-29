from __future__ import annotations

import json

import pytest

from scripts.build_tr_hash_refinement_plan import build_refinement_plan
from scripts.train_tr_hash_text_lineage import (
    make_tr_hash_config,
    validate_lineage_plans,
)


def _pretrain_plan() -> dict:
    selections = {"dclm": [{"file": "tokens-00000.bin", "rows": 8}]}
    return {
        "format": "tr-hash-token-replay-plan-v1",
        "dataset": "hf://datasets/AETHORIA-AI/data-32k-200b-tokens",
        "revision": "main",
        "seq_len": 1024,
        "selection_mode": "manifest_order",
        "row_alignment": 1,
        "unique_tokens": 8192,
        "trained_tokens": 16384,
        "source_unique_tokens": {"dclm": 8192},
        "source_passes": {"dclm": 2},
        "phases": [
            {"name": "unique_core", "passes": 1, "sources": selections},
            {"name": "quality_replay_2", "passes": 1, "sources": selections},
        ],
    }


def test_model_preset_keeps_canonical_multi_hash_contract() -> None:
    config = make_tr_hash_config("complexity-small")

    assert config.vocab_size == 32000
    assert config.routing_strategy == "token_id_multi_hash"
    assert config.route_hash_count == 2
    assert config.top_k == 2
    assert config.top_k_primary_weight == 0.5
    assert config.shared_expert is True
    assert config.tie_word_embeddings is True


def test_refinement_builder_keeps_exact_unique_core_and_drops_replay() -> None:
    pretrain = _pretrain_plan()
    refinement = build_refinement_plan(pretrain)

    assert refinement["trained_tokens"] == refinement["unique_tokens"] == 8192
    assert refinement["source_passes"] == {"dclm": 1}
    assert [phase["name"] for phase in refinement["phases"]] == ["unique_core"]
    assert refinement["phases"][0]["sources"] == pretrain["phases"][0]["sources"]


def test_lineage_validator_accepts_derived_refinement(tmp_path) -> None:
    pretrain = _pretrain_plan()
    refinement = build_refinement_plan(pretrain)
    pretrain_path = tmp_path / "pretrain.json"
    refinement_path = tmp_path / "refinement.json"
    pretrain_path.write_text(json.dumps(pretrain), encoding="utf-8")
    refinement_path.write_text(json.dumps(refinement), encoding="utf-8")

    fingerprint = validate_lineage_plans(
        stage="refinement",
        tokenized_plan=str(refinement_path),
        pretrain_plan=str(pretrain_path),
    )

    assert fingerprint is not None and len(fingerprint) == 64


def test_lineage_validator_rejects_different_refinement_rows(tmp_path) -> None:
    pretrain = _pretrain_plan()
    refinement = build_refinement_plan(pretrain)
    refinement["phases"][0]["sources"]["dclm"][0]["rows"] = 7
    pretrain_path = tmp_path / "pretrain.json"
    refinement_path = tmp_path / "refinement.json"
    pretrain_path.write_text(json.dumps(pretrain), encoding="utf-8")
    refinement_path.write_text(json.dumps(refinement), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_lineage_plans(
            stage="refinement",
            tokenized_plan=str(refinement_path),
            pretrain_plan=str(pretrain_path),
        )
