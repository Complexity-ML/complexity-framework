from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
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


def test_agentic_100m_launcher_bounds_steps_to_audited_plan(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer / "chat_template.jinja").write_text("template", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"trained_tokens": 1_000_003}), encoding="utf-8")
    refinement = tmp_path / "refinement.json"
    refinement.write_text(json.dumps({"trained_tokens": 500_003}), encoding="utf-8")
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    torchrun = binary_directory / "torchrun"
    torchrun.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    torchrun.chmod(0o755)
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [str(repository / "scripts/run_tr_hash_agentic_100m.sh"), "pretraining"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{binary_directory}:{os.environ['PATH']}",
            "REPO_ROOT": str(tmp_path),
            "VENV_ACTIVATE": str(tmp_path / "missing-venv"),
            "TOKENIZER": str(tokenizer),
            "PRETRAIN_PLAN": str(plan),
            "REFINEMENT_PLAN": str(refinement),
            "NPROC_PER_NODE": "4",
            "BATCH_SIZE_PER_GPU": "2",
            "GRADIENT_ACCUMULATION": "1",
            "SEQ_LEN": "8",
        },
    )

    assert "exact bounded schedule=15625 steps" in result.stdout
    assert "unused_tail=3" in result.stdout
    assert "--max-steps\n15625\n" in result.stdout
