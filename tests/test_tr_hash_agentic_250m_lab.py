from __future__ import annotations

import json
import tomllib
from pathlib import Path

from complexity.training.finetuning import validate_refinement_plan


PROJECT_ROOT = Path(__file__).parents[1]
PLAN_DIR = PROJECT_ROOT / "configs" / "replay_plans"
JOB_DIR = PROJECT_ROOT / "configs" / "jobs" / "home"
DATASET = "hf://datasets/AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K"
REVISION = "fc738b3a10c5c093e3b34b48bcf1cb7066184706"


def _json(name: str) -> dict:
    return json.loads((PLAN_DIR / name).read_text(encoding="utf-8"))


def _job(name: str) -> dict:
    with (JOB_DIR / name).open("rb") as handle:
        return tomllib.load(handle)


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_agentic_pretraining_plan_preserves_selective_replay_recipe() -> None:
    plan = _json("tr_hash_agentic_250m_quality_replay.json")

    assert plan["dataset"] == DATASET
    assert plan["revision"] == REVISION
    assert plan["seq_len"] == 2048
    assert plan["unique_tokens"] == 249_888_768
    assert plan["trained_tokens"] == 481_771_520
    assert [phase["name"] for phase in plan["phases"]] == [
        "unique_core",
        "quality_replay_2",
        "quality_replay_3",
    ]
    assert plan["source_passes"]["dclm_foundation"] == 1
    assert plan["source_passes"]["fineweb2_french_foundation"] == 1
    assert plan["source_passes"]["stack_java_agentic"] == 2
    assert plan["source_passes"]["nemotron_tool_calling"] == 2
    assert plan["source_passes"]["finemath_4plus_agentic"] == 3
    assert plan["source_passes"]["infiwebmath_3plus_foundation"] == 3


def test_agentic_refinement_is_exactly_one_unique_pass() -> None:
    pretrain = _json("tr_hash_agentic_250m_quality_replay.json")
    refinement = _json("tr_hash_agentic_250m_refinement.json")

    validate_refinement_plan(refinement, pretrain)
    assert refinement["trained_tokens"] == refinement["unique_tokens"] == 249_888_768
    assert [phase["name"] for phase in refinement["phases"]] == ["unique_core"]
    assert set(refinement["source_passes"].values()) == {1}


def test_home_jobs_pin_lineage_and_fit_the_audited_token_budget() -> None:
    smoke = _job("tr_hash_agentic_250m_smoke.toml")
    pretrain = _job("tr_hash_agentic_250m_pretrain.toml")
    refinement = _job("tr_hash_agentic_250m_refinement.toml")

    for job in (smoke, pretrain, refinement):
        command = job["command"]
        assert job["gpu_devices"] == ["1"]
        assert _arg(command, "--tokenized-data") == DATASET
        assert _arg(command, "--tokenized-revision") == REVISION
        assert _arg(command, "--seq-len") == "2048"
        assert _arg(command, "--batch-size") == "4"
        assert _arg(command, "--gradient-accumulation") == "2"
        assert _arg(command, "--num-workers") == "0"
        assert "--require-cuda" in command
        assert job["checkpoint"]["pattern"] == "*_[0-9]*"
        assert job["environment"]["COMPLEXITY_REQUIRE_LIGER"] == "1"
        assert job["egpu"]["stable_seconds"] >= 120
        assert job["egpu"]["power_limit_w"] == 150

    assert _arg(smoke["command"], "--max-steps") == "20"
    assert _arg(pretrain["command"], "--target-tokens") == "481771520"
    assert "--save-steps" not in pretrain["command"]
    assert 481_771_520 // (4 * 2 * 2048) == 29_405
    assert _arg(refinement["command"], "--target-tokens") == "249888768"
    assert "--save-steps" not in refinement["command"]
    assert "--save-steps" not in smoke["command"]
    assert 249_888_768 // (4 * 2 * 2048) == 15_252
    assert _arg(refinement["command"], "--lr") == "3e-5"
    assert _arg(refinement["command"], "--init-checkpoint").endswith(
        "/artifacts/tr_hash_agentic_250m_pretrain/final"
    )
