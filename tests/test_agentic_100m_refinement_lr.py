from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.tr_hash_agentic_recipe import (
    REFINEMENT_LR_SCHEDULER,
    REFINEMENT_PEAK_LR,
    REFINEMENT_WARMUP_TOKENS,
    REFINEMENT_WEIGHT_DECAY,
    validate_refinement_recipe,
)


@pytest.fixture
def launch(tmp_path: Path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "chat_template.jinja").write_text("template")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"trained_tokens": 8_000_000}))
    checkpoint = tmp_path / "base_final"
    checkpoint.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python").symlink_to(sys.executable)
    marker = tmp_path / "launched.json"
    torchrun = binary / "torchrun"
    torchrun.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['LAUNCH_MARKER']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    torchrun.chmod(0o755)
    repository = Path(__file__).resolve().parents[1]

    def run(*arguments: str, stage: str = "refinement", env_overrides=None):
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "BATCH_SIZE_PER_GPU",
                "GRADIENT_ACCUMULATION",
                "LR",
                "LR_SCHEDULER",
                "WARMUP_TOKENS",
                "WEIGHT_DECAY",
            }
        }
        env.update(
            PATH=f"{binary}:{os.environ['PATH']}",
            REPO_ROOT=str(tmp_path),
            VENV_ACTIVATE=str(tmp_path / "missing-venv"),
            TOKENIZER=str(tokenizer),
            PRETRAIN_PLAN=str(plan),
            REFINEMENT_PLAN=str(plan),
            INIT_CHECKPOINT=str(checkpoint),
            NPROC_PER_NODE="4",
            BATCH_SIZE_PER_GPU="16",
            SEQ_LEN="2048",
            LAUNCH_MARKER=str(marker),
        )
        if env_overrides:
            env.update(env_overrides)
        marker.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "bash",
                str(repository / "scripts/run_tr_hash_agentic_100m.sh"),
                stage,
                *arguments,
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        return result, json.loads(marker.read_text()) if marker.exists() else None

    return run


def test_refinement_defaults_reproduce_observed_200m_optimizer_regime(launch):
    result, arguments = launch()

    assert result.returncode == 0, result.stderr
    assert arguments is not None
    assert int(arguments[arguments.index("--gradient-accumulation") + 1]) == 30
    assert float(arguments[arguments.index("--lr") + 1]) == REFINEMENT_PEAK_LR
    assert arguments[arguments.index("--lr-scheduler") + 1] == REFINEMENT_LR_SCHEDULER
    assert int(arguments[arguments.index("--warmup-tokens") + 1]) == REFINEMENT_WARMUP_TOKENS
    assert float(arguments[arguments.index("--weight-decay") + 1]) == REFINEMENT_WEIGHT_DECAY
    assert "optimizer_updates=2 tokens/update=3932160" in result.stdout


def test_eight_gpu_refinement_derives_fifteen_accumulation_steps(launch):
    result, arguments = launch(env_overrides={"NPROC_PER_NODE": "8"})

    assert result.returncode == 0, result.stderr
    assert arguments is not None
    assert int(arguments[arguments.index("--gradient-accumulation") + 1]) == 15


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"LR": "3e-5"}, "lr=3e-05"),
        ({"LR": "2e-4"}, "lr=0.0002"),
        ({"LR_SCHEDULER": "wsd"}, "lr_scheduler='wsd'"),
        ({"WARMUP_TOKENS": "1000000000"}, "warmup must use 500000000 tokens"),
        ({"WEIGHT_DECAY": "0.01"}, "weight_decay=0.01"),
        ({"GRADIENT_ACCUMULATION": "1"}, "tokens_per_step=131,072"),
    ],
)
def test_non_reference_refinement_is_rejected_before_torchrun(launch, override, expected):
    result, launched = launch(env_overrides=override)

    assert result.returncode != 0
    assert expected in result.stderr
    assert launched is None


def test_trailing_cli_override_cannot_bypass_refinement_contract(launch):
    result, launched = launch("--gradient-accumulation", "1")

    assert result.returncode != 0
    assert "tokens_per_step=131,072" in result.stderr
    assert launched is None


def test_validation_does_not_change_pretraining_or_other_models() -> None:
    validate_refinement_recipe(
        stage="pretraining",
        model_preset="complexity-100m",
        learning_rate=3e-4,
        lr_scheduler="wsd",
        warmup_tokens=1_000_000_000,
        warmup_steps=None,
        weight_decay=0.1,
        tokens_per_step=262_144,
    )
    validate_refinement_recipe(
        stage="refinement",
        model_preset="complexity-200m",
        learning_rate=3e-5,
        lr_scheduler="wsd",
        warmup_tokens=1,
        warmup_steps=None,
        weight_decay=1.0,
        tokens_per_step=1,
    )
