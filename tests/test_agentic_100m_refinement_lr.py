from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from complexity.training.lineage import build_agentic_recipe
from complexity.training.lineage_runner import TRHashTextLineageRunner
from complexity.training.refinement import (
    REFINEMENT_LR_SCHEDULER,
    REFINEMENT_PEAK_LR,
    REFINEMENT_WARMUP_TOKENS,
    REFINEMENT_WEIGHT_DECAY,
    validate_refinement_recipe,
)


@pytest.fixture
def launch(tmp_path: Path, monkeypatch):
    plan = tmp_path / "plan.json"
    tokens = 3906 * 2048
    payload = {
        "format": "tr-hash-token-replay-plan-v1",
        "seq_len": 2048,
        "unique_tokens": tokens,
        "trained_tokens": tokens,
        "source_unique_tokens": {"agentic": tokens},
        "source_passes": {"agentic": 1},
        "phases": [
            {
                "name": "unique_core",
                "passes": 1,
                "sources": {"agentic": [{"file": "tokens-00000.bin", "rows": 3906}]},
            }
        ],
    }
    plan.write_text(json.dumps(payload))
    checkpoint = tmp_path / "base_final"
    checkpoint.mkdir()
    dispatched = []

    def run(*arguments: str, stage: str = "refinement", env_overrides=None):
        override = env_overrides or {}
        world = int(override.get("NPROC_PER_NODE", 4))
        monkeypatch.setenv("WORLD_SIZE", str(world))
        translations = {
            "LR": ("learning_rate", float),
            "LR_SCHEDULER": ("lr_scheduler", str),
            "WARMUP_TOKENS": ("warmup_tokens", int),
            "WEIGHT_DECAY": ("weight_decay", float),
            "GRADIENT_ACCUMULATION": ("gradient_accumulation", int),
        }
        typed = {
            translations[name][0]: translations[name][1](value)
            for name, value in override.items()
            if name in translations
        }
        dispatched.clear()
        try:
            settings = build_agentic_recipe(stage, payload, world_size=world, overrides=typed)
            argv = [
                "--model-preset",
                "complexity-100m",
                "--stage",
                stage,
                "--tokenized-data",
                str(tmp_path / "not_loaded"),
                "--tokenized-plan",
                str(plan),
                "--tokenized-revision",
                "pinned",
                "--batch-size",
                str(settings["batch_size"]),
                "--gradient-accumulation",
                str(settings["gradient_accumulation"]),
                "--seq-len",
                "2048",
                "--lr",
                str(settings["learning_rate"]),
                "--lr-scheduler",
                settings["lr_scheduler"],
                "--warmup-tokens",
                str(settings["warmup_tokens"]),
                "--weight-decay",
                str(settings["weight_decay"]),
            ]
            if stage == "refinement":
                argv += ["--pretrain-plan", str(plan), "--init-checkpoint", str(checkpoint)]
            parsed = TRHashTextLineageRunner()._build_parser().parse_args(argv + list(arguments))
            dispatched.append(parsed)
            stdout = f"optimizer_updates={parsed.max_steps} tokens/update={parsed.lineage_audit['tokens_per_step']}"
            return SimpleNamespace(returncode=0, stderr="", stdout=stdout), parsed
        except ValueError as exc:
            assert not dispatched
            return SimpleNamespace(returncode=2, stderr=str(exc), stdout=""), None

    return run


def test_refinement_defaults_reproduce_observed_200m_optimizer_regime(launch):
    result, arguments = launch()

    assert result.returncode == 0, result.stderr
    assert arguments is not None
    assert arguments.gradient_accumulation == 30
    assert arguments.lr == REFINEMENT_PEAK_LR
    assert arguments.lr_scheduler == REFINEMENT_LR_SCHEDULER
    assert arguments.warmup_tokens == REFINEMENT_WARMUP_TOKENS
    assert arguments.weight_decay == REFINEMENT_WEIGHT_DECAY
    assert "optimizer_updates=2 tokens/update=3932160" in result.stdout


def test_eight_gpu_refinement_derives_fifteen_accumulation_steps(launch):
    result, arguments = launch(env_overrides={"NPROC_PER_NODE": "8"})

    assert result.returncode == 0, result.stderr
    assert arguments is not None
    assert arguments.gradient_accumulation == 15


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
def test_non_reference_refinement_is_rejected_before_dispatch(launch, override, expected):
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
