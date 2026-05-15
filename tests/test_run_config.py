"""Tests for YAML run configuration and resume guards."""

from __future__ import annotations

import argparse

import pytest


def test_yaml_config_defaults_and_cli_override(tmp_path):
    from complexity.training.run_config import parse_args_with_yaml_config

    config = tmp_path / "run.yaml"
    config.write_text(
        """
run:
  steps: 123
  batch-size: 64
  bf16: true
""",
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")

    args = parse_args_with_yaml_config(parser, ["--config", str(config), "--steps", "456"])

    assert args.steps == 456
    assert args.batch_size == 64
    assert args.bf16 is True


def test_yaml_config_rejects_unknown_keys(tmp_path):
    from complexity.training.run_config import parse_args_with_yaml_config

    config = tmp_path / "run.yaml"
    config.write_text("run:\n  nope: 1\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)

    with pytest.raises(ValueError, match="Unknown YAML config keys"):
        parse_args_with_yaml_config(parser, ["--config", str(config)])


def test_resume_guard_allows_volatile_changes(tmp_path):
    from complexity.training.run_config import write_or_validate_run_config

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = {
        "args": {"batch_size": 8, "save_steps": 100, "resume": None, "steps": 1000},
        "model_config": {"hidden_size": 128},
    }
    second = {
        "args": {"batch_size": 8, "save_steps": 10, "resume": "ckpt/latest", "steps": 2000},
        "model_config": {"hidden_size": 128},
    }

    write_or_validate_run_config(run_dir, first, resume=False)
    write_or_validate_run_config(run_dir, second, resume=True)


def test_resume_guard_rejects_training_mismatch(tmp_path):
    from complexity.training.run_config import write_or_validate_run_config

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = {
        "args": {"batch_size": 8, "seq_len": 2048},
        "model_config": {"hidden_size": 128},
    }
    second = {
        "args": {"batch_size": 16, "seq_len": 2048},
        "model_config": {"hidden_size": 128},
    }

    write_or_validate_run_config(run_dir, first, resume=False)
    with pytest.raises(ValueError, match="Resume config mismatch"):
        write_or_validate_run_config(run_dir, second, resume=True)
