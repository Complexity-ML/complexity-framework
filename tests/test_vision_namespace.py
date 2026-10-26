"""The vision tree preserves module identities and canonical training assets."""

import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("area", ["detection", "image", "video", "vision_language", "vision_tasks"])
def test_old_and_new_packages_share_identity(area):
    old = importlib.import_module(f"complexity.generative.{area}")
    new = importlib.import_module(f"complexity.vision.{area}")
    assert old is new
    assert new.__spec__.name == f"complexity.vision.{area}"


def test_nested_module_identity_and_legacy_pickle_resolution():
    old = importlib.import_module("complexity.generative.detection.config")
    new = importlib.import_module("complexity.vision.detection.config")
    assert old is new
    # Protocol 0 GLOBAL reproduces the old saved class path without rewriting
    # any class metadata or trusted checkpoint contents.
    payload = b"ccomplexity.generative.detection.config\nTRHashDetectorConfig\n."
    assert pickle.loads(payload) is new.TRHashDetectorConfig


def test_legacy_launcher_help_still_works_without_training():
    result = subprocess.run(
        [sys.executable, "-m", "complexity.generative.detection.training", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--annotations" in result.stdout


@pytest.mark.parametrize(
    "old, new",
    [
        ("generate_tr_hash_image", "image.generation"),
        ("edit_tr_hash_image", "image.edit_inference"),
    ],
)
def test_compatibility_script_shares_operation_module(old, new):
    assert importlib.import_module(f"scripts.{old}") is importlib.import_module(
        f"complexity.vision.{new}"
    )


def test_vision_import_and_workflow_listing_do_not_load_runtimes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import complexity.vision; import complexity.vision.workflows; "
                "assert 'torch' not in sys.modules; "
                "assert 'complexity.vision.detection' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_dispatch_restores_argv_and_forwards_args(monkeypatch):
    from complexity.vision import workflows

    calls = []

    class Runtime:
        def main(self):
            calls.append(sys.argv.copy())
            raise ValueError("preflight failed")

    monkeypatch.setattr(workflows, "import_module", lambda module: Runtime())
    previous = sys.argv
    with pytest.raises(ValueError, match="preflight failed"):
        workflows.run_workflow("train", "detector", ["--annotations", "local.json"])
    assert sys.argv is previous
    assert calls == [["complexity vision train detector", "--annotations", "local.json"]]


def test_cli_lists_and_forwards_runtime_help():
    from typer.testing import CliRunner

    from complexity.cli.app import app

    runner = CliRunner()
    assert "complexity vision train detector" in runner.invoke(app, ["vision", "list"]).stdout
    result = runner.invoke(app, ["vision", "train", "detector", "--", "--help"])
    assert result.exit_code == 0, result.output
    assert "--annotations" in result.stdout


def test_fsdp_assets_match_the_canonical_model_and_preserve_the_data_schedule():
    import hashlib
    import json

    from complexity.config import ModelConfig

    root = Path(__file__).parents[1] / "scripts/pretrain_1b_fsdp"
    model_path = root / "model.json"
    recipe = json.loads((root / "recipe.json").read_text())
    raw = json.loads(model_path.read_text())
    config = ModelConfig.from_dict(raw, strict=True)

    assert recipe["model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert (config.num_attention_heads, config.num_key_value_heads) == (4, 1)
    assert (config.num_experts, config.top_k) == (4, 2)
    assert config.routing_strategy == "token_id_composed_modulo"
    assert config.top_k_primary_weight == 0.5
    assert config.routed_output_scale == 2.0
    assert config.use_qk_norm and config.shared_expert and config.use_sdpa
    assert (config.hidden_size, config.num_hidden_layers, config.vocab_size) == (1536, 24, 32000)
    assert (config.intermediate_size, config.shared_intermediate_size) == (3072, 4352)
    assert config.max_position_embeddings == 16384
    assert config.rope_theta == 1000000.0

    # Architecture replacement changes its model digest, while the pinned data
    # and the complete optimizer/update schedule keep their declared values.
    unchanged = {
        "model_repo": "Pacific-i64/TR-HASH-MoE-1B-70B-Agentic-Pretraining",
        "data_plan_sha256": "b16c633b8f17e66db37a04858dd1995216b0ef827e8a5f36e0ce9590689c5c85",
        "world_size": 4,
        "microbatch": 2,
        "context": 16384,
        "seed": 42,
        "max_updates": 534057,
        "lr": 0.0003,
        "min_lr_ratio": 0.1,
        "warmup_updates": 2000,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "grad_clip": 1.0,
        "save_every": 1000,
        "eval_every": 1000,
        "first_checkpoint": 10,
        "cache_gib": 64,
        "keep_checkpoints": 3,
    }
    assert {key: recipe[key] for key in unchanged} == unchanged
    assert recipe["world_size"] * recipe["microbatch"] * recipe["context"] == 131072
    assert recipe["max_updates"] * 131072 == 69999919104
