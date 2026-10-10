"""The vision tree keeps existing qualified imports and checkpoints usable."""

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
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--annotations" in result.stdout


@pytest.mark.parametrize("old, new", [
    ("generate_tr_hash_image", "image.generation"),
    ("edit_tr_hash_image", "image.edit_inference"),
])
def test_compatibility_script_shares_operation_module(old, new):
    assert importlib.import_module(f"scripts.{old}") is importlib.import_module(f"complexity.vision.{new}")


def test_vision_import_and_workflow_listing_do_not_load_runtimes():
    result = subprocess.run([sys.executable, "-c", (
        "import sys; import complexity.vision; import complexity.vision.workflows; "
        "assert 'torch' not in sys.modules; "
        "assert 'complexity.vision.detection' not in sys.modules"
    )], capture_output=True, text=True, timeout=15)
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


def test_published_fsdp_assets_remain_unchanged():
    import hashlib
    root = Path(__file__).parents[1] / "scripts/pretrain_1b_fsdp"
    assert hashlib.sha256((root / "recipe.json").read_bytes()).hexdigest() == "8cb72103026174e88be6c271e2147cf39f70e70d30cf70b181b1761755c0cf8d"
    assert hashlib.sha256((root / "model.json").read_bytes()).hexdigest() == "a614a990ddd587dcda3a4ec0cf0b998c3de18ea0806892fe5e9c29f9300c3a8d"
