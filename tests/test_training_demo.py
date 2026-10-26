"""Real tiny-model integration, continuation and explicit device selection."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from complexity.cli.app import app
from complexity.training.config import TrainingConfig
from complexity.training.demo import (
    FORMAT,
    HORIZON,
    PAUSE_UPDATE,
    _build_trainer,
    _rng,
    _seed,
    run_training_demo,
    synthetic_fixtures,
)
from complexity.training.trainer import Trainer
from complexity.utils.checkpointing import CheckpointManager


def test_native_demo_really_restores_adamw_cursor_rng_and_export(tmp_path):
    before_rng = _rng("cpu")
    report = run_training_demo(tmp_path / "demo", device="cpu")
    assert report["format"] == FORMAT and report["passed"]
    assert report["synthetic_data"] and not report["downloaded_assets"]
    assert report["release_eligible"] is False
    assert report["model_parameters"] == 30_912
    assert report["planned_updates"] == report["final_update"] == report["data_cursor"] == 4
    assert report["pause_update"] == report["resumed_from_update"] == 2
    assert all(check["passed"] for check in report["checks"].values())
    assert all(check.get("max_abs_diff", 0.0) == 0.0 for check in report["checks"].values())
    assert [row["step"] for row in report["updates"]] == [1, 2, 3, 4]
    assert [row["epoch"] for row in report["updates"]] == [1, 1, 1, 1]
    assert [row["row_ids"] for row in report["updates"]] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert report["updates"][-1]["learning_rate"] == pytest.approx(0.0002)
    assert math.isfinite(report["initial_eval_loss"]) and math.isfinite(report["final_eval_loss"])
    checkpoint = Path(report["checkpoint"])
    marker = json.loads((checkpoint / "complete.json").read_text())
    assert marker["contract"]["training"]["max_steps"] == HORIZON
    assert {"rng_rank0.pt", "optimizer_rank0.pt", "checkpoint.pt", "training_state.json"} <= marker[
        "files"
    ].keys()
    state = json.loads((checkpoint / "training_state.json").read_text())
    assert state["step"] == PAUSE_UPDATE and state["epoch"] == 1
    assert state["total_tokens"] == 48
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    assert payload["scheduler"]["last_epoch"] == PAUSE_UPDATE
    assert payload["scheduler"]["_schedulers"][1]["T_max"] == 3
    assert (Path(report["export"]) / "model.safetensors").is_file()
    assert json.loads(Path(report["report"]).read_text()) == report
    assert torch.equal(torch.get_rng_state(), before_rng["torch"])


def test_demo_fixtures_have_aligned_targets_supervised_eos_and_masks():
    fixtures = synthetic_fixtures()
    assert torch.equal(fixtures["labels"][:, 1:-1], fixtures["input_ids"][:, 2:])
    assert fixtures["labels"][:, 0].tolist() == [-100] * 12
    assert fixtures["labels"][:, -1].tolist() == [0] * 12
    assert fixtures["input_ids"].min() >= 1
    assert fixtures["input_ids"].max() < 32


def test_cpu_demo_does_not_seed_unused_device_generators(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU demo must not seed other device generators")

    monkeypatch.setattr(torch, "manual_seed", unexpected)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected)
    monkeypatch.setattr(torch.mps, "manual_seed", unexpected)
    assert run_training_demo(tmp_path / "cpu-only", device="cpu")["passed"]


def test_fixtures_stay_on_cpu_under_another_default_device():
    with torch.device("meta"):
        fixtures = synthetic_fixtures()
    assert all(tensor.device.type == "cpu" for tensor in fixtures.values())
    assert fixtures["labels"][:, -1].tolist() == [0] * 12


def test_demo_cli_is_one_command_with_json_stdout(tmp_path):
    result = CliRunner().invoke(
        app, ["train", "demo", "--output", str(tmp_path / "cli"), "--device", "cpu"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["passed"] and report["device"] == "cpu"
    assert report["checks"]["export_reload"]["passed"]


def test_existing_demo_output_is_preserved_without_overwrite(tmp_path):
    sentinel = tmp_path / "important.json"
    sentinel.write_text("preserve me")
    with pytest.raises(FileExistsError, match="new or empty"):
        run_training_demo(tmp_path)
    assert sentinel.read_text() == "preserve me"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_unavailable_mps_fails_before_output_or_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    output = tmp_path / "missing-device"
    with pytest.raises(RuntimeError, match="MPS.*unavailable.*refusing CPU fallback"):
        run_training_demo(output, device="mps")
    assert not output.exists()
    result = CliRunner().invoke(app, ["train", "demo", "--output", str(output), "--device", "mps"])
    assert result.exit_code == 2 and "refusing CPU fallback" in result.stderr
    assert not output.exists()


def test_explicit_cpu_does_not_select_available_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "device_count", lambda: pytest.fail("CPU should not enumerate CUDA")
    )
    trainer, _ = _build_trainer(tmp_path, synthetic_fixtures(), "cpu")
    assert trainer.device.type == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in trainer.model.parameters())


def test_explicit_mps_does_not_claim_an_unsupported_mixed_precision(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="MPS Trainer.*precision='fp32'"):
        Trainer(torch.nn.Linear(2, 2), TrainingConfig(), [], device="mps")


def test_omitted_device_keeps_the_historical_cpu_choice_with_available_mps(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    config = TrainingConfig(
        use_fsdp=False,
        precision="fp32",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )
    trainer = Trainer(torch.nn.Linear(2, 2), config, [])
    assert trainer.device.type == "cpu"


def test_demo_resume_rejects_a_changed_schedule_before_loading_weights(tmp_path):
    fixtures = synthetic_fixtures()
    _seed("cpu")
    first, _ = _build_trainer(tmp_path / "first", fixtures, "cpu", stop_at=PAUSE_UPDATE)
    first.train()
    checkpoint = tmp_path / "first" / "checkpoints" / "interrupted_2"
    resumed, _ = _build_trainer(tmp_path / "resume", fixtures, "cpu", resume=checkpoint)
    resumed.checkpoint_manager.resume_contract["training"]["max_steps"] += 1
    before = {name: value.clone() for name, value in resumed.model.state_dict().items()}
    with pytest.raises(ValueError, match="Resume contract"):
        resumed.train()
    assert all(
        torch.equal(value, resumed.model.state_dict()[name]) for name, value in before.items()
    )
    assert resumed.global_step == 0


def test_demo_checkpoint_corruption_is_rejected_before_model_mutation(tmp_path):
    fixtures = synthetic_fixtures()
    first, _ = _build_trainer(tmp_path / "first", fixtures, "cpu", stop_at=PAUSE_UPDATE)
    first.train()
    checkpoint = tmp_path / "first" / "checkpoints" / "interrupted_2"
    (checkpoint / "rng_rank0.pt").write_bytes(b"broken")
    resumed, _ = _build_trainer(tmp_path / "resume", fixtures, "cpu", resume=checkpoint)
    before = {name: value.clone() for name, value in resumed.model.state_dict().items()}
    with pytest.raises(ValueError, match="Checkpoint integrity failure: rng_rank0.pt"):
        resumed.train()
    assert all(
        torch.equal(value, resumed.model.state_dict()[name]) for name, value in before.items()
    )


def test_missing_mps_rng_is_rejected_before_restore_without_mps_hardware(tmp_path, monkeypatch):
    fixtures = synthetic_fixtures()
    first, _ = _build_trainer(tmp_path / "first", fixtures, "cpu", stop_at=PAUSE_UPDATE)
    first.train()
    checkpoint = tmp_path / "first" / "checkpoints" / "interrupted_2"
    manager = first.checkpoint_manager
    manager.model = SimpleNamespace(
        parameters=lambda: [SimpleNamespace(device=torch.device("mps"))]
    )
    monkeypatch.setattr(
        manager, "_load_regular", lambda *args: pytest.fail("No weights may be mutated")
    )
    with pytest.raises(ValueError, match="Exact MPS resume requires a valid MPS RNG"):
        manager.load(checkpoint)


def test_mps_rng_cannot_be_silently_dropped_by_cpu_restore(tmp_path, monkeypatch):
    first, _ = _build_trainer(tmp_path / "first", synthetic_fixtures(), "cpu", stop_at=PAUSE_UPDATE)
    first.train()
    checkpoint = tmp_path / "first" / "checkpoints" / "interrupted_2"
    path = checkpoint / "rng_rank0.pt"
    rng = torch.load(path, weights_only=False)
    rng["mps"] = torch.ones(8, dtype=torch.uint8)
    torch.save(rng, path)
    marker_path = checkpoint / "complete.json"
    marker = json.loads(marker_path.read_text())
    marker["files"][path.name] = {
        "bytes": path.stat().st_size,
        "sha256": CheckpointManager._digest(path),
    }
    marker_path.write_text(json.dumps(marker))
    monkeypatch.setattr(
        first.checkpoint_manager,
        "_load_regular",
        lambda *args: pytest.fail("No weights may be mutated"),
    )
    with pytest.raises(ValueError, match="MPS RNG.*non-MPS"):
        first.checkpoint_manager.load(checkpoint)


def test_incompatible_mps_rng_layout_is_rejected_before_model_mutation(tmp_path, monkeypatch):
    first, _ = _build_trainer(tmp_path / "first", synthetic_fixtures(), "cpu", stop_at=PAUSE_UPDATE)
    first.train()
    checkpoint = tmp_path / "first" / "checkpoints" / "interrupted_2"
    path = checkpoint / "rng_rank0.pt"
    rng = torch.load(path, weights_only=False)
    rng["mps"] = torch.ones(2, dtype=torch.uint8)
    torch.save(rng, path)
    marker_path = checkpoint / "complete.json"
    marker = json.loads(marker_path.read_text())
    marker["files"][path.name] = {
        "bytes": path.stat().st_size,
        "sha256": CheckpointManager._digest(path),
    }
    marker_path.write_text(json.dumps(marker))
    manager = first.checkpoint_manager
    manager.model = SimpleNamespace(
        parameters=lambda: [SimpleNamespace(device=torch.device("mps"))]
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.zeros(8, dtype=torch.uint8))
    monkeypatch.setattr(
        manager, "_load_regular", lambda *args: pytest.fail("No weights may be mutated")
    )
    with pytest.raises(ValueError, match="current MPS RNG layout"):
        manager.load(checkpoint)


def test_local_demo_rejects_distributed_launch_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one process"):
        run_training_demo(tmp_path / "distributed")
    assert not (tmp_path / "distributed").exists()
