"""Real reload and distributed scope checks for complete native model exports."""

import datetime
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from complexity import ComplexityModel, ModelConfig


def _model():
    return ComplexityModel(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=16,
            mlp_type="tr_hash_engine",
            num_experts=4,
            top_k=2,
            use_custom_kernels=False,
        )
    ).eval()


def test_failed_full_tensor_never_writes_a_local_shard(tmp_path, monkeypatch):
    model = _model()

    class FailedShard:
        def full_tensor(self):
            raise RuntimeError("injected all-gather failure")

        def to_local(self):
            raise AssertionError("a local shard must never become a complete export")

    monkeypatch.setattr(model, "state_dict", lambda: {"embed_tokens.weight": FailedShard()})
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Cannot export complete model.*all-gather failure"):
        model.save_pretrained(output)
    assert not output.exists()


def test_serialization_failure_preserves_existing_export(tmp_path, monkeypatch):
    import safetensors.torch

    model = _model()
    model.save_pretrained(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def fail_save(state, path):
        Path(path).write_bytes(b"unfinished tensor file")
        raise OSError("injected write failure")

    monkeypatch.setattr(safetensors.torch, "save_file", fail_save)
    with pytest.raises(OSError, match="injected write failure"):
        model.save_pretrained(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_commit_failure_restores_old_weights_config_and_format(tmp_path, monkeypatch):
    model = _model()
    model.save_pretrained(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with torch.no_grad():
        model.embed_tokens.weight.add_(0.25)
    replace = Path.replace

    def fail_config_commit(path, target):
        if path.suffix == ".tmp" and Path(target).name == "config.json":
            raise OSError("injected config commit failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_config_commit)
    with pytest.raises(OSError, match="injected config commit failure"):
        model.save_pretrained(tmp_path, safe_serialization=False)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_failed_rollback_keeps_original_files_recoverable(tmp_path, monkeypatch):
    model = _model()
    model.save_pretrained(tmp_path)
    original_weights = (tmp_path / "model.safetensors").read_bytes()
    replace = Path.replace

    def fail_commit_and_rollback(path, target):
        if (
            path.suffix == ".tmp" and Path(target).name == "config.json"
        ) or path.suffix == ".backup":
            raise OSError("injected filesystem failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_commit_and_rollback)
    with pytest.raises(RuntimeError, match="rollback failed; preserved original files"):
        model.save_pretrained(tmp_path)
    backups = list(tmp_path.glob("*.backup"))
    assert len(backups) == 2
    assert original_weights in {path.read_bytes() for path in backups}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("uniform", [True, False])
def test_runtime_routing_and_capacity_survive_native_export(tmp_path, uniform):
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=32,
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        use_custom_kernels=False,
    )
    model = ComplexityModel(config).eval()
    original_config = config.to_dict()
    original_keys = set(model.state_dict())
    for index, layer in enumerate(model.layers):
        layer.mlp.set_top_k_primary_weight(0.5)
        layer.mlp.set_active_experts(4)
        layer.mlp.set_active_expert_width(3 if uniform or index == 0 else 5)
    ids = torch.tensor([[1, 5, 9, 13]])
    expected = model(ids)["logits"].detach()
    model.save_pretrained(tmp_path)
    native_config = json.loads((tmp_path / "config.json").read_text())
    assert ("layer_mlp_controls" in native_config) is (not uniform)
    loaded = ComplexityModel.from_pretrained(tmp_path).eval()
    assert torch.equal(loaded(ids)["logits"], expected)
    assert set(loaded.state_dict()) == original_keys
    assert config.to_dict() == original_config


@pytest.mark.parametrize(
    "invalid",
    [
        {"top_k_primary_weight": float("nan")},
        {"active_num_experts": 3},
        {"active_expert_width": 99},
        {"extra_control": 1},
    ],
)
def test_invalid_second_layer_controls_never_change_first_layer(invalid):
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=32,
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        use_custom_kernels=False,
    )
    model = ComplexityModel(config)
    before = [layer.mlp.training_telemetry() for layer in model.layers]
    control = {"top_k_primary_weight": 0.5, "active_num_experts": 4, "active_expert_width": 3}
    raw = {**model.export_config(), "layer_mlp_controls": [control, {**control, **invalid}]}
    with pytest.raises(ValueError, match="layer_mlp_controls"):
        model.apply_export_controls(raw)
    assert [layer.mlp.training_telemetry() for layer in model.layers] == before


@pytest.mark.parametrize("initial_safe", [True, False])
def test_switching_weight_format_reloads_new_weights(tmp_path, initial_safe):
    model = _model()
    model.save_pretrained(tmp_path, safe_serialization=initial_safe)
    with torch.no_grad():
        model.embed_tokens.weight.add_(0.25)
    model.save_pretrained(tmp_path, safe_serialization=not initial_safe)
    loaded = ComplexityModel.from_pretrained(tmp_path).eval()
    for key, expected in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected), key
    assert (tmp_path / "model.safetensors").exists() != (tmp_path / "pytorch_model.bin").exists()


def _export_worker(rank, directory, rendezvous, collective):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=25),
    )
    try:
        torch.manual_seed(733)
        model = _model()
        ids = torch.tensor([[1, 2, 3, 4]])
        expected = model(ids)["logits"].detach()
        if collective:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import Replicate, Shard, distribute_tensor

            mesh = init_device_mesh("cpu", (2,))
            raw = model.state_dict()
            distributed_state = {
                name: distribute_tensor(
                    value, mesh, [Shard(0) if value.ndim and value.shape[0] >= 2 else Replicate()]
                )
                for name, value in raw.items()
            }
            model.state_dict = lambda: distributed_state
            model.save_pretrained(directory)
        elif rank == 0:
            # Rank one deliberately does not enter the model export. The
            # external barrier belongs to the runner, not replicated weights.
            model.save_pretrained(directory)
        if rank == 0:
            loaded = ComplexityModel.from_pretrained(directory).eval()
            assert torch.equal(loaded(ids)["logits"], expected)
            (Path(directory) / "verified.json").write_text(json.dumps({"passed": True}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("collective", [False, True])
def test_distributed_export_scopes_and_exact_reload(tmp_path, collective):
    output = tmp_path / "export"
    mp.spawn(
        _export_worker,
        args=(str(output), str(tmp_path / "rendezvous"), collective),
        nprocs=2,
        join=True,
    )
    assert json.loads((output / "verified.json").read_text()) == {"passed": True}
