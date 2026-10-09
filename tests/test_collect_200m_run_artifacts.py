from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from complexity.export import provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_collects_model_only_release_and_provenance(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = checkpoint_root / "step_000010"
    checkpoint_dir.mkdir(parents=True)
    tied = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    torch.save(
        {
            "step": 10,
            "model": {"embed.weight": tied, "lm_head.weight": tied},
            "optimizer": {"state": {}},
            "config": {"hidden_size": 4},
        },
        checkpoint_path,
    )
    (checkpoint_root / "latest").write_text("step_000010\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    _write_json(run_dir / "run_config.json", {"schema_version": 1})
    (run_dir / "metrics.csv").write_text("step,eval_loss\n10,1.0\n", encoding="utf-8")

    data_root = tmp_path / "data"
    _write_json(data_root / "dataset_manifest.json", {"schema_version": 1})
    for partition in ("train", "eval"):
        _write_json(
            data_root / partition / "tokens.idx.json",
            {"partition": partition, "sha256": partition},
        )

    tokenizer = tmp_path / "tokenizer"
    _write_json(tokenizer / "tokenizer.json", {"version": "1.0"})
    log_path = tmp_path / "training.log"
    log_path.write_text("finished\n", encoding="utf-8")
    output_dir = tmp_path / "release"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "complexity.export.provenance",
            "--run-name",
            "test-run",
            "--checkpoint-root",
            str(checkpoint_root / "latest"),
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(data_root),
            "--tokenizer",
            str(tokenizer),
            "--output-dir",
            str(output_dir),
            "--log",
            str(log_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    weights = load_file(output_dir / "model.safetensors")
    assert torch.equal(weights["embed.weight"], tied)
    assert torch.equal(weights["lm_head.weight"], tied)

    model_config = json.loads((output_dir / "model_config.json").read_text())
    assert model_config["checkpoint_step"] == 10
    assert model_config["model"]["hidden_size"] == 4

    artifact_manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
    assert (
        artifact_manifest["checkpoint"]["sha256"]
        == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    )
    released_paths = {entry["path"] for entry in artifact_manifest["files"]}
    assert "model.safetensors" in released_paths
    assert "metrics.csv" in released_paths
    assert "dataset/train/tokens.idx.json" in released_paths
    assert artifact_manifest["model_only"] is True
    assert artifact_manifest["resume_supported"] is False
    assert artifact_manifest["checkpoint"]["included_in_release"] is False
    assert not any(path.suffix == ".pt" for path in output_dir.rglob("*"))
    assert set(weights) == {"embed.weight", "lm_head.weight"}
    for entry in artifact_manifest["files"]:
        path = output_dir / entry["path"]
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def collection_inputs(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        {
            "model": {"weight": torch.arange(4, dtype=torch.float32)},
            "step": 4,
            "config": {"hidden_size": 4},
        },
        checkpoint / "checkpoint.pt",
    )
    run = tmp_path / "run"
    _write_json(run / "run_config.json", {"run": "fixture"})
    (run / "metrics.csv").write_text("step,loss\n4,1\n")
    data = tmp_path / "data"
    for name in provenance.DEFAULT_DATA_FILES:
        _write_json(data / name, {"fixture": name})
    tokenizer = tmp_path / "tokenizer"
    _write_json(tokenizer / "tokenizer.json", {"fixture": "tokenizer"})
    return {
        "run_name": "fixture",
        "checkpoint_root": checkpoint,
        "run_dir": run,
        "data_root": data,
        "tokenizer": tokenizer,
        "output_dir": tmp_path / "output",
    }


def test_collection_does_not_overwrite_existing_output(collection_inputs) -> None:
    output = collection_inputs["output_dir"]
    output.mkdir()
    retained = output / "retained.txt"
    retained.write_text("keep")
    with pytest.raises(FileExistsError):
        provenance.collect_run_artifacts(**collection_inputs)
    assert retained.read_text() == "keep"


def test_missing_metadata_leaves_no_partial_bundle(collection_inputs) -> None:
    (collection_inputs["run_dir"] / "metrics.csv").unlink()
    with pytest.raises(FileNotFoundError):
        provenance.collect_run_artifacts(**collection_inputs)
    assert not collection_inputs["output_dir"].exists()


def test_collection_copies_only_known_tokenizer_assets(collection_inputs) -> None:
    (collection_inputs["tokenizer"] / "hf_token.txt").write_text("fixture-secret")
    provenance.collect_run_artifacts(**collection_inputs)
    assert {path.name for path in (collection_inputs["output_dir"] / "tokenizer").iterdir()} == {
        "tokenizer.json"
    }


@pytest.mark.parametrize("relative", ["../outside.json", "/outside.json", "checkpoint.pt"])
def test_collection_rejects_escaped_or_binary_metadata(collection_inputs, relative) -> None:
    with pytest.raises(ValueError):
        provenance.collect_run_artifacts(**collection_inputs, run_files=[relative])
    assert not collection_inputs["output_dir"].exists()


def test_collection_rejects_escaped_tokenizer_symlink(collection_inputs, tmp_path) -> None:
    asset = collection_inputs["tokenizer"] / "tokenizer.json"
    asset.unlink()
    _write_json(tmp_path / "outside.json", {"fixture": "outside"})
    asset.symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escapes"):
        provenance.collect_run_artifacts(**collection_inputs)
    assert not collection_inputs["output_dir"].exists()


def test_collection_rejects_escaped_latest_pointer(collection_inputs) -> None:
    root = collection_inputs["checkpoint_root"]
    latest = root / "latest"
    latest.write_text("../checkpoint\n")
    collection_inputs["checkpoint_root"] = latest
    with pytest.raises(ValueError, match="inside its root"):
        provenance.collect_run_artifacts(**collection_inputs)
    assert not collection_inputs["output_dir"].exists()


def test_collection_does_not_fall_back_to_unrestricted_pickle(collection_inputs) -> None:
    torch.save(
        {"model": {"weight": torch.ones(1)}, "config": {"hidden_size": 1}, "extra": object()},
        collection_inputs["checkpoint_root"] / "checkpoint.pt",
    )
    with pytest.raises(ValueError, match="restricted weights-only"):
        provenance.collect_run_artifacts(**collection_inputs)
    assert not collection_inputs["output_dir"].exists()


def test_native_collection_preserves_safetensors_bytes(collection_inputs) -> None:
    root = collection_inputs["checkpoint_root"]
    (root / "checkpoint.pt").unlink()
    _write_json(root / "config.json", {"hidden_size": 4})
    weights = root / "model.safetensors"
    save_file(
        {"weight": torch.arange(4, dtype=torch.float32)}, weights, metadata={"fixture": "keep"}
    )
    original = weights.read_bytes()
    manifest_path = provenance.collect_run_artifacts(**collection_inputs)
    assert (collection_inputs["output_dir"] / "model.safetensors").read_bytes() == original
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_format"] == "native-weights"
    assert manifest["resume_supported"] is False


def test_changed_source_is_not_published(collection_inputs, monkeypatch) -> None:
    source_checkpoint = collection_inputs["checkpoint_root"] / "checkpoint.pt"
    copy = provenance.shutil.copy2

    def change_after_copy(source, destination):
        result = copy(source, destination)
        with source_checkpoint.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(provenance.shutil, "copy2", change_after_copy)
    with pytest.raises(RuntimeError, match="changed during collection"):
        provenance.collect_run_artifacts(**collection_inputs)
    assert not collection_inputs["output_dir"].exists()


def test_checkpoint_collect_cli_resolves_config_paths(collection_inputs, tmp_path) -> None:
    config = tmp_path / "collection.json"
    config.write_text(
        json.dumps(
            {
                "format": "complexity-artifact-collection-v1",
                **{
                    key: value.name if isinstance(value, Path) else value
                    for key, value in collection_inputs.items()
                },
            }
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "complexity", "checkpoint", "collect", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (collection_inputs["output_dir"] / "artifact_manifest.json").is_file()


def test_collection_metadata_layout_is_not_fixed_to_200m(collection_inputs) -> None:
    _write_json(collection_inputs["run_dir"] / "experiment.json", {"experiment": "fixture"})
    _write_json(collection_inputs["data_root"] / "train" / "sft.idx.json", {"format": "fixture"})
    provenance.collect_run_artifacts(
        **collection_inputs,
        run_files=["experiment.json"],
        data_files=["train/sft.idx.json"],
    )
    output = collection_inputs["output_dir"]
    assert (output / "experiment.json").is_file()
    assert (output / "dataset" / "train" / "sft.idx.json").is_file()


def test_native_sft_identity_checkpoint_collects_reloadable_weights(collection_inputs) -> None:
    from complexity import ComplexityModel, ModelConfig
    from complexity.training.sft import _save_run_checkpoint

    config = ModelConfig.load(REPO_ROOT / "configs/examples/tr_gqa_cpu.json", strict=True)
    model = ComplexityModel(config).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    root = collection_inputs["checkpoint_root"]
    _save_run_checkpoint(
        root,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=4,
        identity={"plan": {"model_config": config.to_dict()}},
        device=torch.device("cpu"),
        history=[],
    )
    collection_inputs["checkpoint_root"] = root / "latest"
    provenance.collect_run_artifacts(**collection_inputs)
    restored = ComplexityModel.from_pretrained(collection_inputs["output_dir"]).eval()
    tokens = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        assert torch.equal(model(tokens)["logits"], restored(tokens)["logits"])
    assert (
        json.loads((collection_inputs["output_dir"] / "config.json").read_text())
        == config.to_dict()
    )
    assert not list(collection_inputs["output_dir"].rglob("*.pt"))


@pytest.mark.parametrize("uniform", [True, False])
def test_generic_checkpoint_collection_preserves_live_layer_controls(collection_inputs, uniform):
    from complexity import ComplexityModel, ModelConfig
    from complexity.utils.checkpointing import CheckpointManager

    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
        use_custom_kernels=False,
    )
    model = ComplexityModel(config).eval()
    for index, layer in enumerate(model.layers):
        layer.mlp.set_top_k_primary_weight(0.9 if uniform or index == 0 else 0.7)
        layer.mlp.set_active_experts(2 if uniform or index == 0 else 4)
        layer.mlp.set_active_expert_width(3 if uniform or index == 0 else 5)
    manager = CheckpointManager(collection_inputs["checkpoint_root"], model)
    collection_inputs["checkpoint_root"] = manager.save(4)
    provenance.collect_run_artifacts(**collection_inputs)
    restored = ComplexityModel.from_pretrained(collection_inputs["output_dir"]).eval()
    tokens = torch.tensor([[1, 5, 9, 13]])
    with torch.no_grad():
        assert torch.equal(model(tokens)["logits"], restored(tokens)["logits"])
    assert (
        json.loads((collection_inputs["output_dir"] / "config.json").read_text())
        == model.export_config()
    )
    assert not list(collection_inputs["output_dir"].rglob("*.pt"))


def test_collection_rejects_invalid_runtime_controls_before_publication(collection_inputs):
    path = collection_inputs["checkpoint_root"] / "checkpoint.pt"
    state = torch.load(path, weights_only=True)
    state["runtime_mlp_controls"] = {"format": "unknown", "model_config": {"hidden_size": 4}}
    torch.save(state, path)
    with pytest.raises(ValueError, match="runtime MLP controls"):
        provenance.collect_run_artifacts(**collection_inputs)
    assert not collection_inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "identity",
    [
        {"plan": {"model_config": []}},
        {"plan": "invalid"},
        "invalid",
        {"plan": {"model_config": {}}},
    ],
)
def test_sft_config_fallback_rejects_malformed_identity(collection_inputs, identity) -> None:
    torch.save(
        {"model": {"weight": torch.ones(1)}, "identity": identity},
        collection_inputs["checkpoint_root"] / "checkpoint.pt",
    )
    with pytest.raises(ValueError, match="model configuration"):
        provenance.collect_run_artifacts(**collection_inputs)
    assert not collection_inputs["output_dir"].exists()
