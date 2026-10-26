"""CLI acceptance checks: read-only inspection, useful failures and chat wiring."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from complexity.cli.app import app

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def test_native_export_cli_preserves_source_and_uses_tr_hash_adapter(tmp_path):
    import torch
    from safetensors.torch import load_file

    from complexity import ComplexityModel, ModelConfig

    source, output = tmp_path / "native", tmp_path / "bundle"
    config = ModelConfig.load(ROOT / "configs/examples/tr_gqa_cpu.json", strict=True)
    model = ComplexityModel(config)
    model.save_pretrained(source)
    before = _fingerprints(source)
    result = RUNNER.invoke(app, ["convert", "to-hf", str(source), str(output)])
    assert result.exit_code == 0, result.output
    assert _fingerprints(source) == before
    exported = json.loads((output / "config.json").read_text())
    assert exported["model_type"] == "tr_hash_moe"
    assert exported["num_experts_per_tok"] == config.top_k
    assert (output / "modeling_tr_hash_moe.py").is_file()
    weights = load_file(str(output / "model.safetensors"))
    for name, value in model.state_dict().items():
        assert torch.equal(weights[name], value)
    result = RUNNER.invoke(app, ["convert", "to-hf", str(source), str(output)])
    assert result.exit_code == 1
    assert "not empty" in result.output


def test_missing_native_export_input_does_not_create_output(tmp_path):
    output = tmp_path / "bundle"
    result = RUNNER.invoke(app, ["convert", "to-hf", str(tmp_path / "missing"), str(output)])
    assert result.exit_code == 1
    assert not output.exists()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fingerprints(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in path.rglob("*")
        if file.is_file()
    }


@pytest.fixture
def fsdp_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Use JSON and arbitrary bytes so inspection cannot rely on deserialization."""
    recipe = tmp_path / "recipe.json"
    _write_json(
        recipe,
        {
            "world_size": 2,
            "microbatch": 2,
            "context": 8,
            "max_updates": 100,
            "data_plan_sha256": "a" * 64,
        },
    )
    checkpoint = tmp_path / "step_0000010"
    distributed = checkpoint / "distributed"
    distributed.mkdir(parents=True)
    _write_json(
        checkpoint / "state.json",
        {
            "step": 10,
            "next_update": 10,
            "tokens_seen": 320,
            "world_size": 2,
            "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
            "data_plan_sha256": "a" * 64,
        },
    )
    (checkpoint / "config.json").write_bytes(
        (ROOT / "scripts/pretrain_1b_fsdp/model.json").read_bytes()
    )
    (distributed / ".metadata").write_bytes(b"not a pickle; inspection must never deserialize")
    (distributed / "__0_0.distcp").write_bytes(b"synthetic shard bytes")
    for rank in range(2):
        (checkpoint / f"rng_rank{rank}.pt").write_bytes(b"synthetic RNG bytes")
    files = {
        str(file.relative_to(checkpoint)): {
            "bytes": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        for file in checkpoint.rglob("*")
        if file.is_file()
    }
    _write_json(checkpoint / "complete.json", {"step": 10, "files": files})
    return checkpoint, recipe


def test_module_help_and_version_need_no_torch() -> None:
    code = """
import sys
from typer.testing import CliRunner
from complexity.cli.app import app
runner = CliRunner()
for args in (["--version"], ["--help"], ["vision", "list"], ["checkpoint", "--help"]):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (args, result.output, result.exception)
assert "torch" not in sys.modules
print("metadata-help-ok")
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "metadata-help-ok"
    module = subprocess.run(
        [sys.executable, "-m", "complexity", "--version"], cwd=ROOT, text=True, capture_output=True
    )
    assert module.returncode == 0, module.stderr
    assert module.stdout.startswith("complexity ")


def test_real_pinned_recipe_inspection_is_read_only_and_clean_json() -> None:
    recipe_dir = ROOT / "scripts/pretrain_1b_fsdp"
    before = _fingerprints(recipe_dir)
    result = subprocess.run(
        [sys.executable, "-m", "complexity", "train", "inspect", str(recipe_dir), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["tokens_per_update"] == 131072
    assert report["max_updates"] == 534057
    assert report["scheduled_tokens"] == 69_999_919_104
    assert report["training_started"] is False
    assert report["assets"]["model.json"]["sha256_matches"] is True
    assert _fingerprints(recipe_dir) == before


@pytest.mark.parametrize(
    "value", [None, [], "not a mapping", {"model": None}, {"run": {"steps": 10}}]
)
def test_invalid_document_shapes_are_actionable(tmp_path: Path, value: object) -> None:
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(value))
    result = RUNNER.invoke(app, ["train", "validate", str(config), "--json"])
    assert result.exit_code == 1
    assert "error" in json.loads(result.stderr)
    assert "Traceback" not in result.output


def test_cli_strict_validation_rejects_misspelled_model_fields(tmp_path: Path) -> None:
    config = tmp_path / "typo.json"
    data = json.loads((ROOT / "scripts/pretrain_1b_fsdp/model.json").read_text())
    data["hidden_sze"] = data.pop("hidden_size")
    _write_json(config, data)
    result = RUNNER.invoke(app, ["train", "validate", str(config), "--json"])
    assert result.exit_code == 1
    # Missing architecture keys must not silently become defaults.
    assert "hidden_sze" in result.stderr or "ModelConfig" in result.stderr


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_create_validate_dry_run_and_overwrite_protection(tmp_path: Path, suffix: str) -> None:
    config = tmp_path / f"experiment{suffix}"
    result = RUNNER.invoke(app, ["train", "create-config", str(config), "--size", "tiny"])
    assert result.exit_code == 0, result.output
    before = config.read_bytes()
    validated = RUNNER.invoke(app, ["train", "validate", str(config), "--json"])
    assert validated.exit_code == 0, validated.output
    report = json.loads(validated.stdout)
    assert report["model"]["num_hidden_layers"] == 6
    assert report["training"]["precision"] == "bf16"
    assert report["training_started"] is False
    assert RUNNER.invoke(app, ["train", "run", str(config), "--dry-run"]).exit_code == 0
    assert RUNNER.invoke(app, ["train", "run", str(config)]).exit_code == 2
    overwrite = RUNNER.invoke(app, ["train", "create-config", str(config)])
    assert overwrite.exit_code == 1
    assert config.read_bytes() == before


def test_old_cli_template_fields_still_validate_explicitly(tmp_path: Path) -> None:
    config = tmp_path / "legacy.json"
    _write_json(
        config,
        {
            "model": {
                "name": "complexity-tiny",
                "hidden_size": 256,
                "num_layers": 6,
                "num_heads": 4,
                "num_kv_heads": 1,
                "ff_multiplier": 4,
                "max_seq_length": 2048,
                "use_sliding_window": False,
            },
            "training": {"mixed_precision": "bf16", "gradient_clip": 1.0},
        },
    )
    result = RUNNER.invoke(app, ["train", "validate", str(config), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["model"]["intermediate_size"] == 1024
    assert any("num_layers" in item for item in report["warnings"])


@pytest.mark.parametrize("value", [0, "32", True])
def test_training_batch_invalid_types_and_ranges_fail(tmp_path: Path, value: object) -> None:
    config = tmp_path / "training.json"
    _write_json(config, {"model": {"hidden_size": 768}, "training": {"batch_size": value}})
    result = RUNNER.invoke(app, ["train", "validate", str(config), "--json"])
    assert result.exit_code == 1
    assert "batch_size" in result.stderr


def test_checkpoint_verification_checks_manifest_without_deserializing(
    fsdp_checkpoint: tuple[Path, Path],
) -> None:
    path, recipe = fsdp_checkpoint
    before = _fingerprints(path)
    result = RUNNER.invoke(
        app, ["checkpoint", "inspect", str(path), "--verify", "--recipe", str(recipe), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["sha256_verified"] is True
    assert report["recipe_matches"] is True
    assert report["checkpoint_restored"] is False
    assert _fingerprints(path) == before


def test_same_size_corruption_requires_hash_verification(
    fsdp_checkpoint: tuple[Path, Path],
) -> None:
    path, _ = fsdp_checkpoint
    shard = path / "distributed/__0_0.distcp"
    shard.write_bytes(b"x" * shard.stat().st_size)
    sizes = RUNNER.invoke(app, ["checkpoint", "inspect", str(path), "--json"])
    assert sizes.exit_code == 0, sizes.output
    assert json.loads(sizes.stdout)["sha256_verified"] is False
    hashes = RUNNER.invoke(app, ["checkpoint", "inspect", str(path), "--verify", "--json"])
    assert hashes.exit_code == 1
    assert "Checksum mismatch" in json.loads(hashes.stderr)["error"]


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "distributed/../../outside", "distributed\\outside"]
)
def test_checkpoint_manifest_rejects_unsafe_paths(
    fsdp_checkpoint: tuple[Path, Path], name: str
) -> None:
    path, _ = fsdp_checkpoint
    manifest = json.loads((path / "complete.json").read_text())
    manifest["files"][name] = {"bytes": 0, "sha256": "a" * 64}
    _write_json(path / "complete.json", manifest)
    result = RUNNER.invoke(app, ["checkpoint", "inspect", str(path), "--json"])
    assert result.exit_code == 1
    assert "Unsafe" in json.loads(result.stderr)["error"]


def test_checkpoint_symlink_escape_is_rejected(
    fsdp_checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
    path, _ = fsdp_checkpoint
    outside = tmp_path / "outside"
    outside.write_bytes(b"synthetic shard bytes")
    shard = path / "distributed/__0_0.distcp"
    shard.unlink()
    shard.symlink_to(outside)
    result = RUNNER.invoke(app, ["checkpoint", "inspect", str(path), "--json"])
    assert result.exit_code == 1
    assert "escapes" in json.loads(result.stderr)["error"]


def test_checkpoint_recipe_mismatch_is_rejected(fsdp_checkpoint: tuple[Path, Path]) -> None:
    path, recipe = fsdp_checkpoint
    recipe.write_bytes(recipe.read_bytes() + b"\n")
    result = RUNNER.invoke(
        app, ["checkpoint", "inspect", str(path), "--recipe", str(recipe), "--json"]
    )
    assert result.exit_code == 1
    assert "recipe_sha256" in json.loads(result.stderr)["error"]


def test_chat_preserves_history_and_clear_without_native_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("complexity.inference")
    calls = []
    options = {}

    class FakeServer:
        def chat(self, messages, config):
            calls.append(list(messages))
            return "server answer"

    def create(backend, **kwargs):
        options.update(backend=backend, **kwargs)
        return FakeServer()

    monkeypatch.setattr(api, "create_external_backend", create)
    result = RUNNER.invoke(
        app,
        [
            "inference",
            "chat",
            "org/my-model",
            "--backend",
            "sglang",
            "--base-url",
            "http://localhost:30000",
            "--system",
            "help",
        ],
        input="first\nsecond\nclear\nthird\nexit\n",
    )
    assert result.exit_code == 0, result.output
    assert options["model"] == "org/my-model"
    assert options["backend"] == "sglang"
    assert calls[1] == [
        {"role": "system", "content": "help"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "server answer"},
        {"role": "user", "content": "second"},
    ]
    assert calls[2] == [{"role": "system", "content": "help"}, {"role": "user", "content": "third"}]


def test_chat_legacy_local_options_have_migration_guidance() -> None:
    result = RUNNER.invoke(app, ["inference", "chat", "model.pt", "--device", "cuda"])
    assert result.exit_code == 2
    assert "server-side" in result.output


def test_batch_publishes_jsonl_and_preserves_existing_output_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = importlib.import_module("complexity.inference")
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"
    input_file.write_text('{"prompt": "one"}\n{"text": "deux é"}\n')
    calls = []

    class FakeServer:
        fail = False

        def complete(self, prompt, config):
            calls.append(prompt)
            if self.fail:
                raise RuntimeError("endpoint stopped")
            return prompt.upper()

    server = FakeServer()
    monkeypatch.setattr(api, "create_external_backend", lambda *args, **kwargs: server)
    args = ["inference", "batch", "my-model", str(input_file), str(output_file)]
    result = RUNNER.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert [json.loads(line) for line in output_file.read_text().splitlines()] == [
        {"prompt": "one", "output": "ONE"},
        {"prompt": "deux é", "output": "DEUX É"},
    ]
    old = output_file.read_bytes()
    count = len(calls)
    assert RUNNER.invoke(app, args).exit_code == 1
    assert len(calls) == count  # Reject overwrite before contacting the endpoint.
    server.fail = True
    failed = RUNNER.invoke(app, args + ["--force"])
    assert failed.exit_code == 1
    assert output_file.read_bytes() == old
    assert not list(tmp_path.glob(".complexity-batch-*"))


def test_batch_rejects_invalid_input_before_contacting_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = importlib.import_module("complexity.inference")

    def forbidden(*args, **kwargs):
        raise AssertionError("No endpoint should be created for invalid input")

    monkeypatch.setattr(api, "create_external_backend", forbidden)
    input_file = tmp_path / "invalid.jsonl"
    input_file.write_text('{"prompt": "valid"}\n{"prompt": 123}\n')
    result = RUNNER.invoke(
        app, ["inference", "batch", "m", str(input_file), str(tmp_path / "output.jsonl")]
    )
    assert result.exit_code == 1
    assert "line 2" in result.output
    assert not (tmp_path / "output.jsonl").exists()


def test_endpoint_benchmark_counts_failures_and_sends_exact_request_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("complexity.inference")
    from itertools import count

    calls = count()

    class FakeServer:
        def complete(self, prompt, config):
            if next(calls) == 1:
                raise RuntimeError("one request failed")
            return "ok"

    monkeypatch.setattr(api, "create_external_backend", lambda *args, **kwargs: FakeServer())
    result = RUNNER.invoke(
        app,
        [
            "serve",
            "benchmark-server",
            "--model",
            "m",
            "--requests",
            "3",
            "--concurrency",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["successful_requests"] == 2
    assert report["failed_requests"] == 1
    assert next(calls) == 3  # No undisclosed warmup traffic.
    assert report["token_throughput_measured"] is False
    assert report["mean_latency_seconds"] >= 0


@pytest.mark.parametrize(
    "args",
    [
        ["serve", "start", "model.pt"],
        ["serve", "openai-proxy", "model.pt"],
        ["inference", "benchmark", "model.pt"],
        ["profile", "throughput", "model.pt"],
    ],
)
def test_removed_native_execution_commands_explain_migration(args: list[str]) -> None:
    result = RUNNER.invoke(app, args)
    assert result.exit_code == 2
    assert "not supported" in result.output
    assert "Missing dependency" not in result.output


def test_model_info_never_opens_pickle_weights(tmp_path: Path) -> None:
    import pickle

    marker = tmp_path / "must-not-exist"

    class Unsafe:
        def __reduce__(self):
            return (eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",))

    model = tmp_path / "model.pt"
    model.write_bytes(pickle.dumps(Unsafe()))
    result = RUNNER.invoke(app, ["info", "model", str(model), "--detailed", "--json"])
    assert result.exit_code == 0, result.output
    assert not marker.exists()
    report = json.loads(result.stdout)
    assert report["weights_loaded"] is False
    assert report["tensor_elements_from_safetensors_headers"] is None


def test_model_info_uses_safetensors_header_shapes(tmp_path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    model = tmp_path / "model.safetensors"
    save_file(
        {"weight": np.zeros((3, 4), dtype=np.float32), "buffer": np.zeros(2, dtype=np.int64)},
        str(model),
    )
    result = RUNNER.invoke(app, ["info", "model", str(model), "--detailed", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["tensor_elements_from_safetensors_headers"] == 14
    assert report["weights_loaded"] is False
    assert {item["name"] for item in report["tensor_headers"]} == {"weight", "buffer"}


def test_model_info_routes_fsdp_checkpoints_to_read_only_inspection(
    fsdp_checkpoint: tuple[Path, Path],
) -> None:
    path, recipe = fsdp_checkpoint
    result = RUNNER.invoke(
        app, ["info", "model", str(path), "--verify", "--recipe", str(recipe), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["format"] == "tr_hash_fsdp_checkpoint"
    assert report["checkpoint_restored"] is False


def test_meta_parameter_profile_matches_cpu_model_counts(tmp_path: Path) -> None:
    from complexity import ComplexityModel, ModelConfig

    config = ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=48,
        shared_intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=32,
        num_experts=4,
        top_k=2,
        tie_word_embeddings=True,
        mlp_type="tr_hash_engine",
        routing_strategy="token_id_composed_modulo",
    )
    path = tmp_path / "config.json"
    config.save(path)
    cpu_model = ComplexityModel(config)
    expected = sum(parameter.numel() for parameter in cpu_model.parameters())
    result = RUNNER.invoke(app, ["profile", "model", str(path), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["parameters"] == expected
    assert sum(report["components"].values()) == expected
    assert report["weights_allocated"] is False
    assert report["execution_measured"] is False
    assert report["memory_measured"] is False
