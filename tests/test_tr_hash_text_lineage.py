from __future__ import annotations

import json

import pytest
import torch

from complexity.training.lineage import (
    make_tr_hash_config,
    validate_lineage_plans,
)
from complexity.training.refinement import build_refinement_plan


def _pretrain_plan() -> dict:
    selections = {"dclm": [{"file": "tokens-00000.bin", "rows": 8}]}
    return {
        "format": "tr-hash-token-replay-plan-v1",
        "dataset": "hf://datasets/AETHORIA-AI/data-32k-200b-tokens",
        "revision": "main",
        "seq_len": 1024,
        "selection_mode": "manifest_order",
        "row_alignment": 1,
        "unique_tokens": 8192,
        "trained_tokens": 16384,
        "source_unique_tokens": {"dclm": 8192},
        "source_passes": {"dclm": 2},
        "phases": [
            {"name": "unique_core", "passes": 1, "sources": selections},
            {"name": "quality_replay_2", "passes": 1, "sources": selections},
        ],
    }


def test_model_preset_keeps_canonical_multi_hash_contract() -> None:
    config = make_tr_hash_config("complexity-small")

    assert config.vocab_size == 32000
    assert config.routing_strategy == "token_id_multi_hash"
    assert config.route_hash_count == 2
    assert config.top_k == 2
    assert config.top_k_primary_weight == 0.5
    assert config.shared_expert is True
    assert config.tie_word_embeddings is True


def test_refinement_builder_keeps_exact_unique_core_and_drops_replay() -> None:
    pretrain = _pretrain_plan()
    refinement = build_refinement_plan(pretrain)

    assert refinement["trained_tokens"] == refinement["unique_tokens"] == 8192
    assert refinement["source_passes"] == {"dclm": 1}
    assert [phase["name"] for phase in refinement["phases"]] == ["unique_core"]
    assert refinement["phases"][0]["sources"] == pretrain["phases"][0]["sources"]


def test_lineage_validator_accepts_derived_refinement(tmp_path) -> None:
    pretrain = _pretrain_plan()
    refinement = build_refinement_plan(pretrain)
    pretrain_path = tmp_path / "pretrain.json"
    refinement_path = tmp_path / "refinement.json"
    pretrain_path.write_text(json.dumps(pretrain), encoding="utf-8")
    refinement_path.write_text(json.dumps(refinement), encoding="utf-8")

    fingerprint = validate_lineage_plans(
        stage="refinement",
        tokenized_plan=str(refinement_path),
        pretrain_plan=str(pretrain_path),
    )

    assert fingerprint is not None and len(fingerprint) == 64


def test_lineage_validator_rejects_different_refinement_rows(tmp_path) -> None:
    pretrain = _pretrain_plan()
    refinement = build_refinement_plan(pretrain)
    refinement["phases"][0]["sources"]["dclm"][0]["rows"] = 7
    pretrain_path = tmp_path / "pretrain.json"
    refinement_path = tmp_path / "refinement.json"
    pretrain_path.write_text(json.dumps(pretrain), encoding="utf-8")
    refinement_path.write_text(json.dumps(refinement), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_lineage_plans(
            stage="refinement",
            tokenized_plan=str(refinement_path),
            pretrain_plan=str(pretrain_path),
        )


def test_refinement_plan_is_detached_and_rejects_inconsistent_core_counts():
    original = _pretrain_plan()
    derived = build_refinement_plan(original)
    derived["phases"][0]["sources"]["dclm"][0]["rows"] = 7
    assert original["phases"][0]["sources"]["dclm"][0]["rows"] == 8
    original["unique_tokens"] += 1024
    with pytest.raises(ValueError, match="exact unique_core coverage"):
        build_refinement_plan(original)


def test_parser_audits_schedule_before_tokenizer_model_or_dataset(tmp_path, monkeypatch):
    from complexity.training.lineage_runner import TRHashTextLineageRunner

    plan = _pretrain_plan()
    path = tmp_path / "pretrain.json"
    path.write_text(json.dumps(plan))
    runner = TRHashTextLineageRunner()
    monkeypatch.setenv("WORLD_SIZE", "4")
    arguments = [
        "--model-preset",
        "complexity-small",
        "--stage",
        "pretraining",
        "--tokenizer",
        str(tmp_path / "absent_tokenizer"),
        "--tokenized-data",
        str(tmp_path / "absent_corpus"),
        "--tokenized-plan",
        str(path),
        "--tokenized-revision",
        "main",
        "--batch-size",
        "2",
        "--gradient-accumulation",
        "1",
        "--seq-len",
        "1024",
    ]
    parsed = runner._build_parser().parse_args(arguments)
    assert parsed.max_steps == 2
    assert parsed.target_tokens == 16384
    with pytest.raises(ValueError, match="exceeds audited plan"):
        runner._build_parser().parse_args(arguments + ["--max-steps", "3"])
    assert not (tmp_path / "absent_corpus").exists()


def test_native_lineage_dataset_uses_exact_selected_rows(tmp_path):
    from types import SimpleNamespace

    from complexity.training.corpus_mixture import TextCorpusSource
    from complexity.training.corpus_preparation import TokenShardWriter, write_mixture_manifest
    from complexity.training.lineage_runner import TRHashTextLineageRunner

    source = TextCorpusSource("tiny", 1.0, data_files="unused.jsonl")
    root = tmp_path / "corpus"
    writer = TokenShardWriter(root / "corpora" / "tiny", seq_len=4, total_rows=5, rows_per_shard=2)
    writer.feed(torch.arange(21).numpy())
    writer.write_manifest(source=source)
    write_mixture_manifest(
        output_root=root,
        sources=(source,),
        seq_len=4,
        requested_tokens=20,
        actual_tokens=20,
        global_batch_sequences=1,
        rows_by_source={"tiny": 5},
    )
    plan = {
        "format": "tr-hash-token-replay-plan-v1",
        "seq_len": 4,
        "unique_tokens": 8,
        "trained_tokens": 8,
        "phases": [
            {
                "name": "unique_core",
                "passes": 1,
                "sources": {"tiny": [{"file": "tokens-00000.bin", "rows": 2}]},
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        stage="pretraining",
        model_preset="complexity-small",
        lr=3e-4,
        lr_scheduler="cosine",
        warmup_tokens=None,
        warmup_steps=0,
        weight_decay=0.1,
        batch_size=1,
        gradient_accumulation=1,
        seq_len=4,
        init_checkpoint=None,
        pretrain_plan=None,
        tokenized_plan=path,
        tokenized_data=root,
        tokenized_revision="pinned",
        max_steps=2,
        target_tokens=8,
        num_workers=0,
        tokenized_cache_dir=tmp_path / "cache",
        tokenized_cache_gb=0.01,
        tokenized_hf_token_env="HF_TOKEN",
        tokenized_prefetch_shards=0,
        resume=None,
    )
    dataset = TRHashTextLineageRunner().build_dataset(None, args, rank=0, world_size=1)
    samples = list(dataset)
    assert len(samples) == 2
    assert samples[-1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert samples[-1]["labels"].tolist() == [5, 6, 7, 8]


def test_native_lineage_loss_supports_tied_head_and_aligned_targets():
    from types import SimpleNamespace

    from torch.nn import functional as F

    from complexity import ComplexityModel, ModelConfig
    from complexity.training.lineage_runner import TRHashTextLineageRunner

    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        shared_intermediate_size=16,
        vocab_size=32,
        num_experts=2,
        top_k=2,
        max_position_embeddings=8,
        use_custom_kernels=False,
        tie_word_embeddings=True,
    )
    model = ComplexityModel(config).eval()
    batch = {"input_ids": torch.tensor([[2, 3, 4, 5]]), "labels": torch.tensor([[3, 4, 5, 0]])}
    compute = TRHashTextLineageRunner().build_compute_loss(
        SimpleNamespace(device=torch.device("cpu")), model
    )
    loss = compute(model, batch)
    expected = F.cross_entropy(
        model(batch["input_ids"])["logits"].reshape(-1, 32), batch["labels"].reshape(-1)
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.embed_tokens.weight.grad is not None


def _lineage_config(tmp_path, *, profile="complexity-small", world_size=1):
    (tmp_path / "plan.json").write_text(json.dumps(_pretrain_plan()))
    path = tmp_path / "lineage.json"
    data = {
        "format": "complexity-text-lineage-v1",
        "profile": profile,
        "stage": "pretraining",
        "world_size": world_size,
        "settings": {
            "tokenizer": "absent_tokenizer",
            "tokenized_data": "absent_corpus",
            "tokenized_plan": "plan.json",
            "tokenized_revision": "main",
            "checkpoint_dir": "outputs",
            "batch_size": 2,
            "seq_len": 1024,
            "gradient_accumulation": 1,
            "precision": "fp32",
            "use_custom_kernels": "false",
        },
    }
    path.write_text(json.dumps(data))
    return path, data


def test_lineage_json_preflight_is_read_only_and_preserves_floor_tail(tmp_path, monkeypatch):
    from complexity.training.lineage_runner import TRHashTextLineageRunner, execute_lineage_config

    path, _ = _lineage_config(tmp_path, world_size=3)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not start training, initialize a device or read data")

    monkeypatch.setattr(TRHashTextLineageRunner, "run", forbidden)
    monkeypatch.setattr(TRHashTextLineageRunner, "build_dataset", forbidden)
    monkeypatch.setattr(TRHashTextLineageRunner, "_load_tokenizer", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    before = {file.name: file.read_bytes() for file in tmp_path.iterdir()}
    report = execute_lineage_config(path)
    assert report["training_started"] is report["data_verified"] is False
    assert report["world_size"] == 3
    assert report["schedule"]["max_steps"] == 2
    assert report["schedule"]["scheduled_tokens"] == 12288
    assert report["schedule"]["unused_tail"] == 4096
    assert report["settings"]["tokenized_plan"] == str(tmp_path / "plan.json")
    assert report["launch_command"][-5:] == ["complexity", "train", "lineage", str(path), "--run"]
    assert report["launch_command"][report["launch_command"].index("--nproc_per_node") + 1] == "3"
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"batch_size": True}, "JSON integer"),
        ({"gradient_accumulation_steps": 2}, "unknown lineage settings"),
        ({"precision": "quarter"}, "precision must be one of"),
        ({"require_cuda": "false"}, "JSON boolean"),
        ({"tokenized_cache_gb": float("nan")}, "finite JSON number"),
        ({"save_steps": -1}, "save_steps must be nonnegative"),
        ({"warmup_steps": -1}, "warmup_steps must be nonnegative"),
    ],
)
def test_lineage_json_rejects_coercions_and_settings_typos(tmp_path, settings, message):
    from complexity.training.lineage_runner import execute_lineage_config

    path, data = _lineage_config(tmp_path)
    data["settings"].update(settings)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        execute_lineage_config(path)
    assert not (tmp_path / "outputs").exists()


def test_lineage_json_run_requires_matching_torchrun_geometry(tmp_path, monkeypatch):
    from complexity.training.lineage_runner import execute_lineage_config

    path, _ = _lineage_config(tmp_path, world_size=3)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match="launch world size 1 differs from config 3"):
        execute_lineage_config(path, run=True)


@pytest.mark.parametrize("profile", ["complexity-small", "tr-hash-200m"])
def test_lineage_cli_explicit_run_reuses_exact_reviewed_arguments(tmp_path, monkeypatch, profile):
    import sys

    from typer.testing import CliRunner

    from complexity.cli.app import app
    from complexity.training.runner import TrainRunner

    path, _ = _lineage_config(tmp_path, profile=profile)
    invoked = []
    original_argv = list(sys.argv)

    def capture(self):
        args = self._build_parser().parse_args()
        invoked.append(args)
        assert sys.argv == original_argv
        assert args.max_steps == 8
        assert args.target_tokens == 16384
        assert args.tokenized_data == str(tmp_path / "absent_corpus")

    monkeypatch.setattr(TrainRunner, "run", capture)
    runner = CliRunner()
    default = runner.invoke(app, ["train", "lineage", str(path)])
    assert default.exit_code == 0, default.output
    assert json.loads(default.stdout)["training_started"] is False
    assert not invoked
    result = runner.invoke(app, ["train", "lineage", str(path), "--run"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["training_started"] is True
    assert len(invoked) == 1
    if profile == "tr-hash-200m":
        assert report["model"]["hidden_size"] == 896
        assert report["model"]["shared_intermediate_size"] == 3072
        assert invoked[0].stage == "pretraining"
    assert not (tmp_path / "outputs").exists()


def test_lineage_cli_subprocess_json_is_clean_and_errors_are_actionable(tmp_path):
    import subprocess
    import sys

    path, data = _lineage_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "complexity", "train", "lineage", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["training_started"] is False
    data["settings"]["tokenized_revison"] = "misspelled"
    path.write_text(json.dumps(data))
    result = subprocess.run(
        [sys.executable, "-m", "complexity", "train", "lineage", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown lineage settings" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "outputs").exists()


def test_lineage_preflight_rejects_ambiguous_warmup_before_execution(tmp_path):
    from complexity.training.lineage_runner import execute_lineage_config

    path, data = _lineage_config(tmp_path)
    data["settings"].update(warmup_tokens=2048, warmup_steps=1)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mutually exclusive"):
        execute_lineage_config(path)


@pytest.mark.parametrize("profile", ["complexity-small", "tr-hash-200m"])
def test_lineage_preflight_rejects_remote_loader_workers(tmp_path, profile):
    from complexity.training.lineage_runner import execute_lineage_config

    path, data = _lineage_config(tmp_path, profile=profile)
    data["settings"].update(tokenized_data=_pretrain_plan()["dataset"], num_workers=2)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="remote pretokenized mixtures require"):
        execute_lineage_config(path)
