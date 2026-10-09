from __future__ import annotations

import inspect
import json
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from complexity.models import ComplexityModel
from complexity.training import (
    SupervisorProgram,
    WeightedStreamingTextDataset,
    resolve_warmup_steps,
    text_pretraining,
)
from complexity.training.packing import resolve_token_pack_schedule
from complexity.training.runner import log_routing_diagnostic, routing_diagnostic

SCRIPT = Path(text_pretraining.__file__)


def _profile():
    return text_pretraining


def _replay_command(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"trained_tokens": 129_995_112_448}))
    command = text_pretraining.build_replay_command(
        plan,
        tokenizer="tokenizer",
        data="hf://datasets/Pacific-i64/data-32k-200b-tokens",
        revision="pinned",
        output="artifacts/run",
        cache_dir="cache",
    )
    options = {
        command[index][2:]: command[index + 1]
        for index in range(len(command) - 1)
        if command[index].startswith("--") and not command[index + 1].startswith("--")
    }
    return plan, command, options


def test_true_200b_token_budget_is_not_compressed() -> None:
    schedule = resolve_token_pack_schedule(
        target_tokens=200_000_000_000,
        tokens_per_step=8 * 8 * 8 * 1024,
        token_packs=40,
    )

    assert schedule.actual_tokens >= 200_000_000_000
    assert schedule.actual_tokens - 200_000_000_000 < schedule.tokens_per_step
    assert len(schedule.boundaries) == 40
    assert max(schedule.pack_step_counts) - min(schedule.pack_step_counts) <= 1


def test_one_billion_token_warmup_scales_with_global_batch() -> None:
    tokens_per_step = 8 * 8 * 8 * 1024
    assert (
        resolve_warmup_steps(
            max_steps=381_470,
            tokens_per_step=tokens_per_step,
            warmup_steps=None,
            warmup_tokens=1_000_000_000,
        )
        == 1908
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_warmup_steps(
            max_steps=100,
            tokens_per_step=tokens_per_step,
            warmup_steps=10,
            warmup_tokens=1_000_000,
        )


def test_200m_profile_keeps_dense_shared_swiglu_and_narrow_experts() -> None:
    module = _profile()
    config = module.make_200m_config()
    with torch.device("meta"):
        parameter_count = ComplexityModel(config).num_parameters()

    assert 195_000_000 <= parameter_count <= 205_000_000
    assert config.hidden_size == 896
    assert config.num_hidden_layers == 16
    assert config.mlp_type == "tr_hash_engine"
    assert config.shared_expert is True
    assert config.num_experts == 4
    assert config.top_k == 2
    assert config.routing_strategy == "token_id_multi_hash"
    assert config.route_hash_count == 2
    assert config.shared_intermediate_size == 3072
    assert config.intermediate_size // config.num_experts == 64
    assert config.shared_intermediate_size > config.intermediate_size // config.num_experts

    defaults = (
        module.build_200m_runner()
        ._build_parser()
        .parse_args(["--stack-edu-data", "stack/*.jsonl.gz"])
    )
    assert defaults.target_tokens == 200_000_000_000
    assert defaults.token_packs == 40
    assert defaults.seq_len == 1024
    assert defaults.gradient_accumulation == 8
    assert defaults.gradient_checkpointing is False
    assert defaults.distributed_mode == "ddp"
    assert defaults.tokenized_cache_gb == 32.0
    assert defaults.tokenized_prefetch_shards == 1
    assert defaults.tokenized_revision == "main"


def test_200m_routing_banner_exposes_multi_hash_without_forcing_other_profiles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _profile()
    config = module.make_200m_config()

    assert routing_diagnostic(config) == {
        "strategy": "token_id_multi_hash",
        "multi_hash": True,
        "configured_hashes": 2,
        "effective_hashes": 2,
    }
    with caplog.at_level(logging.INFO, logger="complexity.training.runner"):
        log_routing_diagnostic(config)
    assert (
        "TR-Hash routing: strategy=token_id_multi_hash multi_hash=true "
        "effective_hashes=2 configured_route_hash_count=2 top_k=2"
    ) in caplog.messages

    config.routing_strategy = "token_id_balanced_hash"
    assert routing_diagnostic(config) == {
        "strategy": "token_id_balanced_hash",
        "multi_hash": False,
        "configured_hashes": 2,
        "effective_hashes": 1,
    }


def test_pretokenized_schedule_cannot_silently_repeat_after_shard_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _profile()
    dataset = SimpleNamespace(seq_len=1024, trained_tokens=1024)
    monkeypatch.setattr(
        module,
        "PretokenizedCorpusMixtureDataset",
        lambda *_args, **_kwargs: dataset,
    )
    args = SimpleNamespace(
        tokenized_data="hf://datasets/owner/repo",
        tokenized_cache_dir="cache",
        tokenized_cache_gb=32.0,
        tokenized_revision="main",
        tokenized_hf_token_env="HF_TOKEN",
        tokenized_prefetch_shards=1,
        tokenized_plan="plan.json",
        seq_len=1024,
        batch_size=1,
        gradient_accumulation=1,
        max_steps=2,
        target_tokens=2048,
    )

    with pytest.raises(ValueError, match="automatic dataset repetition is forbidden"):
        module.build_200m_runner().build_dataset(None, args, rank=0, world_size=1)


def test_pretokenized_schedule_may_underuse_coverage_that_does_not_divide_evenly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: a replay plan's trained_tokens is fixed by its
    corpus budgets, not by any particular GPU count. world_size=10 hit this
    live — 129,995,112,448 tokens has no factor of 5, so no batch/accum
    choice divides it evenly at world_size=10, and the old strict '!='
    check made every world_size but the one the plan happened to be built
    for unusable. Falling a little short of full coverage is fine; only
    exceeding it (repetition) is forbidden."""
    module = _profile()
    dataset = SimpleNamespace(seq_len=1024, trained_tokens=129_995_112_448)
    monkeypatch.setattr(
        module,
        "PretokenizedCorpusMixtureDataset",
        lambda *_args, **_kwargs: dataset,
    )
    args = SimpleNamespace(
        tokenized_data="hf://datasets/owner/repo",
        tokenized_cache_dir="cache",
        tokenized_cache_gb=32.0,
        tokenized_revision="main",
        tokenized_hf_token_env="HF_TOKEN",
        tokenized_prefetch_shards=1,
        tokenized_plan="plan.json",
        seq_len=1024,
        batch_size=8,
        gradient_accumulation=8,
        max_steps=None,
        target_tokens=129_995_112_448,
    )

    # world_size=10 -> tokens_per_step=655,360, which does not divide
    # 129,995,112,448 evenly. This must no longer raise.
    result = module.build_200m_runner().build_dataset(None, args, rank=0, world_size=10)
    assert result is dataset


def test_200b_corpus_budget_and_distribution_are_exact() -> None:
    module = _profile()
    sources = module.corpus_sources("stack/*.jsonl.gz")
    budgets = module.CORPUS_TOKEN_BUDGETS

    assert sum(budgets.values()) == 200_000_000_000
    assert budgets == {
        "dclm": 90_000_000_000,
        "fineweb_edu_dedup": 60_000_000_000,
        "stack_edu": 20_000_000_000,
        "finemath": 10_000_000_000,
        "infiwebmath": 10_000_000_000,
        "cosmopedia_v2": 10_000_000_000,
    }
    assert {source.name: source.weight for source in sources} == {
        name: tokens / 200_000_000_000 for name, tokens in budgets.items()
    }
    assert sum(source.weight for source in sources) == pytest.approx(1.0)


def test_weighted_scheduler_realizes_exact_distribution_every_20_chunks() -> None:
    module = _profile()
    sources = module.corpus_sources("stack.jsonl")
    counts = Counter({source.name: 0 for source in sources})

    for _ in range(20):
        source = WeightedStreamingTextDataset._next_source(sources, counts)
        counts[source.name] += 1

    assert counts == Counter(
        dclm=9,
        fineweb_edu_dedup=6,
        stack_edu=2,
        finemath=1,
        infiwebmath=1,
        cosmopedia_v2=1,
    )


def test_stack_edu_materialization_is_mandatory() -> None:
    module = _profile()
    with pytest.raises(ValueError, match="SWHIDs"):
        module.corpus_sources("")


def test_package_replay_command_streams_without_worker_cache_thrash(tmp_path) -> None:
    plan, command, options = _replay_command(tmp_path)
    planned = json.loads(plan.read_text())["trained_tokens"]
    scheduled = planned // (8 * 8 * 8 * 1024) * (8 * 8 * 8 * 1024)
    assert int(options["target-tokens"]) == scheduled
    assert int(options["max-steps"]) * (8 * 8 * 8 * 1024) == scheduled <= planned
    assert options["token-packs"] == "40"
    assert options["nproc_per_node"] == "8"
    assert options["gradient-accumulation"] == "8"
    assert options["warmup-tokens"] == "1000000000"
    assert options["distributed-mode"] == "ddp"
    assert "--no-gradient-checkpointing" in command
    assert options["lr-scheduler"] == "wsd"
    assert options["tokenized-data"].startswith("hf://")
    assert options["tokenized-cache-dir"] == "cache"
    assert options["tokenized-cache-gb"] == "24.0"
    assert options["tokenized-prefetch-shards"] == "1"
    assert options["num-workers"] == "0"
    assert options["save-steps"] == "0"
    assert "-u" in command
    assert "TR_HASH_LINE_PROGRESS" not in command
    assert "complexity.training.text_pretraining" in command
    with pytest.raises(ValueError, match="conflicts with replay plan"):
        text_pretraining.build_replay_command(
            plan,
            tokenizer="tokenizer",
            data="data",
            revision="pinned",
            output="run",
            cache_dir="cache",
            target_tokens=1,
        )


def test_tokenizer_upload_uses_explicit_approved_dataset_repo_and_workers(tmp_path) -> None:
    from complexity.training.corpus_preparation import resolve_layout, upload_dataset_subset

    _, actual, _ = resolve_layout(
        target_tokens=text_pretraining.TARGET_TOKENS,
        seq_len=1024,
        global_batch_sequences=512,
        sources=text_pretraining.corpus_sources("stack.jsonl"),
    )
    assert 200_000_000_000 <= actual < 200_000_000_000 + 512 * 1024
    calls = []
    upload_dataset_subset(
        output_root=tmp_path,
        repo_id="Pacific-i64/data-32k-200b-tokens",
        allow_patterns=["corpora/*", "mixture_manifest.json"],
        token=None,
        workers=64,
        api=SimpleNamespace(upload_large_folder=lambda **kwargs: calls.append(kwargs)),
    )
    assert calls[0]["repo_id"] == "Pacific-i64/data-32k-200b-tokens"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["num_workers"] == 64
    assert calls[0]["allow_patterns"] == ["corpora/*", "mixture_manifest.json"]


def test_parallel_tokenization_flags_are_not_silently_claimed_by_portable_preparer():
    from complexity.training.corpus_preparation import prepare_corpus_mixture

    # The removed 360-vCPU orchestrator exposed these flags. The canonical
    # portable writer is sequential; unsupported parallel settings fail rather
    # than suggesting that this historical operational pipeline was validated.
    for option in ("parallel_corpora", "stack_download_workers"):
        assert option not in inspect.signature(prepare_corpus_mixture).parameters
        with pytest.raises(TypeError, match="unexpected keyword"):
            prepare_corpus_mixture(**{option: 3})


def test_package_replay_supervisor_resumes_without_secrets_or_shell_pipeline(tmp_path) -> None:
    _, command, options = _replay_command(tmp_path)
    supervisor = SupervisorProgram(
        name="tr_hash_200m_70b_replay",
        command=command,
        directory=Path("/workspace/complexity-framework"),
        stdout_logfile=Path(
            "/workspace/complexity-framework/artifacts/tr_hash_200m_70b_replay.log"
        ),
    ).render()
    assert "[program:tr_hash_200m_70b_replay]" in supervisor
    assert "autostart=true" in supervisor and "autorestart=unexpected" in supervisor
    assert options["resume"] == "auto"
    assert "torch.distributed.run" in supervisor and "--standalone" in command
    assert "complexity.training.text_pretraining" in supervisor
    assert (
        "/dev/stdout" not in supervisor and "pty " not in supervisor and "| tee" not in supervisor
    )
    assert "environment=" not in supervisor and "HF_TOKEN=" not in supervisor
    assert "stdout_logfile=/workspace/complexity-framework/artifacts/" in supervisor
    assert "-u" in command


def test_successful_run_cleans_numbered_checkpoints_but_keeps_final_export(
    tmp_path: Path,
) -> None:
    from complexity.training.checkpoint_lifecycle import cleanup_numbered_checkpoints as cleanup

    output = tmp_path / "run"
    final = output / "final"
    final.mkdir(parents=True)
    from safetensors.torch import save_file

    save_file({"weight": torch.ones(1)}, str(final / "model.safetensors"))
    preserved_weights = (final / "model.safetensors").read_bytes()
    (output / "tensorboard").mkdir()
    for name in ("token_pack_001", "step_100", "interrupted_90", "final_90"):
        checkpoint = output / name
        checkpoint.mkdir()
        (checkpoint / "checkpoint.pt").write_bytes(b"checkpoint")

    removed, reclaimed = cleanup(output)

    assert removed == ["final_90", "interrupted_90", "step_100", "token_pack_001"]
    assert reclaimed == 4 * len(b"checkpoint")
    assert (final / "model.safetensors").read_bytes() == preserved_weights
    assert (output / "tensorboard").is_dir()


def test_checkpoint_cleanup_refuses_to_run_without_final_export(tmp_path: Path) -> None:
    from complexity.training.checkpoint_lifecycle import cleanup_numbered_checkpoints as cleanup

    checkpoint = tmp_path / "run" / "token_pack_001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.pt").write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="final export is missing"):
        cleanup(tmp_path / "run")

    assert checkpoint.is_dir()


def test_checkpoint_cleanup_dry_run_and_corrupt_export_leave_checkpoints(tmp_path):
    from safetensors.torch import save_file

    from complexity.training.checkpoint_lifecycle import cleanup_numbered_checkpoints

    final = tmp_path / "final"
    final.mkdir()
    save_file({"weight": torch.ones(1)}, str(final / "model.safetensors"))
    checkpoint = tmp_path / "step_000001"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.pt").write_bytes(b"state")
    assert cleanup_numbered_checkpoints(tmp_path, dry_run=True) == (["step_000001"], 5)
    assert checkpoint.is_dir()
    (final / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="export is unreadable"):
        cleanup_numbered_checkpoints(tmp_path)
    assert checkpoint.is_dir()


def test_checkpoint_cleanup_rejects_symlink_before_removing_any_candidate(tmp_path):
    from safetensors.torch import save_file

    from complexity.training.checkpoint_lifecycle import cleanup_numbered_checkpoints

    root = tmp_path / "run"
    (root / "final").mkdir(parents=True)
    save_file({"weight": torch.ones(1)}, str(root / "final" / "model.safetensors"))
    (root / "step_000001").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / "step_000002").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink checkpoint"):
        cleanup_numbered_checkpoints(root)
    assert (root / "step_000001").is_dir()
    assert external.is_dir()


def test_text_pretraining_writes_a_live_metrics_jsonl_artifact() -> None:
    runner = Path("complexity/training/runner.py").read_text(encoding="utf-8")

    assert 'metrics_path = os.path.join(args.checkpoint_dir, "metrics.jsonl")' in runner
    assert 'with open(metrics_path, "a", encoding="utf-8")' in runner
    assert "step % args.log_steps == 0" in runner


def test_tqdm_callback_does_not_pin_stdout() -> None:
    source = Path("complexity/training/callbacks.py").read_text(encoding="utf-8")

    # stdout is fully buffered when piped (non-tty); stderr is not. Pinning
    # the bar to stdout silently starved every log tail under Supervisor.
    assert "file=sys.stdout" not in source


def test_remote_pretokenized_training_rejects_dataloader_workers() -> None:
    module = _profile()
    runner = module.build_200m_runner()
    args = SimpleNamespace(tokenized_data="hf://datasets/owner/repo", num_workers=4)

    with pytest.raises(ValueError, match="require --num-workers 0"):
        runner.build_dataset(None, args, rank=0, world_size=1)


def test_text_pretraining_profile_has_no_mosaic_or_epoch_schedule() -> None:
    forbidden = (
        "text_mosaic",
        "TEXT_MOSAIC",
        "--packed-epochs",
        "PACKED_EPOCHS",
        "packed_epochs",
        "packed-epoch",
    )
    offenders = [
        f"{path}:{token}"
        for path in (SCRIPT,)
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]
    assert not offenders, "invalid text-pretraining schedule: " + ", ".join(offenders)


def test_obsolete_300m_1t_launchers_are_removed() -> None:
    assert not Path("scripts/train_tr_hash_300m_1t.py").exists()
    assert not Path("scripts/vast_pretrain_tr_hash_300m_1t.sh").exists()
    assert not Path("scripts/vast_pretrain_tr_hash_200m_200b.sh").exists()
