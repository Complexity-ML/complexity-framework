from __future__ import annotations

import importlib.util
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from complexity.models import ComplexityModel
from complexity.training import (
    SupervisorProgram,
    WeightedStreamingTextDataset,
    resolve_warmup_steps,
)
from complexity.training.packing import resolve_token_pack_schedule
from complexity.training.runner import log_routing_diagnostic, routing_diagnostic

SCRIPT = Path("scripts/train_tr_hash_200m_200b.py")
TOKENIZER_LAUNCHER = Path("scripts/cpu_tokenize_tr_hash_200m_200b.sh")
REPLAY_LAUNCHER = Path("scripts/vast_pretrain_tr_hash_200m_70b_replay.sh")


def _load_training_script():
    spec = importlib.util.spec_from_file_location("train_tr_hash_200m_200b", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    module = _load_training_script()
    config = module.make_config()
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
        module.build_runner()._build_parser().parse_args(["--stack-edu-data", "stack/*.jsonl.gz"])
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
    module = _load_training_script()
    config = module.make_config()

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
    module = _load_training_script()
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
        module.build_runner().build_dataset(None, args, rank=0, world_size=1)


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
    module = _load_training_script()
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
    result = module.build_runner().build_dataset(None, args, rank=0, world_size=10)
    assert result is dataset


def test_200b_corpus_budget_and_distribution_are_exact() -> None:
    module = _load_training_script()
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
    module = _load_training_script()
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
    module = _load_training_script()
    with pytest.raises(ValueError, match="SWHIDs"):
        module.corpus_sources("")


def test_production_launcher_streams_the_70b_replay_without_worker_cache_thrash() -> None:
    source = REPLAY_LAUNCHER.read_text(encoding="utf-8")

    assert 'export TARGET_TOKENS="$planned_tokens"' in source
    assert "TOKEN_PACKS:-40" in source
    assert "NPROC_PER_NODE:-8" in source
    assert "GRADIENT_ACCUMULATION:-8" in source
    assert "WARMUP_TOKENS:-1000000000" in source
    assert "--distributed-mode ddp" in source
    assert "--no-gradient-checkpointing" in source
    assert "--lr-scheduler" in source and "wsd" in source
    assert "--tokenized-data" in source
    assert "--tokenized-cache-dir" in source
    assert "--tokenized-cache-gb" in source
    assert "--tokenized-prefetch-shards" in source
    assert "TOKENIZED_CACHE_GB:-24" in source
    assert "NUM_WORKERS:-0" in source
    assert "export PYTHONUNBUFFERED=1" in source
    assert "TR_HASH_LINE_PROGRESS" not in source
    assert "--save-steps 0" in source


def test_cpu_tokenizer_publishes_to_the_approved_dataset_repo() -> None:
    source = TOKENIZER_LAUNCHER.read_text(encoding="utf-8")

    assert "TARGET_TOKENS:-200000000000" in source
    assert "Pacific-i64/data-32k-200b-tokens" in source
    assert "--hf-repo" in source
    assert "--parallel-corpora" in source
    assert "--stack-download-workers" in source
    assert "HF_TOKEN" in source


def test_70b_replay_supervisor_survives_a_crash_or_reboot_unattended() -> None:
    launcher = REPLAY_LAUNCHER.read_text(encoding="utf-8")
    supervisor = SupervisorProgram(
        name="tr_hash_200m_70b_replay",
        command=("/bin/bash", "/workspace/run_200m.sh"),
        directory=Path("/workspace/complexity-framework"),
        stdout_logfile=Path(
            "/workspace/complexity-framework/artifacts/tr_hash_200m_70b_replay.log"
        ),
    ).render()

    assert "build_tr_hash_70b_replay_plan" in launcher
    assert 'export TARGET_TOKENS="$planned_tokens"' in launcher
    assert "conflicts with replay plan" in launcher
    assert "-m scripts.train_tr_hash_200m_200b" in launcher
    assert not Path("scripts/vast_pretrain_tr_hash_200m_200b.sh").exists()
    assert "[program:tr_hash_200m_70b_replay]" in supervisor
    assert "autostart=true" in supervisor
    assert "autorestart=unexpected" in supervisor

    # Regression guard: an earlier ad-hoc deployment piped the launcher
    # through `pty ... | tee` into the portal's /dev/stdout capture, which
    # fully buffers stdout when it isn't a tty — tqdm's carriage-return
    # redraws never reached the log even though training was progressing
    # normally. Direct `torchrun` + a real stdout_logfile path is what
    # actually surfaces live progress.
    assert "pty " not in launcher
    assert "| tee" not in launcher
    assert "torchrun --standalone" in launcher
    assert "exec torchrun" not in launcher
    assert "/dev/stdout" not in supervisor

    # Regression guard: the env (including HF_TOKEN) used to be inlined
    # directly into this tracked conf file. That conf is committed to a
    # public repo, so any real token value would leak. The run's env now
    # lives only in an instance-local, untracked wrapper script that this
    # conf's command= merely invokes.
    assert "environment=" not in supervisor
    assert "HF_TOKEN=" not in supervisor
    assert "command=/bin/bash /workspace/run_200m.sh" in supervisor
    assert "stdout_logfile=/workspace/complexity-framework/artifacts/" in supervisor
    assert "export PYTHONUNBUFFERED=1" in launcher

    # Unattended resilience: a crash relaunches and resumes from the last
    # checkpoint automatically (autorestart=unexpected — a *clean* exit,
    # i.e. job finished, still won't loop-restart), and a full instance
    # reboot brings the service back up on its own too (autostart=true),
    # rather than silently sitting dead until someone notices.
    assert "autorestart=unexpected" in supervisor
    assert "autostart=true" in supervisor
    assert '--resume "${RESUME:-auto}"' in launcher


def test_successful_run_cleans_numbered_checkpoints_but_keeps_final_export(
    tmp_path: Path,
) -> None:
    from scripts.cleanup_tr_hash_200m_checkpoints import cleanup

    output = tmp_path / "run"
    final = output / "final"
    final.mkdir(parents=True)
    (final / "model.safetensors").write_bytes(b"weights")
    (output / "tensorboard").mkdir()
    for name in ("token_pack_001", "step_100", "interrupted_90", "final_90"):
        checkpoint = output / name
        checkpoint.mkdir()
        (checkpoint / "checkpoint.pt").write_bytes(b"checkpoint")

    removed, reclaimed = cleanup(output)

    assert removed == ["final_90", "interrupted_90", "step_100", "token_pack_001"]
    assert reclaimed == 4 * len(b"checkpoint")
    assert (final / "model.safetensors").read_bytes() == b"weights"
    assert (output / "tensorboard").is_dir()


def test_checkpoint_cleanup_refuses_to_run_without_final_export(tmp_path: Path) -> None:
    from scripts.cleanup_tr_hash_200m_checkpoints import cleanup

    checkpoint = tmp_path / "run" / "token_pack_001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.pt").write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="final export is missing"):
        cleanup(tmp_path / "run")

    assert checkpoint.is_dir()


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
    module = _load_training_script()
    runner = module.build_runner()
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
        for path in (SCRIPT, REPLAY_LAUNCHER)
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]
    assert not offenders, "invalid text-pretraining schedule: " + ", ".join(offenders)


def test_obsolete_300m_1t_launchers_are_removed() -> None:
    assert not Path("scripts/train_tr_hash_300m_1t.py").exists()
    assert not Path("scripts/vast_pretrain_tr_hash_300m_1t.sh").exists()
    assert not Path("scripts/vast_pretrain_tr_hash_200m_200b.sh").exists()
