"""Real two-rank Gloo updates, rank RNG restore and explicit single→DDP migration."""

import copy
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typer.testing import CliRunner

from complexity.cli.app import app
from complexity.training.sft import (
    SFTTokenShardDataset,
    _optimizer_schedule,
    accumulated_sft_update,
    load_sft_model,
    run_sft,
)
from complexity.training.sft_distributed import (
    RUNTIME_FORMAT,
    SFTLossModule,
    distributed_sft_update,
    run_distributed_sft,
    validate_resume_payload,
)
from tests.test_sft_workflow import artifacts as artifacts
from tests.test_sft_workflow import plan_for, write_partition


def _group(rank, world, store):
    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=45),
    )


def _update_worker(rank, world, store, plan, result_dir):
    _group(rank, world, store)
    try:
        model = load_sft_model(plan["checkpoint"])
        model.gradient_checkpointing_enable()
        data = SFTTokenShardDataset(plan["data"], sequence_length=8, vocab_size=32)
        optimizer, scheduler = _optimizer_schedule(model, plan)
        wrapped = DDP(SFTLossModule(model), broadcast_buffers=False, find_unused_parameters=False)
        forwards = []
        wrapped.register_forward_hook(lambda *args: forwards.append(1))
        # Unequal target counts between ranks AND variable target counts inside rank0.
        ranks = [[0, 1, 0], [1, 1, 1]]
        metrics = distributed_sft_update(
            wrapped, data, ranks[rank], optimizer, scheduler, micro_batch_size=2
        )
        torch.save(
            {"model": model.state_dict(), "metrics": metrics, "forwards": len(forwards)},
            Path(result_dir) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_masked_means_and_short_microbatch_match_portable_update(artifacts, tmp_path):
    plan = plan_for(artifacts, world_size=2, batch_size=3)
    result_dir = tmp_path / "updates"
    result_dir.mkdir()
    mp.spawn(
        _update_worker,
        args=(2, str(tmp_path / "update.store"), plan, str(result_dir)),
        nprocs=2,
        join=True,
    )
    model = load_sft_model(plan["checkpoint"])
    model.gradient_checkpointing_enable()
    data = SFTTokenShardDataset(plan["data"], sequence_length=8, vocab_size=32)
    optimizer, scheduler = _optimizer_schedule(model, plan)
    expected = accumulated_sft_update(
        model, data, [[0, 1, 0], [1, 1, 1]], optimizer, scheduler, micro_batch_size=2
    )
    for rank in range(2):
        actual = torch.load(result_dir / f"rank{rank}.pt", weights_only=True)
        assert actual["forwards"] == 2  # Both microbatches use real DDP forward.
        assert actual["metrics"]["loss"] == pytest.approx(expected["loss"], abs=1e-6)
        assert actual["metrics"]["gradient_norm"] == pytest.approx(
            expected["gradient_norm"], abs=1e-6
        )
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(actual["model"][name], parameter, rtol=1e-5, atol=1e-7)


def _run_worker(rank, world, store, plan, output, steps, resume, migrate):
    _group(rank, world, store)
    try:
        result = run_distributed_sft(
            plan,
            output,
            device="cpu",
            micro_batch_size=1,
            gradient_checkpointing=True,
            save_every=100,
            steps=steps,
            resume=resume,
            migrate_single=migrate,
        )
        assert result["step"] == steps
    finally:
        dist.destroy_process_group()


def _spawn_run(tmp_path, name, plan, output, steps, resume=None, migrate=False):
    mp.spawn(
        _run_worker,
        args=(
            2,
            str(tmp_path / f"{name}.store"),
            plan,
            str(output),
            steps,
            str(resume) if resume else None,
            migrate,
        ),
        nprocs=2,
        join=True,
    )


def _larger_plan(artifacts):
    rows = []
    for index in range(7):
        tokens = list(range(2 + index, 10 + index))
        targets = [-100] * (index % 4) + tokens[index % 4 + 1 :] + [0]
        rows.append((tokens, targets))
    # Reuse local fixture assets, replacing only its tiny shard with longer complete rows.
    import shutil

    shutil.rmtree(artifacts[1] / "train")
    write_partition(artifacts[1], "train", rows)
    return plan_for(artifacts, world_size=2, batch_size=2)


def _checkpoint(output, step):
    path = output / f"step_{step:06d}"
    state = torch.load(path / "checkpoint.pt", weights_only=True)
    marker = json.loads((path / "complete.json").read_text())
    assert marker["runtime_format"] == state["runtime_format"] == RUNTIME_FORMAT
    return state, marker


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_tree_equal(first, second)
    else:
        assert left == right


def test_two_rank_resume_restores_model_adamw_scheduler_and_each_rank_rng(artifacts, tmp_path):
    plan = _larger_plan(artifacts)
    assert plan["packed_items"] == 7 and plan["steps_per_epoch"] == 2
    full, split = tmp_path / "full", tmp_path / "split"
    _spawn_run(tmp_path, "full", plan, full, 4)
    _spawn_run(tmp_path, "first", plan, split, 1)
    _spawn_run(tmp_path, "resume", plan, split, 4, resume=split / "latest")
    first, _ = _checkpoint(full, 4)
    second, _ = _checkpoint(split, 4)
    for key in ("model", "optimizer", "scheduler", "rng_by_rank", "history"):
        _assert_tree_equal(first[key], second[key])
    assert [item["rank"] for item in second["rng_by_rank"]] == [0, 1]
    assert len((split / "metrics.jsonl").read_text().splitlines()) == 4
    assert (split / "epoch_02" / "model.safetensors").is_file()
    assert not first["migration"]


def test_portable_checkpoint_requires_explicit_migration_and_preserves_plan(artifacts, tmp_path):
    plan = _larger_plan(artifacts)
    portable, distributed = tmp_path / "portable", tmp_path / "migrated"
    run_sft(plan, portable, device="cpu", micro_batch_size=1, steps=1, gradient_checkpointing=True)
    _spawn_run(
        tmp_path, "migration", plan, distributed, 2, resume=portable / "latest", migrate=True
    )
    state, marker = _checkpoint(distributed, 2)
    assert state["migration"]["source_step"] == 1
    assert state["migration"]["bitwise_equivalence_claimed"] is False
    run_sft(
        plan,
        portable,
        device="cpu",
        micro_batch_size=1,
        steps=2,
        resume=portable / "latest",
        gradient_checkpointing=True,
    )
    reference = torch.load(portable / "step_000002" / "checkpoint.pt", weights_only=True)
    for name, tensor in reference["model"].items():
        torch.testing.assert_close(state["model"][name], tensor, rtol=1e-5, atol=1e-7)
    assert state["scheduler"] == reference["scheduler"]
    corrupted = copy.deepcopy(state)
    corrupted["rng_by_rank"][0]["rank"] = 1
    with pytest.raises(ValueError, match="rank RNG identity"):
        validate_resume_payload(corrupted, marker, state["identity"])
    with pytest.raises(ValueError, match="only to a portable"):
        validate_resume_payload(state, marker, state["identity"], migrate_single=True)
    identity = state["identity"]
    portable_marker = {"step": 2, "experiment": "no_refinement_ablation", "release_eligible": False}
    with pytest.raises(ValueError, match="explicit migrate_single"):
        validate_resume_payload(reference, portable_marker, identity)
    changed = copy.deepcopy(identity)
    changed["scientific_contract"]["seed"] = 99
    with pytest.raises(ValueError, match="scientific plan/data"):
        validate_resume_payload(reference, portable_marker, changed, migrate_single=True)
    for key, value in (("experiment", "release"), ("release_eligible", True)):
        bad_marker = {**marker, key: value}
        with pytest.raises(ValueError, match="marker experiment/release"):
            validate_resume_payload(state, bad_marker, identity)
    with pytest.raises(ValueError, match="runtime/rank identity"):
        validate_resume_payload(state, marker, {**identity, "micro_batch_size": 2})


def _failure_worker(rank, world, store, plan, output, case, resumes, result_dir):
    _group(rank, world, store)
    import complexity.training.sft_distributed as workflow

    original_write, original_verify = Path.write_text, workflow._verify_plan
    try:
        if case == "data" and rank == 1:

            def bad_data(*args):
                raise ValueError("injected missing rank1 data")

            workflow._verify_plan = bad_data
        if case == "save" and rank == 0:

            def fail_marker(path, *args, **kwargs):
                if path.name == "complete.json" and "000001" in str(path):
                    raise OSError("injected complete marker failure")
                return original_write(path, *args, **kwargs)

            Path.write_text = fail_marker
        try:
            workflow.run_distributed_sft(
                plan,
                output,
                device="cpu",
                micro_batch_size=1,
                steps=(rank + 1 if case == "controls" else 1),
                resume=resumes[rank] if resumes else None,
                gradient_checkpointing=True,
            )
        except (ValueError, RuntimeError) as exc:
            original_write(Path(result_dir) / f"rank{rank}.txt", str(exc))
        else:
            raise AssertionError("Both ranks must reject the injected failure")
    finally:
        Path.write_text, workflow._verify_plan = original_write, original_verify
        dist.destroy_process_group()


def _expect_collective_failure(tmp_path, plan, case, output, resumes=None):
    results = tmp_path / f"{case}-errors"
    results.mkdir()
    mp.spawn(
        _failure_worker,
        args=(2, str(tmp_path / f"{case}.store"), plan, str(output), case, resumes, str(results)),
        nprocs=2,
        join=True,
    )
    errors = [(results / f"rank{rank}.txt").read_text() for rank in range(2)]
    assert errors[0] == errors[1]
    return errors[0]


@pytest.mark.parametrize(
    "case, message", [("controls", "run controls"), ("data", "injected missing rank1 data")]
)
def test_rank_control_and_preflight_errors_stop_every_rank(artifacts, tmp_path, case, message):
    plan = _larger_plan(artifacts)
    output = tmp_path / "output"
    assert message in _expect_collective_failure(tmp_path, plan, case, output)
    assert not output.exists()


def test_checkpoint_marker_failure_keeps_committed_cursor_and_can_resume(artifacts, tmp_path):
    plan = _larger_plan(artifacts)
    output = tmp_path / "output"
    assert "injected complete marker failure" in _expect_collective_failure(
        tmp_path, plan, "save", output
    )
    assert (output / "latest").read_text().strip() == "step_000000"
    assert not (output / "step_000001").exists()
    _spawn_run(tmp_path, "recover", plan, output, 1, resume=output / "latest")
    recovered, _ = _checkpoint(output, 1)
    assert recovered["step"] == 1
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 1


def test_valid_but_different_source_checkpoints_cannot_mix_rank_optimizer_states(
    artifacts, tmp_path
):
    plan = _larger_plan(artifacts)
    source = tmp_path / "source"
    _spawn_run(tmp_path, "source", plan, source, 1)
    state, marker = _checkpoint(source, 1)
    alternative = tmp_path / "alternative"
    alternative.mkdir()
    state["rng_by_rank"][0]["state"]["torch"][0] += 1
    torch.save(state, alternative / "checkpoint.pt")
    marker["checkpoint_sha256"] = hashlib.sha256(
        (alternative / "checkpoint.pt").read_bytes()
    ).hexdigest()
    (alternative / "complete.json").write_text(json.dumps(marker))
    output = tmp_path / "rejected"
    resumes = [str(source / "step_000001"), str(alternative)]
    assert "restored checkpoint" in _expect_collective_failure(
        tmp_path, plan, "checkpoint", output, resumes
    )
    assert not output.exists()


def test_distributed_cli_preflight_does_not_select_gpu_or_start_process_group(
    artifacts, tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("preflight initialized CUDA"))
    monkeypatch.setattr(
        dist, "init_process_group", lambda *args, **kwargs: pytest.fail("preflight initialized DDP")
    )
    config = tmp_path / "sft_ddp.json"
    config.write_text(
        json.dumps(
            {
                "format": "complexity-sft-experiment-v1",
                "checkpoint": "base",
                "data": "data",
                "experiment": "no_refinement_ablation",
                "batch_size": 1,
                "world_size": 2,
                "sequence_length": 8,
                "execution": {"device": "cuda", "precision": "bf16", "output": "output"},
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(app, ["train", "sft", str(config), "--distributed"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["training_started"] is False
    assert report["runtime_format"] == RUNTIME_FORMAT
    assert report["launch_command"][-2:] == ["--distributed", "--run"]
    assert not (tmp_path / "output").exists()
    bad = runner.invoke(app, ["train", "sft", str(config), "--migrate-single"])
    assert bad.exit_code == 2 and "requires --distributed" in bad.stderr
    authored = json.loads(config.read_text())
    authored["execution"]["device"] = "mps"
    config.write_text(json.dumps(authored))
    unsupported = runner.invoke(app, ["train", "sft", str(config), "--distributed"])
    assert unsupported.exit_code == 2 and "CPU/fp32 or CUDA" in unsupported.stderr
