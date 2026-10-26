"""Actual CPU updates and exact continuation for the local SFT experiment."""

import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from complexity import ComplexityModel, ModelConfig
from complexity.inference.chat_template import default_chat_template
from complexity.training.finetuning import validate_training_stage_transition
from complexity.training.sft import (
    EXPERIMENT,
    SFTTokenShardDataset,
    _optimizer_schedule,
    accumulated_sft_update,
    build_sft_plan,
    cpu_diagnostic,
    evaluate_sft,
    execute_sft_config,
    load_sft_model,
    reference_epoch_batches,
    run_sft,
    sft_loss,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_partition(root, name, rows):
    partition = root / name
    partition.mkdir(parents=True)
    inputs, labels, examples = [], [], []
    for index, (tokens, targets) in enumerate(rows):
        examples.append(
            {
                "example_id": str(index),
                "offset": len(inputs),
                "num_tokens": len(tokens),
                "supervised_tokens": sum(target != -100 for target in targets),
            }
        )
        inputs.extend(tokens)
        labels.extend(targets)
    np.array(inputs, dtype="<u4").tofile(partition / "input_ids.bin")
    np.array(labels, dtype="<i4").tofile(partition / "labels.bin")
    (partition / "examples.jsonl").write_text(
        "".join(json.dumps(example) + "\n" for example in examples)
    )
    metadata = {
        "format": "complexity-sft-token-shard-v2",
        "assistant_supervision": "final_assistant_only",
        "history_assistant_turns": "masked_context",
        "chat_template_id": "complexity-chat-v2",
        "chat_template_eos_token": "</s>",
        "eos_token_id": 0,
        "input_dtype": "uint32-le",
        "label_dtype": "int32-le",
        "truncation_policy": "fail_closed_no_truncation",
        "vocab_size": 32,
        "sequence_length_cap": 8,
        "num_tokens": len(inputs),
        "examples": len(examples),
        "supervised_tokens": sum(target != -100 for target in labels),
        "files": {
            filename: digest(partition / filename)
            for filename in ("input_ids.bin", "labels.bin", "examples.jsonl")
        },
    }
    (partition / "sft.idx.json").write_text(json.dumps(metadata))
    return partition


@pytest.fixture
def artifacts(tmp_path):
    torch.manual_seed(47)
    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        shared_intermediate_size=16,
        vocab_size=32,
        num_experts=4,
        top_k=2,
        max_position_embeddings=8,
        use_custom_kernels=False,
    )
    checkpoint = tmp_path / "base"
    ComplexityModel(config).save_pretrained(checkpoint)
    data = tmp_path / "data"
    data.mkdir()
    template = default_chat_template()
    template["eos_token"] = "</s>"
    (data / "chat_template.json").write_text(json.dumps(template))
    rows = [
        ([2, 3, 4], [-100, 4, 0]),
        ([5, 6], [6, 0]),
        ([7, 8, 9], [-100, 9, 0]),
        ([10, 11, 12, 13], [-100, 12, 13, 0]),
    ]
    write_partition(data, "train", rows)
    write_partition(data, "eval", rows[:2])
    return checkpoint, data


def plan_for(artifacts, **kwargs):
    checkpoint, data = artifacts
    return build_sft_plan(
        checkpoint,
        data,
        experiment=EXPERIMENT,
        batch_size=kwargs.pop("batch_size", 1),
        world_size=kwargs.pop("world_size", 1),
        expert_lr_scale=1.0,
        sequence_length=8,
        **kwargs,
    )


def test_packing_retains_all_next_token_and_final_eos_targets(artifacts):
    _, root = artifacts
    data = SFTTokenShardDataset(root, sequence_length=8, vocab_size=32)
    assert data.packing_plan.packs == ((0, 1), (2, 3))
    assert data[0]["input_ids"].tolist() == [2, 3, 4, 0, 5, 6, 0, 0]
    assert data[0]["labels"].tolist() == [-100, 4, 0, -100, 6, 0, -100, -100]
    assert sum(int((data[item]["labels"] != -100).sum()) for item in range(len(data))) == 9
    assert sum(int((data[item]["labels"] == 0).sum()) for item in range(len(data))) == 4


def test_loss_matches_all_preshifted_labels_including_final_eos(artifacts):
    checkpoint, root = artifacts
    model = load_sft_model(checkpoint).eval()
    data = SFTTokenShardDataset(root, sequence_length=8, vocab_size=32)
    batch = {name: tensor.unsqueeze(0) for name, tensor in data[1].items()}
    logits = model(batch["input_ids"])["logits"]
    expected = F.cross_entropy(
        logits.reshape(-1, 32), batch["labels"].reshape(-1), ignore_index=-100
    )
    torch.testing.assert_close(sft_loss(model, batch), expected)
    shifted = F.cross_entropy(
        logits[:, :-1].reshape(-1, 32), batch["labels"][:, 1:].reshape(-1), ignore_index=-100
    )
    assert not torch.isclose(expected, shifted)


def test_sampler_preserves_historical_inplace_shuffle_and_sparse_rank_padding():
    plan = {"packed_items": 5, "world_size": 3, "batch_size": 2, "steps_per_epoch": 1, "seed": 42}
    indices = list(range(5))
    random.Random(42).shuffle(indices)
    random.Random(43).shuffle(indices)
    expected = []
    for rank in range(3):
        items = indices[rank::3]
        expected.append((items + items)[:2])
    assert list(reference_epoch_batches(plan, 1)) == [expected]
    fresh = list(range(5))
    random.Random(43).shuffle(fresh)
    assert fresh != indices


def test_microbatch_accumulation_preserves_equal_rank_means(artifacts):
    checkpoint, root = artifacts
    data = SFTTokenShardDataset(root, sequence_length=8, vocab_size=32)
    plan = plan_for(artifacts, batch_size=2, world_size=2)
    model = load_sft_model(checkpoint)
    reference = copy.deepcopy(model)
    optimizer, scheduler = _optimizer_schedule(model, plan)
    reference_optimizer, reference_scheduler = _optimizer_schedule(reference, plan)
    ranks = [[0, 1], [0, 0]]
    actual = accumulated_sft_update(model, data, ranks, optimizer, scheduler, micro_batch_size=1)
    expected = accumulated_sft_update(
        reference, data, ranks, reference_optimizer, reference_scheduler, micro_batch_size=2
    )
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected_parameter, atol=1e-7, rtol=1e-6)
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert {id(parameter) for parameter in optimized} == {
        id(parameter) for parameter in model.parameters()
    }
    assert len(optimized) == len({id(parameter) for parameter in optimized})


def test_cpu_diagnostic_updates_real_model_without_writing_artifacts(artifacts, monkeypatch):
    plan = plan_for(artifacts)
    before = {name: digest(Path(name)) for name in plan["files"]}
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CPU test initialized CUDA"))
    result = cpu_diagnostic(plan)
    assert result["parameters_updated"] is True
    assert result["device"] == "cpu"
    assert result["checkpoint_written"] is False
    assert result["supervised_eos_targets"] > 0
    assert {name: digest(Path(name)) for name in plan["files"]} == before


def test_complete_local_run_and_resume_match_model_optimizer_scheduler_and_rng(artifacts, tmp_path):
    plan = plan_for(artifacts)
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    full = run_sft(plan, continuous, device="cpu", gradient_checkpointing=True, save_every=1)
    bounded = run_sft(plan, split, device="cpu", steps=2, gradient_checkpointing=True, save_every=1)
    resumed = run_sft(
        plan,
        split,
        device="cpu",
        resume=split / "step_000002",
        gradient_checkpointing=True,
        save_every=1,
    )
    assert bounded["complete"] is False
    assert full["complete"] is resumed["complete"] is True
    assert full["evaluations"] == resumed["evaluations"]
    assert [item["step"] for item in full["evaluations"]] == [0, 2, 4, 6]
    assert all(item["evaluation"]["complete"] for item in full["evaluations"])
    first = torch.load(continuous / "step_000006" / "checkpoint.pt", weights_only=True)
    second = torch.load(split / "step_000006" / "checkpoint.pt", weights_only=True)
    assert first["scheduler"] == second["scheduler"]
    torch.testing.assert_close(first["rng"]["torch"], second["rng"]["torch"], rtol=0, atol=0)
    assert first["rng"]["python"] == second["rng"]["python"]
    for name, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][name], rtol=0, atol=0)
    for key, state in first["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, second["optimizer"]["state"][key][name], rtol=0, atol=0
            )
    for epoch in (1, 2, 3):
        exported = split / f"epoch_{epoch:02d}"
        assert (exported / "model.safetensors").is_file()
        assert json.loads((exported / "experiment.json").read_text())["release_eligible"] is False


def test_eval_is_target_weighted_and_step_limit_retains_complete_schedule(artifacts):
    checkpoint, root = artifacts
    model = load_sft_model(checkpoint)
    data = SFTTokenShardDataset(root, sequence_length=8, vocab_size=32)
    assert evaluate_sft(model, data, batch_size=1)["loss"] == pytest.approx(
        evaluate_sft(model, data, batch_size=2)["loss"], abs=1e-6
    )
    plan = plan_for(artifacts, epochs=100)
    assert plan["optimizer_updates"] == 200
    _, schedule = _optimizer_schedule(model, plan)
    assert schedule.get_last_lr()[0] == pytest.approx(plan["learning_rate"] / 6)


def test_recovery_replays_existing_epoch_export_after_checkpoint_write_failure(
    artifacts, tmp_path, monkeypatch
):
    import complexity.training.sft as workflow

    plan = plan_for(artifacts)
    output = tmp_path / "interrupted"
    save = workflow._save_run_checkpoint

    def fail_after_export(*args, **kwargs):
        if kwargs["step"] == 2:
            assert (output / "epoch_01" / "model.safetensors").is_file()
            raise OSError("simulated checkpoint write interruption")
        return save(*args, **kwargs)

    monkeypatch.setattr(workflow, "_save_run_checkpoint", fail_after_export)
    with pytest.raises(OSError, match="simulated"):
        run_sft(plan, output, device="cpu", steps=2, save_every=100)
    exported_hash = digest(output / "epoch_01" / "model.safetensors")
    monkeypatch.setattr(workflow, "_save_run_checkpoint", save)
    result = run_sft(plan, output, device="cpu", resume=output / "step_000000", save_every=100)
    assert result["complete"] is True
    assert digest(output / "epoch_01" / "model.safetensors") == exported_hash


def test_resume_refuses_changed_evaluation_protocol_or_artifacts(artifacts, tmp_path):
    plan = plan_for(artifacts)
    output = tmp_path / "bounded"
    run_sft(plan, output, device="cpu", steps=1, eval_max_batches=1)
    before = digest(output / "run.json")
    with pytest.raises(ValueError, match="identity differ"):
        run_sft(plan, output, device="cpu", steps=2, resume=output / "latest")
    assert digest(output / "run.json") == before
    alternate = artifacts[1].parent / "alternate"
    alternate.mkdir()
    (alternate / "chat_template.json").write_text((artifacts[1] / "chat_template.json").read_text())
    write_partition(alternate, "eval", [([20, 21], [21, 0])])
    with pytest.raises(ValueError, match="identity differ"):
        run_sft(
            plan,
            output,
            device="cpu",
            steps=2,
            resume=output / "latest",
            eval_max_batches=1,
            eval_data=alternate / "eval",
        )


def test_explicit_complete_horizon_cannot_claim_complete_with_partial_evaluation(
    artifacts, tmp_path
):
    plan = plan_for(artifacts)
    with pytest.raises(ValueError, match="complete held-out"):
        run_sft(
            plan,
            tmp_path / "rejected",
            device="cpu",
            steps=plan["optimizer_updates"],
            eval_max_batches=1,
        )
    assert not (tmp_path / "rejected").exists()


def test_marker_write_failure_keeps_latest_complete_checkpoint(artifacts, tmp_path, monkeypatch):
    plan = plan_for(artifacts)
    output = tmp_path / "marker_failure"
    write_text = Path.write_text

    def fail_marker(path, *args, **kwargs):
        if path.name == "complete.json" and path.parent.name == ".step_000001.writing":
            raise OSError("simulated marker write failure")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_marker)
    with pytest.raises(OSError, match="marker write"):
        run_sft(plan, output, device="cpu", steps=1)
    assert (output / "latest").read_text().strip() == "step_000000"
    assert (output / "step_000000" / "complete.json").is_file()
    assert not (output / "step_000001").exists()
    monkeypatch.setattr(Path, "write_text", write_text)
    assert run_sft(plan, output, device="cpu", resume=output / "latest")["complete"] is True


def test_authored_json_configuration_and_canonical_cli(artifacts, tmp_path):
    config = {
        "format": "complexity-sft-experiment-v1",
        "checkpoint": "base",
        "data": "data",
        "experiment": EXPERIMENT,
        "batch_size": 1,
        "world_size": 1,
        "sequence_length": 8,
        "execution": {"output": "configured_run", "device": "cpu", "save_every": 1},
    }
    path = tmp_path / "sft.json"
    path.write_text(json.dumps(config))
    preflight = subprocess.run(
        [sys.executable, "-m", "complexity", "train", "sft", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(preflight.stdout)
    assert report["training_started"] is False
    assert report["execution"]["output"] == str(tmp_path / "configured_run")
    assert not (tmp_path / "configured_run").exists()
    actual = subprocess.run(
        [sys.executable, "-m", "complexity", "train", "sft", str(path), "--run", "--steps", "2"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(actual.stdout)["step"] == 2
    assert (
        execute_sft_config(path, run=True, resume=tmp_path / "configured_run" / "latest")[
            "complete"
        ]
        is True
    )
    config["unknown_silent_override"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unknown SFT"):
        execute_sft_config(path)


def test_unavailable_device_and_invalid_precision_fail_without_device_fallback(
    artifacts, tmp_path, monkeypatch
):
    plan = plan_for(artifacts)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        run_sft(plan, tmp_path / "unavailable", device="cuda")
    with pytest.raises(ValueError, match="only on CUDA"):
        run_sft(plan, tmp_path / "invalid_precision", device="cpu", precision="bf16")
    assert not (tmp_path / "unavailable").exists()


def test_stage_release_contract_stays_strict_and_experiment_is_explicit(artifacts):
    with pytest.raises(ValueError, match="forbidden"):
        validate_training_stage_transition("text", "pretraining", "supervised-finetuning")
    checkpoint, data = artifacts
    with pytest.raises(ValueError, match="explicit experiment"):
        build_sft_plan(
            checkpoint, data, experiment="release", batch_size=1, world_size=1, expert_lr_scale=1
        )
    plan = plan_for(artifacts)
    assert plan["source_stage"] == "pretraining"
    assert plan["skipped_stage"] == "refinement"
    assert plan["release_eligible"] is False


@pytest.mark.parametrize("mutation", ["hash", "supervised", "overlong", "next_token", "eos"])
def test_corrupted_or_invalid_shards_fail_before_execution(artifacts, mutation):
    _, root = artifacts
    partition = root / "train"
    metadata = json.loads((partition / "sft.idx.json").read_text())
    if mutation in {"hash", "next_token", "eos"}:
        labels = np.fromfile(partition / "labels.bin", dtype="<i4")
        labels[1 if mutation != "eos" else 2] = 15
        labels.tofile(partition / "labels.bin")
        if mutation != "hash":
            metadata["files"]["labels.bin"] = digest(partition / "labels.bin")
    elif mutation == "supervised":
        metadata["supervised_tokens"] += 1
    else:
        examples = [
            json.loads(line) for line in (partition / "examples.jsonl").read_text().splitlines()
        ]
        examples[0]["num_tokens"] = 9
        (partition / "examples.jsonl").write_text(
            "".join(json.dumps(example) + "\n" for example in examples)
        )
        metadata["files"]["examples.jsonl"] = digest(partition / "examples.jsonl")
    (partition / "sft.idx.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        SFTTokenShardDataset(root, sequence_length=8, vocab_size=32)


def test_checkpoint_layout_and_alias_conflicts_fail_closed(artifacts):
    checkpoint, _ = artifacts
    raw = json.loads((checkpoint / "config.json").read_text())
    raw["num_experts_per_tok"] = 1
    (checkpoint / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="disagree"):
        plan_for(artifacts)
    raw["num_experts_per_tok"] = raw.pop("top_k")
    (checkpoint / "config.json").write_text(json.dumps(raw))
    assert plan_for(artifacts)["model_config"]["top_k"] == 2
    raw["hidden_size"] *= 2
    (checkpoint / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="tensor layout"):
        plan_for(artifacts)


def test_module_preflight_requires_no_run_and_reports_experiment_json(artifacts):
    checkpoint, data = artifacts
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "complexity.training.sft",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--experiment",
            EXPERIMENT,
            "--batch-size",
            "1",
            "--world-size",
            "1",
            "--expert-lr-scale",
            "1",
            "--sequence-length",
            "8",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["training_started"] is False
    assert report["release_eligible"] is False
