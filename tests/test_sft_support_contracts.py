"""Failure cases and usable semantics of the native SFT support primitives."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import load_file, save_file

from complexity.export.native_checkpoint import (
    load_native_checkpoint,
    normalize_native_checkpoint_config,
)
from complexity.training.sft_evaluation import (
    ValidationState,
    complete_epoch_eligible,
    discover_sft_evaluation_partitions,
    validate_evaluation_coverage,
)
from complexity.training.sft_weighting import WeightedSFTTokenShardDataset
from tests.test_sft_bin import _reader, _write_shard
from tests.test_sft_workflow import artifacts as artifacts
from tests.test_sft_workflow import digest


def test_coverage_threshold_is_an_explicit_policy_not_a_global_ten_percent_rule():
    assert (
        validate_evaluation_coverage(300_000, 3_000, minimum_fraction=0.01, partition_name="eval")[
            "fraction"
        ]
        == 0.01
    )
    with pytest.raises(ValueError, match="required minimum is 10.00%"):
        validate_evaluation_coverage(300_000, 3_000, minimum_fraction=0.1, partition_name="eval")


@pytest.mark.parametrize(
    "train, evaluation, minimum",
    [
        (0, 1, 0.1),
        (True, 1, 0.1),
        (10, -1, 0.1),
        (10, 1, float("nan")),
        (10, 1, float("inf")),
        (10, 1, -0.1),
        (10, 1, 1.1),
    ],
)
def test_invalid_coverage_inputs_never_report_a_passing_audit(train, evaluation, minimum):
    with pytest.raises(ValueError):
        validate_evaluation_coverage(
            train, evaluation, minimum_fraction=minimum, partition_name="eval"
        )


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_validation_never_promotes_or_resets_the_entry_checkpoint(loss):
    state = ValidationState.from_baseline(3.5)
    with pytest.raises(ValueError, match="finite"):
        state.observe(loss, min_delta=0.0)
    assert state.best_loss == 3.5 and state.misses == 0
    with pytest.raises(ValueError, match="finite"):
        ValidationState.from_baseline(loss)


def test_validation_state_can_resume_patience_without_selecting_a_new_baseline():
    state = ValidationState(3.5, misses=2)
    restored = ValidationState(**json.loads(json.dumps(asdict(state))))
    improved, after = restored.observe(3.6, min_delta=0.01)
    assert not improved and after.best_loss == 3.5 and after.misses == 3
    assert after.should_stop(patience=3, step=626, steps_per_epoch=313, minimum_epochs=2)
    assert not after.should_stop(patience=3, step=625, steps_per_epoch=313, minimum_epochs=2)
    unmeasured = ValidationState.from_baseline(None)
    improved, measured = unmeasured.observe(3.5, min_delta=0.01)
    assert improved and measured == ValidationState.from_baseline(3.5)


@pytest.mark.parametrize(
    "step, epoch_steps, epochs",
    [
        (-1, 1, 1),
        (1, 0, 1),
        (1, 1, -1),
        (True, 1, 1),
        (1, 1.0, 1),
    ],
)
def test_epoch_gate_rejects_ambiguous_or_negative_progress(step, epoch_steps, epochs):
    with pytest.raises(ValueError):
        complete_epoch_eligible(step, steps_per_epoch=epoch_steps, minimum_epochs=epochs)


def test_partition_discovery_does_not_promote_a_training_split(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "sft.idx.json").write_text("{}")
    assert discover_sft_evaluation_partitions(train).selection is None
    assert discover_sft_evaluation_partitions(tmp_path).selection is None


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"first_task": 1.0},
        {"first_task": 1.0, "second_task": 1.0, "unknown": 1.0},
        {"first_task": 0.0, "second_task": 1.0},
        {"first_task": -1.0, "second_task": 1.0},
        {"first_task": float("nan"), "second_task": 1.0},
        {"first_task": float("inf"), "second_task": 1.0},
        {"first_task": True, "second_task": 1.0},
    ],
)
def test_loss_coefficients_cannot_drop_rows_or_silently_ignore_tasks(tmp_path, weights):
    dataset = _write_shard(tmp_path)
    with pytest.raises(ValueError):
        WeightedSFTTokenShardDataset(dataset, weights)


def test_changed_primary_task_metadata_after_reader_validation_is_rejected(tmp_path):
    dataset = _write_shard(tmp_path)
    path = tmp_path / "train" / "examples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["task"], rows[1]["task"] = rows[1]["task"], rows[0]["task"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="changes the validated example index"):
        WeightedSFTTokenShardDataset(dataset, {"first_task": 0.5, "second_task": 2.0})


def test_weighting_sidecar_cannot_replace_token_offsets_or_verified_task_labels(tmp_path):
    dataset = _write_shard(tmp_path)
    sidecar = tmp_path / "train" / "loss_metadata.jsonl"
    sidecar.write_text(
        "".join(
            json.dumps(
                {
                    "example_id": str(index),
                    "offset": 1000,
                    "task": "invented",
                    "domain": "added",
                }
            )
            + "\n"
            for index in range(2)
        )
    )
    wrapped = WeightedSFTTokenShardDataset(dataset, {"first_task": 1.0, "second_task": 2.0})
    assert [row["offset"] for row in wrapped.examples] == [0, 4]
    assert [row["task"] for row in wrapped.examples] == ["first_task", "second_task"]
    assert all(row["domain"] == "added" for row in wrapped.examples)
    torch.testing.assert_close(wrapped[0]["labels"], dataset[0]["labels"])


def test_weighted_curriculum_rejects_a_stage_that_discards_shard_rows(tmp_path):
    _write_shard(tmp_path)
    config = tmp_path / "curriculum.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 42,
                "stages": [
                    {
                        "name": "incomplete",
                        "max_examples": 1,
                        "epochs": 1,
                        "lr": 1e-6,
                        "balance_by": "none",
                        "loss_task_targets": {"first_task": 0.5, "second_task": 0.5},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="full-shard loss-target audit"):
        WeightedSFTTokenShardDataset.from_curriculum(_reader(tmp_path, 8), config, "incomplete")


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("alias", "disagree"),
        ("width", "disagrees with stored expert"),
        ("layers", "layers disagree"),
        ("shape", "invalid expert tensor layout"),
        ("hidden", "differs from checkpoint topology"),
        ("missing_hidden", "requires explicit fields"),
        ("missing_experts", "requires explicit fields"),
        ("unknown", "unknown checkpoint architecture fields"),
    ],
)
def test_native_checkpoint_topology_never_guesses_past_conflicting_evidence(mutation, message):
    raw = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "num_experts": 2,
        "top_k": 2,
        "intermediate_size": 16,
        "shared_intermediate_size": 16,
        "use_custom_kernels": False,
    }
    weights = {"layers.0.mlp.engine.expert_gate": torch.empty(2, 16, 8, device="meta")}
    if mutation == "alias":
        raw["num_experts_per_tok"] = 1
    elif mutation == "width":
        raw["intermediate_size"] = 32
    elif mutation == "layers":
        weights["layers.1.mlp.engine.expert_gate"] = torch.empty(2, 16, 16, device="meta")
    elif mutation == "shape":
        weights["layers.0.mlp.engine.expert_gate"] = torch.empty(16, 16, device="meta")
    elif mutation == "hidden":
        raw["hidden_size"] = 32
    elif mutation == "missing_hidden":
        raw.pop("hidden_size")
    elif mutation == "missing_experts":
        raw.pop("num_experts")
    else:
        raw["unknwon_shape"] = 1
    with pytest.raises(ValueError, match=message):
        normalize_native_checkpoint_config(raw, weights)


def test_missing_learned_weights_are_not_treated_as_reconstructible_buffers(artifacts):
    checkpoint, _ = artifacts
    weights = load_file(str(checkpoint / "model.safetensors"))
    weights.pop("embed_tokens.weight")
    save_file(weights, str(checkpoint / "model.safetensors"))
    before = digest(checkpoint / "model.safetensors")
    with pytest.raises(RuntimeError, match="missing=.*embed_tokens.weight"):
        load_native_checkpoint(checkpoint)
    assert digest(checkpoint / "model.safetensors") == before


def test_checkpoint_loading_refuses_a_global_dtype_that_rounds_trained_weights(artifacts):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with pytest.raises(ValueError, match="default dtype float32"):
            load_native_checkpoint(artifacts[0])
    finally:
        torch.set_default_dtype(previous)
