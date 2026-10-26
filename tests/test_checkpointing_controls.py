"""Native runtime controls survive generic checkpoints without tensor changes."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from complexity import ComplexityModel, ModelConfig
from complexity.utils.checkpointing import CheckpointManager, TrainingState


def _model():
    return ComplexityModel(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=32,
            shared_intermediate_size=16,
            mlp_type="tr_hash_engine",
            num_experts=4,
            top_k=2,
            use_custom_kernels=False,
        )
    )


def _set_controls(model, *, uniform=False):
    controls = ((0.5, 4, 4), (0.5, 4, 6))
    for index, layer in enumerate(model.layers):
        primary, experts, width = controls[0 if uniform else index]
        layer.mlp.set_top_k_primary_weight(primary)
        layer.mlp.set_active_experts(experts)
        layer.mlp.set_active_expert_width(width)


def _optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return optimizer, scheduler


def _update(model, optimizer, scheduler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = torch.tensor([[2, 3, 4, 0], [6, 7, 8, 0]])
    F.cross_entropy(model(inputs)["logits"].reshape(-1, 32), labels.reshape(-1)).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


def _logits(model):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor([[1, 2, 3, 4]]))["logits"].clone()


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("transactional", [False, True])
def test_checkpoint_restores_controls_logits_and_next_adamw_update(
    tmp_path, uniform, transactional
):
    torch.manual_seed(129)
    source = _model()
    _set_controls(source, uniform=uniform)
    # Every restored checkpoint must retain the canonical routing anchor.
    anchors = [layer.mlp.engine.route_table.clone() for layer in source.layers]
    assert all(
        torch.equal(layer.mlp.engine.route_table, anchor)
        for layer, anchor in zip(source.layers, anchors)
    )
    optimizer, scheduler = _optimizer(source)
    _update(source, optimizer, scheduler)
    saved_config = copy.deepcopy(source.config.to_dict())
    saved_controls = source.export_config()
    saved_optimizer, saved_scheduler = (
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
    )
    expected_logits = _logits(source)
    manager = CheckpointManager(str(tmp_path / "source"), source, optimizer, scheduler)
    if transactional:
        manager.resume_contract = {"fixture": "runtime-controls"}
    state = TrainingState(step=1, epoch=2, total_tokens=16, learning_rate=0.0016)
    checkpoint = Path(manager.save(1, state))
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    metadata = json.loads((checkpoint / "runtime_mlp_controls.json").read_text())
    assert payload["runtime_mlp_controls"] == metadata
    assert metadata["model_config"] == saved_controls
    assert source.config.to_dict() == saved_config
    assert json.loads((checkpoint / "training_state.json").read_text()) == state.to_dict()
    _update(source, optimizer, scheduler)
    expected_model, expected_optimizer = (
        copy.deepcopy(source.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )

    torch.manual_seed(551)
    restored = _model()
    fresh_config = copy.deepcopy(restored.config.to_dict())
    loaded_optimizer, loaded_scheduler = _optimizer(restored)
    loader = CheckpointManager(
        str(tmp_path / "restore"), restored, loaded_optimizer, loaded_scheduler
    )
    if transactional:
        loader.resume_contract = manager.resume_contract
    assert loader.load(str(checkpoint)).to_dict() == state.to_dict()
    assert restored.export_config() == saved_controls
    assert restored.config.to_dict() == fresh_config
    assert torch.equal(_logits(restored), expected_logits)
    _assert_tree_equal(loaded_optimizer.state_dict(), saved_optimizer)
    _assert_tree_equal(loaded_scheduler.state_dict(), saved_scheduler)
    _update(restored, loaded_optimizer, loaded_scheduler)
    _assert_tree_equal(restored.state_dict(), expected_model)
    _assert_tree_equal(loaded_optimizer.state_dict(), expected_optimizer)


@pytest.mark.parametrize("native_export", [False, True])
def test_weights_only_restores_controls_and_leaves_optimizer_scheduler_untouched(
    tmp_path, native_export
):
    source = _model()
    _set_controls(source)
    if native_export:
        checkpoint = tmp_path / "native"
        source.save_pretrained(checkpoint)
    else:
        checkpoint = Path(CheckpointManager(str(tmp_path / "checkpoints"), source).save(1))
        assert not (checkpoint / "config.json").exists()
    restored = _model()
    optimizer, scheduler = _optimizer(restored)
    _update(restored, optimizer, scheduler)
    before_optimizer, before_scheduler = (
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
    )
    manager = CheckpointManager(str(tmp_path / "restore"), restored, optimizer, scheduler)
    manager.load_weights_only(str(checkpoint))
    assert torch.equal(_logits(restored), _logits(source))
    assert restored.export_config() == source.export_config()
    _assert_tree_equal(optimizer.state_dict(), before_optimizer)
    _assert_tree_equal(scheduler.state_dict(), before_scheduler)


@pytest.mark.parametrize(
    "mutation",
    [
        "format",
        "late_layer",
        "dimensions",
        "nonfinite",
        "tensor",
        "unknown",
        "null",
        "missing_route",
    ],
)
def test_invalid_control_metadata_is_rejected_before_model_or_optimizer_mutation(
    tmp_path, mutation
):
    source = _model()
    _set_controls(source)
    checkpoint = Path(CheckpointManager(str(tmp_path / "source"), source).save(1))
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    controls = payload["runtime_mlp_controls"]
    if mutation == "format":
        controls["format"] = "unsupported"
    elif mutation == "late_layer":
        controls["model_config"]["layer_mlp_controls"][1]["active_expert_width"] = 999
    elif mutation == "dimensions":
        controls["model_config"]["hidden_size"] *= 2
    elif mutation == "nonfinite":
        controls["model_config"]["layer_mlp_controls"][1]["top_k_primary_weight"] = float("nan")
    elif mutation == "tensor":
        controls["model_config"]["layer_mlp_controls"][1]["top_k_primary_weight"] = torch.tensor(
            0.2
        )
    elif mutation == "unknown":
        controls["model_config"]["not_a_model_field"] = 1
    elif mutation == "null":
        payload["runtime_mlp_controls"] = None
    else:
        payload["model"].pop("layers.1.mlp.engine.route_table")
    torch.save(payload, checkpoint / "checkpoint.pt")
    # Removing the JSON mirror isolates payload validation, including refusal
    # of tensor-valued metadata despite its safe Torch deserialization.
    (checkpoint / "runtime_mlp_controls.json").unlink()
    target = _model()
    optimizer, scheduler = _optimizer(target)
    _update(target, optimizer, scheduler)
    before_model, before_controls = copy.deepcopy(target.state_dict()), target.export_config()
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    loader = CheckpointManager(str(tmp_path / "target"), target, optimizer, scheduler)
    with pytest.raises((ValueError, TypeError)):
        loader.load(str(checkpoint))
    _assert_tree_equal(target.state_dict(), before_model)
    assert target.export_config() == before_controls
    _assert_tree_equal(optimizer.state_dict(), before_optimizer)


def test_checkpoint_without_optional_width_controls_retains_current_width(tmp_path):
    source = _model()
    checkpoint = Path(CheckpointManager(str(tmp_path / "source"), source).save(1))
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    payload.pop("runtime_mlp_controls")
    torch.save(payload, checkpoint / "checkpoint.pt")
    (checkpoint / "runtime_mlp_controls.json").unlink()
    restored = _model()
    _set_controls(restored)
    controls = restored.export_config()
    CheckpointManager(str(tmp_path / "restore"), restored).load(str(checkpoint))
    assert restored.export_config() == controls
    _assert_tree_equal(restored.state_dict(), source.state_dict())


def test_plain_torch_model_payload_and_training_metadata_are_unchanged(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    manager = CheckpointManager(str(tmp_path), model, optimizer, scheduler)
    state = TrainingState(step=3, epoch=1, best_loss=2.5, total_tokens=24)
    checkpoint = Path(manager.save(3, state))
    assert set(torch.load(checkpoint / "checkpoint.pt", weights_only=True)) == {
        "model",
        "scheduler",
    }
    assert not (checkpoint / "runtime_mlp_controls.json").exists()
    assert json.loads((checkpoint / "training_state.json").read_text()) == state.to_dict()
    with torch.no_grad():
        model.weight.zero_()
    assert manager.load(str(checkpoint)).to_dict() == state.to_dict()


def test_plain_model_replacing_tag_does_not_retain_old_control_metadata(tmp_path):
    CheckpointManager(str(tmp_path), _model()).save(1)
    model = torch.nn.Linear(3, 2)
    manager = CheckpointManager(str(tmp_path), model)
    checkpoint = Path(manager.save(1))
    assert not (checkpoint / "runtime_mlp_controls.json").exists()
    assert "runtime_mlp_controls" not in torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    manager.load(str(checkpoint))


def test_payload_and_json_control_snapshots_must_agree_before_copy(tmp_path):
    source = _model()
    checkpoint = Path(CheckpointManager(str(tmp_path / "source"), source).save(1))
    path = checkpoint / "runtime_mlp_controls.json"
    metadata = json.loads(path.read_text())
    metadata["model_config"]["active_expert_width"] = 3
    path.write_text(json.dumps(metadata))
    target = _model()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="metadata disagree"):
        CheckpointManager(str(tmp_path / "target"), target).load(str(checkpoint))
    _assert_tree_equal(target.state_dict(), before)
