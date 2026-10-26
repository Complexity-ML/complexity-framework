"""Public HF helpers preserve TR-HASH weights and reject lossy dense conversion."""

import json

import pytest
import torch

from complexity import ComplexityModel, ModelConfig
from complexity.utils import hf_conversion as hf


def small_model():
    return ComplexityModel(
        ModelConfig(
            hidden_size=16,
            num_hidden_layers=2,
            vocab_size=32,
            max_position_embeddings=16,
            use_custom_kernels=False,
            use_cggr=False,
        )
    ).eval()


def test_public_hf_bundle_roundtrip_preserves_all_tensors_and_logits(tmp_path):
    from safetensors.torch import load_file
    from complexity.export.transformers_adapter.modeling_tr_hash_moe import TRHashForCausalLM

    model = small_model()
    output = tmp_path / "bundle"
    hf.save_to_huggingface(model, str(output))
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "tr_hash_moe"
    assert config["architectures"] == ["TRHashForCausalLM"]
    assert config["routing_strategy"] == "token_id_composed_modulo"
    for filename in (
        "configuration_tr_hash_moe.py",
        "modeling_tr_hash_moe.py",
        "composed_routing.py",
    ):
        assert (output / filename).is_file()
    tensors = load_file(str(output / "model.safetensors"))
    assert set(tensors) == set(model.state_dict())
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)

    restored, restored_config = hf.from_pretrained(str(output))
    target = small_model()
    assert hf.load_from_huggingface(str(output), target) is target
    exported = TRHashForCausalLM.from_pretrained(output).eval()
    ids = torch.tensor([[1, 5, 7, 9], [3, 2, 8, 6]])
    with torch.no_grad():
        expected = model(ids)["logits"]
        torch.testing.assert_close(restored.eval()(ids)["logits"], expected, rtol=0, atol=0)
        torch.testing.assert_close(target.eval()(ids)["logits"], expected, rtol=0, atol=0)
        torch.testing.assert_close(exported(ids).logits, expected, rtol=3e-5, atol=3e-6)
    assert restored_config.routing_strategy == "token_id_composed_modulo"


def test_tensor_conversion_preserves_experts_and_routing_anchors():
    state = small_model().state_dict()
    assert any(name.endswith(".expert_gate") for name in state)
    assert any(name.endswith(".route_table") for name in state)
    for conversion in (hf.convert_to_hf_state_dict, hf.convert_hf_state_dict):
        converted = conversion(state)
        assert converted.keys() == state.keys()
        assert all(converted[name] is tensor for name, tensor in state.items())
        with pytest.raises(ValueError, match="dense weight remapping was removed"):
            conversion(state, model_type="llama", num_layers=2)


def test_dense_source_is_rejected_without_allocating_a_model_or_changing_target(
    tmp_path, monkeypatch
):
    source = tmp_path / "dense"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 16}))
    target = small_model()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    monkeypatch.setattr(
        ComplexityModel,
        "__init__",
        lambda *args: pytest.fail("allocated a model for a dense checkpoint"),
    )
    with pytest.raises(ValueError, match="not a canonical TR-HASH"):
        hf.load_from_huggingface(str(source), target)
    with pytest.raises(ValueError, match="not a canonical TR-HASH"):
        hf.from_pretrained(str(source))
    for name, expected in before.items():
        torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)


def test_lossy_export_and_mismatched_config_fail_before_output_creation(tmp_path):
    model = small_model()
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="dense weight remapping was removed"):
        hf.save_to_huggingface(model, str(output), model_type="llama")
    altered = {**model.export_config(), "rope_theta": 12345}
    with pytest.raises(ValueError, match="must match"):
        hf.save_to_huggingface(model, str(output), config=altered)
    assert not output.exists()


def test_missing_expert_is_rejected_without_mutating_target(tmp_path):
    source, target = small_model(), small_model()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    output = tmp_path / "incomplete"
    source.save_pretrained(output, safe_serialization=False)
    weights = torch.load(output / "pytorch_model.bin", weights_only=True)
    weights.pop("layers.0.mlp.engine.expert_gate")
    torch.save(weights, output / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="expert_gate"):
        hf.load_from_huggingface(str(output), target)
    for name, expected in before.items():
        torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)


def test_non_main_rank_participates_in_native_save_without_exporting_or_publishing(
    tmp_path, monkeypatch
):
    import torch.distributed as dist

    model = small_model()
    calls = []
    native_save = model.save_pretrained

    def save_on_worker(path):
        calls.append(path)
        return native_save(path)

    monkeypatch.setattr(model, "save_pretrained", save_on_worker)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        hf, "require_optional", lambda *args, **kwargs: pytest.fail("worker accessed the Hub")
    )
    output = tmp_path / "bundle"
    assert hf.save_to_huggingface(model, str(output)) == str(output)
    assert hf.push_to_hub(model, "unused/tr-hash") is None
    assert len(calls) == 2
    assert not output.exists()
