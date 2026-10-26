"""Canonical public model creation, strict validation and serialization."""
import json
import pytest
import torch
from complexity.api.model import Model
from complexity.config import ModelConfig


def small_config():
    return ModelConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                       max_position_embeddings=16, use_custom_kernels=False, use_cggr=False)


def test_create_copies_and_validates_without_changing_caller():
    original = small_config()
    before = original.to_dict()
    created = Model.create(original, num_hidden_layers=3)
    assert created.config is not original and original.to_dict() == before
    assert created.config.num_hidden_layers == 3
    assert created.config.routing_strategy == "token_id_composed_modulo"
    assert created.config.num_experts == created.config.num_attention_heads == 4
    assert created.config.top_k == 2 and created.config.num_key_value_heads == 1


@pytest.mark.parametrize("overrides", [{"num_experts":8},{"num_attention_heads":8},{"top_k_primary_weight":0.9},{"unknown_option":True}])
def test_invalid_overrides_fail_before_weight_allocation(monkeypatch, overrides):
    import complexity.api.model as api
    config = small_config()
    before = config.to_dict()
    monkeypatch.setattr(api, "ComplexityModel", lambda _: pytest.fail("invalid config allocated weights"))
    with pytest.raises(ValueError):
        Model.create(config, **overrides)
    assert config.to_dict() == before


def test_save_and_load_preserve_canonical_weights(tmp_path):
    model = Model.create(small_config()).eval()
    ids = torch.tensor([[1,3,7,5]])
    expected = model(ids)["logits"]
    model.save(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["routing_strategy"] == "token_id_composed_modulo"
    restored = Model.load(tmp_path).eval()
    torch.testing.assert_close(restored(ids)["logits"], expected, rtol=0, atol=0)


def test_checkpoint_requires_explicit_canonical_routing(tmp_path):
    model = Model.create(small_config())
    model.save(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text()); config.pop("routing_strategy")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="must declare"):
        Model.load(tmp_path)


def test_public_mlp_defaults_use_canonical_expert_count_and_widths():
    from complexity.api.core.mlp import MLP

    module = MLP.moe(hidden_size=16, vocab_size=32, use_custom_kernels=False)
    config = module.mlp.engine.config
    assert (config.num_experts, config.top_k) == (4, 2)
    assert (config.shared_width, config.expert_width) == (32, 4)
    output, auxiliary = module(
        torch.randn(1, 3, 16), token_ids=torch.tensor([[1, 2, 3]])
    )
    assert output.shape == (1, 3, 16)
    assert auxiliary.item() == 0
