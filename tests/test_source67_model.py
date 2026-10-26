"""CPU contracts for the Source67 default, exact reference, and routing cache."""

import json
from pathlib import Path

import pytest
import torch

from complexity import ComplexityModel, ModelConfig
from complexity.core.attention import AttentionConfig
from complexity.core.attention.gqa import GroupedQueryAttention


def small_config(**overrides):
    return ModelConfig(
        hidden_size=64, num_hidden_layers=2, vocab_size=256,
        max_position_embeddings=256, use_custom_kernels=False, **overrides,
    )


def reference_model():
    fixture = json.loads((Path(__file__).parent / "fixtures/source67_reference.json").read_text())
    namespace = {}
    # Source setup below this import changes process defaults and loads archive
    # paths. Only its definitions are needed for the independent model oracle.
    exec(fixture["best-config"].split("\nimport csv,")[0], namespace)
    exec(fixture["best-model"], namespace)
    exec(fixture["simple-head1-helper"], namespace)
    namespace["fixer_seed"](314)
    return namespace["ModeleLangage"](namespace["REFERENCE"], namespace["MODE_BEST42"])


def parameter_pairs(reference, native):
    yield reference.embedding.weight, native.embed_tokens.weight, False
    yield reference.norm.weight, native.norm.weight, False
    for old, new in zip(reference.layers, native.layers):
        yield old.norm1.weight, new.input_layernorm.weight, False
        yield old.norm2.weight, new.post_attention_layernorm.weight, False
        for old_name, new_name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj"), ("norm_q", "q_norm"), ("norm_k", "k_norm")):
            yield getattr(old.attention, old_name).weight, getattr(new.self_attn, new_name).weight, False
        for name in ("gate", "up", "down"):
            yield getattr(old.mlp.shared, name).weight, getattr(new.mlp.engine, "shared_" + name).weight, False
            for index, expert in enumerate(old.mlp.experts):
                yield getattr(expert, name).weight, getattr(new.mlp.engine, "expert_" + name)[index], True


def test_new_default_materializes_the_complete_source67_profile():
    config = small_config()
    assert (config.architecture_profile, config.architecture_profile_version) == ("source67", 1)
    assert (config.num_attention_heads, config.num_key_value_heads) == (4, 1)
    assert (config.num_experts, config.top_k, config.top_k_primary_weight) == (4, 2, 0.5)
    assert (config.intermediate_size, config.shared_intermediate_size) == (64, 128)
    assert (config.shared_output_scale, config.routed_output_scale) == (1, 2)
    assert config.routing_strategy == "token_id_composed_modulo"
    assert config.use_qk_norm and config.shared_expert
    assert ComplexityModel(config).num_parameters() == 110976


@pytest.mark.parametrize("overrides", [
    {"num_experts": 2}, {"top_k": 1}, {"top_k_primary_weight": 0.9},
    {"shared_output_scale": 2.0}, {"routed_output_scale": 1.0},
    {"architecture_profile_version": 2},
])
def test_explicit_source67_rejects_incompatible_recipe_or_version(overrides):
    with pytest.raises(ValueError):
        small_config(architecture_profile="source67", **overrides)


def test_historical_strategy_explicitly_selects_old_attention_and_scales():
    config = small_config(routing_strategy="modulo_cyclic", num_attention_heads=4)
    assert config.architecture_profile == "legacy"
    assert (config.num_experts, config.top_k, config.routed_output_scale) == (1, 1, 1)
    model = ComplexityModel(config)
    assert not any(layer.self_attn.source67 for layer in model.layers)


def test_exact_source67_logits_and_parameter_gradients():
    reference = reference_model().eval()
    native = ComplexityModel(small_config()).eval()
    assert reference.num_parameters() == native.num_parameters() == 110976
    with torch.no_grad():
        for old, new, transpose in parameter_pairs(reference, native):
            new.copy_(old.T if transpose else old)
    ids = torch.tensor([[0, 1, 3, 12, 19, 255, 44, 73, 2], [37, 91, 8, 4, 122, 11, 93, 13, 80]])
    expected = reference(ids)
    actual = native(ids)["logits"]
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    # Expert tensor slices are views; inspect the owning parameter's gradient.
    for old, new in zip(reference.layers, native.layers):
        for name in ("gate", "up", "down"):
            for index, expert in enumerate(old.mlp.experts):
                torch.testing.assert_close(
                    getattr(new.mlp.engine, "expert_" + name).grad[index],
                    getattr(expert, name).weight.grad.T, atol=2e-8, rtol=2e-4,
                )
    old_parameters = dict(reference.named_parameters())
    new_parameters = dict(native.named_parameters())
    direct = [("embedding.weight", "embed_tokens.weight"), ("norm.weight", "norm.weight")]
    for index in range(2):
        prefix = f"layers.{index}."
        direct.extend((prefix + old, prefix + new) for old, new in (
            ("norm1.weight", "input_layernorm.weight"), ("norm2.weight", "post_attention_layernorm.weight"),
            *((f"attention.{a}.weight", f"self_attn.{b}.weight") for a, b in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj"), ("norm_q", "q_norm"), ("norm_k", "k_norm"))),
            *((f"mlp.shared.{name}.weight", f"mlp.engine.shared_{name}.weight") for name in ("gate", "up", "down")),
        ))
    for old_name, new_name in direct:
        torch.testing.assert_close(new_parameters[new_name].grad, old_parameters[old_name].grad, atol=2e-8, rtol=2e-4)


@pytest.mark.parametrize("chunk_sizes", [(1, 1, 1, 1, 1, 1, 1), (2, 3, 2), (1, 4, 2)])
@pytest.mark.parametrize("use_sdpa", [True, False])
def test_source67_cache_matches_full_forward_across_chunk_boundaries(chunk_sizes, use_sdpa):
    model = ComplexityModel(small_config(use_sdpa=use_sdpa)).eval()
    ids = torch.tensor([[17, 5, 92, 13, 46, 39, 8], [29, 1, 54, 24, 19, 7, 32]])
    with torch.no_grad():
        expected = model(ids)["logits"]
        past, pieces, offset = None, [], 0
        for size in chunk_sizes:
            output = model(ids[:, offset:offset + size], past_key_values=past, use_cache=True)
            pieces.append(output["logits"])
            past = output["past_key_values"]
            offset += size
            for key, value, history in past:
                assert key.shape == value.shape == (2, 1, offset, 16)
                expected_history = torch.cat((torch.zeros((2, 2), dtype=torch.long), ids[:, :offset]), dim=1)[:, -2:]
                assert torch.equal(history, expected_history)
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=3e-7, rtol=3e-6)


def test_source67_rejects_missing_original_ids_and_incomplete_cache():
    model = ComplexityModel(small_config()).eval()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="explicit routing_ids"):
        model(inputs_embeds=model.embed_tokens(ids))
    with pytest.raises(ValueError, match="torch.long"):
        model(inputs_embeds=model.embed_tokens(ids), routing_ids=ids.float())
    with pytest.raises(ValueError, match="vocabulary"):
        model(ids, routing_ids=ids + 256)
    prefix = model(ids[:, :2], use_cache=True)["past_key_values"]
    with pytest.raises(ValueError, match="legacy KV cache"):
        model(ids[:, 2:], past_key_values=[entry[:2] for entry in prefix])


def test_source67_can_start_from_an_empty_legacy_cache_allocation():
    model = ComplexityModel(small_config()).eval()
    ids = torch.tensor([[1, 2, 3]])
    empty = [(torch.empty((1, 1, 0, 16)), torch.empty((1, 1, 0, 16))) for _ in model.layers]
    actual = model(ids, past_key_values=empty, use_cache=True)
    torch.testing.assert_close(actual["logits"], model(ids)["logits"], atol=0, rtol=0)
    assert torch.equal(actual["past_key_values"][0][2], ids[:, -2:])


def test_original_routing_ids_take_precedence_with_external_embeddings():
    model = ComplexityModel(small_config()).eval()
    ids = torch.tensor([[3, 12, 19, 44]])
    embeddings = model.embed_tokens(ids)
    expected = model(inputs_embeds=embeddings, routing_ids=ids)["logits"]
    actual = model(torch.zeros_like(ids), inputs_embeds=embeddings, routing_ids=ids)["logits"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.allclose(actual, model(inputs_embeds=embeddings, routing_ids=torch.zeros_like(ids))["logits"])


@pytest.mark.parametrize("heads,kv", [(8, 1), (12, 3), (16, 4)])
def test_larger_head_counts_are_supported_as_explicit_unmeasured_extensions(heads, kv):
    model = ComplexityModel(ModelConfig(hidden_size=8 * heads, vocab_size=32, num_hidden_layers=3,
        num_attention_heads=heads, num_key_value_heads=kv, use_custom_kernels=False))
    assert model(torch.tensor([[1, 12, 25]]))["logits"].shape == (1, 3, 32)


def test_default_profile_roundtrip_and_unversioned_legacy_loading(tmp_path):
    model = ComplexityModel(small_config()).eval()
    ids = torch.tensor([[1, 12, 3]])
    model.save_pretrained(tmp_path / "source")
    restored = ComplexityModel.from_pretrained(tmp_path / "source").eval()
    assert restored.config.to_dict() == model.config.to_dict()
    torch.testing.assert_close(restored(ids)["logits"], model(ids)["logits"], atol=0, rtol=0)
    legacy = ComplexityModel(ModelConfig(architecture_profile="legacy", hidden_size=48, vocab_size=32, num_hidden_layers=1)).eval()
    assert (legacy.config.num_attention_heads, legacy.config.num_key_value_heads) == (12, 3)
    assert (legacy.config.num_experts, legacy.config.top_k, legacy.config.intermediate_size) == (1, 1, 256)
    legacy.save_pretrained(tmp_path / "legacy")
    config_path = tmp_path / "legacy/config.json"
    old_config = json.loads(config_path.read_text())
    for key in ("architecture_profile", "architecture_profile_version", "num_attention_heads", "num_key_value_heads", "num_experts", "top_k", "intermediate_size", "routed_output_scale"):
        old_config.pop(key)
    config_path.write_text(json.dumps(old_config))
    restored_legacy = ComplexityModel.from_pretrained(tmp_path / "legacy").eval()
    assert restored_legacy.config.architecture_profile == "legacy"
    torch.testing.assert_close(restored_legacy(ids)["logits"], legacy(ids)["logits"], atol=0, rtol=0)
    assert len(restored_legacy(ids, use_cache=True)["past_key_values"][0]) == 2


@pytest.mark.parametrize("sdpa", [True, False])
def test_gqa_combines_causal_and_padding_masks_with_cached_chunks(sdpa):
    attention = GroupedQueryAttention(AttentionConfig(hidden_size=16, num_attention_heads=2,
        num_key_value_heads=1, use_sdpa=sdpa)).eval()
    x = torch.randn(2, 6, 16)
    padding = torch.tensor([[1, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        full = attention(x, attention_mask=padding)[0]
        prefix, cache = attention(x[:, :2], attention_mask=padding[:, :2], use_cache=True)
        suffix = attention(x[:, 2:], attention_mask=padding, past_key_value=cache)[0]
    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), full, atol=2e-7, rtol=3e-6)


def test_gradient_checkpointing_preserves_source67_forward_and_backward():
    model = ComplexityModel(small_config()).train()
    model.gradient_checkpointing_enable()
    output = model(torch.tensor([[3, 1, 12, 7]]))["logits"]
    output.square().mean().backward()
    assert model.layers[1].mlp.engine.expert_down.grad is not None
