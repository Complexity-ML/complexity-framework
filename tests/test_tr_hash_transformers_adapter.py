import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from complexity import ComplexityModel, ModelConfig
from complexity.export.transformers import build_transformers_config, export_bundle

ADAPTER = Path("integrations/transformers/tr_hash_moe")


def _load_adapter_modules():
    package_name = "tr_hash_adapter_test"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        ADAPTER / "__init__.py",
        submodule_search_locations=[str(ADAPTER)],
    )
    package = importlib.util.module_from_spec(package_spec)
    import sys

    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    return package.TRHashConfig, package.TRHashForCausalLM


def _tiny_config():
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        shared_intermediate_size=48,
        vocab_size=64,
        max_position_embeddings=64,
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        routing_strategy="token_id_composed_modulo",
        use_custom_kernels=False,
        use_cggr=False,
    )


def test_adapter_logits_and_cache_match_native_model():
    TRHashConfig, TRHashForCausalLM = _load_adapter_modules()
    native = ComplexityModel(_tiny_config()).eval()
    raw_config = native.config.to_dict()
    raw_config["num_experts_per_tok"] = raw_config.pop("top_k")
    config = TRHashConfig(**raw_config)
    adapted = TRHashForCausalLM(config).eval()
    adapted.load_state_dict(native.state_dict(), strict=True)
    input_ids = torch.tensor([[1, 7, 13, 2]])

    with torch.no_grad():
        expected = native(input_ids)["logits"]
        actual = adapted(input_ids).logits
        cached = adapted(input_ids[:, :3], use_cache=True)
        final = adapted(
            input_ids[:, 3:],
            past_key_values=cached.past_key_values,
            use_cache=True,
        ).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final[:, -1], expected[:, -1], rtol=1e-5, atol=1e-6)


def test_adapter_multi_token_cache_preserves_causality():
    TRHashConfig, TRHashForCausalLM = _load_adapter_modules()
    native = ComplexityModel(_tiny_config()).eval()
    raw_config = native.config.to_dict()
    raw_config["num_experts_per_tok"] = raw_config.pop("top_k")
    adapted = TRHashForCausalLM(TRHashConfig(**raw_config)).eval()
    adapted.load_state_dict(native.state_dict(), strict=True)
    input_ids = torch.tensor([[1, 7, 13, 2]])

    with torch.no_grad():
        full = adapted(input_ids).logits
        prefix = adapted(input_ids[:, :2], use_cache=True)
        continuation = adapted(
            input_ids[:, 2:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits

    torch.testing.assert_close(
        continuation,
        full[:, 2:],
        rtol=1e-5,
        atol=1e-5,
    )


def test_exported_bundle_loads_through_auto_model(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(
        transformers.dynamic_module_utils,
        "HF_MODULES_CACHE",
        str(tmp_path / "hf-modules"),
    )
    native = ComplexityModel(_tiny_config()).eval()
    source = tmp_path / "native"
    native.save_pretrained(source)
    output = tmp_path / "hub"
    export_bundle(
        config_path=source / "config.json",
        weights_path=source / "model.safetensors",
        output=output,
    )

    config = transformers.AutoConfig.from_pretrained(output, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True).eval()
    raw_exported_config = __import__("json").loads(
        (output / "config.json").read_text(encoding="utf-8")
    )
    assert config.model_type == "tr_hash_moe"
    assert config.num_experts_per_tok == 2
    assert raw_exported_config["top_k"] == 2
    assert raw_exported_config["num_experts_per_tok"] == 2
    assert raw_exported_config["top_k_primary_weight"] == 0.5
    assert "top_k" not in config.to_dict()
    with safe_open(output / "model.safetensors", framework="pt") as checkpoint:
        assert checkpoint.metadata()["format"] == "pt"
    with torch.no_grad():
        ids = torch.tensor([[1, 5, 9]])
        torch.testing.assert_close(model(ids).logits, native(ids)["logits"], rtol=1e-5, atol=1e-6)
        output_ids = model.generate(
            torch.tensor([[1, 5, 9]]),
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
    assert output_ids.shape == (1, 5)


def test_export_infers_special_token_ids(tmp_path):
    native = ComplexityModel(_tiny_config()).eval()
    source = tmp_path / "native"
    native.save_pretrained(source)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text(
        """{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "pad_token": "<pad>",
  "unk_token": "<unk>",
  "added_tokens_decoder": {
    "0": {"content": "</s>"},
    "1": {"content": "<pad>"},
    "2": {"content": "<s>"},
    "3": {"content": "<unk>"}
  }
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "hub"
    export_bundle(
        config_path=source / "config.json",
        weights_path=source / "model.safetensors",
        output=output,
        tokenizer_dir=tokenizer,
    )

    import json

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["bos_token_id"] == 2
    assert config["eos_token_id"] == 0
    assert config["pad_token_id"] == 1
    assert config["unk_token_id"] == 3


@pytest.mark.parametrize(
    "override",
    [
        {"attention_type": "lexical_gqa"},
        {"tie_word_embeddings": False},
        {"top_k": 1},
        {"learn_hash_pair_gates": True},
        {"active_expert_width": 2},
        {"layer_mlp_controls": []},
        {"rope_type": "yarn"},
    ],
)
def test_export_rejects_unsupported_runtime_behavior(override):
    config = _tiny_config().to_dict()
    config.update(override)
    with pytest.raises(ValueError, match="Unsupported Transformers export"):
        build_transformers_config(config)


def test_export_preserves_source_bytes_and_requires_a_separate_destination(tmp_path):
    import hashlib

    native = ComplexityModel(_tiny_config()).eval()
    source = tmp_path / "native"
    native.save_pretrained(source)
    config_file = source / "config.json"
    weights_file = source / "model.safetensors"
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in [config_file, weights_file]]
    with pytest.raises(ValueError, match="separate from source"):
        export_bundle(config_path=config_file, weights_path=weights_file, output=source)
    output = tmp_path / "export"
    export_bundle(config_path=config_file, weights_path=weights_file, output=output)
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in [config_file, weights_file]
    ] == hashes
    with (
        safe_open(weights_file, framework="pt") as original,
        safe_open(output / "model.safetensors", framework="pt") as exported,
    ):
        assert original.keys() == exported.keys()
        for key in original.keys():
            assert torch.equal(original.get_tensor(key), exported.get_tensor(key))


def test_current_1b_recipe_is_in_export_supported_family():
    import json

    config = json.loads(Path("scripts/pretrain_1b_fsdp/model.json").read_text())
    exported = build_transformers_config(config)
    assert exported["num_experts_per_tok"] == 2
    assert exported["num_experts"] == 4
    assert exported["hidden_size"] == 1536


def test_adapter_embeddings_require_ids_and_match_native_gradients():
    from torch.nn import functional as F

    Config, CausalLM = _load_adapter_modules()
    native = ComplexityModel(_tiny_config()).eval()
    adapted = CausalLM(Config(**native.config.to_dict())).eval()
    adapted.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 7, 3, 5]])
    embeddings = adapted.get_input_embeddings()(ids)
    with pytest.raises(ValueError, match="routing_ids"):
        adapted(inputs_embeds=embeddings)
    actual = adapted(inputs_embeds=embeddings, routing_ids=ids, labels=ids, use_cache=False)
    expected = native(ids)["logits"]
    loss = F.cross_entropy(expected[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(actual.logits, expected, rtol=1e-5, atol=1e-6)
    actual.loss.backward()
    loss.backward()
    for name, parameter in native.named_parameters():
        gradient = dict(adapted.named_parameters())[name].grad
        if parameter.grad is None:
            assert gradient is None
        else:
            torch.testing.assert_close(gradient, parameter.grad, rtol=1e-4, atol=1e-6)


def test_adapter_cache_beam_reorder_and_crop_preserve_routing_ids():
    Config, CausalLM = _load_adapter_modules()
    native = ComplexityModel(_tiny_config()).eval()
    model = CausalLM(Config(**native.config.to_dict())).eval()
    model.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 7, 3, 5], [8, 4, 2, 6]])
    with torch.inference_mode():
        cache = model(ids[:, :3], use_cache=True).past_key_values
        order = torch.tensor([1, 0])
        cache = model._reorder_cache(cache, order)
        actual = model(ids[order, 3:], past_key_values=cache, use_cache=True).logits
        expected = model(ids[order]).logits[:, 3:]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        if hasattr(cache, "crop"):
            cache.crop(2)
            actual = model(ids[order, 2:], past_key_values=cache, use_cache=True).logits
            torch.testing.assert_close(
                actual, model(ids[order]).logits[:, 2:], rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize("legacy_tuple_cache", (False, True))
def test_generation_preparation_uses_decoded_ids_and_preserves_prompt_history(legacy_tuple_cache):
    Config, CausalLM = _load_adapter_modules()
    native = ComplexityModel(_tiny_config()).eval()
    model = CausalLM(Config(**native.config.to_dict())).eval()
    model.load_state_dict(native.state_dict(), strict=True)
    prompt = torch.tensor([[1, 7, 3], [8, 4, 2]])
    prompt_routes = torch.tensor([[11, 22, 35], [6, 27, 42]])
    sequence = prompt.clone()
    routes = prompt_routes.clone()
    with torch.inference_mode():
        cache = model(prompt, routing_ids=prompt_routes, use_cache=True).past_key_values
        if legacy_tuple_cache and hasattr(cache, "routing_history"):
            key_values = (
                cache.to_legacy_cache()
                if hasattr(cache, "to_legacy_cache")
                else tuple((layer.keys, layer.values) for layer in cache.layers)
            )
            cache = tuple((*pair, cache.routing_history.clone()) for pair in key_values)
        for decoded in (torch.tensor([[9], [13]]), torch.tensor([[17], [23]])):
            sequence = torch.cat((sequence, decoded), dim=1)
            routes = torch.cat((routes, decoded), dim=1)
            prepared = model.prepare_inputs_for_generation(
                sequence,
                past_key_values=cache,
                routing_ids=prompt_routes,
                attention_mask=torch.ones_like(sequence),
                use_cache=True,
            )
            assert torch.equal(prepared["input_ids"], decoded)
            assert torch.equal(prepared["routing_ids"], decoded)
            assert prepared["past_key_values"] is cache
            expected_history = routes[:, -3:-1]
            if hasattr(cache, "routing_history"):
                assert torch.equal(cache.routing_history, expected_history)
            else:
                assert all(torch.equal(layer[2], expected_history) for layer in cache)
            actual = model(**prepared)
            expected = model(sequence, routing_ids=routes, use_cache=False).logits[:, -1:]
            torch.testing.assert_close(actual.logits, expected, atol=1e-6, rtol=1e-5)
            cache = actual.past_key_values
            if hasattr(cache, "routing_history"):
                assert torch.equal(cache.routing_history, routes[:, -2:])
            else:
                assert all(torch.equal(layer[2], routes[:, -2:]) for layer in cache)


@pytest.mark.parametrize("use_cache", (False, True))
def test_generate_with_explicit_prompt_routes_matches_full_sequence_scores(use_cache):
    Config, CausalLM = _load_adapter_modules()
    native = ComplexityModel(_tiny_config()).eval()
    model = CausalLM(Config(**native.config.to_dict())).eval()
    model.load_state_dict(native.state_dict(), strict=True)
    prompt = torch.tensor([[1, 7, 3]])
    prompt_routes = torch.tensor([[11, 22, 35]])
    expected = prompt.clone()
    expected_scores = []
    with torch.inference_mode():
        for _ in range(3):
            routes = torch.cat((prompt_routes, expected[:, prompt.shape[1] :]), dim=1)
            scores = model(expected, routing_ids=routes, use_cache=False).logits[:, -1]
            expected_scores.append(scores)
            expected = torch.cat((expected, scores.argmax(dim=-1, keepdim=True)), dim=1)
        actual = model.generate(
            prompt,
            routing_ids=prompt_routes,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=3,
            do_sample=False,
            use_cache=use_cache,
            eos_token_id=None,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_scores=True,
        )
    assert torch.equal(actual.sequences, expected)
    for actual_scores, expected_step in zip(actual.scores, expected_scores):
        torch.testing.assert_close(actual_scores, expected_step, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("routing_marker", (None, "token_id_balanced_hash"))
def test_adapter_config_reload_rejects_missing_or_obsolete_routing_marker(tmp_path, routing_marker):
    import json

    Config, _ = _load_adapter_modules()
    raw = _tiny_config().to_dict()
    if routing_marker is None:
        raw.pop("routing_strategy")
    else:
        raw["routing_strategy"] = routing_marker
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="explicitly declare routing_strategy"):
        Config.from_dict(raw)
    with pytest.raises(ValueError, match="explicitly declare routing_strategy"):
        Config.from_pretrained(tmp_path)


@pytest.mark.parametrize("corruption", ("values", "dtype", "missing"))
def test_adapter_rejects_noncanonical_anchors_before_assigning_weights(tmp_path, corruption):
    from safetensors.torch import load_file, save_file

    Config, CausalLM = _load_adapter_modules()
    model = CausalLM(Config(**_tiny_config().to_dict())).eval()
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    corrupted = {key: value.clone() for key, value in original.items()}
    key = "layers.0.mlp.engine.route_table"
    if corruption == "values":
        corrupted[key] = corrupted[key].roll(1, dims=1)
    elif corruption == "dtype":
        corrupted[key] = corrupted[key].to(torch.int32)
    else:
        corrupted.pop(key)
    corrupted["embed_tokens.weight"].fill_(7)
    with pytest.raises(RuntimeError, match="canonical zero-history routing anchor"):
        model.load_state_dict(corrupted)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())

    model.save_pretrained(tmp_path)
    checkpoint_path = tmp_path / "model.safetensors"
    saved = load_file(checkpoint_path)
    if corruption == "values":
        saved[key] = saved[key].roll(1, dims=1)
    elif corruption == "dtype":
        saved[key] = saved[key].to(torch.int32)
    else:
        saved.pop(key)
    save_file(saved, checkpoint_path, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="canonical zero-history routing anchor"):
        CausalLM.from_pretrained(tmp_path)
