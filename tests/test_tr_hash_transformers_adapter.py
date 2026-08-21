import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from complexity import ComplexityModel, ModelConfig
from scripts.export_tr_hash_transformers import export_bundle

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
        num_key_value_heads=2,
        intermediate_size=16,
        shared_intermediate_size=48,
        vocab_size=64,
        max_position_embeddings=64,
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        routing_strategy="token_id_multi_hash",
        route_hash_count=2,
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
    assert config.model_type == "tr_hash_moe"
    assert config.num_experts_per_tok == 2
    assert "top_k" not in config.to_dict()
    with safe_open(output / "model.safetensors", framework="pt") as checkpoint:
        assert checkpoint.metadata()["format"] == "pt"
    with torch.no_grad():
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
