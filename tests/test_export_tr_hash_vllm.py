from scripts.export_tr_hash_vllm import build_config, vllm_tensor_name


def _engine_config():
    return {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 16,
        "vocab_size": 128,
        "mlp_type": "tr_hash_engine",
        "use_shared_routed_gates": True,
    }


def test_tr_hash_engine_config_uses_compatible_vllm_runtime_contract():
    config = build_config(_engine_config())

    assert config["source_mlp_type"] == "tr_hash_engine"
    assert config["mlp_type"] == "token_routed"
    assert config["use_shared_routed_gates"] is False


def test_tr_hash_engine_tensors_map_to_token_routed_runtime_names():
    prefix = "layers.0.mlp.engine"
    assert vllm_tensor_name(f"{prefix}.expert_gate") == (
        "layers.0.mlp.gate_proj_w"
    )
    assert vllm_tensor_name(f"{prefix}.expert_up") == "layers.0.mlp.up_proj_w"
    assert vllm_tensor_name(f"{prefix}.expert_down") == (
        "layers.0.mlp.down_proj_w"
    )
    assert vllm_tensor_name(f"{prefix}.route_table") == (
        "layers.0.mlp.topk_token_to_expert"
    )
    assert vllm_tensor_name(f"{prefix}.shared_gate.weight") == (
        "layers.0.mlp.shared_gate.weight"
    )
