import json

import torch

from scripts.convert_pt_to_mlx import (
    build_mlx_config,
    copy_tokenizer_files,
    remap_state_dict,
)


def test_build_mlx_config_maps_canonical_engine_to_mlx_routed_layout():
    config = build_mlx_config(
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 32,
            "mlp_type": "tr_hash_engine",
            "routing_strategy": "token_id_balanced_hash",
            "use_shared_routed_gates": True,
        }
    )

    assert config["mlp_type"] == "token_routed"
    assert config["routing_strategy"] == "token_id_balanced_hash"
    assert config["use_shared_routed_gates"] is False


def test_remap_state_dict_preserves_canonical_engine_routes_and_weights():
    routes = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
    expert_gate = torch.randn(4, 16, 2)
    shared_gate = torch.randn(32, 16)
    state = {
        "layers.0.mlp.engine.expert_gate": expert_gate,
        "layers.0.mlp.engine.shared_gate.weight": shared_gate,
        "layers.0.mlp.engine.route_table": routes,
        "layers.0.mlp.engine.fused_route_codes": torch.ones(4, dtype=torch.uint8),
        "layers.0.mlp.engine.fused_expert_pairs": torch.ones(2, 2, dtype=torch.int32),
    }

    converted = remap_state_dict(state, dtype="float16")

    assert set(converted) == {
        "model.layers.0.mlp.gate_proj_w",
        "model.layers.0.mlp.shared_gate.weight",
        "model.layers.0.mlp.topk_token_to_expert",
        "model.layers.0.mlp.token_to_expert",
    }
    assert converted["model.layers.0.mlp.gate_proj_w"].dtype == torch.float16
    assert converted["model.layers.0.mlp.shared_gate.weight"].dtype == torch.float16
    assert torch.equal(
        converted["model.layers.0.mlp.topk_token_to_expert"],
        routes.to(torch.int32),
    )
    assert torch.equal(
        converted["model.layers.0.mlp.token_to_expert"],
        routes[0].to(torch.int32),
    )
    assert (
        converted["model.layers.0.mlp.token_to_expert"].untyped_storage().data_ptr()
        != converted[
            "model.layers.0.mlp.topk_token_to_expert"
        ].untyped_storage().data_ptr()
    )


def test_copy_tokenizer_files_does_not_overwrite_mlx_config(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "config.json").write_text(
        json.dumps({"model_type": "deep"}), encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps({"model_type": "complexity"}), encoding="utf-8"
    )

    copied = copy_tokenizer_files(source, output)

    assert copied == ["tokenizer.json", "tokenizer_config.json"]
    assert json.loads((output / "config.json").read_text(encoding="utf-8")) == {
        "model_type": "complexity"
    }
