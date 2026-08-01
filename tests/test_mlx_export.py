from __future__ import annotations

import json

import torch

from complexity.inference.chat_template import default_chat_template
from scripts.convert_pt_to_mlx import (
    build_mlx_config,
    remap_state_dict,
    write_chat_template,
)


def test_mlx_config_keeps_tr_hash_fields() -> None:
    config = build_mlx_config(
        {
            "hidden_size": 64,
            "vocab_size": 256,
            "routing_strategy": "token_id_balanced_hash",
            "learn_hash_channel_modulation": True,
            "routed_output_scale": 2.0,
            "ignored": "not portable",
        }
    )
    assert config == {
        "model_type": "complexity",
        "hidden_size": 64,
        "vocab_size": 256,
        "routing_strategy": "token_id_balanced_hash",
        "learn_hash_channel_modulation": True,
        "routed_output_scale": 2.0,
        "rope_traditional": False,
    }


def test_mlx_state_remap_casts_route_tables_and_drops_rope() -> None:
    remapped = remap_state_dict(
        {
            "layers.0.mlp.topk_token_to_expert": torch.tensor([[0, 1]]),
            "layers.0.self_attn.rotary_emb.inv_freq": torch.ones(2),
            "embed_tokens.weight": torch.ones(2, 3),
        },
        dtype="float16",
    )
    assert set(remapped) == {
        "model.layers.0.mlp.topk_token_to_expert",
        "model.embed_tokens.weight",
    }
    assert remapped["model.layers.0.mlp.topk_token_to_expert"].dtype == torch.int32
    assert remapped["model.embed_tokens.weight"].dtype == torch.float16


def test_mlx_export_preserves_chat_template(tmp_path) -> None:
    template = default_chat_template()
    path = write_chat_template({"chat_template": template}, tmp_path)
    assert path == tmp_path / "chat_template.json"
    assert json.loads(path.read_text()) == template

