from __future__ import annotations

import json

import torch

from complexity.tr_hash import TRHashEngineConfig, TRHashStrategy
from complexity.tr_hash.routing import build_route_table, compile_top2_pair_metadata
from scripts.migrate_tr_hash_vocab_32004 import (
    NEW_VOCAB_SIZE,
    OLD_VOCAB_SIZE,
    expand_routing_state,
    update_tokenizer_metadata,
)


def test_expansion_preserves_every_old_route_and_fused_code() -> None:
    raw = {
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "intermediate_size": 8,
        "num_experts": 4,
        "top_k": 2,
        "shared_intermediate_size": 32,
        "initializer_range": 0.02,
        "routing_strategy": "token_id_multi_hash",
        "route_hash_count": 2,
        "shared_output_scale": 1.0,
        "routed_output_scale": 2.0,
    }
    state: dict[str, torch.Tensor] = {}
    old_snapshots: dict[str, torch.Tensor] = {}
    for layer in range(raw["num_hidden_layers"]):
        config = TRHashEngineConfig(
            hidden_size=raw["hidden_size"],
            vocab_size=OLD_VOCAB_SIZE,
            num_experts=raw["num_experts"],
            top_k=raw["top_k"],
            shared_width=raw["shared_intermediate_size"],
            expert_width=raw["intermediate_size"] // raw["num_experts"],
            routing_strategy=TRHashStrategy.MULTI_HASH,
            layer_index=layer,
            route_hash_count=raw["route_hash_count"],
        )
        routes = build_route_table(config)
        codes, pairs = compile_top2_pair_metadata(routes, num_experts=raw["num_experts"])
        prefix = f"layers.{layer}.mlp.engine"
        state[f"{prefix}.route_table"] = routes
        state[f"{prefix}.fused_route_codes"] = codes
        state[f"{prefix}.fused_expert_pairs"] = pairs
        old_snapshots.update({key: value.clone() for key, value in state.items()})

    report = expand_routing_state(state, raw)

    assert len(report) == raw["num_hidden_layers"]
    for layer in range(raw["num_hidden_layers"]):
        prefix = f"layers.{layer}.mlp.engine"
        assert state[f"{prefix}.route_table"].shape == (2, NEW_VOCAB_SIZE)
        assert state[f"{prefix}.fused_route_codes"].shape == (NEW_VOCAB_SIZE,)
        assert torch.equal(
            state[f"{prefix}.route_table"][:, :OLD_VOCAB_SIZE],
            old_snapshots[f"{prefix}.route_table"],
        )
        assert torch.equal(
            state[f"{prefix}.fused_route_codes"][:OLD_VOCAB_SIZE],
            old_snapshots[f"{prefix}.fused_route_codes"],
        )
        assert torch.equal(
            state[f"{prefix}.fused_expert_pairs"],
            old_snapshots[f"{prefix}.fused_expert_pairs"],
        )
        new_routes = state[f"{prefix}.route_table"][:, OLD_VOCAB_SIZE:]
        assert torch.all(new_routes[0] != new_routes[1])
        assert int(new_routes.min()) >= 0
        assert int(new_routes.max()) < raw["num_experts"]


def test_tokenizer_metadata_adds_boundaries_without_replacing_core_roles(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "extra_special_tokens": {"stale_role": "<stale>"},
            }
        ),
        encoding="utf-8",
    )
    (source / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )

    update_tokenizer_metadata(source, output)

    config = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
    special = json.loads((output / "special_tokens_map.json").read_text(encoding="utf-8"))
    assert "extra_special_tokens" not in config
    assert config["bos_token"] == special["bos_token"] == "<s>"
    assert config["eos_token"] == special["eos_token"] == "</s>"
    assert config["pad_token"] == special["pad_token"] == "<pad>"
    assert config["unk_token"] == special["unk_token"] == "<unk>"
    assert config["additional_special_tokens"] == [
        "<|think_start|>",
        "<|think_end|>",
        "<|final_start|>",
        "<|final_end|>",
    ]


def test_tokenizer_metadata_restores_core_roles_when_source_has_no_sidecars(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    update_tokenizer_metadata(source, output)

    config = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
    special = json.loads((output / "special_tokens_map.json").read_text(encoding="utf-8"))
    assert config["bos_token"] == special["bos_token"] == "<s>"
    assert config["eos_token"] == special["eos_token"] == "</s>"
    assert config["pad_token"] == special["pad_token"] == "<pad>"
    assert config["unk_token"] == special["unk_token"] == "<unk>"
    assert set(config["added_tokens_decoder"]) == {
        "0",
        "1",
        "2",
        "3",
        "32000",
        "32001",
        "32002",
        "32003",
    }
