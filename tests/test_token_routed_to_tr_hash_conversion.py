"""token_routed -> tr_hash_engine checkpoint conversion.

TokenRoutedMLP (the historical dispatch implementation, since removed) and
TRHashEngineMLP (the canonical TR-Hash path) stored per-expert weights in the
same tensor layout, but their route-table constructions were independent
algorithms with no guarantee of agreeing. The converter must transplant the
exact trained routing table rather than let TRHashEngine regenerate its own.

``ModelConfig(mlp_type="token_routed")`` is itself rejected by the mlp_type
guardrail now that the class is gone, so these tests build a synthetic
"legacy checkpoint" (state dict + raw config dict) by hand, matching the
historical serialization contract, instead of instantiating the deleted
class. The non-MLP tensors (attention/norm/embedding) are identical
regardless of mlp_type, so they're borrowed from a real tr_hash_engine model.
"""

from __future__ import annotations

import pytest
import torch

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.utils.token_routed_conversion import (
    convert_token_routed_checkpoint,
    convert_token_routed_config,
    convert_token_routed_state_dict,
)

_BASE_CONFIG = dict(
    vocab_size=97,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=64,
    num_experts=4,
    top_k=2,
    shared_expert=True,
    shared_intermediate_size=48,
    routing_strategy="token_id_balanced_hash",
)


def _legacy_checkpoint(seed: int = 0, **overrides):
    """Synthesize a token_routed-shaped (state_dict, raw_config_dict) pair."""

    fields = dict(_BASE_CONFIG)
    fields.update(overrides)
    expert_width = fields["intermediate_size"] // fields["num_experts"]

    skeleton = ComplexityModel(ModelConfig(mlp_type="tr_hash_engine", **fields))
    state_dict = {
        k: v.clone() for k, v in skeleton.state_dict().items() if ".mlp.engine." not in k
    }

    generator = torch.Generator().manual_seed(seed)
    for layer_idx in range(fields["num_hidden_layers"]):
        prefix = f"layers.{layer_idx}.mlp"
        state_dict[f"{prefix}.gate_proj_w"] = torch.randn(
            fields["num_experts"], fields["hidden_size"], expert_width, generator=generator
        )
        state_dict[f"{prefix}.up_proj_w"] = torch.randn(
            fields["num_experts"], fields["hidden_size"], expert_width, generator=generator
        )
        state_dict[f"{prefix}.down_proj_w"] = torch.randn(
            fields["num_experts"], expert_width, fields["hidden_size"], generator=generator
        )
        state_dict[f"{prefix}.shared_gate.weight"] = torch.randn(
            fields["shared_intermediate_size"], fields["hidden_size"], generator=generator
        )
        state_dict[f"{prefix}.shared_up.weight"] = torch.randn(
            fields["shared_intermediate_size"], fields["hidden_size"], generator=generator
        )
        state_dict[f"{prefix}.shared_down.weight"] = torch.randn(
            fields["hidden_size"], fields["shared_intermediate_size"], generator=generator
        )
        # Each token's top_k routes must select distinct experts (as any real
        # trained TokenRoutedMLP table would) — argsort a random score per
        # (token, expert) to get a random permutation of experts per token.
        scores = torch.rand(fields["vocab_size"], fields["num_experts"], generator=generator)
        state_dict[f"{prefix}.topk_token_to_expert"] = (
            torch.argsort(scores, dim=1)[:, : fields["top_k"]].t().contiguous()
        )

    config_dict = dict(fields)
    config_dict["mlp_type"] = "token_routed"
    return state_dict, config_dict


def test_convert_token_routed_config_switches_mlp_type():
    _, config_dict = _legacy_checkpoint()
    converted_config = convert_token_routed_config(config_dict)
    assert converted_config.mlp_type == "tr_hash_engine"
    assert converted_config.num_experts == config_dict["num_experts"]
    assert converted_config.top_k == config_dict["top_k"]


def test_convert_token_routed_config_rejects_non_token_routed_input():
    _, config_dict = _legacy_checkpoint()
    config_dict["mlp_type"] = "tr_hash_engine"
    with pytest.raises(ValueError, match="expects a config dict with mlp_type"):
        convert_token_routed_config(config_dict)


def test_converted_checkpoint_builds_a_tr_hash_engine_model():
    state_dict, config_dict = _legacy_checkpoint()
    model = convert_token_routed_checkpoint(state_dict, config_dict)
    assert model.layers[0].mlp.__class__.__name__ == "TRHashEngineMLP"
    assert model.config.mlp_type == "tr_hash_engine"

    input_ids = torch.randint(0, 97, (2, 6))
    with torch.no_grad():
        out = model(input_ids)["logits"]
    assert out.shape == (2, 6, 97)


def test_converted_route_table_matches_the_legacy_trained_routing():
    state_dict, config_dict = _legacy_checkpoint(seed=3)
    model = convert_token_routed_checkpoint(state_dict, config_dict)
    for layer_idx in range(config_dict["num_hidden_layers"]):
        expected = state_dict[f"layers.{layer_idx}.mlp.topk_token_to_expert"]
        actual = model.layers[layer_idx].mlp.engine.route_table
        assert torch.equal(expected, actual)


def test_converted_expert_weights_match_the_legacy_tensors_exactly():
    state_dict, config_dict = _legacy_checkpoint(seed=7)
    model = convert_token_routed_checkpoint(state_dict, config_dict)
    engine = model.layers[0].mlp.engine
    assert torch.equal(engine.expert_gate, state_dict["layers.0.mlp.gate_proj_w"])
    assert torch.equal(engine.expert_up, state_dict["layers.0.mlp.up_proj_w"])
    assert torch.equal(engine.expert_down, state_dict["layers.0.mlp.down_proj_w"])
    assert torch.equal(engine.shared_gate.weight, state_dict["layers.0.mlp.shared_gate.weight"])


def test_convert_state_dict_rejects_unmapped_tensors_instead_of_dropping_them():
    state_dict, _ = _legacy_checkpoint()
    state_dict["layers.0.mlp.hash_pair_gate_logits"] = torch.zeros(3)
    with pytest.raises(ValueError, match="don't know how to convert"):
        convert_token_routed_state_dict(state_dict)
