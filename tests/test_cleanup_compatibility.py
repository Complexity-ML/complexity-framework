"""Compatibility contracts protected while the framework is modularized."""

import torch


def _legacy_mlp():
    from complexity.core.mlp import MLPConfig, TokenRoutedMLP

    return TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            vocab_size=17,
            num_experts=4,
            top_k=2,
            top_k_primary_weight=0.5,
            shared_expert=True,
            use_cggr=False,
            routing_strategy="token_id_balanced_hash",
        )
    )


def test_historical_routing_helpers_remain_import_compatible():
    from complexity.core.mlp import token_routed
    from complexity.core.mlp import token_routing

    assert (
        token_routed._create_pair_coverage_hash_metadata
        is token_routing.create_pair_coverage_hash_metadata
    )
    assert (
        token_routed._create_pair_coverage_hash_routes
        is token_routing.create_pair_coverage_hash_routes
    )
    assert (
        token_routed._encode_pair_coverage_hash_routes
        is token_routing.encode_pair_coverage_hash_routes
    )


def test_legacy_token_routed_state_dict_keys_and_shapes_are_stable():
    state = _legacy_mlp().state_dict()
    assert {name: tuple(value.shape) for name, value in state.items()} == {
        "gate_proj_w": (4, 8, 4),
        "up_proj_w": (4, 8, 4),
        "down_proj_w": (4, 4, 8),
        "token_to_expert": (17,),
        "topk_token_to_expert": (2, 17),
        "pair_hash_route_codes": (17,),
        "pair_hash_expert_pairs": (6, 2),
        "shared_gate.weight": (16, 8),
        "shared_up.weight": (16, 8),
        "shared_down.weight": (8, 16),
    }


def test_legacy_token_routed_state_dict_round_trip_is_bitwise():
    torch.manual_seed(47)
    source = _legacy_mlp()
    target = _legacy_mlp()
    target.load_state_dict(source.state_dict())

    hidden = torch.randn(2, 5, 8)
    token_ids = torch.randint(0, 17, (2, 5))
    assert torch.equal(
        source(hidden, token_ids=token_ids),
        target(hidden, token_ids=token_ids),
    )


def test_cuda_extras_remain_available_from_historical_module():
    from complexity_cuda.token_routed_extras import (
        RoboticsTokenRoutedLayer as MovedRoboticsLayer,
        benchmark_token_routed_mlp as moved_benchmark,
    )
    from complexity_cuda.triton_token_routed import (
        RoboticsTokenRoutedLayer,
        benchmark_token_routed_mlp,
    )

    assert RoboticsTokenRoutedLayer is MovedRoboticsLayer
    assert benchmark_token_routed_mlp is moved_benchmark
