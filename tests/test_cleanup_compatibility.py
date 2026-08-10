"""Compatibility contracts protected while the framework is modularized.

The historical ``TokenRoutedMLP`` dispatch implementation (and its
``token_routing.py`` helper module) were removed once the canonical
``TRHashEngineMLP`` covered its default behavior. Existing on-disk
``token_routed``-format checkpoints still need their exact tensor
names/shapes honored, so that contract now lives in
``complexity.utils.token_routed_conversion`` — these tests lock it in there
instead of against a live legacy instance.
"""

import torch


def test_legacy_token_routed_tensor_names_are_all_accounted_for():
    """Every tensor name a real token_routed checkpoint can contain must be
    either renamed, transplanted separately (route_table), or explicitly
    dropped by the converter — never silently ignored."""
    from complexity.utils.token_routed_conversion import (
        _DROPPED_SUFFIXES,
        _RENAME,
        _ROUTE_TABLE_SUFFIX,
    )

    historical_tensor_names = {
        "gate_proj_w",
        "up_proj_w",
        "down_proj_w",
        "token_to_expert",
        "topk_token_to_expert",
        "pair_hash_route_codes",
        "pair_hash_expert_pairs",
        "shared_gate.weight",
        "shared_up.weight",
        "shared_down.weight",
    }
    accounted_for = set(_RENAME) | set(_DROPPED_SUFFIXES) | {_ROUTE_TABLE_SUFFIX}
    assert historical_tensor_names <= accounted_for


def test_legacy_token_routed_expert_weight_shapes_match_tr_hash_engine():
    """gate_proj_w/up_proj_w/down_proj_w's [num_experts, hidden, width] /
    [num_experts, width, hidden] layout must still match TRHashEngineMLP's
    expert_gate/expert_up/expert_down — the converter only renames keys, it
    never reshapes or transposes."""
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP

    mlp = TRHashEngineMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            vocab_size=17,
            num_experts=4,
            top_k=2,
            top_k_primary_weight=0.5,
            shared_expert=True,
            shared_intermediate_size=16,
            routing_strategy="token_id_balanced_hash",
        )
    )
    assert tuple(mlp.engine.expert_gate.shape) == (4, 8, 4)
    assert tuple(mlp.engine.expert_up.shape) == (4, 8, 4)
    assert tuple(mlp.engine.expert_down.shape) == (4, 4, 8)
    assert tuple(mlp.engine.shared_gate.weight.shape) == (16, 8)


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
