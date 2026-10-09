import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from complexity.config import ModelConfig
from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from complexity.models import ComplexityModel
from complexity.profiling.routing import (
    audit_full_probe_routing,
    load_tr_hash_model,
    parse_layers,
    routed_contributions,
    stratified_token_sample,
)


def _small_mlp() -> TRHashEngineMLP:
    return TRHashEngineMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=32,
            routing_strategy="token_id_balanced_hash",
            top_k=2,
            top_k_primary_weight=0.5,
            shared_expert=False,
            routed_output_scale=2.0,
            use_custom_kernels=False,
        )
    )


def test_parse_layers_preserves_order_and_removes_duplicates():
    assert parse_layers("0, 5,11,5,23") == (0, 5, 11, 23)


def test_stratified_sample_represents_each_primary_expert():
    mlp = _small_mlp()
    token_ids = torch.arange(32)
    selected = stratified_token_sample(
        mlp.engine.route_table,
        token_ids,
        count=16,
        rng=np.random.default_rng(42),
    )
    primary = mlp.engine.route_table[0, token_ids[selected]]
    assert len(selected) == 16
    assert sorted(primary.unique().tolist()) == [0, 1, 2, 3]


def test_routed_contributions_follow_both_persisted_routes():
    mlp = _small_mlp()
    hidden = torch.randn(32, 8, generator=torch.Generator().manual_seed(7))
    token_ids = torch.arange(32)
    selected = np.arange(12, dtype=np.int64)

    packed, norms = routed_contributions(
        mlp,
        hidden,
        token_ids,
        selected,
        torch.device("cpu"),
    )
    vectors, experts, route_ranks, source_indices = packed
    expected = mlp.engine.route_table[:, token_ids[selected]].reshape(-1).numpy()

    assert vectors.shape == (24, 8)
    assert np.array_equal(experts, expected)
    assert np.array_equal(route_ranks, np.repeat([0, 1], 12))
    assert np.array_equal(source_indices, np.tile(selected, 2))
    assert np.all(norms > 0)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_full_probe_audit_uses_unsampled_routes_and_reports_finite_norm_ratios():
    mlp = _small_mlp()
    hidden = torch.randn(32, 8, generator=torch.Generator().manual_seed(11))
    residual = torch.randn(32, 8, generator=torch.Generator().manual_seed(12))
    token_ids = torch.arange(32)

    audit = audit_full_probe_routing(
        mlp,
        hidden,
        residual,
        token_ids,
        torch.device("cpu"),
        batch_size=7,
    )

    expected_routes = mlp.engine.route_table[:, token_ids]
    expected_counts = [torch.bincount(route, minlength=4).tolist() for route in expected_routes]
    assert audit["tokens"] == 32
    assert audit["unique_token_ids"] == 32
    assert audit["sampling"] == "all_collected_probe_tokens"
    assert audit["route_counts"] == expected_counts
    assert sum(audit["combined_route_counts"]) == 64
    assert audit["tokens_with_repeated_expert_across_routes"] == 0
    assert audit["all_experts_observed_per_route"]
    for metric in (
        "routed_branch_norm",
        "residual_stream_norm",
        "mlp_input_norm",
        "routed_to_residual_norm_ratio",
        "routed_to_mlp_input_norm_ratio",
    ):
        assert all(np.isfinite(value) for value in audit[metric].values())
        assert audit[metric]["max"] >= audit[metric]["p99"]
        assert audit[metric]["p99"] >= audit[metric]["p50"]


def test_load_tr_hash_model_accepts_current_engine_bundle(tmp_path):
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=8,
        shared_intermediate_size=24,
        mlp_type="tr_hash_engine",
        routing_strategy="token_id_multi_hash",
        route_hash_count=2,
        num_experts=4,
        top_k=2,
        shared_expert=True,
        tie_word_embeddings=False,
        use_custom_kernels=False,
    )
    source = ComplexityModel(config)
    (tmp_path / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    save_file(
        {key: value.detach().contiguous() for key, value in source.state_dict().items()},
        tmp_path / "model.safetensors",
    )

    loaded = load_tr_hash_model(tmp_path, torch.device("cpu"))

    assert len(loaded.layers) == 1
    assert isinstance(loaded.layers[0].mlp, TRHashEngineMLP)
    assert torch.equal(
        loaded.layers[0].mlp.engine.route_table,
        source.layers[0].mlp.engine.route_table,
    )


def test_contributions_reconstruct_forward_with_reduced_active_capacity():
    mlp = _small_mlp()
    mlp.set_active_experts(2)
    mlp.set_active_expert_width(2)
    hidden = torch.randn(32, 8, generator=torch.Generator().manual_seed(71))
    token_ids = torch.arange(32)
    packed, norms = routed_contributions(mlp, hidden, token_ids, np.arange(32), "cpu")
    vectors, experts, _, _ = packed
    reconstructed = (vectors * norms[:, None]).reshape(2, 32, 8).sum(axis=0)
    with torch.no_grad():
        actual = mlp(hidden.unsqueeze(0), token_ids.unsqueeze(0)).squeeze(0).numpy()
    assert set(experts) == {0, 1}
    np.testing.assert_allclose(reconstructed, actual, atol=1e-9, rtol=1e-5)
    audit = audit_full_probe_routing(mlp, hidden, hidden, token_ids, "cpu", batch_size=5)
    assert audit["active_num_experts"] == 2
    assert audit["active_expert_width"] == 2
    assert sum(audit["combined_route_counts"]) == 64


def test_sampling_rejects_a_count_that_cannot_represent_every_expert():
    mlp = _small_mlp()
    with pytest.raises(ValueError, match="every observed primary expert"):
        stratified_token_sample(
            mlp.engine.route_table, torch.arange(32), 2, np.random.default_rng(0)
        )


def test_primary_weight_control_changes_the_executed_route_mix():
    mlp = _small_mlp()
    hidden = torch.randn(32, 8, generator=torch.Generator().manual_seed(19))
    token_ids = torch.arange(32)
    before = mlp(hidden.unsqueeze(0), token_ids.unsqueeze(0)).detach()
    mlp.set_top_k_primary_weight(0.9)
    after = mlp(hidden.unsqueeze(0), token_ids.unsqueeze(0)).detach()
    assert not torch.equal(before, after)
    packed, norms = routed_contributions(mlp, hidden, token_ids, np.arange(32), "cpu")
    reconstructed = (packed[0] * norms[:, None]).reshape(2, 32, 8).sum(axis=0)
    np.testing.assert_allclose(reconstructed, after.squeeze(0).numpy(), atol=1e-9, rtol=1e-5)
