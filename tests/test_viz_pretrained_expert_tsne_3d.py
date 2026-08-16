import numpy as np
import torch

from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from scripts.viz_pretrained_expert_tsne_3d import (
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
