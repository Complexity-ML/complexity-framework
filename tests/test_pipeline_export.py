"""Tests for export-based pipeline helpers."""

from __future__ import annotations

import pytest
import torch


def test_pipeline_split_spec_even_layers():
    from torch.distributed.pipelining import SplitPoint

    from complexity.parallel.pipeline_export import pipeline_split_spec

    assert pipeline_split_spec(32, 8) == {
        "layers.4": SplitPoint.BEGINNING,
        "layers.8": SplitPoint.BEGINNING,
        "layers.12": SplitPoint.BEGINNING,
        "layers.16": SplitPoint.BEGINNING,
        "layers.20": SplitPoint.BEGINNING,
        "layers.24": SplitPoint.BEGINNING,
        "layers.28": SplitPoint.BEGINNING,
    }


def test_pipeline_split_spec_rejects_uneven_by_default():
    from complexity.parallel.pipeline_export import pipeline_split_spec

    with pytest.raises(ValueError, match="divisible"):
        pipeline_split_spec(30, 8)


def test_static_tr_dispatch_matches_sparse_dispatch():
    from complexity.core.mlp import MLPConfig, TokenRoutedMLP

    torch.manual_seed(0)
    sparse_config = MLPConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        vocab_size=64,
        shared_expert=False,
        top_k=2,
        top_k_primary_weight=0.5,
    )
    static_config = MLPConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        vocab_size=64,
        shared_expert=False,
        top_k=2,
        top_k_primary_weight=0.5,
        static_expert_capacity=True,
    )
    sparse = TokenRoutedMLP(sparse_config)
    static = TokenRoutedMLP(static_config)
    static.load_state_dict(sparse.state_dict(), strict=False)

    hidden = torch.randn(3, 7, 16)
    token_ids = torch.randint(0, 64, (3, 7))

    assert torch.allclose(
        sparse(hidden, token_ids=token_ids),
        static(hidden, token_ids=token_ids),
        atol=1e-6,
        rtol=1e-5,
    )


def test_trace_pipeline_accepts_token_routed_model():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel
    from complexity.parallel.pipeline_export import trace_pipeline

    config = ModelConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=128,
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=True,
        shared_intermediate_size=64,
        use_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.1,
        top_k=2,
        top_k_primary_weight=0.5,
        use_mu_guidance=False,
        use_cache=False,
    )

    model = ComplexityModel(config)
    example_input_ids = torch.randint(0, 128, (2, 8))
    pipe = trace_pipeline(model, example_input_ids, pp_size=2)

    assert pipe is not None
    assert config.static_expert_capacity is True

