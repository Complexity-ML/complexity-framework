from __future__ import annotations

import pytest
import torch


def test_routed_expert_initialization_respects_initializer_range():
    """TRHashEngineMLP (unlike the removed TokenRoutedMLP) has no
    expert_initialization="legacy_kaiming" ablation-replay mode — it always
    inits gpt_normal-style, scaled by initializer_range."""
    from complexity.config.model_config import ModelConfig
    from complexity.models.builder import ComplexityModel

    def build(initializer_range: float) -> ComplexityModel:
        torch.manual_seed(11)
        return ComplexityModel(
            ModelConfig(
                vocab_size=128,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=32,
                num_experts=4,
                mlp_type="tr_hash_engine",
                shared_expert=True,
                shared_intermediate_size=96,
                initializer_range=initializer_range,
            )
        )

    narrow = build(0.02)
    wide = build(0.1)

    narrow_mlp = narrow.layers[0].mlp
    wide_mlp = wide.layers[0].mlp
    narrow_std = narrow_mlp.engine.expert_gate.detach().std().item()
    shared_std = narrow_mlp.engine.shared_gate.weight.detach().std().item()
    wide_std = wide_mlp.engine.expert_gate.detach().std().item()

    assert narrow_std == pytest.approx(0.02, rel=0.03)
    assert shared_std == pytest.approx(0.02, rel=0.03)
    assert narrow_std == pytest.approx(shared_std, rel=0.04)
    assert wide_std == pytest.approx(0.1, rel=0.03)


def test_expert_initialization_rejects_unknown_mode():
    from complexity.core.mlp.base import MLPConfig

    with pytest.raises(ValueError, match="expert_initialization"):
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            expert_initialization="mystery",
        )


def test_non_hash_routing_strategies_are_rejected():
    """zipf/round_robin/random/lsh_hidden are not token-ID/hash-table routing
    and were removed to keep this framework TR-Hash-only (see base.py's
    ``_removed_routing_strategies`` guard)."""
    from complexity.core.mlp.base import MLPConfig

    for removed in ("zipf", "round_robin", "random", "lsh_hidden"):
        with pytest.raises(ValueError, match="was removed"):
            MLPConfig(
                hidden_size=8,
                intermediate_size=16,
                num_experts=4,
                vocab_size=8,
                routing_strategy=removed,
                shared_expert=False,
            )


