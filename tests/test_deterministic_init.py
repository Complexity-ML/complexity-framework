"""
Tests for the deterministic, RNG-free initialisation scheme.

The whole point of ``deterministic_gaussian_init_`` is that it produces
identical weights every time from a pure function ``(shape, key)``
without consulting any global PRNG state. These tests pin that
guarantee and the downstream behaviour (bit-identical models under
``mlp_type="dense_deterministic"``).
"""

import pytest
import torch

from complexity.config import ModelConfig
from complexity.core.mlp import (
    DenseDeterministicMLP,
    MLPConfig,
    deterministic_gaussian_init_,
)
from complexity.models import ComplexityModel


# --------------------------------------------------------------------------
# deterministic_gaussian_init_ unit tests
# --------------------------------------------------------------------------

def test_deterministic_gaussian_is_bit_identical_across_calls():
    w1 = torch.empty(64, 128)
    w2 = torch.empty(64, 128)
    deterministic_gaussian_init_(w1, key=3, std=0.02)
    deterministic_gaussian_init_(w2, key=3, std=0.02)
    assert torch.equal(w1, w2), "same key must produce identical output"


def test_deterministic_gaussian_differs_by_key():
    w1 = torch.empty(64, 128)
    w2 = torch.empty(64, 128)
    deterministic_gaussian_init_(w1, key=1, std=0.02)
    deterministic_gaussian_init_(w2, key=2, std=0.02)
    assert not torch.equal(w1, w2), "different keys must produce different output"


def test_deterministic_gaussian_independent_of_global_rng():
    """The whole point — the output must be immune to torch.manual_seed."""
    w_a = torch.empty(64, 128)
    torch.manual_seed(1)
    deterministic_gaussian_init_(w_a, key=42, std=0.02)

    w_b = torch.empty(64, 128)
    torch.manual_seed(999)
    # Consume some global RNG state before the call.
    _ = torch.randn(1000)
    deterministic_gaussian_init_(w_b, key=42, std=0.02)

    assert torch.equal(w_a, w_b), (
        "deterministic init must not depend on global PRNG state"
    )


def test_deterministic_gaussian_target_std():
    """Entry std should land within a couple percent of the target."""
    w = torch.empty(512, 512)
    deterministic_gaussian_init_(w, key=7, std=0.05)
    assert abs(w.std().item() - 0.05) < 0.003


def test_deterministic_gaussian_non_power_of_two_shape():
    """Shapes that aren't powers of two are fine."""
    w = torch.empty(129, 257)
    deterministic_gaussian_init_(w, key=11, std=0.02)
    assert w.shape == (129, 257)
    assert torch.isfinite(w).all()


def test_deterministic_gaussian_rejects_non_2d():
    with pytest.raises(ValueError):
        deterministic_gaussian_init_(torch.empty(8, 8, 8), key=1)


# --------------------------------------------------------------------------
# DenseDeterministicMLP + ComplexityModel integration
# --------------------------------------------------------------------------

def _make_dense_deterministic_model() -> ComplexityModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1024,
        max_position_embeddings=128,
        attention_type="gqa",
        mlp_type="dense_deterministic",
        intermediate_size=256,
        num_experts=1,
        shared_expert=False,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=False,
    )
    return ComplexityModel(cfg)


def test_model_uses_dense_deterministic_mlp():
    m = _make_dense_deterministic_model()
    for b in m.layers:
        assert isinstance(b.mlp, DenseDeterministicMLP)


def test_two_models_have_bit_identical_weights():
    """The central invariant: two instantiations produce identical weights,
    without any seed management, on all of {embedding, attention, FFN}."""
    m1 = _make_dense_deterministic_model()
    m2 = _make_dense_deterministic_model()
    for (name, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), f"divergence on {name}"


def test_global_rng_does_not_affect_deterministic_model():
    """Changing torch.manual_seed must not alter the deterministic init."""
    torch.manual_seed(0)
    m_a = _make_dense_deterministic_model()
    torch.manual_seed(9999)
    _ = torch.randn(1000)  # churn the global RNG
    m_b = _make_dense_deterministic_model()
    for (name, p_a), (_, p_b) in zip(m_a.named_parameters(), m_b.named_parameters()):
        # LayerNorm weights are all-ones constant init → trivially equal.
        # What matters is that attn/FFN/embedding weights match too.
        assert torch.equal(p_a, p_b), f"global RNG leaked into {name}"


def test_swiglu_model_is_not_deterministic():
    """Sanity check: a regular swiglu model consumes global RNG, so two
    instantiations without a fixed seed diverge."""
    cfg = ModelConfig(
        hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=1024, max_position_embeddings=128,
        attention_type="gqa", mlp_type="swiglu",
        intermediate_size=256, num_experts=1, shared_expert=False,
        norm_type="rmsnorm", use_qk_norm=True, use_mu_guidance=False,
    )
    torch.manual_seed(1)
    m1 = ComplexityModel(cfg)
    torch.manual_seed(2)
    m2 = ComplexityModel(cfg)
    # Different seeds → different random draws → different weights.
    some_param = "layers.0.mlp.gate_proj.weight"
    p1 = dict(m1.named_parameters())[some_param]
    p2 = dict(m2.named_parameters())[some_param]
    assert not torch.equal(p1, p2)
