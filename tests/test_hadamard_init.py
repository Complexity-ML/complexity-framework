"""
Tests for the deterministic Hadamard initialisation.

The whole point of this init is that it produces identical weights every
time from a pure `(shape, layer_idx)` → matrix function, without any
RNG. These tests pin that guarantee and the orthogonality / Xavier-std
properties that downstream hyperparameters rely on.
"""

import torch
import pytest

from complexity.core.mlp import (
    DenseHadamardMLP,
    MLPConfig,
    hadamard_init_,
    hadamard_sylvester,
)


# --------------------------------------------------------------------------
# Hadamard matrix properties
# --------------------------------------------------------------------------

def test_hadamard_sylvester_small_sizes():
    """H_1 is trivial, H_2 is the canonical 2×2 Hadamard."""
    H1 = hadamard_sylvester(1)
    assert H1.shape == (1, 1) and H1[0, 0].item() == 1.0
    H2 = hadamard_sylvester(2)
    expected = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    assert torch.equal(H2, expected)


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64, 128, 256])
def test_hadamard_is_orthogonal(n):
    """H_n has pairwise orthogonal rows: H H^T = n · I."""
    H = hadamard_sylvester(n)
    product = H @ H.t()
    assert torch.allclose(product, n * torch.eye(n), atol=1e-5)


def test_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        hadamard_sylvester(6)
    with pytest.raises(ValueError):
        hadamard_sylvester(0)


# --------------------------------------------------------------------------
# hadamard_init_ properties
# --------------------------------------------------------------------------

def test_hadamard_init_is_fully_deterministic():
    """Two calls with identical args produce identical tensors — no RNG."""
    w1 = torch.empty(64, 128)
    w2 = torch.empty(64, 128)
    hadamard_init_(w1, layer_idx=3)
    hadamard_init_(w2, layer_idx=3)
    assert torch.equal(w1, w2), "Hadamard init must be bit-identical across calls"


def test_hadamard_init_differs_by_layer_idx():
    """Different layer_idx produce different weights (distinguishable layers)."""
    w1 = torch.empty(64, 128)
    w2 = torch.empty(64, 128)
    hadamard_init_(w1, layer_idx=0)
    hadamard_init_(w2, layer_idx=1)
    assert not torch.equal(w1, w2)


def test_hadamard_init_non_square():
    """Works on rectangular shapes (typical for FFN: hidden ≠ intermediate)."""
    w = torch.empty(256, 1024)
    hadamard_init_(w, layer_idx=5)
    assert w.shape == (256, 1024)
    assert torch.isfinite(w).all()


def test_hadamard_init_non_power_of_two_shape():
    """Shapes not matching a power of two are handled via truncation."""
    w = torch.empty(129, 257)
    hadamard_init_(w, layer_idx=7)
    assert w.shape == (129, 257)
    assert torch.isfinite(w).all()


def test_hadamard_init_matches_xavier_std():
    """With the default `std_target=None`, entry magnitudes match Xavier."""
    out_f, in_f = 512, 2048
    w = torch.empty(out_f, in_f)
    hadamard_init_(w)
    expected_std = (2.0 / (out_f + in_f)) ** 0.5
    # Entries are ±expected_std by construction (Hadamard entries ±1 × scale).
    assert torch.allclose(w.abs(), torch.full_like(w, expected_std), atol=1e-6)


def test_hadamard_init_rows_orthogonal_after_scaling():
    """Row orthogonality survives the sign flip and the std scaling."""
    out_f, in_f = 64, 128
    w = torch.empty(out_f, in_f)
    hadamard_init_(w, layer_idx=2, std_target=1.0)
    gram = w @ w.t()
    off_diag = gram - torch.diag(torch.diagonal(gram))
    # Rows are pairwise orthogonal → off-diagonal is zero.
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-4)


# --------------------------------------------------------------------------
# DenseHadamardMLP end-to-end
# --------------------------------------------------------------------------

def test_dense_hadamard_forward_shape():
    cfg = MLPConfig(hidden_size=128, intermediate_size=512, layer_idx=0)
    mlp = DenseHadamardMLP(cfg)
    x = torch.randn(4, 32, 128)
    y = mlp(x)
    assert y.shape == x.shape


def test_dense_hadamard_bit_identical_across_instances():
    """Two MLPs with identical config get bit-identical weights — the whole point."""
    cfg = MLPConfig(hidden_size=128, intermediate_size=512, layer_idx=4)
    a = DenseHadamardMLP(cfg)
    b = DenseHadamardMLP(cfg)
    for name in ("gate_proj", "up_proj", "down_proj"):
        w_a = getattr(a, name).weight
        w_b = getattr(b, name).weight
        assert torch.equal(w_a, w_b), f"{name} weights diverge between runs"


def test_dense_hadamard_distinct_layers():
    """Different layer_idx values produce different weights."""
    cfg0 = MLPConfig(hidden_size=128, intermediate_size=512, layer_idx=0)
    cfg1 = MLPConfig(hidden_size=128, intermediate_size=512, layer_idx=1)
    m0 = DenseHadamardMLP(cfg0)
    m1 = DenseHadamardMLP(cfg1)
    assert not torch.equal(m0.gate_proj.weight, m1.gate_proj.weight)


def test_dense_hadamard_gate_up_down_distinct_within_layer():
    """The three projections in a single block must not share identical weights."""
    cfg = MLPConfig(hidden_size=128, intermediate_size=512, layer_idx=0)
    m = DenseHadamardMLP(cfg)
    # gate and up have identical shape — their weights must still differ.
    assert not torch.equal(m.gate_proj.weight, m.up_proj.weight)
