"""
Tests for low-level integer ops (complexity.core.integer_ops).

The I64 attention/MLP/normalization/model variants were removed to keep
this framework TR-Hash-only; the general-purpose integer ops utilities
they used are unrelated and still live here.
"""

import pytest
import torch


# =========================================================================
# Integer Ops
# =========================================================================

class TestIntegerOps:
    """Test core integer operations."""

    def test_quantize_weight_int8(self):
        from complexity.core.integer_ops import quantize_weight_int8
        w = torch.randn(64, 32)
        wq, ws = quantize_weight_int8(w)
        assert wq.dtype == torch.int8
        assert wq.shape == (64, 32)
        assert ws.shape == (64,)

    def test_quantize_activation_int8(self):
        from complexity.core.integer_ops import quantize_activation_int8
        x = torch.randn(8, 32)
        xq, xs = quantize_activation_int8(x)
        assert xq.dtype == torch.int8
        assert xq.shape == (8, 32)
        assert xs.shape == (8,)

    def test_int8_linear(self):
        from complexity.core.integer_ops import int8_linear, quantize_weight_int8
        w = torch.randn(64, 32)
        wq, ws = quantize_weight_int8(w)
        x = torch.randn(4, 32)
        out = int8_linear(x, wq, ws)
        assert out.shape == (4, 64)

    def test_int8_linear_3d(self):
        from complexity.core.integer_ops import int8_linear, quantize_weight_int8
        w = torch.randn(64, 32)
        wq, ws = quantize_weight_int8(w)
        x = torch.randn(2, 8, 32)
        out = int8_linear(x, wq, ws)
        assert out.shape == (2, 8, 64)

    def test_int8_fused_gate_up(self):
        from complexity.core.integer_ops import int8_fused_gate_up, quantize_weight_int8
        gate_w = torch.randn(64, 32)
        up_w = torch.randn(64, 32)
        fused = torch.cat([gate_w, up_w], dim=0)
        fq, fs = quantize_weight_int8(fused)
        x = torch.randn(4, 32)
        gate, up = int8_fused_gate_up(x, fq, fs, 64)
        assert gate.shape == (4, 64)
        assert up.shape == (4, 64)

    def test_silu_lut(self):
        from complexity.core.integer_ops import silu_integer
        x_q7 = torch.tensor([0, 128, -128, 256, -256], dtype=torch.int32)
        out = silu_integer(x_q7)
        assert out.dtype == torch.int32
        assert out.shape == x_q7.shape
        # silu(0) = 0
        assert out[0].item() == 0

    def test_sigmoid_lut(self):
        from complexity.core.integer_ops import sigmoid_integer
        x_q7 = torch.tensor([0, 1024, -1024], dtype=torch.int32)
        out = sigmoid_integer(x_q7)
        # sigmoid(0) = 0.5 -> 0.5 * 128 = 64
        assert abs(out[0].item() - 64) <= 1

    def test_silu_multiply_integer(self):
        from complexity.core.integer_ops import silu_multiply_integer
        gate = torch.randn(4, 32)
        up = torch.randn(4, 32)
        out = silu_multiply_integer(gate, up)
        assert out.shape == (4, 32)

    def test_int8_linear_accuracy(self):
        """Verify INT8 linear is close to float linear."""
        from complexity.core.integer_ops import int8_linear, quantize_weight_int8
        w = torch.randn(64, 32)
        x = torch.randn(4, 32)
        ref = x @ w.t()
        wq, ws = quantize_weight_int8(w)
        out = int8_linear(x, wq, ws)
        # Should be close (within quantization error — larger on CPU fallback)
        rel_err = (ref - out).abs() / (ref.abs() + 1e-6)
        assert rel_err.mean() < 0.15, f"Mean relative error {rel_err.mean():.4f} too high"
