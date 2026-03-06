"""
Tests for I64 Integer components.

Tests integer ops, I64 attention, I64 MLP, I64 normalization,
full model creation, forward pass, and quantization.
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


# =========================================================================
# I64 Attention
# =========================================================================

class TestI64Attention:
    """Test I64 integer attention."""

    def _make_attn(self, hidden=256, heads=4, kv_heads=2):
        from complexity.core.attention import AttentionConfig, I64Attention
        config = AttentionConfig(
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            max_position_embeddings=512,
        )
        return I64Attention(config)

    def test_forward(self):
        attn = self._make_attn()
        x = torch.randn(2, 16, 256)
        out, kv = attn(x)
        assert out.shape == (2, 16, 256)
        assert kv is None

    def test_forward_with_cache(self):
        attn = self._make_attn()
        x = torch.randn(2, 16, 256)
        out, kv = attn(x, use_cache=True)
        assert out.shape == (2, 16, 256)
        assert kv is not None
        assert kv[0].shape[2] == 16  # cached seq_len

    def test_forward_with_mu(self):
        attn = self._make_attn()
        x = torch.randn(2, 16, 256)
        mu = torch.randn(2, 16, 256)
        out, kv = attn(x, mu_prev=mu)
        assert out.shape == (2, 16, 256)

    def test_quantize(self):
        attn = self._make_attn()
        x = torch.randn(2, 8, 256)
        # Float forward
        with torch.no_grad():
            out_float, _ = attn(x)
        # Quantize
        attn.quantize()
        assert hasattr(attn, 'qkv_int8')
        assert hasattr(attn, 'o_int8')
        # INT8 forward
        with torch.no_grad():
            out_int8, _ = attn(x)
        assert out_int8.shape == out_float.shape

    def test_registry(self):
        from complexity.core.registry import ATTENTION_REGISTRY
        assert "i64" in ATTENTION_REGISTRY
        assert "integer" in ATTENTION_REGISTRY


# =========================================================================
# I64 MLP
# =========================================================================

class TestI64SwiGLUMLP:
    """Test I64 integer SwiGLU MLP."""

    def _make_mlp(self, hidden=256, inter=704):
        from complexity.core.mlp import MLPConfig, I64SwiGLUMLP
        config = MLPConfig(hidden_size=hidden, intermediate_size=inter)
        return I64SwiGLUMLP(config)

    def test_forward_float(self):
        mlp = self._make_mlp()
        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == (2, 16, 256)

    def test_quantize(self):
        mlp = self._make_mlp()
        x = torch.randn(2, 8, 256)
        with torch.no_grad():
            out_float = mlp(x)
        mlp.quantize()
        assert hasattr(mlp, 'gate_up_int8')
        assert hasattr(mlp, 'down_int8')
        with torch.no_grad():
            out_int8 = mlp(x)
        assert out_int8.shape == out_float.shape

    def test_registry(self):
        from complexity.core.registry import MLP_REGISTRY
        assert "i64_swiglu" in MLP_REGISTRY
        assert "integer_swiglu" in MLP_REGISTRY


class TestI64TokenRoutedMLP:
    """Test I64 integer token-routed MLP."""

    def _make_moe(self, hidden=256, inter=704, experts=4):
        from complexity.core.mlp import MLPConfig, I64TokenRoutedMLP
        config = MLPConfig(
            hidden_size=hidden, intermediate_size=inter,
            num_experts=experts, vocab_size=32000,
        )
        return I64TokenRoutedMLP(config)

    def test_forward_with_token_ids(self):
        moe = self._make_moe()
        x = torch.randn(2, 16, 256)
        ids = torch.randint(0, 32000, (2, 16))
        out = moe(x, token_ids=ids)
        assert out.shape == (2, 16, 256)

    def test_forward_without_token_ids(self):
        moe = self._make_moe()
        x = torch.randn(2, 16, 256)
        out = moe(x)
        assert out.shape == (2, 16, 256)

    def test_routing_deterministic(self):
        moe = self._make_moe()
        ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        expert_ids = moe.route(ids, 8, torch.device('cpu'))
        expected = ids % 4
        assert torch.equal(expert_ids, expected)

    def test_quantize(self):
        moe = self._make_moe()
        x = torch.randn(2, 8, 256)
        ids = torch.randint(0, 32000, (2, 8))
        with torch.no_grad():
            out_float = moe(x, token_ids=ids)
        moe.quantize()
        assert hasattr(moe, 'gate_up_int8_experts')
        with torch.no_grad():
            out_int8 = moe(x, token_ids=ids)
        assert out_int8.shape == out_float.shape

    def test_registry(self):
        from complexity.core.registry import MLP_REGISTRY
        assert "i64_token_routed" in MLP_REGISTRY
        assert "integer_moe" in MLP_REGISTRY


# =========================================================================
# I64 Normalization
# =========================================================================

class TestI64RMSNorm:
    """Test I64 integer RMSNorm."""

    def test_forward_float(self):
        from complexity.core.normalization import I64RMSNorm
        norm = I64RMSNorm(256)
        x = torch.randn(2, 16, 256)
        out = norm(x)
        assert out.shape == (2, 16, 256)

    def test_quantize_weight(self):
        from complexity.core.normalization import I64RMSNorm
        norm = I64RMSNorm(256)
        x = torch.randn(2, 8, 256)
        with torch.no_grad():
            out_float = norm(x)
        norm.quantize_weight()
        assert hasattr(norm, 'weight_q12')
        assert norm.weight_q12.dtype == torch.int16
        with torch.no_grad():
            out_int = norm(x)
        assert out_int.shape == out_float.shape
        # Should be close
        assert torch.allclose(out_float, out_int, atol=0.01)

    def test_registry(self):
        from complexity.core.registry import NORMALIZATION_REGISTRY
        assert "i64_rmsnorm" in NORMALIZATION_REGISTRY
        assert "integer_rmsnorm" in NORMALIZATION_REGISTRY


# =========================================================================
# I64 Full Model
# =========================================================================

class TestI64Model:
    """Test full I64 model creation and inference."""

    def _make_model(self):
        from complexity.config import ModelConfig
        from complexity.models.builder import ComplexityModel
        config = ModelConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=704,
            vocab_size=1000,
            max_position_embeddings=512,
            attention_type="i64",
            mlp_type="i64_swiglu",
            norm_type="i64_rmsnorm",
            use_inl_dynamics=True,
        )
        return ComplexityModel(config)

    def test_model_creation(self):
        model = self._make_model()
        assert model.num_parameters() > 0

    def test_forward_pass(self):
        model = self._make_model()
        ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            outputs = model(ids)
        assert outputs["logits"].shape == (2, 32, 1000)

    def test_forward_with_cache(self):
        model = self._make_model()
        ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            outputs = model(ids, use_cache=True)
        assert outputs["past_key_values"] is not None
        assert len(outputs["past_key_values"]) == 4  # 4 layers

    def test_velocity_states(self):
        model = self._make_model()
        ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            outputs = model(ids)
        assert outputs["velocity_states"] is not None
        assert len(outputs["velocity_states"]) == 4

    def test_quantize_all(self):
        model = self._make_model()
        ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            out_float = model(ids)
        model.quantize_all()
        with torch.no_grad():
            out_int8 = model(ids)
        assert out_int8["logits"].shape == out_float["logits"].shape

    def test_generate(self):
        model = self._make_model()
        ids = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            generated = model.generate(input_ids=ids, max_new_tokens=10, do_sample=False)
        assert generated.shape[1] == 18  # 8 + 10

    def test_generate_quantized(self):
        model = self._make_model()
        model.quantize_all()
        ids = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            generated = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False)
        assert generated.shape[1] == 13  # 8 + 5


# =========================================================================
# I64 Presets
# =========================================================================

class TestI64Presets:
    """Test I64 config presets exist and are valid."""

    @pytest.mark.parametrize("name", ["i64-1b", "i64-3b", "i64-7b"])
    def test_preset_exists(self, name):
        from complexity.config import get_preset
        config = get_preset(name)
        assert config.attention_type == "i64"
        assert config.mlp_type == "i64_swiglu"
        assert config.norm_type == "i64_rmsnorm"
        assert config.use_inl_dynamics is True

    def test_i64_1b_params(self):
        from complexity.config import get_preset
        config = get_preset("i64-1b")
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24

    def test_i64_3b_params(self):
        from complexity.config import get_preset
        config = get_preset("i64-3b")
        assert config.hidden_size == 2560
        assert config.num_hidden_layers == 32

    def test_i64_7b_params(self):
        from complexity.config import get_preset
        config = get_preset("i64-7b")
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
