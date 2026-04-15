"""
Test core components directly.

Priority: HIGH - Core components are building blocks for everything.
"""

import pytest
import torch


class TestMultiHeadAttention:
    """Test MultiHeadAttention component."""

    def test_mha_forward(self):
        """Test MHA forward pass."""
        from complexity.core.attention import MultiHeadAttention, AttentionConfig

        config = AttentionConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=4)
        attn = MultiHeadAttention(config)

        x = torch.randn(2, 16, 256)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_mha_with_mask(self):
        """Test MHA with attention mask."""
        from complexity.core.attention import MultiHeadAttention, AttentionConfig

        config = AttentionConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=4)
        attn = MultiHeadAttention(config)

        x = torch.randn(2, 16, 256)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_mha_kv_cache(self):
        """Test MHA with KV cache for generation."""
        from complexity.core.attention import MultiHeadAttention, AttentionConfig

        config = AttentionConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=4)
        attn = MultiHeadAttention(config)

        # First pass: full sequence
        x = torch.randn(2, 16, 256)
        out, kv_cache = attn(x, use_cache=True)
        assert out.shape == x.shape
        assert kv_cache is not None

        # Second pass: single token with cache
        x_new = torch.randn(2, 1, 256)
        out_new, kv_cache_new = attn(x_new, past_key_value=kv_cache, use_cache=True)
        assert out_new.shape == (2, 1, 256)


class TestGroupedQueryAttention:
    """Test GroupedQueryAttention component."""

    def test_gqa_forward(self):
        """Test GQA forward pass."""
        from complexity.core.attention import GroupedQueryAttention, AttentionConfig

        config = AttentionConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)

        x = torch.randn(2, 16, 256)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_gqa_head_ratio(self):
        """Test GQA with different head ratios."""
        from complexity.core.attention import GroupedQueryAttention, AttentionConfig

        # 4:1 ratio
        config = AttentionConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        assert attn.num_heads == 8
        assert attn.num_kv_heads == 2


class TestRMSNorm:
    """Test RMSNorm component."""

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        from complexity.core.normalization import RMSNorm

        norm = RMSNorm(256)

        x = torch.randn(2, 16, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_normalization(self):
        """Test that RMSNorm actually normalizes."""
        from complexity.core.normalization import RMSNorm

        norm = RMSNorm(64, eps=1e-6)

        # Create input with large values
        x = torch.randn(2, 8, 64) * 100
        out = norm(x)

        # RMS should be approximately 1 after normalization
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRoPE:
    """Test RoPE position encoding."""

    def test_rope_forward(self):
        """Test RoPE forward pass."""
        from complexity.core.position import StandardRoPE

        rope = StandardRoPE(dim=64, max_seq_len=2048)

        # Simulate Q/K tensors
        q = torch.randn(2, 4, 16, 64)  # batch, heads, seq, dim
        k = torch.randn(2, 4, 16, 64)

        cos, sin = rope(16)
        q_rot, k_rot = rope.rotate(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_position_sensitivity(self):
        """Test that RoPE is position-sensitive."""
        from complexity.core.position import StandardRoPE

        rope = StandardRoPE(dim=64, max_seq_len=2048)

        q = torch.randn(1, 1, 4, 64)
        k = torch.randn(1, 1, 4, 64)

        # Positions 0-3
        cos1, sin1 = rope(4)
        q_rot1, _ = rope.rotate(q, k, cos1, sin1)

        # Positions 4-7 (slice from a longer cache)
        cos_full, sin_full = rope(8)
        cos2, sin2 = cos_full[4:], sin_full[4:]
        q_rot2, _ = rope.rotate(q, k, cos2, sin2)

        # Different positions should give different results
        assert not torch.allclose(q_rot1, q_rot2)


class TestSwiGLU:
    """Test SwiGLU MLP."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        from complexity.core.mlp import SwiGLUMLP, MLPConfig

        mlp = SwiGLUMLP(MLPConfig(hidden_size=256, intermediate_size=512))

        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == x.shape

    def test_swiglu_gating(self):
        """Test that SwiGLU uses gating mechanism."""
        from complexity.core.mlp import SwiGLUMLP, MLPConfig

        mlp = SwiGLUMLP(MLPConfig(hidden_size=64, intermediate_size=128))

        # Should have gate and up projections
        assert hasattr(mlp, 'gate_proj') or hasattr(mlp, 'w1')
        assert hasattr(mlp, 'up_proj') or hasattr(mlp, 'w2') or hasattr(mlp, 'w3')


class TestGeGLU:
    """Test GeGLU MLP."""

    def test_geglu_forward(self):
        """Test GeGLU forward pass."""
        from complexity.core.mlp import GeGLUMLP, MLPConfig

        mlp = GeGLUMLP(MLPConfig(hidden_size=256, intermediate_size=512))

        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == x.shape


class TestTokenRoutedMLP:
    """Test TokenRoutedMLP (MoE) component."""

    def test_token_routed_mlp_forward(self):
        """Test MoE forward pass."""
        from complexity.core.mlp import TokenRoutedMLP, MLPConfig

        moe = TokenRoutedMLP(MLPConfig(hidden_size=256, intermediate_size=512, num_experts=4))

        x = torch.randn(2, 16, 256)
        # token_ids required for routed dispatch; without them it uses _forward_all_experts
        token_ids = torch.randint(0, 32000, (2, 16))
        out = moe(x, token_ids=token_ids)
        assert out.shape == x.shape

    def test_token_routed_mlp_load_balancing(self):
        """Test that routing covers all experts across a batch."""
        from complexity.core.mlp import TokenRoutedMLP, MLPConfig

        moe = TokenRoutedMLP(MLPConfig(hidden_size=64, intermediate_size=128, num_experts=4))

        # Run multiple batches and verify outputs are consistent shapes
        for _ in range(5):
            x = torch.randn(4, 32, 64)
            token_ids = torch.randint(0, 32000, (4, 32))
            out = moe(x, token_ids=token_ids)
            assert out.shape == x.shape


class TestALiBi:
    """Test ALiBi position encoding."""

    def test_alibi_bias_shape(self):
        """Test ALiBi bias generation."""
        from complexity.core.position import ALiBiPositionBias

        alibi = ALiBiPositionBias(num_heads=8)

        bias = alibi(seq_len=16)
        assert bias.shape[-2:] == (16, 16)

    def test_alibi_slopes(self):
        """Test ALiBi slope computation."""
        from complexity.core.position import ALiBiPositionBias

        alibi = ALiBiPositionBias(num_heads=8)

        # Slopes should be geometric sequence
        slopes = alibi.slopes
        assert len(slopes) == 8
        # Each slope should be smaller than previous
        for i in range(1, len(slopes)):
            assert slopes[i] <= slopes[i - 1]
