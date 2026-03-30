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
        from complexity.core.attention import MultiHeadAttention

        attn = MultiHeadAttention(
            hidden_size=256,
            num_heads=4,
        )

        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_mha_with_mask(self):
        """Test MHA with attention mask."""
        from complexity.core.attention import MultiHeadAttention

        attn = MultiHeadAttention(
            hidden_size=256,
            num_heads=4,
        )

        x = torch.randn(2, 16, 256)
        # Causal mask
        mask = torch.tril(torch.ones(16, 16))
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_mha_kv_cache(self):
        """Test MHA with KV cache for generation."""
        from complexity.core.attention import MultiHeadAttention

        attn = MultiHeadAttention(
            hidden_size=256,
            num_heads=4,
        )

        # First pass: full sequence
        x = torch.randn(2, 16, 256)
        out, kv_cache = attn(x, use_cache=True)
        assert out.shape == x.shape
        assert kv_cache is not None

        # Second pass: single token with cache
        x_new = torch.randn(2, 1, 256)
        out_new, kv_cache_new = attn(x_new, past_kv=kv_cache, use_cache=True)
        assert out_new.shape == (2, 1, 256)


class TestGroupedQueryAttention:
    """Test GroupedQueryAttention component."""

    def test_gqa_forward(self):
        """Test GQA forward pass."""
        from complexity.core.attention import GroupedQueryAttention

        attn = GroupedQueryAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
        )

        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_head_ratio(self):
        """Test GQA with different head ratios."""
        from complexity.core.attention import GroupedQueryAttention

        # 4:1 ratio
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
        )
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
        from complexity.core.position import RoPE

        rope = RoPE(dim=64, max_seq_len=2048)

        # Simulate Q/K tensors
        q = torch.randn(2, 4, 16, 64)  # batch, heads, seq, dim
        k = torch.randn(2, 4, 16, 64)

        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_position_sensitivity(self):
        """Test that RoPE is position-sensitive."""
        from complexity.core.position import RoPE

        rope = RoPE(dim=64, max_seq_len=2048)

        q = torch.randn(1, 1, 4, 64)
        k = torch.randn(1, 1, 4, 64)

        q_rot1, k_rot1 = rope(q, k, position_ids=torch.tensor([[0, 1, 2, 3]]))
        q_rot2, k_rot2 = rope(q, k, position_ids=torch.tensor([[4, 5, 6, 7]]))

        # Different positions should give different results
        assert not torch.allclose(q_rot1, q_rot2)


class TestSwiGLU:
    """Test SwiGLU MLP."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        from complexity.core.mlp import SwiGLU

        mlp = SwiGLU(hidden_size=256, intermediate_size=512)

        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == x.shape

    def test_swiglu_gating(self):
        """Test that SwiGLU uses gating mechanism."""
        from complexity.core.mlp import SwiGLU

        mlp = SwiGLU(hidden_size=64, intermediate_size=128)

        # Should have gate and up projections
        assert hasattr(mlp, 'gate_proj') or hasattr(mlp, 'w1')
        assert hasattr(mlp, 'up_proj') or hasattr(mlp, 'w2') or hasattr(mlp, 'w3')


class TestGeGLU:
    """Test GeGLU MLP."""

    def test_geglu_forward(self):
        """Test GeGLU forward pass."""
        from complexity.core.mlp import GeGLU

        mlp = GeGLU(hidden_size=256, intermediate_size=512)

        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == x.shape


class TestTokenRoutedMLP:
    """Test TokenRoutedMLP (MoE) component."""

    def test_token_routed_mlp_forward(self):
        """Test MoE forward pass."""
        from complexity.core.mlp import TokenRoutedMLP

        moe = TokenRoutedMLP(
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            top_k=2,
        )

        x = torch.randn(2, 16, 256)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss >= 0

    def test_token_routed_mlp_load_balancing(self):
        """Test that load balancing loss encourages even distribution."""
        from complexity.core.mlp import TokenRoutedMLP

        moe = TokenRoutedMLP(
            hidden_size=64,
            intermediate_size=128,
            num_experts=8,
            top_k=2,
            aux_loss_weight=0.01,
        )

        # Run multiple batches
        total_aux_loss = 0
        for _ in range(10):
            x = torch.randn(4, 32, 64)
            _, aux_loss = moe(x)
            total_aux_loss += aux_loss.item()

        # Aux loss should be positive (encouraging balance)
        assert total_aux_loss > 0


class TestALiBi:
    """Test ALiBi position encoding."""

    def test_alibi_bias_shape(self):
        """Test ALiBi bias generation."""
        from complexity.core.position import ALiBi

        alibi = ALiBi(num_heads=8)

        bias = alibi.get_bias(seq_len=16)
        assert bias.shape[-2:] == (16, 16)

    def test_alibi_slopes(self):
        """Test ALiBi slope computation."""
        from complexity.core.position import ALiBi

        alibi = ALiBi(num_heads=8)

        # Slopes should be geometric sequence
        slopes = alibi.slopes
        assert len(slopes) == 8
        # Each slope should be smaller than previous
        for i in range(1, len(slopes)):
            assert slopes[i] <= slopes[i - 1]
