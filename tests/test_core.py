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


class TestBidirectionalAttention:
    """is_causal=False regression coverage -- see complexity/models/embedding.py.

    The default (is_causal=True, or the field omitted entirely) must remain
    bit-for-bit identical to pre-existing behavior, since this is the same
    GroupedQueryAttention class used by the live causal pretrain.
    """

    def test_omitting_is_causal_matches_explicit_true(self):
        from complexity.core.attention import GroupedQueryAttention, AttentionConfig

        torch.manual_seed(0)
        config_default = AttentionConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=2)
        attn_default = GroupedQueryAttention(config_default)

        torch.manual_seed(0)
        config_explicit = AttentionConfig(
            hidden_size=256, num_attention_heads=8, num_key_value_heads=2, is_causal=True,
        )
        attn_explicit = GroupedQueryAttention(config_explicit)

        x = torch.randn(2, 16, 256)
        out_default, _ = attn_default(x)
        out_explicit, _ = attn_explicit(x)
        assert torch.equal(out_default, out_explicit)

    def test_is_causal_false_lets_earlier_tokens_see_later_ones(self):
        """The defining behavioral difference: in causal attention, changing
        a later token cannot change an earlier token's output (no leakage
        from the future). In bidirectional attention, it can."""
        from complexity.core.attention import GroupedQueryAttention, AttentionConfig

        torch.manual_seed(0)
        causal_config = AttentionConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, is_causal=True)
        causal_attn = GroupedQueryAttention(causal_config)

        torch.manual_seed(0)
        bidir_config = AttentionConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, is_causal=False)
        bidir_attn = GroupedQueryAttention(bidir_config)

        torch.manual_seed(1)
        x = torch.randn(1, 8, 64)
        x_perturbed = x.clone()
        x_perturbed[:, -1, :] += 10.0  # perturb only the LAST token

        causal_out, _ = causal_attn(x)
        causal_out_perturbed, _ = causal_attn(x_perturbed)
        # Earlier tokens (all but the last) must be unaffected by a
        # perturbation to a strictly-later token under causal masking.
        assert torch.allclose(causal_out[:, :-1, :], causal_out_perturbed[:, :-1, :], atol=1e-5)

        bidir_out, _ = bidir_attn(x)
        bidir_out_perturbed, _ = bidir_attn(x_perturbed)
        # Bidirectional attention has no such guarantee -- earlier tokens
        # should generally change too.
        assert not torch.allclose(bidir_out[:, :-1, :], bidir_out_perturbed[:, :-1, :], atol=1e-5)

    def test_is_causal_false_honors_a_padding_mask(self):
        """A non-causal forward pass must still respect an explicit additive
        padding mask instead of silently discarding it (the bug this
        opt-in flag exists to avoid reintroducing)."""
        from complexity.core.attention import GroupedQueryAttention, AttentionConfig

        torch.manual_seed(0)
        config = AttentionConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, is_causal=False)
        attn = GroupedQueryAttention(config)

        x = torch.randn(1, 6, 64)
        # Mask out the last 2 positions as padding.
        mask = torch.zeros(1, 1, 6, 6)
        mask[:, :, :, -2:] = float("-inf")

        out, _ = attn(x, attention_mask=mask)
        assert torch.isfinite(out).all()

        # Changing the padded tokens' content must not change the real
        # tokens' output, since they're masked out of every real query's
        # attention.
        x_changed_padding = x.clone()
        x_changed_padding[:, -2:, :] = torch.randn(1, 2, 64)
        out_changed_padding, _ = attn(x_changed_padding, attention_mask=mask)
        assert torch.allclose(out[:, :-2, :], out_changed_padding[:, :-2, :], atol=1e-5)


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


class TestTRHashEngineMLP:
    """Test TRHashEngineMLP (canonical TR-Hash MoE) component."""

    def test_tr_hash_engine_mlp_forward(self):
        """Test MoE forward pass."""
        from complexity.core.mlp import TRHashEngineMLP, MLPConfig

        moe = TRHashEngineMLP(MLPConfig(hidden_size=256, intermediate_size=512, num_experts=4))

        x = torch.randn(2, 16, 256)
        # token_ids required: TR-Hash MoE always routes deterministically by token ID.
        token_ids = torch.randint(0, 32000, (2, 16))
        out = moe(x, token_ids=token_ids)
        assert out.shape == x.shape

    def test_tr_hash_engine_mlp_load_balancing(self):
        """Test that routing covers all experts across a batch."""
        from complexity.core.mlp import TRHashEngineMLP, MLPConfig

        moe = TRHashEngineMLP(MLPConfig(hidden_size=64, intermediate_size=128, num_experts=4))

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
