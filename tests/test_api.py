"""
Test public API factories and namespaces.

Priority: HIGH - API is the main user interface.
"""

import pytest
import torch


class TestAttentionFactory:
    """Test Attention factory methods."""

    def test_attention_mha(self):
        """Test MHA creation via factory."""
        from complexity.api import Attention

        attn = Attention.mha(hidden_size=256, num_heads=4)
        assert attn is not None

        # Test forward
        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_attention_gqa(self):
        """Test GQA creation via factory."""
        from complexity.api import Attention

        attn = Attention.gqa(hidden_size=256, num_heads=4, num_kv_heads=2)
        assert attn is not None

        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_attention_mqa(self):
        """Test MQA creation via factory."""
        from complexity.api import Attention

        attn = Attention.mqa(hidden_size=256, num_heads=4)
        assert attn is not None

        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == x.shape


class TestMLPFactory:
    """Test MLP factory methods."""

    def test_mlp_moe(self):
        """Test MoE MLP creation."""
        from complexity.api import MLP

        moe = MLP.moe(hidden_size=256, num_experts=4, top_k=2)
        assert moe is not None

        x = torch.randn(2, 16, 256)
        token_ids = torch.randint(0, 100000, (2, 16))
        out, aux_loss = moe(x, token_ids=token_ids)
        assert out.shape == x.shape
        assert aux_loss.ndim == 0  # scalar


class TestBlockFactory:
    def test_public_moe_block_uses_canonical_attention_routes_and_cache(self):
        from complexity.api import Block
        from complexity.core.attention.gqa import TRHashAttention
        from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP

        torch.manual_seed(17)
        block = Block.moe(
            hidden_size=32,
            vocab_size=64,
            num_hidden_layers=2,
            layer_idx=1,
            use_custom_kernels=False,
        ).eval()
        assert isinstance(block.self_attn, TRHashAttention)
        assert isinstance(block.mlp, TRHashEngineMLP)
        assert (block.config.num_attention_heads, block.config.num_key_value_heads) == (4, 1)
        assert (block.config.num_experts, block.config.top_k) == (4, 2)
        assert block.config.mlp_type == "tr_hash_engine"
        assert block.mlp.engine.config.layer_index == block.layer_idx == 1
        assert block.mlp.engine.config.route_weights == (0.5, 0.5)
        assert (
            block.mlp.engine.config.shared_output_scale,
            block.mlp.engine.config.routed_output_scale,
        ) == (1, 2)
        hidden = torch.randn(2, 6, 32)
        ids = torch.tensor([[1, 7, 13, 2, 3, 5], [8, 6, 4, 2, 9, 7]])
        with torch.no_grad():
            full, _, _ = block(hidden, token_ids=ids)
            prefix, cache, _ = block(hidden[:, :4], token_ids=ids[:, :4], use_cache=True)
            assert len(cache) == 3
            assert torch.equal(cache[2], ids[:, 2:4])
            suffix, cache, _ = block(
                hidden[:, 4:],
                token_ids=ids[:, 4:],
                past_key_value=cache,
                use_cache=True,
            )
        torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), full, atol=2e-7, rtol=3e-6)
        assert torch.equal(cache[2], ids[:, -2:])

    @pytest.mark.parametrize(
        "options",
        (
            {"num_experts": 8},
            {"top_k": 1},
            {"num_heads": 8},
            {"num_key_value_heads": 2},
            {"mlp_type": "token_routed"},
        ),
    )
    def test_public_moe_block_rejects_incompatible_architectures(self, options):
        from complexity.api import Block

        with pytest.raises(ValueError, match="TR-HASH"):
            Block.moe(hidden_size=32, vocab_size=64, **options)


class TestCUDANamespace:
    """Test CUDA namespace."""

    def test_cuda_flash_factory(self):
        """Test CUDA.flash() returns attention module."""
        from complexity.api import CUDA

        # Should return fallback if flash-attn not installed
        attn = CUDA.flash(hidden_size=256, num_heads=4)
        assert attn is not None

        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_cuda_sliding_factory(self):
        """Test CUDA.sliding() returns attention module."""
        from complexity.api import CUDA

        attn = CUDA.sliding(hidden_size=256, num_heads=4, window_size=512)
        assert attn is not None

    def test_cuda_sparse_factory(self):
        """Test CUDA.sparse() returns attention module."""
        from complexity.api import CUDA

        attn = CUDA.sparse(hidden_size=256, num_heads=4)
        assert attn is not None


class TestEfficientNamespace:
    """Test Efficient namespace."""

    def test_efficient_nano_llm(self):
        """Test Efficient.nano_llm() creates small model."""
        from complexity.api import Efficient

        model = Efficient.nano_llm(vocab_size=1000)
        assert model is not None

        # Check it's small
        params = sum(p.numel() for p in model.parameters())
        assert params < 50_000_000  # < 50M

    def test_efficient_tiny_llm(self):
        """Test Efficient.tiny_llm() creates model."""
        from complexity.api import Efficient

        model = Efficient.tiny_llm(vocab_size=1000)
        assert model is not None

    def test_efficient_checkpointing(self):
        """Test Efficient.enable_checkpointing()."""
        from complexity.api import Efficient

        model = Efficient.nano_llm(vocab_size=1000)
        Efficient.enable_checkpointing(model)
        # Should not raise

    @pytest.mark.parametrize("size", ("nano", "micro", "tiny", "small", "medium"))
    def test_efficient_presets_keep_canonical_tr_hash_attention(self, size):
        from complexity.api import SmallModels
        from complexity.core.attention.gqa import TRHashAttention

        with torch.device("meta"):
            model = SmallModels.create(size, vocab_size=32)
        config = model.config
        assert (config.num_attention_heads, config.num_key_value_heads) == (4, 1)
        assert (config.num_experts, config.top_k) == (4, 2)
        assert config.intermediate_size == config.hidden_size
        assert all(isinstance(layer.self_attn, TRHashAttention) for layer in model.layers)

    @pytest.mark.parametrize("use_flash", (False, True))
    def test_efficient_flash_policy_preserves_gqa(self, use_flash):
        from complexity.api import SmallModels

        with torch.device("meta"):
            model = SmallModels.nano(vocab_size=32, use_flash=use_flash)
        assert model.config.attention_type == "gqa"
        assert model.config.use_sdpa == use_flash

    def test_efficient_rejects_incompatible_tr_attention(self):
        from complexity.api import SmallModels

        with pytest.raises(ValueError, match="requires GQA"):
            SmallModels.nano(vocab_size=32, use_gqa=False)
        with pytest.raises(ValueError, match="four query heads"):
            SmallModels.nano(vocab_size=32, num_attention_heads=8)


class TestTokenRoutedMLP:
    """Test TokenRoutedMLP (MoE) directly."""

    def test_token_routed_mlp_forward(self):
        """Test TokenRoutedMLP forward pass."""
        from complexity.api import TokenRoutedMLP

        moe = TokenRoutedMLP(
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            top_k=2,
        )

        x = torch.randn(2, 16, 256)
        token_ids = torch.randint(0, 100000, (2, 16))
        out, aux_loss = moe(x, token_ids=token_ids)

        assert out.shape == x.shape
        assert aux_loss.ndim == 0
        assert aux_loss >= 0  # aux loss should be non-negative

    def test_token_routed_mlp_expert_selection(self):
        """Test that different tokens route to different experts."""
        from complexity.api import TokenRoutedMLP

        moe = TokenRoutedMLP(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            top_k=2,
        )

        x = torch.randn(4, 32, 64)
        token_ids = torch.arange(128).reshape(4, 32)
        out, _ = moe(x, token_ids=token_ids)
        out.square().mean().backward()

        assert not torch.allclose(out, x)
        gradient = moe.mlp.engine.expert_gate.grad
        assert gradient is not None
        assert gradient.abs().sum(dim=(1, 2)).gt(0).all()
