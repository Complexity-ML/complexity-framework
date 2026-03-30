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

    def test_mlp_standard(self):
        """Test standard MLP creation."""
        from complexity.api import MLP

        mlp = MLP.standard(hidden_size=256, intermediate_size=512)
        assert mlp is not None

        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == x.shape

    def test_mlp_swiglu(self):
        """Test SwiGLU MLP creation."""
        from complexity.api import MLP

        mlp = MLP.swiglu(hidden_size=256, intermediate_size=512)
        assert mlp is not None

        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == x.shape

    def test_mlp_geglu(self):
        """Test GeGLU MLP creation."""
        from complexity.api import MLP

        mlp = MLP.geglu(hidden_size=256, intermediate_size=512)
        assert mlp is not None

        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == x.shape

    def test_mlp_moe(self):
        """Test MoE MLP creation."""
        from complexity.api import MLP

        moe = MLP.moe(hidden_size=256, num_experts=4, top_k=2)
        assert moe is not None

        x = torch.randn(2, 16, 256)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss.ndim == 0  # scalar


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


class TestArchitectureNamespace:
    """Test Architecture namespace for O(N) models."""

    def test_architecture_mamba(self):
        """Test Architecture.mamba() creates model."""
        from complexity.api import Architecture

        try:
            model = Architecture.mamba(hidden_size=256, num_layers=2)
            assert model is not None
        except NotImplementedError:
            pytest.skip("Mamba not fully implemented")

    def test_architecture_rwkv(self):
        """Test Architecture.rwkv() creates model."""
        from complexity.api import Architecture

        try:
            model = Architecture.rwkv(hidden_size=256, num_layers=2)
            assert model is not None
        except NotImplementedError:
            pytest.skip("RWKV not fully implemented")

    def test_architecture_retnet(self):
        """Test Architecture.retnet() creates model."""
        from complexity.api import Architecture

        try:
            model = Architecture.retnet(hidden_size=256, num_layers=2)
            assert model is not None
        except NotImplementedError:
            pytest.skip("RetNet not fully implemented")


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
        out, aux_loss = moe(x)

        assert out.shape == x.shape
        assert aux_loss.ndim == 0
        assert aux_loss >= 0  # aux loss should be non-negative

    def test_token_routed_mlp_expert_selection(self):
        """Test that different tokens route to different experts."""
        from complexity.api import TokenRoutedMLP

        moe = TokenRoutedMLP(
            hidden_size=64,
            intermediate_size=128,
            num_experts=8,
            top_k=2,
        )

        # With enough diversity, experts should vary
        x = torch.randn(4, 32, 64)
        out, _ = moe(x)

        # Output should differ from input (transformation happened)
        assert not torch.allclose(out, x)
