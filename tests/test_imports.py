"""
Test that all imports work correctly.

Priority: CRITICAL - If imports fail, nothing works.
"""

import pytest


class TestCoreImports:
    """Test core module imports."""

    def test_import_complexity(self):
        """Test main package import."""
        import complexity
        assert hasattr(complexity, '__version__')

    def test_import_core_attention(self):
        """Test attention imports."""
        from complexity.core.attention import (
            MultiHeadAttention,
            GroupedQueryAttention,
            MultiQueryAttention,
        )
        assert MultiHeadAttention is not None
        assert GroupedQueryAttention is not None
        assert MultiQueryAttention is not None

    def test_import_core_mlp(self):
        """Test MLP imports."""
        from complexity.core.mlp import (
            MLP,
            SwiGLU,
            GeGLU,
            TokenRoutedMLP,
        )
        assert MLP is not None
        assert SwiGLU is not None
        assert GeGLU is not None
        assert TokenRoutedMLP is not None

    def test_import_core_normalization(self):
        """Test normalization imports."""
        from complexity.core.normalization import (
            RMSNorm,
            LayerNorm,
        )
        assert RMSNorm is not None
        assert LayerNorm is not None

    def test_import_core_position(self):
        """Test position encoding imports."""
        from complexity.core.position import (
            RoPE,
            ALiBi,
            YaRN,
        )
        assert RoPE is not None
        assert ALiBi is not None
        assert YaRN is not None


class TestAPIImports:
    """Test public API imports."""

    def test_import_api_building_blocks(self):
        """Test building blocks from API."""
        from complexity.api import (
            Attention,
            MLP,
            RMSNorm,
            RoPE,
        )
        assert Attention is not None
        assert MLP is not None
        assert RMSNorm is not None
        assert RoPE is not None

    def test_import_api_dynamics(self):
        """Test INL Dynamics from API."""
        from complexity.api import INLDynamics
        assert INLDynamics is not None

    def test_import_api_cuda(self):
        """Test CUDA namespace from API."""
        from complexity.api import CUDA
        assert CUDA is not None
        assert hasattr(CUDA, 'flash')

    def test_import_api_efficient(self):
        """Test Efficient namespace from API."""
        from complexity.api import Efficient
        assert Efficient is not None
        assert hasattr(Efficient, 'tiny_llm')

    def test_import_api_architecture(self):
        """Test Architecture namespace from API."""
        from complexity.api import Architecture
        assert Architecture is not None
        assert hasattr(Architecture, 'mamba')
        assert hasattr(Architecture, 'rwkv')


class TestModelsImports:
    """Test models module imports."""

    def test_import_transformer(self):
        """Test Transformer import."""
        from complexity.models import Transformer, TransformerConfig
        assert Transformer is not None
        assert TransformerConfig is not None


class TestDynamicsImports:
    """Test dynamics module imports."""

    def test_import_inl_dynamics(self):
        """Test INLDynamics import from dynamics module."""
        from complexity.core.dynamics import INLDynamics
        assert INLDynamics is not None

    def test_import_inl_dynamics_lite(self):
        """Test INLDynamicsLite import."""
        from complexity.core.dynamics import INLDynamicsLite
        assert INLDynamicsLite is not None

    def test_import_dynamics_config(self):
        """Test DynamicsConfig import."""
        from complexity.core.dynamics import DynamicsConfig
        assert DynamicsConfig is not None


class TestCLIImports:
    """Test CLI imports (optional - requires typer)."""

    def test_import_cli_app(self):
        """Test CLI app import."""
        try:
            from complexity.cli.app import app, main
            assert main is not None
            # app may be None if typer not installed
        except ImportError:
            pytest.skip("typer not installed")


class TestOptionalImports:
    """Test optional module imports."""

    def test_import_cuda_flash(self):
        """Test flash attention import (optional)."""
        try:
            from complexity.cuda import FlashAttention
            assert FlashAttention is not None
        except ImportError:
            pytest.skip("CUDA extensions not available")

    def test_import_linear_mamba(self):
        """Test Mamba import."""
        try:
            from complexity.linear import Mamba
            assert Mamba is not None
        except ImportError:
            pytest.skip("Linear architectures not available")

    def test_import_linear_rwkv(self):
        """Test RWKV import."""
        try:
            from complexity.linear import RWKV
            assert RWKV is not None
        except ImportError:
            pytest.skip("Linear architectures not available")
