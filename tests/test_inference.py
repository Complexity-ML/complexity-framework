"""Tests for complexity.inference module."""

import pytest
import torch


class TestKVCache:
    """Test KV Cache implementations."""

    def test_kv_cache_basic(self):
        """Test basic KV cache."""
        from complexity.inference import KVCache

        cache = KVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_seq_len=128,
        )

        assert cache is not None
        assert cache.num_layers == 2

    def test_kv_cache_update(self):
        """Test updating KV cache."""
        from complexity.inference import KVCache

        cache = KVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_seq_len=128,
        )

        # Simulate update
        batch_size = 2
        seq_len = 10
        k = torch.randn(batch_size, 4, seq_len, 32)
        v = torch.randn(batch_size, 4, seq_len, 32)

        cache.update(layer_idx=0, key=k, value=v)
        cached_k, cached_v = cache.get(layer_idx=0)

        assert cached_k.shape[-2] == seq_len
        assert cached_v.shape[-2] == seq_len

    def test_paged_kv_cache(self):
        """Test paged KV cache."""
        from complexity.inference import PagedKVCache

        cache = PagedKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            page_size=16,
            max_pages=100,
        )

        assert cache is not None

    def test_sliding_window_cache(self):
        """Test sliding window cache."""
        from complexity.inference import SlidingWindowCache

        cache = SlidingWindowCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            window_size=64,
        )

        assert cache is not None
        assert cache.window_size == 64


class TestInferenceEngine:
    """Test inference engine."""

    def test_create_engine(self):
        """Test creating inference engine."""
        from complexity.inference import InferenceEngine, InferenceConfig
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(config)

        engine = InferenceEngine(model)
        assert engine is not None

    def test_engine_with_config(self):
        """Test engine with custom config."""
        from complexity.inference import InferenceEngine, InferenceConfig
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        model_config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = ComplexityModel(model_config)

        inference_config = InferenceConfig(
            cache_type="standard",
            max_batch_size=4,
        )

        engine = InferenceEngine(model, inference_config)
        assert engine is not None


class TestGenerationConfig:
    """Test generation config."""

    def test_default_config(self):
        """Test default generation config."""
        from complexity.inference import GenerationConfig

        config = GenerationConfig()

        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9

    def test_custom_config(self):
        """Test custom generation config."""
        from complexity.inference import GenerationConfig

        config = GenerationConfig(
            max_new_tokens=200,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
        )

        assert config.max_new_tokens == 200
        assert config.temperature == 0.7


class TestSpeculativeDecoder:
    """Test speculative decoding."""

    def test_speculative_config(self):
        """Test speculative decoding config."""
        from complexity.inference import SpeculativeConfig

        config = SpeculativeConfig(
            num_speculative_tokens=4,
            acceptance_threshold=0.8,
        )

        assert config.num_speculative_tokens == 4

    @pytest.mark.skip(reason="Requires two models - expensive test")
    def test_speculative_decoder(self):
        """Test speculative decoder creation."""
        from complexity.inference import SpeculativeDecoder, SpeculativeConfig
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        # Create target (large) and draft (small) models
        target_config = ModelConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=512,
            vocab_size=1000,
        )
        draft_config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )

        target = ComplexityModel(target_config)
        draft = ComplexityModel(draft_config)

        decoder = SpeculativeDecoder(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(num_speculative_tokens=4),
        )

        assert decoder is not None


class TestContinuousBatcher:
    """Test continuous batching."""

    def test_batch_config(self):
        """Test batch config."""
        from complexity.inference import BatchConfig

        config = BatchConfig(
            max_batch_size=32,
            max_tokens_per_batch=4096,
        )

        assert config.max_batch_size == 32

    def test_request(self):
        """Test request creation."""
        from complexity.inference import Request

        request = Request(
            input_ids=torch.randint(0, 1000, (1, 16)),
            max_tokens=100,
        )

        assert request.input_ids.shape == (1, 16)
        assert request.max_tokens == 100
