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
        from complexity.inference import InferenceEngine
        from complexity.models import ComplexityModel
        from complexity.config import ModelConfig

        config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
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
            num_key_value_heads=1,
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

    @pytest.mark.parametrize("prefix_length", (1, 5))
    def test_generation_preserves_kv_and_two_token_routing_history(self, prefix_length):
        from complexity.config import ModelConfig
        from complexity.inference import GenerationConfig, InferenceConfig, InferenceEngine
        from complexity.inference.engine import DecodingStrategy
        from complexity.models import ComplexityModel

        torch.manual_seed(21)
        model = ComplexityModel(
            ModelConfig(
                hidden_size=32,
                num_hidden_layers=2,
                vocab_size=128,
                max_position_embeddings=32,
                use_custom_kernels=False,
            )
        )
        cached = InferenceEngine(model, InferenceConfig(max_cache_length=32))
        full = InferenceEngine(model, InferenceConfig(use_kv_cache=False))
        assert cached.cache.num_heads == 1
        assert cached.cache.head_dim == 8
        prompt = torch.randint(0, 128, (2, prefix_length))
        generation = GenerationConfig(
            strategy=DecodingStrategy.GREEDY,
            max_new_tokens=4,
            return_logits=True,
        )
        seen = None
        lengths = []

        def verify_history(module, args, kwargs, output):
            nonlocal seen
            token_ids = args[0]
            incoming = kwargs.get("past_key_values")
            if seen is None:
                assert incoming is None
                seen = token_ids.clone()
            else:
                expected_previous = torch.nn.functional.pad(seen, (max(0, 2 - seen.size(1)), 0))[
                    :, -2:
                ]
                assert incoming is not None
                assert all(len(layer_cache) == 3 for layer_cache in incoming)
                assert all(
                    torch.equal(layer_cache[2], expected_previous) for layer_cache in incoming
                )
                seen = torch.cat((seen, token_ids), dim=1)
            expected = torch.nn.functional.pad(seen, (max(0, 2 - seen.size(1)), 0))[:, -2:]
            assert all(len(layer_cache) == 3 for layer_cache in output["past_key_values"])
            assert all(
                torch.equal(layer_cache[2], expected) for layer_cache in output["past_key_values"]
            )
            lengths.append(token_ids.size(1))

        hook = model.register_forward_hook(verify_history, with_kwargs=True)
        cached_result = cached.generate(prompt, generation)
        hook.remove()
        assert lengths == [prefix_length, 1, 1, 1]
        full_result = full.generate(prompt, generation)
        assert torch.equal(cached_result["sequences"], full_result["sequences"])
        torch.testing.assert_close(
            cached_result["logits"], full_result["logits"], atol=2e-5, rtol=2e-5
        )

        seen = None
        lengths.clear()
        hook = model.register_forward_hook(verify_history, with_kwargs=True)
        streamed = torch.stack(list(cached.generate_stream(prompt, generation)), dim=1)
        hook.remove()
        assert lengths == [prefix_length, 1, 1, 1]
        assert torch.equal(streamed, full_result["sequences"][:, prefix_length:])
        assert torch.equal(
            cached.generate(prompt, generation)["sequences"], full_result["sequences"]
        )


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
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=512,
            vocab_size=1000,
        )
        draft_config = ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
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
