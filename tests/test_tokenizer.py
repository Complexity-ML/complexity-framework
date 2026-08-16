"""Tests for complexity.tokenizer module."""

import pytest


class TestTokenizerConfig:
    """Test TokenizerConfig."""

    def test_default_config(self):
        from complexity.tokenizer import TokenizerConfig
        config = TokenizerConfig()
        assert config.vocab_size == 32000
        assert config.format == "complexity"
        assert config.method == "bpe"

    def test_custom_config(self):
        from complexity.tokenizer import TokenizerConfig
        config = TokenizerConfig(vocab_size=100000, format="llama3")
        assert config.vocab_size == 100000
        assert config.format == "llama3"


class TestTokenizerPresets:
    """Test tokenizer presets."""

    def test_presets_exist(self):
        from complexity.tokenizer import TOKENIZER_PRESETS
        assert "complexity-7b" in TOKENIZER_PRESETS
        assert "llama-7b" in TOKENIZER_PRESETS
        assert "llama3-8b" in TOKENIZER_PRESETS
        assert "mistral-7b" in TOKENIZER_PRESETS

    def test_preset_values(self):
        from complexity.tokenizer import TOKENIZER_PRESETS
        assert TOKENIZER_PRESETS["llama3-8b"]["vocab_size"] == 128256
        assert TOKENIZER_PRESETS["complexity-7b"]["vocab_size"] == 65536


class TestSpecialTokens:
    """Test special tokens configuration."""

    def test_format_tokens(self):
        from complexity.tokenizer import FORMAT_SPECIAL_TOKENS, get_special_tokens

        # Complexity format
        tokens = get_special_tokens("complexity")
        assert "<|begin|>" in tokens
        assert "<|end|>" in tokens
        assert "<|user|>" in tokens
        assert "<|assistant|>" in tokens

        # Llama3 format
        tokens = get_special_tokens("llama3")
        assert "<|begin_of_text|>" in tokens
        assert "<|end_of_text|>" in tokens

    def test_bos_eos(self):
        from complexity.tokenizer import get_bos_eos

        bos_eos = get_bos_eos("complexity")
        assert bos_eos["bos"] == "<|begin|>"
        assert bos_eos["eos"] == "<|end|>"

        bos_eos = get_bos_eos("mistral")
        assert bos_eos["bos"] == "<s>"
        assert bos_eos["eos"] == "</s>"


class TestTokenizer:
    """Test Tokenizer class."""

    @pytest.mark.skipif(
        not pytest.importorskip("tokenizers", reason="tokenizers not installed"),
        reason="tokenizers library required"
    )
    def test_load_preset(self):
        from complexity.tokenizer import Tokenizer
        try:
            tok = Tokenizer.load("complexity-7b")
            assert tok.vocab_size == 65536
            assert tok.format == "complexity"
        except ImportError:
            pytest.skip("tokenizers library not installed")

    def test_tokenizer_repr(self):
        from complexity.tokenizer import Tokenizer, TokenizerConfig
        # Create minimal tokenizer for repr test
        config = TokenizerConfig(vocab_size=32000, format="complexity")
        # Note: Full tokenizer tests require tokenizers library
