"""
Complexity Tokenizer - Fast BPE/Unigram/WordPiece tokenization.

Uses HuggingFace tokenizers (Rust backend) for speed.

Usage:
    from complexity.tokenizer import Tokenizer, TokenizerConfig

    # Load preset
    tok = Tokenizer.load("llama3-8b")

    # Train new tokenizer
    tok = Tokenizer.train("./data/", vocab_size=100000)

    # Encode/decode
    ids = tok.encode("Hello world")
    text = tok.decode(ids)
"""

from .config import (
    TokenizerConfig,
    TOKENIZER_PRESETS,
    FORMAT_SPECIAL_TOKENS,
    get_special_tokens,
    get_bos_eos,
)
from .tokenizer import Tokenizer

__all__ = [
    "Tokenizer",
    "TokenizerConfig",
    "TOKENIZER_PRESETS",
    "FORMAT_SPECIAL_TOKENS",
    "get_special_tokens",
    "get_bos_eos",
]
