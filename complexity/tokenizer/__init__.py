"""
Complexity Tokenizer - BPE tokenizer for LLM training.

Usage:
    from complexity.tokenizer import ComplexityTokenizer

    # Train a new tokenizer
    tokenizer = ComplexityTokenizer.train(
        files=["corpus.txt"],
        vocab_size=32000,
    )
    tokenizer.save("my_tokenizer")

    # Load existing tokenizer
    tokenizer = ComplexityTokenizer.load("my_tokenizer")

    # Encode/Decode
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)
"""

from .bpe_tokenizer import ComplexityTokenizer

__all__ = ["ComplexityTokenizer"]
