"""
BPE Tokenizer - Byte-Pair Encoding for Complexity framework.

Implements BPE from scratch (no external dependencies except regex).
Compatible with the training pipeline.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter


class ComplexityTokenizer:
    """
    BPE Tokenizer for Complexity framework.

    Features:
    - Train from scratch on any text corpus
    - Save/load tokenizer
    - Encode text to token IDs
    - Decode token IDs to text
    - Special tokens support (PAD, BOS, EOS, UNK)
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize tokenizer with vocabulary and merges.

        Args:
            vocab: Token to ID mapping
            merges: List of BPE merge pairs
            special_tokens: Special token to ID mapping
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or {}

        # Build reverse vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

        # Build merge ranks for fast lookup
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}

        # Regex pattern for tokenization (GPT-2 style)
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens.get(self.PAD_TOKEN, 0)

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens.get(self.BOS_TOKEN, 1)

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens.get(self.EOS_TOKEN, 2)

    @property
    def unk_token_id(self) -> int:
        return self.special_tokens.get(self.UNK_TOKEN, 3)

    @classmethod
    def train(
        cls,
        files: Union[str, List[str]],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> "ComplexityTokenizer":
        """
        Train a new BPE tokenizer on text files.

        Args:
            files: Path(s) to training text files
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for BPE merges
            special_tokens: Additional special tokens
            verbose: Print progress

        Returns:
            Trained ComplexityTokenizer
        """
        if isinstance(files, str):
            files = [files]

        # Default special tokens
        default_special = [cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN]
        all_special = default_special + (special_tokens or [])

        # Read all text
        if verbose:
            print(f"Reading {len(files)} file(s)...")

        text = ""
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                text += fp.read() + "\n"

        if verbose:
            print(f"Total characters: {len(text):,}")

        # Tokenize into words (pre-tokenization)
        pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )
        words = pattern.findall(text)

        # Count word frequencies
        word_freqs = Counter(words)

        if verbose:
            print(f"Unique words: {len(word_freqs):,}")

        # Initialize vocab with bytes (0-255) + special tokens
        vocab = {token: i for i, token in enumerate(all_special)}
        for i in range(256):
            vocab[bytes([i]).decode("utf-8", errors="replace")] = len(vocab)

        # Convert words to byte sequences
        splits = {}
        for word, freq in word_freqs.items():
            # Convert word to list of characters/bytes
            chars = list(word.encode("utf-8").decode("utf-8", errors="replace"))
            splits[word] = (chars, freq)

        # BPE merges
        merges = []
        num_merges = vocab_size - len(vocab)

        if verbose:
            print(f"Training {num_merges} BPE merges...")

        for i in range(num_merges):
            # Count pairs
            pair_freqs = Counter()
            for word, (chars, freq) in splits.items():
                for j in range(len(chars) - 1):
                    pair = (chars[j], chars[j + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            if best_freq < min_frequency:
                break

            # Merge pair
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            merges.append(best_pair)

            # Update splits
            for word in splits:
                chars, freq = splits[word]
                new_chars = []
                j = 0
                while j < len(chars):
                    if j < len(chars) - 1 and chars[j] == best_pair[0] and chars[j + 1] == best_pair[1]:
                        new_chars.append(new_token)
                        j += 2
                    else:
                        new_chars.append(chars[j])
                        j += 1
                splits[word] = (new_chars, freq)

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Merge {i + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' (freq: {best_freq})")

        if verbose:
            print(f"Final vocab size: {len(vocab)}")

        special_token_ids = {token: vocab[token] for token in all_special}

        return cls(vocab, merges, special_token_ids)

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply BPE to a single word."""
        # Convert to characters
        chars = list(word)

        if len(chars) <= 1:
            return chars

        # Apply merges
        while len(chars) > 1:
            # Find best merge
            best_pair = None
            best_rank = float("inf")

            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            # Apply merge
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == best_pair[0] and chars[i + 1] == best_pair[1]:
                    new_chars.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars

        return chars

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end

        Returns:
            List of token IDs
        """
        # Pre-tokenize
        words = self.pattern.findall(text)

        # Tokenize each word
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)

        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.unk_token_id)

        if add_eos:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special: Skip special tokens in output

        Returns:
            Decoded text
        """
        tokens = []
        special_ids = set(self.special_tokens.values()) if skip_special else set()

        for id in ids:
            if id in special_ids:
                continue
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                tokens.append(self.UNK_TOKEN)

        return "".join(tokens)

    def save(self, path: str):
        """
        Save tokenizer to directory.

        Args:
            path: Directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocab
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merges as JSON (handles spaces in tokens)
        with open(path / "merges.json", "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False)

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ComplexityTokenizer":
        """
        Load tokenizer from directory.

        Args:
            path: Directory path

        Returns:
            Loaded ComplexityTokenizer
        """
        path = Path(path)

        # Load vocab (ensure values are ints)
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            vocab = {k: int(v) for k, v in json.load(f).items()}

        # Load merges
        with open(path / "merges.json", "r", encoding="utf-8") as f:
            merges_list = json.load(f)
            merges = [tuple(pair) for pair in merges_list]

        # Load config
        with open(path / "tokenizer_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(vocab, merges, config.get("special_tokens"))

    def __repr__(self) -> str:
        return f"ComplexityTokenizer(vocab_size={self.vocab_size})"
