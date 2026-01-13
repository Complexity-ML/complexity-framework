"""
Tokenizer - Fast BPE/Unigram/WordPiece using HuggingFace tokenizers (Rust).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Iterator, Callable

import torch

from .config import (
    TokenizerConfig,
    TOKENIZER_PRESETS,
    FORMAT_SPECIAL_TOKENS,
    FORMAT_BOS_EOS,
    get_special_tokens,
    get_bos_eos,
)


class Tokenizer:
    """
    Fast tokenizer using HuggingFace tokenizers (Rust backend).

    Supports multiple formats and training methods.

    Examples:
        # Load preset
        tok = Tokenizer.load("llama3-8b")

        # Train new tokenizer
        tok = Tokenizer.train("./data/", vocab_size=100000)

        # Encode/decode
        ids = tok.encode("Hello world")
        text = tok.decode(ids)

        # Chat format
        ids = tok.encode_chat([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ])
    """

    def __init__(self, tokenizer: Any, config: TokenizerConfig = None, **kwargs):
        self._tokenizer = tokenizer
        self._config = config or TokenizerConfig(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    @property
    def format(self) -> str:
        return self._config.format

    # ==================== Load ====================

    @classmethod
    def load(cls, path: str, **kwargs) -> "Tokenizer":
        """
        Load tokenizer from preset or path.

        Args:
            path: Preset name or path to tokenizer
            **kwargs: Override config values
        """
        if path in TOKENIZER_PRESETS:
            preset = TOKENIZER_PRESETS[path].copy()
            preset.update(kwargs)
            config = TokenizerConfig(**preset)
            special_tokens = get_special_tokens(config.format)
            tokenizer = cls._make_tokenizer(config.vocab_size, special_tokens)
            print(f"[Tokenizer] Loaded preset '{path}'")
            return cls(tokenizer, config)

        # From file
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Not found: {path}")

        config_file = p / "config.json" if p.is_dir() else None
        if config_file and config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            cfg.update(kwargs)
            config = TokenizerConfig(**cfg)
        else:
            config = TokenizerConfig(**kwargs)

        # Load tokenizer
        tok_file = p / "tokenizer.json" if p.is_dir() else p
        if tok_file.exists() and tok_file.suffix == ".json":
            from tokenizers import Tokenizer as HFTok
            tokenizer = HFTok.from_file(str(tok_file))
        else:
            raise ValueError(f"No tokenizer in {path}")

        print(f"[Tokenizer] Loaded from {path}")
        return cls(tokenizer, config)

    # ==================== Train ====================

    @classmethod
    def train(
        cls,
        data: Union[str, Path, List[str]],
        config: TokenizerConfig = None,
        **kwargs,
    ) -> "Tokenizer":
        """
        Train tokenizer on data.

        Args:
            data: File, directory, or list of files
            config: Optional config
            **kwargs: Override config (vocab_size, format, method, ...)
        """
        if config:
            for k, v in kwargs.items():
                setattr(config, k, v)
        else:
            config = TokenizerConfig(**kwargs)

        # Collect files
        files = cls._get_files(data)
        if not files:
            raise ValueError(f"No files in {data}")

        print(f"[Tokenizer] Training {config.method.upper()}...")
        print(f"  Files: {len(files)}, Vocab: {config.vocab_size}, Format: {config.format}")

        special_tokens = get_special_tokens(config.format)

        # Train by method
        if config.method == "bpe":
            tokenizer = cls._train_bpe(files, config.vocab_size, special_tokens, config.min_frequency)
        elif config.method == "unigram":
            tokenizer = cls._train_unigram(files, config.vocab_size, special_tokens)
        elif config.method == "wordpiece":
            tokenizer = cls._train_wordpiece(files, config.vocab_size, special_tokens, config.min_frequency)
        else:
            raise ValueError(f"Unknown method: {config.method}")

        print(f"[Tokenizer] Done!")
        return cls(tokenizer, config)

    @classmethod
    def train_from_iterator(
        cls,
        iterator: Union[Iterator[str], Callable[[], Iterator[str]]],
        config: TokenizerConfig = None,
        **kwargs,
    ) -> "Tokenizer":
        """
        Train tokenizer from an iterator of texts.

        Perfect for streaming datasets (HuggingFace datasets, generators, etc.)

        Args:
            iterator: Iterator yielding text strings, or callable returning iterator
            config: Optional config
            **kwargs: Override config (vocab_size, format, method, ...)

        Examples:
            # From HuggingFace dataset
            from datasets import load_dataset
            ds = load_dataset("Pacific-Prime/edu-web", split="train", streaming=True)

            def text_iter():
                for ex in ds:
                    yield ex["text"]

            tokenizer = Tokenizer.train_from_iterator(text_iter, vocab_size=100000)

            # From generator
            def my_texts():
                yield "Hello world"
                yield "This is a test"

            tokenizer = Tokenizer.train_from_iterator(my_texts(), vocab_size=32000)
        """
        if config:
            for k, v in kwargs.items():
                setattr(config, k, v)
        else:
            config = TokenizerConfig(**kwargs)

        print(f"[Tokenizer] Training {config.method.upper()} from iterator...")
        print(f"  Vocab: {config.vocab_size}, Format: {config.format}")

        special_tokens = get_special_tokens(config.format)

        # Get iterator (call if callable)
        if callable(iterator) and not hasattr(iterator, '__next__'):
            iterator = iterator()

        # Train by method
        if config.method == "bpe":
            tokenizer = cls._train_bpe_from_iterator(iterator, config.vocab_size, special_tokens, config.min_frequency)
        elif config.method == "unigram":
            tokenizer = cls._train_unigram_from_iterator(iterator, config.vocab_size, special_tokens)
        elif config.method == "wordpiece":
            tokenizer = cls._train_wordpiece_from_iterator(iterator, config.vocab_size, special_tokens, config.min_frequency)
        else:
            raise ValueError(f"Unknown method: {config.method}")

        print(f"[Tokenizer] Done!")
        return cls(tokenizer, config)

    # ==================== Encode/Decode ====================

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if hasattr(self._tokenizer, "encode_batch"):
            tokens = self._tokenizer.encode(text).ids
        else:
            tokens = self._tokenizer.encode(text)

        if add_special_tokens:
            bos = self._get_special_id("bos")
            eos = self._get_special_id("eos")
            if bos is not None:
                tokens = [bos] + list(tokens)
            if eos is not None:
                tokens = list(tokens) + [eos]

        return list(tokens)

    def decode(self, tokens: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if hasattr(self._tokenizer, "decode_batch"):
            return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        return self._tokenizer.decode(tokens)

    def encode_chat(self, messages: List[Dict], **kwargs) -> List[int]:
        """Encode chat messages to token IDs."""
        text = self._format_chat(messages)
        return self.encode(text, **kwargs)

    # ==================== Save ====================

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if hasattr(self._tokenizer, "save"):
            self._tokenizer.save(str(path / "tokenizer.json"))

        cfg = {k: v for k, v in self._config.__dict__.items() if k != "extra"}
        with open(path / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"[Tokenizer] Saved to {path}")

    # ==================== Properties ====================

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._get_special_id("bos")

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._get_special_id("eos")

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._get_special_id("pad")

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._get_special_id("unk")

    # ==================== Private ====================

    @staticmethod
    def _get_files(data) -> List[Path]:
        if isinstance(data, list):
            return [Path(f) for f in data if Path(f).exists()]
        p = Path(data)
        if p.is_file():
            return [p]
        return list(p.glob("**/*.txt")) + list(p.glob("**/*.jsonl")) + list(p.glob("**/*.json"))

    @staticmethod
    def _make_tokenizer(vocab_size: int, special_tokens: List[str]):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        tok = HFTok(BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.add_special_tokens(special_tokens)
        return tok

    @staticmethod
    def _train_bpe(files, vocab_size, special_tokens, min_freq):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as BLD
        tok = HFTok(BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = BLD()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tok.train([str(f) for f in files], trainer)
        return tok

    @staticmethod
    def _train_unigram(files, vocab_size, special_tokens):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import Unigram
        from tokenizers.trainers import UnigramTrainer
        from tokenizers.pre_tokenizers import Metaspace
        tok = HFTok(Unigram())
        tok.pre_tokenizer = Metaspace()
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            unk_token="<|unk|>",
        )
        tok.train([str(f) for f in files], trainer)
        return tok

    @staticmethod
    def _train_wordpiece(files, vocab_size, special_tokens, min_freq):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        tok = HFTok(WordPiece(unk_token="<|unk|>"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tok.train([str(f) for f in files], trainer)
        return tok

    # ==================== Train from Iterator ====================

    @staticmethod
    def _train_bpe_from_iterator(iterator, vocab_size, special_tokens, min_freq):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as BLD
        tok = HFTok(BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = BLD()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tok.train_from_iterator(iterator, trainer)
        return tok

    @staticmethod
    def _train_unigram_from_iterator(iterator, vocab_size, special_tokens):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import Unigram
        from tokenizers.trainers import UnigramTrainer
        from tokenizers.pre_tokenizers import Metaspace
        tok = HFTok(Unigram())
        tok.pre_tokenizer = Metaspace()
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            unk_token="<|unk|>",
        )
        tok.train_from_iterator(iterator, trainer)
        return tok

    @staticmethod
    def _train_wordpiece_from_iterator(iterator, vocab_size, special_tokens, min_freq):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        tok = HFTok(WordPiece(unk_token="<|unk|>"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tok.train_from_iterator(iterator, trainer)
        return tok

    def _get_special_id(self, typ: str) -> Optional[int]:
        m = get_bos_eos(self._config.format)
        name = m.get(typ)
        if name and hasattr(self._tokenizer, "token_to_id"):
            return self._tokenizer.token_to_id(name)
        return None

    def _format_chat(self, messages: List[Dict]) -> str:
        fmt = self._config.format
        parts = []
        for m in messages:
            r, c = m["role"], m["content"]
            if fmt == "complexity":
                parts.append(f"<|{r}|>{c}<|/{r}|>" if r != "system" else f"<|system|>{c}")
            elif fmt == "llama3":
                parts.append(f"<|start_header_id|>{r}<|end_header_id|>\n\n{c}<|eot_id|>")
            elif fmt == "mistral":
                parts.append(f"[INST]{c}[/INST]" if r == "user" else c)
            elif fmt == "chatml":
                parts.append(f"<|im_start|>{r}\n{c}<|im_end|>")
            elif fmt == "gemma":
                gr = "model" if r == "assistant" else "user"
                parts.append(f"<start_of_turn>{gr}\n{c}<end_of_turn>")
            else:
                parts.append(f"{r}: {c}")
        return "".join(parts) if fmt != "chatml" else "\n".join(parts)

    def __repr__(self):
        return f"Tokenizer(vocab={self._config.vocab_size}, format='{self._config.format}')"

    def __len__(self):
        """Return vocabulary size."""
        if hasattr(self._tokenizer, "get_vocab_size"):
            return self._tokenizer.get_vocab_size()
        return self._config.vocab_size
