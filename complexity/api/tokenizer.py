"""
Tokenizer API - Flexible avec defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field

import torch


@dataclass
class TokenizerConfig:
    """Config tokenizer - tous les champs ont des defaults, tout est overridable."""
    vocab_size: int = 32000
    format: str = "complexity"
    method: str = "bpe"
    min_frequency: int = 2
    max_length: int = 2048
    # Override n'importe quoi via **kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Merge extra dans les attributs
        for k, v in self.extra.items():
            setattr(self, k, v)


class Tokenizer:
    """
    Tokenizer flexible.

    Defaults sensibles, mais TOUT est overridable via **kwargs.

    Examples:
        # Simple
        tok = Tokenizer.load("llama-7b")

        # Override ce qu'on veut
        tok = Tokenizer.train("./data/", vocab_size=50000, method="unigram", min_frequency=5)

        # Passer une config custom
        tok = Tokenizer.train("./data/", config=TokenizerConfig(vocab_size=100000))
    """

    PRESETS = {
        "complexity-7b": {"vocab_size": 65536, "format": "complexity"},
        "llama-7b": {"vocab_size": 32000, "format": "llama3"},
        "llama3-8b": {"vocab_size": 128256, "format": "llama3"},
        "mistral-7b": {"vocab_size": 32768, "format": "mistral"},
        "gemma-7b": {"vocab_size": 256000, "format": "gemma"},
    }

    FORMAT_TOKENS = {
        "complexity": ["<|begin|>", "<|end|>", "<|pad|>", "<|unk|>", "<|system|>", "<|user|>", "<|assistant|>", "<|reason|>", "<|step|>", "<|conclude|>"],
        "llama3": ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        "mistral": ["<s>", "</s>", "[INST]", "[/INST]"],
        "chatml": ["<|im_start|>", "<|im_end|>"],
        "gemma": ["<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>"],
    }

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
        Charge tokenizer. Override n'importe quoi via kwargs.

        Args:
            path: Preset ou chemin
            **kwargs: Override config (vocab_size, format, ...)
        """
        if path in cls.PRESETS:
            preset = cls.PRESETS[path].copy()
            preset.update(kwargs)  # kwargs override preset
            config = TokenizerConfig(**preset)
            special_tokens = cls.FORMAT_TOKENS.get(config.format, [])
            tokenizer = cls._make_tokenizer(config.vocab_size, special_tokens)
            print(f"[Tokenizer] Loaded preset '{path}'")
            return cls(tokenizer, config)

        # Depuis fichier
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Not found: {path}")

        config_file = p / "config.json" if p.is_dir() else None
        if config_file and config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            cfg.update(kwargs)  # kwargs override
            config = TokenizerConfig(**cfg)
        else:
            config = TokenizerConfig(**kwargs)

        # Charger tokenizer
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
        Entraîne tokenizer. Override tout via kwargs.

        Args:
            data: Données (fichier, dossier, liste)
            config: Config (optionnel)
            **kwargs: Override (vocab_size, format, method, min_frequency, ...)
        """
        # Config = fournie ou créée, puis kwargs override
        if config:
            for k, v in kwargs.items():
                setattr(config, k, v)
        else:
            config = TokenizerConfig(**kwargs)

        # Collecter fichiers
        files = cls._get_files(data)
        if not files:
            raise ValueError(f"No files in {data}")

        print(f"[Tokenizer] Training {config.method.upper()}...")
        print(f"  Files: {len(files)}, Vocab: {config.vocab_size}, Format: {config.format}")

        special_tokens = cls.FORMAT_TOKENS.get(config.format, [])

        # Train selon méthode
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

    # ==================== Encode/Decode ====================

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode. kwargs override defaults (add_special_tokens, ...)"""
        add_special = kwargs.get("add_special_tokens", True)

        if hasattr(self._tokenizer, "encode_batch"):
            tokens = self._tokenizer.encode(text).ids
        else:
            tokens = self._tokenizer.encode(text)

        if add_special:
            bos = self._get_special_id("bos")
            eos = self._get_special_id("eos")
            if bos is not None:
                tokens = [bos] + list(tokens)
            if eos is not None:
                tokens = list(tokens) + [eos]

        return list(tokens)

    def decode(self, tokens: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode. kwargs override defaults."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        skip_special = kwargs.get("skip_special_tokens", True)

        if hasattr(self._tokenizer, "decode_batch"):
            return self._tokenizer.decode(tokens, skip_special_tokens=skip_special)
        return self._tokenizer.decode(tokens)

    def encode_chat(self, messages: List[Dict], **kwargs) -> List[int]:
        """Encode chat. kwargs passés à encode."""
        text = self._format_chat(messages)
        return self.encode(text, **kwargs)

    # ==================== Save ====================

    def save(self, path: Union[str, Path]) -> None:
        """Sauvegarde."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if hasattr(self._tokenizer, "save"):
            self._tokenizer.save(str(path / "tokenizer.json"))

        cfg = {k: v for k, v in self._config.__dict__.items() if k != "extra"}
        with open(path / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"[Tokenizer] Saved to {path}")

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
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=special_tokens, show_progress=True)
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
        trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True, unk_token="<|unk|>")
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
        trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=special_tokens, show_progress=True)
        tok.train([str(f) for f in files], trainer)
        return tok

    def _get_special_id(self, typ: str) -> Optional[int]:
        MAP = {
            "complexity": {"bos": "<|begin|>", "eos": "<|end|>"},
            "llama3": {"bos": "<|begin_of_text|>", "eos": "<|end_of_text|>"},
            "mistral": {"bos": "<s>", "eos": "</s>"},
            "chatml": {"bos": "<|im_start|>", "eos": "<|im_end|>"},
            "gemma": {"bos": "<bos>", "eos": "<eos>"},
        }
        m = MAP.get(self._config.format, {})
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
