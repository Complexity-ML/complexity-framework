"""
Tokenizer Configuration - Formats, special tokens, presets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class TokenizerConfig:
    """Config tokenizer - tous les champs ont des defaults, tout est overridable."""
    vocab_size: int = 32000
    format: str = "complexity"
    method: str = "bpe"
    min_frequency: int = 2
    max_length: int = 2048
    num_reserved_tokens: int = 256  # Reserved for future special tokens
    # Override n'importe quoi via **kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Merge extra dans les attributs
        for k, v in self.extra.items():
            setattr(self, k, v)


# === Presets ===
TOKENIZER_PRESETS = {
    "complexity-7b": {"vocab_size": 65536, "format": "complexity"},
    "llama-7b": {"vocab_size": 32000, "format": "llama3"},
    "llama3-8b": {"vocab_size": 128256, "format": "llama3"},
    "mistral-7b": {"vocab_size": 32768, "format": "mistral"},
    "gemma-7b": {"vocab_size": 256000, "format": "gemma"},
}


# === Format-specific special tokens ===
FORMAT_SPECIAL_TOKENS = {
    "complexity": [
        # Control tokens (0-9)
        "<|begin|>", "<|end|>", "<|pad|>", "<|unk|>",
        # Role tokens (10-19)
        "<|system|>", "<|user|>", "<|assistant|>",
        # Reasoning tokens (20-29)
        "<|reason|>", "<|step|>", "<|conclude|>",
        # Tool tokens (30-39)
        "<|tool_call|>", "<|tool_result|>",
        # Code tokens (40-49)
        "<|code|>", "<|/code|>",
    ],
    "llama3": [
        "<|begin_of_text|>", "<|end_of_text|>",
        "<|start_header_id|>", "<|end_header_id|>",
        "<|eot_id|>",
    ],
    "mistral": [
        "<s>", "</s>",
        "[INST]", "[/INST]",
    ],
    "chatml": [
        "<|im_start|>", "<|im_end|>",
    ],
    "gemma": [
        "<bos>", "<eos>",
        "<start_of_turn>", "<end_of_turn>",
    ],
}


# === BOS/EOS mapping per format ===
FORMAT_BOS_EOS = {
    "complexity": {"bos": "<|begin|>", "eos": "<|end|>", "pad": "<|pad|>", "unk": "<|unk|>"},
    "llama3": {"bos": "<|begin_of_text|>", "eos": "<|end_of_text|>"},
    "mistral": {"bos": "<s>", "eos": "</s>"},
    "chatml": {"bos": "<|im_start|>", "eos": "<|im_end|>"},
    "gemma": {"bos": "<bos>", "eos": "<eos>"},
}


def get_special_tokens(format: str) -> List[str]:
    """Get special tokens for a format."""
    return FORMAT_SPECIAL_TOKENS.get(format, [])


def get_bos_eos(format: str) -> Dict[str, str]:
    """Get BOS/EOS tokens for a format."""
    return FORMAT_BOS_EOS.get(format, {"bos": "<s>", "eos": "</s>"})
