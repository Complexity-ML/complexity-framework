"""Tokenizer configuration and implementations loaded on demand.

    from complexity.tokenizer import Tokenizer, TokenizerConfig
    # tokenizer = Tokenizer.load("/path/to/existing/tokenizer")

Agentic tokenizer construction is available in complexity.tokenizer.training.
Existing tokenizer/checkpoint identifiers are preserved; training a new tokenizer
is a separate operation from resuming a model that already has a pinned one.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "Tokenizer": (".tokenizer", "Tokenizer"),
    "TokenizerConfig": (".config", "TokenizerConfig"),
    "TOKENIZER_PRESETS": (".config", "TOKENIZER_PRESETS"),
    "FORMAT_SPECIAL_TOKENS": (".config", "FORMAT_SPECIAL_TOKENS"),
    "get_special_tokens": (".config", "get_special_tokens"),
    "get_bos_eos": (".config", "get_bos_eos"),
}

__all__ = [
    "Tokenizer",
    "TokenizerConfig",
    "TOKENIZER_PRESETS",
    "FORMAT_SPECIAL_TOKENS",
    "get_special_tokens",
    "get_bos_eos",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(_import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
