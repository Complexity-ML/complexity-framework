"""Self-contained Transformers classes, imported only when requested."""

from importlib import import_module

_EXPORTS = {
    "TRHashConfig": ".configuration_tr_hash_moe",
    "TRHashForCausalLM": ".modeling_tr_hash_moe",
}

__all__ = ["TRHashConfig", "TRHashForCausalLM"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
