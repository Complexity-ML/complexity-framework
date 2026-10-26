"""Low-level inference helpers and external clients, imported on demand."""

from importlib import import_module

_EXPORTS = {
    "KVCache": (".kv_cache", "KVCache"),
    "PagedKVCache": (".kv_cache", "PagedKVCache"),
    "SlidingWindowCache": (".kv_cache", "SlidingWindowCache"),
    "SpeculativeDecoder": (".speculative", "SpeculativeDecoder"),
    "SpeculativeConfig": (".speculative", "SpeculativeConfig"),
    "ContinuousBatcher": (".batching", "ContinuousBatcher"),
    "Request": (".batching", "Request"),
    "BatchConfig": (".batching", "BatchConfig"),
    "InferenceEngine": (".engine", "InferenceEngine"),
    "InferenceConfig": (".engine", "InferenceConfig"),
    "GenerationConfig": (".engine", "GenerationConfig"),
    "ExternalBackendName": (".external", "ExternalBackendName"),
    "ExternalGenerationConfig": (".external", "ExternalGenerationConfig"),
    "OpenAICompatibleBackend": (".external", "OpenAICompatibleBackend"),
    "create_external_backend": (".external", "create_external_backend"),
    "SharedOnlineRLLoop": (".shared_online_rl", "SharedOnlineRLLoop"),
    "SharedOnlineRLConfig": (".shared_online_rl", "SharedOnlineRLConfig"),
    "SharedRLEvent": (".shared_online_rl", "SharedRLEvent"),
    "VerifiedToolEpisode": (".tool_rewards", "VerifiedToolEpisode"),
    "build_calculator_episode": (".tool_rewards", "build_calculator_episode"),
    "build_datetime_episode": (".tool_rewards", "build_datetime_episode"),
    "build_verified_tool_episode": (".tool_rewards", "build_verified_tool_episode"),
    "extract_arithmetic_expression": (".tool_rewards", "extract_arithmetic_expression"),
    "extract_datetime_hint": (".tool_rewards", "extract_datetime_hint"),
    "safe_calculator": (".tool_rewards", "safe_calculator"),
    "safe_datetime": (".tool_rewards", "safe_datetime"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
