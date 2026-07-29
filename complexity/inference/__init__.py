"""
Inference optimization module for framework-complexity.

Provides low-level inference utilities plus external serving clients.
Production text generation is delegated to vLLM/SGLang OpenAI-compatible
servers; Complexity's PyTorch model layer does not own a native generate loop.

Usage:
    from complexity.inference import (
        KVCache,
        SpeculativeDecoder,
        ContinuousBatcher,
        InferenceEngine,
    )

    # Production generation through vLLM/SGLang
    backend = create_external_backend("vllm", base_url="http://localhost:8000", model="my-model")
    output = backend.complete("Hello", ExternalGenerationConfig(max_tokens=100))

    # Speculative decoding (2-4x faster)
    decoder = SpeculativeDecoder(target_model, draft_model)
    output = decoder.generate(input_ids)

    # High-throughput serving
    batcher = ContinuousBatcher(model, max_batch_size=32)
    batcher.add_request(input_ids)
    outputs = batcher.step()
"""

from .kv_cache import (
    KVCache,
    PagedKVCache,
    SlidingWindowCache,
)

from .speculative import (
    SpeculativeDecoder,
    SpeculativeConfig,
)

from .batching import (
    ContinuousBatcher,
    Request,
    BatchConfig,
)

from .engine import (
    InferenceEngine,
    InferenceConfig,
    GenerationConfig,
)

from .external import (
    ExternalBackendName,
    ExternalGenerationConfig,
    OpenAICompatibleBackend,
    create_external_backend,
)

from .shared_online_rl import (
    SharedOnlineRLLoop,
    SharedOnlineRLConfig,
    SharedRLEvent,
)

from .mps_online_rl_engine import (
    MPSOnlineRLEngine,
    MPSOnlineRLEngineConfig,
)

from .tool_rewards import (
    VerifiedToolEpisode,
    build_calculator_episode,
    build_datetime_episode,
    build_verified_tool_episode,
    extract_arithmetic_expression,
    extract_datetime_hint,
    safe_calculator,
    safe_datetime,
)

__all__ = [
    # KV Cache
    "KVCache",
    "PagedKVCache",
    "SlidingWindowCache",
    # Speculative
    "SpeculativeDecoder",
    "SpeculativeConfig",
    # Batching
    "ContinuousBatcher",
    "Request",
    "BatchConfig",
    # Engine
    "InferenceEngine",
    "InferenceConfig",
    "GenerationConfig",
    "ExternalBackendName",
    "ExternalGenerationConfig",
    "OpenAICompatibleBackend",
    "create_external_backend",
    # Shared online RL
    "SharedOnlineRLLoop",
    "SharedOnlineRLConfig",
    "SharedRLEvent",
    "MPSOnlineRLEngine",
    "MPSOnlineRLEngineConfig",
    "VerifiedToolEpisode",
    "build_calculator_episode",
    "build_datetime_episode",
    "build_verified_tool_episode",
    "extract_arithmetic_expression",
    "extract_datetime_hint",
    "safe_calculator",
    "safe_datetime",
]
