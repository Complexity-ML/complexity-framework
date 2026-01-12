"""
Inference optimization module for framework-complexity.

Provides:
- KV-Cache for efficient autoregressive generation
- Speculative decoding for faster inference
- Continuous batching for high throughput
- Tensor parallelism for inference

Usage:
    from complexity.inference import (
        KVCache,
        SpeculativeDecoder,
        ContinuousBatcher,
        InferenceEngine,
    )

    # Simple inference with KV cache
    engine = InferenceEngine(model)
    output = engine.generate(input_ids, max_tokens=100)

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
]
