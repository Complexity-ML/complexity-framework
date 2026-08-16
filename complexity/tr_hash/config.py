"""Configuration and capability contracts for the TR-Hash execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Tuple

SUPPORTED_EXPERT_COUNTS = (
    1,
    2,
    4,
    8,
    16,
)  # 1 = degenerate single-route case (no dense/standalone fallback exists anymore)
SUPPORTED_TOP_K = (1, 2, 4)


class TRHashBackend(str, Enum):
    """Execution backend requested for a TR-Hash layer."""

    AUTO = "auto"
    PYTORCH = "pytorch"
    CGGR = "cggr"
    FUSED_CUDA = "fused_cuda"
    CUDA_GRAPH = "cuda_graph"


class TRHashPrecision(str, Enum):
    """Storage/compute precision requested by the engine."""

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"


class TRHashPhase(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class AttentionBackbone(str, Enum):
    """Upstream attention metadata; routing itself is attention-agnostic."""

    GQA = "gqa"
    MHA = "mha"


class TRHashStrategy(str, Enum):
    """Deterministic token-ID route families implemented by the engine."""

    MODULO_CYCLIC = "modulo_cyclic"
    BALANCED_HASH = "token_id_balanced_hash"
    AFFINE_HASH = "token_id_affine_hash"
    MULTI_HASH = "token_id_multi_hash"
    HIERARCHICAL_HASH = "token_id_hierarchical_hash"


@dataclass(frozen=True)
class GraphBucket:
    """One static CUDA Graph shape.

    Batch and sequence dimensions are padded to this shape before capture.
    """

    batch_size: int
    sequence_length: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("graph bucket batch_size must be positive")
        if self.sequence_length <= 0:
            raise ValueError("graph bucket sequence_length must be positive")

    @property
    def token_capacity(self) -> int:
        return self.batch_size * self.sequence_length


def _normalize_buckets(values: Iterable[GraphBucket]) -> Tuple[GraphBucket, ...]:
    unique = {(int(v.batch_size), int(v.sequence_length)) for v in values}
    return tuple(
        GraphBucket(batch, sequence)
        for batch, sequence in sorted(
            unique,
            key=lambda item: (item[0] * item[1], item[0], item[1]),
        )
    )


@dataclass(frozen=True)
class TRHashEngineConfig:
    """Shape, routing and runtime contract for a TR-Hash layer."""

    hidden_size: int
    vocab_size: int
    num_experts: int = 4
    top_k: int = 2
    shared_width: int = 128
    expert_width: int = 48
    initializer_range: float = 0.02
    routing_strategy: TRHashStrategy = TRHashStrategy.BALANCED_HASH
    layer_index: int = 0
    route_seed: int = 0x71D5A17
    route_hash_count: int = 2
    route_weights: Tuple[float, ...] = ()
    shared_output_scale: float = 1.0
    routed_output_scale: float = 1.0
    attention_backbone: AttentionBackbone = AttentionBackbone.GQA
    phase: TRHashPhase = TRHashPhase.TRAINING
    precision: TRHashPrecision = TRHashPrecision.BF16
    backend: TRHashBackend = TRHashBackend.AUTO
    world_size: int = 1
    graph_buckets: Tuple[GraphBucket, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "routing_strategy",
            TRHashStrategy(self.routing_strategy),
        )
        object.__setattr__(
            self,
            "attention_backbone",
            AttentionBackbone(self.attention_backbone),
        )
        object.__setattr__(self, "phase", TRHashPhase(self.phase))
        object.__setattr__(self, "precision", TRHashPrecision(self.precision))
        object.__setattr__(self, "backend", TRHashBackend(self.backend))
        object.__setattr__(
            self,
            "graph_buckets",
            _normalize_buckets(self.graph_buckets),
        )

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_experts not in SUPPORTED_EXPERT_COUNTS:
            raise ValueError(
                f"num_experts must be one of {SUPPORTED_EXPERT_COUNTS}, got {self.num_experts}"
            )
        if self.top_k not in SUPPORTED_TOP_K:
            raise ValueError(f"top_k must be one of {SUPPORTED_TOP_K}, got {self.top_k}")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.shared_width < 0:
            raise ValueError("shared_width must be non-negative")
        if self.expert_width <= 0:
            raise ValueError("expert_width must be positive")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if not 2 <= self.route_hash_count <= 8:
            raise ValueError("route_hash_count must be between 2 and 8")
        if self.shared_output_scale < 0:
            raise ValueError("shared_output_scale must be non-negative")
        if self.routed_output_scale < 0:
            raise ValueError("routed_output_scale must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")

        weights = self.route_weights
        if not weights:
            weights = tuple(1.0 / self.top_k for _ in range(self.top_k))
            object.__setattr__(self, "route_weights", weights)
        if len(weights) != self.top_k:
            raise ValueError("route_weights must contain exactly top_k values")
        if any(weight < 0 for weight in weights):
            raise ValueError("route_weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("route_weights must sum to 1")

        if self.backend is TRHashBackend.CUDA_GRAPH:
            if self.phase is not TRHashPhase.INFERENCE:
                raise ValueError("CUDA Graph execution is inference-only")
            if not self.graph_buckets:
                raise ValueError("CUDA Graph execution requires at least one graph bucket")

    @property
    def stored_routed_width(self) -> int:
        return self.num_experts * self.expert_width

    @property
    def active_routed_width(self) -> int:
        return self.top_k * self.expert_width

    @property
    def stored_mlp_width(self) -> int:
        return self.shared_width + self.stored_routed_width

    @property
    def active_mlp_width(self) -> int:
        return self.shared_width + self.active_routed_width

    @property
    def active_width_fraction(self) -> float:
        return self.active_mlp_width / self.stored_mlp_width
