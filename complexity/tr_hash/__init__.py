"""Public TR-Hash execution-engine API."""

from .buckets import GraphBucketPlanner
from .capabilities import BackendDecision, precision_to_torch_dtype, select_backend
from .config import (
    SUPPORTED_EXPERT_COUNTS,
    SUPPORTED_TOP_K,
    AttentionBackbone,
    GraphBucket,
    TRHashBackend,
    TRHashEngineConfig,
    TRHashPhase,
    TRHashPrecision,
    TRHashStrategy,
)
from .engine import TRHashEngine, reference_topk_swiglu
from .routing import build_route_table

__all__ = [
    "AttentionBackbone",
    "BackendDecision",
    "GraphBucket",
    "GraphBucketPlanner",
    "SUPPORTED_EXPERT_COUNTS",
    "SUPPORTED_TOP_K",
    "TRHashBackend",
    "TRHashEngine",
    "TRHashEngineConfig",
    "TRHashPhase",
    "TRHashPrecision",
    "TRHashStrategy",
    "build_route_table",
    "precision_to_torch_dtype",
    "reference_topk_swiglu",
    "select_backend",
]
