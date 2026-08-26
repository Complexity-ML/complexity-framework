"""Stable output schema for ONNX detector inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One decoded object detection."""

    box_norm: Box
    box_pixel: Box
    class_id: int
    score: float


@dataclass(frozen=True)
class TimingBreakdown:
    """Wall-clock timing captured around the ONNX inference pipeline."""

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
        }


@dataclass(frozen=True)
class DetectionResult:
    """Stable result returned by the ONNX detector pipeline."""

    detections: Sequence[Detection] = field(default_factory=tuple)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    provider_used: str = ""
    branch_type: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
