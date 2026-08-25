"""ONNX Runtime deployment helpers for TR-Hash Vision detectors."""

from .metadata import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    BranchType,
    OnnxDetectorMetadata,
)
from .pipeline import OnnxDetectorPipeline
from .preprocess import ImageGeometry, PreprocessResult, preprocess_image, restore_boxes
from .session import OnnxDetectorSession, OrtSessionConfig
from .types import Detection, DetectionResult, TimingBreakdown

__all__ = [
    "BranchType",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_MAX_DETECTIONS",
    "Detection",
    "DetectionResult",
    "ImageGeometry",
    "OnnxDetectorMetadata",
    "OnnxDetectorPipeline",
    "OnnxDetectorSession",
    "OrtSessionConfig",
    "PreprocessResult",
    "TimingBreakdown",
    "preprocess_image",
    "restore_boxes",
]
