"""TR-Hash model variants for seven common vision task families."""

from .model import (
    SUPPORTED_VISION_TASKS,
    TRHashDepthEstimator,
    TRHashImageClassifier,
    TRHashInstanceSegmenter,
    TRHashOBBDetector,
    TRHashPoseEstimator,
    TRHashSemanticSegmenter,
    VisionTask,
    create_vision_model,
)

__all__ = [
    "SUPPORTED_VISION_TASKS",
    "VisionTask",
    "TRHashDepthEstimator",
    "TRHashImageClassifier",
    "TRHashInstanceSegmenter",
    "TRHashOBBDetector",
    "TRHashPoseEstimator",
    "TRHashSemanticSegmenter",
    "create_vision_model",
]
