"""TR-Hash model variants for seven common vision task families."""

from .checkpoint import load_vision_task_checkpoint, save_vision_task_checkpoint
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
    "load_vision_task_checkpoint",
    "save_vision_task_checkpoint",
]
