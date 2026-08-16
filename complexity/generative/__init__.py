"""Generative models beyond text-only autoregression."""

from .audio import (
    TRHashAudioConfig,
    TRHashSpeechToText,
    TRHashSpeechToTextConfig,
    TRHashTextToSpeech,
)
from .detection import TRHashDetectorConfig, TRHashObjectDetector
from .sensor_fusion import TRHashSensorFusionClassifier, TRHashSensorFusionConfig
from .video import TRHashVideoClassifier, TRHashVideoTower, TRHashVideoTowerConfig
from .vision_language import (
    TRHashImageTextToText,
    TRHashVisionClassifier,
    TRHashVisionLanguageConfig,
    TRHashVisionTower,
    TRHashVisionTowerConfig,
)
from .vision_tasks import (
    SUPPORTED_VISION_TASKS,
    TRHashDepthEstimator,
    TRHashImageClassifier,
    TRHashInstanceSegmenter,
    TRHashOBBDetector,
    TRHashPoseEstimator,
    TRHashSemanticSegmenter,
    create_vision_model,
    load_vision_task_checkpoint,
    save_vision_task_checkpoint,
)

__all__ = [
    "TRHashImageTextToText",
    "TRHashVisionLanguageConfig",
    "TRHashVisionTower",
    "TRHashVisionTowerConfig",
    "TRHashVisionClassifier",
    "TRHashSpeechToText",
    "TRHashSpeechToTextConfig",
    "TRHashTextToSpeech",
    "TRHashAudioConfig",
    "TRHashVideoTower",
    "TRHashVideoTowerConfig",
    "TRHashVideoClassifier",
    "TRHashObjectDetector",
    "TRHashDetectorConfig",
    "TRHashSensorFusionClassifier",
    "TRHashSensorFusionConfig",
    "SUPPORTED_VISION_TASKS",
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
