"""Generative models beyond text-only autoregression."""

from .audio import (
    TRHashAudioConfig,
    TRHashSpeechToText,
    TRHashSpeechToTextConfig,
    TRHashTextToSpeech,
)
from .detection import TRHashDetectorConfig, TRHashObjectDetector
from .video import TRHashVideoClassifier, TRHashVideoTower, TRHashVideoTowerConfig
from .vision_language import (
    TRHashImageTextToText,
    TRHashVisionClassifier,
    TRHashVisionLanguageConfig,
    TRHashVisionTower,
    TRHashVisionTowerConfig,
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
]
