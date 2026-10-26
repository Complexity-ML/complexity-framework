"""Audio, sensor fusion and compatibility imports for visual models.

Visual implementations now live in ``complexity.vision``. Existing qualified
imports continue to resolve to the same module objects.
"""
from importlib import import_module

from ._compat import VISION_AREAS, install_aliases

install_aliases()
_AUDIO = {"TRHashAudioConfig", "TRHashSpeechToText", "TRHashSpeechToTextConfig", "TRHashTextToSpeech"}
_SENSOR = {"TRHashSensorFusionClassifier", "TRHashSensorFusionConfig"}
_VISUAL = {
    "TRHashImageTextToText", "TRHashVisionLanguageConfig", "TRHashVisionTower",
    "TRHashVisionTowerConfig", "TRHashVisionClassifier", "TRHashVideoTower",
    "TRHashVideoTowerConfig", "TRHashVideoClassifier", "TRHashObjectDetector",
    "TRHashDetectorConfig", "SUPPORTED_VISION_TASKS", "TRHashDepthEstimator",
    "TRHashImageClassifier", "TRHashInstanceSegmenter", "TRHashOBBDetector",
    "TRHashPoseEstimator", "TRHashSemanticSegmenter", "create_vision_model",
    "load_vision_task_checkpoint", "save_vision_task_checkpoint",
}
__all__ = sorted(_AUDIO | _SENSOR | _VISUAL)


def __getattr__(name):
    if name in _AUDIO:
        value = getattr(import_module("complexity.generative.audio"), name)
    elif name in _SENSOR:
        value = getattr(import_module("complexity.generative.sensor_fusion"), name)
    elif name in _VISUAL:
        value = getattr(import_module("complexity.vision"), name)
    elif name in VISION_AREAS:
        value = import_module(f"complexity.generative.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(VISION_AREAS))
