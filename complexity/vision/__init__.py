"""Visual models and workflows, loaded only when requested.

Detection, image generation/editing, video, visual-language models and task
heads live in this tree. Audio and sensor fusion remain in ``generative``.
"""

from importlib import import_module

_AREAS = ("detection", "image", "video", "vision_language", "vision_tasks")
_EXPORTS = {
    "TRHashDetectorConfig": "detection",
    "TRHashObjectDetector": "detection",
    "TRHashImageConfig": "image",
    "TRHashImageEditConfig": "image",
    "TRHashImageEditor": "image",
    "TRHashTextToImage": "image",
    "TRHashVideoClassifier": "video",
    "TRHashVideoTower": "video",
    "TRHashVideoTowerConfig": "video",
    "TRHashImageTextToText": "vision_language",
    "TRHashVisionClassifier": "vision_language",
    "TRHashVisionLanguageConfig": "vision_language",
    "TRHashVisionTower": "vision_language",
    "TRHashVisionTowerConfig": "vision_language",
    "SUPPORTED_VISION_TASKS": "vision_tasks",
    "TRHashDepthEstimator": "vision_tasks",
    "TRHashImageClassifier": "vision_tasks",
    "TRHashInstanceSegmenter": "vision_tasks",
    "TRHashOBBDetector": "vision_tasks",
    "TRHashPoseEstimator": "vision_tasks",
    "TRHashSemanticSegmenter": "vision_tasks",
    "create_vision_model": "vision_tasks",
    "load_vision_task_checkpoint": "vision_tasks",
    "save_vision_task_checkpoint": "vision_tasks",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _AREAS:
        value = import_module(f"{__name__}.{name}")
    elif name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_AREAS) | set(__all__))
