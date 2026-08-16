"""TR-Hash MoE video model.

Replaces ``complexity.multimodal.video``'s position-routed
``VideoTokenRoutedMLP`` (a single fixed routing reused at every layer) with
a real ``TRHashEngine`` per block — tubelets route through multiple experts
by their fixed spatio-temporal position, with a per-layer route permutation.
Same principle as ``complexity.generative.vision_language.TRHashVisionTower``,
generalized from image patches to video tubelets.

Usage:
    from complexity.generative.video import (
        TRHashVideoTower, TRHashVideoClassifier, TRHashVideoTowerConfig,
    )

    tower = TRHashVideoTower(TRHashVideoTowerConfig())
    features = tower(video)  # [batch, tubelets, hidden_size]

    classifier = TRHashVideoClassifier(TRHashVideoTowerConfig(), num_classes=400)
    out = classifier(video, labels=labels)
"""

from .config import TRHashVideoTowerConfig
from .tower import TRHashVideoBlock, TRHashVideoClassifier, TRHashVideoTower

__all__ = [
    "TRHashVideoTowerConfig",
    "TRHashVideoTower",
    "TRHashVideoBlock",
    "TRHashVideoClassifier",
]
