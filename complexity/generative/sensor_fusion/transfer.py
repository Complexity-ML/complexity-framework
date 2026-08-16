"""Compatible vision-tower transfer for TR-Hash sensor fusion."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file

from ..detection.config import TRHashDetectorConfig
from .model import TRHashSensorFusionClassifier

_VISION_FIELDS = (
    "image_size",
    "patch_size",
    "vision_hidden_size",
    "vision_layers",
    "vision_heads",
    "vision_num_experts",
    "vision_top_k",
    "vision_shared_width",
    "vision_expert_width",
    "vision_stage_depths",
    "vision_window_size",
    "route_seed",
)


def load_pretrained_visual_tower(
    model: TRHashSensorFusionClassifier,
    checkpoint: str | Path,
) -> int:
    """Load a compatible pretrained TR-Hash vision tower into ``model``."""

    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "tower.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "vision checkpoint must contain config.json and tower.safetensors"
        )
    payload = json.loads(config_path.read_text())
    # Vision-only checkpoints wrap the detector configuration under ``tower``;
    # detection checkpoints store it directly. Supporting both lets downstream
    # tasks reuse the native COCO detection stage without a separate classifier
    # pretraining dependency.
    source = TRHashDetectorConfig.from_dict(payload.get("tower", payload))
    target = model.vision_tower.config
    mismatches = [
        name for name in _VISION_FIELDS if getattr(source, name) != getattr(target, name)
    ]
    if mismatches:
        raise ValueError("incompatible pretrained vision tower: " + ", ".join(mismatches))
    state = load_file(str(weights_path), device="cpu")
    model.vision_tower.load_state_dict(state, strict=True)
    return len(state)
