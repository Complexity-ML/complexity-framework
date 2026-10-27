"""Versioned detector presets with stable, testable architecture budgets."""

from __future__ import annotations

from typing import Any

from .config import TRHashDetectorConfig

COCO_V8_NANO_NAME = "coco-v8-nano"


def coco_v8_nano_config(**overrides: Any) -> TRHashDetectorConfig:
    """Build the compact v8 detector with TR-HASH routing by default.

    The existing explicit 216/27 shared/private widths, feature pyramid,
    progressive stem and task heads are retained. This is an unmeasured
    reconstructs the former eight-expert, MHA preset and its capacity budget.
    """
    values: dict[str, Any] = {
        "architecture_version": 8,
        "image_size": 640,
        "patch_size": 8,
        "num_classes": 80,
        "vision_hidden_size": 128,
        "vision_layers": 7,
        "vision_heads": 4,
        "vision_num_experts": 4,
        "vision_top_k": 2,
        "vision_shared_width": 216,
        "vision_expert_width": 27,
        "vision_stage_depths": (2, 2, 3),
        "vision_window_size": 8,
        "neck_mode": "pan",
        "neck_normalized_fusion": True,
        "neck_repeats": 2,
        "p2_head": True,
        "assignment_top_k": 8,
        "end_to_end": False,
        "head_hidden_size": 96,
        "head_spatial_mixing": True,
        "regression_logit_scale": True,
        "box_loss_weight": 7.5,
        "dfl_loss_weight": 1.5,
        "quality_loss_weight": 0.75,
        "level_adapters_enabled": True,
        "level_adapter_ratio": 0.25,
        "class_level_hash_gate_enabled": True,
        "class_level_gate_temperature": 1.0,
        "object_weighting_enabled": True,
        "object_weighting_beta": 0.999,
        "object_weighting_max": 4.0,
        "level_aux_loss_weight": 0.1,
        "gate_calibration_loss_weight": 0.1,
        "object_contrastive_loss_weight": 0.05,
        "object_contrastive_temperature": 0.1,
    }
    values.update(overrides)
    return TRHashDetectorConfig(**values)


__all__ = ["COCO_V8_NANO_NAME", "coco_v8_nano_config"]
