"""Versioned detector presets with stable, testable architecture budgets."""

from __future__ import annotations

from typing import Any

from .config import TRHashDetectorConfig

COCO_V8_NANO_NAME = "coco-v8-nano"


def coco_v8_nano_config(**overrides: Any) -> TRHashDetectorConfig:
    """Return the competitive v8 COCO configuration, ~2.3-2.5M parameters.

    Every routed-MoE stage stays TR-Hash: deterministic position-keyed
    routing, never learned or contextual. Within that constraint, v8 spends
    its budget differently from a routed-only tower:

    - A dominant dense shared branch plus finer-grained routed specialization:
      ``vision_num_experts=8`` with ``vision_top_k=2`` (25% of routed capacity
      fires per token) and a narrow ``vision_expert_width=27`` -- each expert
      is a small position-keyed correction on top of ``vision_shared_width=216``
      of capacity every token gets unconditionally (shared:expert = 8:1).
    - Extra depth: ``vision_stage_depths=(2, 2, 3)``, one more block than the
      previous 5-layer shallow tower. Narrow experts make additional layers
      cheap relative to the capacity they add, and depth generally helps
      dense prediction more than raw width at a fixed budget.
    - A progressive stride-2 conv stem (``patch_size`` factored into
      ``log2(patch_size)`` depthwise-separable stages) instead of one
      non-overlapping large-stride conv, preserving more fine detail into
      the first tower stage.
    - Windowed attention now carries a learned relative position bias
      (Swin-style) instead of relying on absolute position embeddings alone
      to encode local structure inside a window.
    - ``p2_head`` enabled by default: the stride-4 prediction level is the
      single biggest lever for small-object AP, the weakest of the three
      COCO size buckets on the earlier (P2-disabled) checkpoint.
    """

    values: dict[str, Any] = {
        "architecture_version": 8,
        "image_size": 640,
        "patch_size": 8,
        "num_classes": 80,
        "vision_hidden_size": 128,
        "vision_layers": 7,
        "vision_heads": 4,
        "vision_num_experts": 8,
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
        "level_aux_loss_weight": 0.10,
        "gate_calibration_loss_weight": 0.10,
        "object_contrastive_loss_weight": 0.05,
        "object_contrastive_temperature": 0.10,
    }
    values.update(overrides)
    return TRHashDetectorConfig(**values)


__all__ = [
    "COCO_V8_NANO_NAME",
    "coco_v8_nano_config",
]
