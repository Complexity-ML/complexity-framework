"""Behavioral contracts for the next detector architecture generalization.

These tests deliberately avoid naming a future preset.  They exercise the
capabilities that the next preset is expected to compose: finer hash expert
partitions at a fixed storage budget, a trainable P2 prediction level, deeper
hierarchical stages, and frontend/position modules that preserve rectangular
multi-scale geometry and gradients.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.generative.detection.hierarchical_tower import (
    HierarchicalTRHashVisionTower,
)
from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
)


@pytest.mark.parametrize(
    ("num_experts", "shared_width", "expert_width", "expected_active_width"),
    (
        (4, 264, 66, 396),
        (8, 264, 33, 330),
        (16, 272, 16, 304),
    ),
)
def test_finer_expert_partitions_preserve_the_stored_budget(
    num_experts: int,
    shared_width: int,
    expert_width: int,
    expected_active_width: int,
) -> None:
    """More experts must mean finer specialization, not parameter inflation."""

    config = TRHashEngineConfig(
        hidden_size=16,
        vocab_size=64,
        num_experts=num_experts,
        top_k=2,
        shared_width=shared_width,
        expert_width=expert_width,
        precision=TRHashPrecision.FP32,
        backend=TRHashBackend.PYTORCH,
    )
    engine = TRHashEngine(config)
    summary = engine.capability_summary("cpu")

    assert summary["stored_width"] == 528
    assert summary["active_width"] == expected_active_width
    assert summary["experts"] == num_experts
    assert summary["top_k"] == 2
    assert torch.unique(engine.route_table).numel() == num_experts

    values = torch.randn(2, 32, 16, requires_grad=True)
    token_ids = torch.arange(32).expand(2, -1)
    engine(values, token_ids).square().mean().backward()
    assert values.grad is not None
    assert engine.shared_gate.weight.grad is not None
    assert engine.expert_gate.grad is not None


def _candidate_config(**overrides) -> TRHashDetectorConfig:
    values = {
        "architecture_version": 8,
        "image_size": 64,
        "patch_size": 8,
        "vision_hidden_size": 32,
        "vision_layers": 8,
        "vision_heads": 4,
        "vision_num_experts": 8,
        "vision_top_k": 2,
        "vision_shared_width": 32,
        "vision_expert_width": 8,
        "vision_stage_depths": (2, 2, 4),
        "vision_window_size": 2,
        "num_classes": 3,
        "head_hidden_size": 16,
        "end_to_end": False,
        "level_adapters_enabled": False,
        "class_level_hash_gate_enabled": False,
        "object_weighting_enabled": False,
        "object_contrastive_loss_weight": 0.0,
    }
    values.update(overrides)
    return TRHashDetectorConfig(**values)


def test_p2_is_a_real_trainable_stride_four_prediction_level() -> None:
    config = _candidate_config(p2_head=True)
    model = TRHashObjectDetector(config)
    pixels = torch.randn(2, 3, 64, 64)

    tower_features = model.tower(pixels)
    pyramid = model._feature_pyramid(tower_features)
    raw = model(pixels)
    raw.square().mean().backward()

    assert config.grid_sizes == (16, 8, 4, 2)
    assert [tuple(level.shape[-2:]) for level in pyramid] == [
        (16, 16),
        (8, 8),
        (4, 4),
        (2, 2),
    ]
    assert raw.shape == (2, config.num_cells, config.prediction_width)
    assert len(model.head.regression_heads) == 4
    assert len(model.head.classification_heads) == 4
    assert model.fpn_upsample is not None
    assert model.fpn_upsample[0].weight.grad is not None
    assert model.head.regression_heads[0][-1].weight.grad is not None
    assert model.head.classification_heads[0][-1].weight.grad is not None


def test_deeper_stage_schedule_materializes_every_requested_block() -> None:
    config = _candidate_config()
    tower = HierarchicalTRHashVisionTower(config)

    assert [len(stage) for stage in tower.stages] == [2, 2, 4]
    assert len(tower.blocks) == config.vision_layers == 8
    assert [block.shift for block in tower.stages[0]] == [0, 1]
    assert [block.shift for block in tower.stages[1]] == [0, 1]
    assert all(block.window_size == 0 and block.shift == 0 for block in tower.stages[2])
    assert [block.mlp.config.layer_index for block in tower.blocks] == list(range(8))


def test_frontend_and_position_encoding_preserve_rectangular_geometry_and_gradients() -> None:
    """A progressive stem or relative bias may replace today's implementation.

    The implementation is intentionally not prescribed here.  The stable
    contract is that rectangular inputs retain the configured stride pyramid,
    shifted windows remain usable, and both the frontend and whichever
    positional parameters exist participate in backpropagation.
    """

    config = _candidate_config()
    tower = HierarchicalTRHashVisionTower(config)
    pixels = torch.randn(2, 3, 48, 32, requires_grad=True)
    outputs = tower(pixels)
    sum(level.square().mean() for level in outputs).backward()

    assert [tuple(level.shape[-2:]) for level in outputs] == [
        (6, 4),
        (3, 2),
        (2, 1),
    ]
    assert pixels.grad is not None and pixels.grad.abs().sum() > 0

    frontend_parameters = [
        parameter
        for name, parameter in tower.named_parameters()
        if name.startswith(("patch_embed.", "stem."))
    ]
    assert frontend_parameters
    assert any(parameter.grad is not None for parameter in frontend_parameters)

    position_parameters = [
        parameter
        for name, parameter in tower.named_parameters()
        if "position" in name or "relative_bias" in name
    ]
    assert position_parameters
    assert any(parameter.grad is not None for parameter in position_parameters)

    frontend_convolutions = [
        module
        for name, module in tower.named_modules()
        if name.startswith(("patch_embed", "stem")) and isinstance(module, nn.Conv2d)
    ]
    assert frontend_convolutions
    assert outputs[0].shape[-2:] == (
        pixels.shape[-2] // config.patch_size,
        pixels.shape[-1] // config.patch_size,
    )
