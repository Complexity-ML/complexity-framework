from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

from complexity.generative.detection import (
    TRHashDetectorConfig,
    TRHashObjectDetector,
    coco_v8_nano_config,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _tiny_v8(**overrides) -> TRHashDetectorConfig:
    values = {
        "architecture_version": 8,
        "image_size": 32,
        "patch_size": 8,
        "vision_hidden_size": 32,
        "vision_layers": 3,
        "vision_heads": 4,
        "vision_expert_width": 8,
        "vision_stage_depths": (1, 1, 1),
        "vision_window_size": 2,
        "num_classes": 3,
        "neck_mode": "pan",
        "neck_normalized_fusion": True,
        "neck_repeats": 2,
        "head_hidden_size": 16,
        "head_spatial_mixing": True,
        "regression_logit_scale": True,
        "end_to_end": False,
    }
    values.update(overrides)
    return TRHashDetectorConfig(**values)


def test_v8_nano_budget_and_contract() -> None:
    config = coco_v8_nano_config()
    model = TRHashObjectDetector(config)

    assert config.architecture_version == 8
    assert config.grid_sizes == (160, 80, 40, 20)
    assert config.vision_num_experts == 8
    assert config.vision_top_k == 2
    assert config.vision_shared_width > 0
    assert config.p2_head
    assert config.neck_normalized_fusion
    assert config.neck_repeats == 2
    assert config.head_spatial_mixing
    assert config.regression_logit_scale
    assert model.one_to_one_head is None
    assert len(model.extra_necks) == 1
    for block in model.tower.blocks:
        assert block.mlp.shared_gate is not None
        assert block.mlp.shared_up is not None
        assert block.mlp.shared_down is not None
        summary = block.mlp.capability_summary("cpu")
        assert summary["shared_width"] == config.vision_shared_width
        assert summary["expert_width"] == config.vision_expert_width
        assert summary["stored_width"] == 432
        assert summary["active_width"] == 270
    assert 2_300_000 <= model.num_parameters() <= 2_500_000
    assert TRHashDetectorConfig.from_dict(config.to_dict()) == config


def test_v8_p2_is_an_independent_ablation() -> None:
    default = coco_v8_nano_config()
    no_p2 = coco_v8_nano_config(p2_head=False)
    assert default.p2_head
    assert default.grid_sizes == (160, 80, 40, 20)
    assert no_p2.grid_sizes == (80, 40, 20)
    assert no_p2.to_dict() | {"p2_head": True} == default.to_dict()


def test_v8_spatial_head_pan_and_regression_scales_receive_gradients() -> None:
    torch.manual_seed(7)
    model = TRHashObjectDetector(_tiny_v8())
    raw = model(torch.randn(2, 3, 32, 32))
    raw.square().mean().backward()

    assert raw.shape == (2, 21, 71)
    assert model.head.regression_spatial[0].depthwise.weight.grad is not None
    assert model.head.classification_spatial[0].depthwise.weight.grad is not None
    assert model.head.regression_log_scales is not None
    assert model.head.regression_log_scales.grad is not None
    assert model.neck is not None
    assert model.neck.top_down_gates.grad is not None
    assert model.extra_necks[0].bottom_up_gates.grad is not None


@pytest.mark.parametrize(
    "launcher",
    (
        "scripts/vast_train_detector_coco_v08_nano.sh",
        "scripts/vast_train_detector_coco_v08_nano_p2.sh",
    ),
)
def test_v8_launchers_encode_the_controlled_recipe(launcher: str) -> None:
    result = subprocess.run(
        ["bash", launcher],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "VENV_ACTIVATE": "/dev/null",
            "REPO_ROOT": str(PROJECT_ROOT),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    command = result.stdout
    expected = coco_v8_nano_config()
    assert "--architecture-version 8" in command
    assert f"--vision-hidden-size {expected.vision_hidden_size}" in command
    assert f"--vision-layers {expected.vision_layers}" in command
    assert f"--vision-num-experts {expected.vision_num_experts}" in command
    assert f"--vision-shared-width {expected.vision_shared_width}" in command
    assert f"--vision-expert-width {expected.vision_expert_width}" in command
    assert "--neck-normalized-fusion" in command
    assert "--neck-repeats 2" in command
    assert "--head-spatial-mixing" in command
    assert "--regression-logit-scale" in command
    assert "--nominal-batch-size 64" in command
    assert "--warmup-epochs 3.0" in command
    assert "--box-loss-weight 7.5" in command
    assert "--dfl-loss-weight 1.5" in command
    assert "--quality-loss-weight 0.75" in command
    assert "--mosaic 1.0" in command
    assert "--mosaic-tiles 16" in command
    assert "--mosaic-canvas-size 1280" in command
    assert "--mosaic-packed-epoch" in command
    assert "--packed-epochs 2" in command
    assert "--close-mosaic-epochs 10" in command
    assert "--multi-scale-min 512" in command
    assert "--multi-scale-max 640" in command
    assert "--copy-paste 0.0" in command
    assert "--p2-head" in command
    assert "--no-end-to-end" in command


def test_v8_launcher_closes_mosaic_for_the_final_ten_epochs() -> None:
    launcher = Path("scripts/vast_train_detector_coco_v08_nano.sh").read_text(encoding="utf-8")

    assert 'CLOSE_MOSAIC_EPOCHS="${CLOSE_MOSAIC_EPOCHS:-10}"' in launcher
    assert 'CLOSE_MOSAIC_EPOCHS="${CLOSE_MOSAIC_EPOCHS:-0}"' not in launcher
    assert 'PACKED_EPOCHS="${PACKED_EPOCHS:-2}"' in launcher
    assert 'MOSAIC_CANVAS_SIZE="${MOSAIC_CANVAS_SIZE:-1280}"' in launcher
