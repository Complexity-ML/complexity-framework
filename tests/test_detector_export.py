from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from complexity.generative.detection import (
    TRHashDetectorConfig,
    TRHashObjectDetector,
    load_detector_checkpoint,
)
from complexity.generative.detection.exporting import RawDetectorExport
from scripts.check_onnx_parity import check_parity
from scripts.export_onnx import export_onnx


def tiny_config(*, end_to_end: bool) -> TRHashDetectorConfig:
    return TRHashDetectorConfig(
        architecture_version=8,
        image_size=32,
        patch_size=8,
        vision_hidden_size=16,
        vision_layers=3,
        vision_heads=4,
        vision_num_experts=2,
        vision_top_k=1,
        vision_expert_width=8,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
        vision_precision="fp32",
        num_classes=3,
        reg_max=4,
        head_hidden_size=16,
        end_to_end=end_to_end,
    )


def write_checkpoint(path: Path, *, end_to_end: bool) -> TRHashObjectDetector:
    torch.manual_seed(7)
    model = TRHashObjectDetector(tiny_config(end_to_end=end_to_end)).eval()
    path.mkdir()
    (path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), str(path / "model.safetensors"))
    return model


def test_public_loader_accepts_v8_export_architecture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "detector-v8"
    expected = write_checkpoint(checkpoint, end_to_end=False)
    loaded = load_detector_checkpoint(checkpoint, device="cpu").eval()

    assert loaded.config.architecture_version == 8
    values = torch.randn(1, 3, 32, 32)
    torch.testing.assert_close(
        RawDetectorExport(loaded, "o2m")(values),
        RawDetectorExport(expected, "o2m")(values),
    )


def test_auto_export_selects_the_production_branch() -> None:
    values = torch.randn(2, 3, 32, 32)
    end_to_end = TRHashObjectDetector(tiny_config(end_to_end=True)).eval()
    one_to_many, one_to_one = end_to_end.forward_branches(values)

    assert one_to_one is not None
    assert RawDetectorExport(end_to_end).branch == "nms-free"
    torch.testing.assert_close(RawDetectorExport(end_to_end)(values), one_to_one)
    torch.testing.assert_close(RawDetectorExport(end_to_end, "o2m")(values), one_to_many)

    classic = TRHashObjectDetector(tiny_config(end_to_end=False)).eval()
    assert RawDetectorExport(classic).branch == "o2m"
    with pytest.raises(ValueError, match="end_to_end=True"):
        RawDetectorExport(classic, "nms-free")


@pytest.mark.parametrize("branch", ("nms-free", "o2m"))
def test_dynamic_onnx_export_matches_the_selected_branch(tmp_path: Path, branch: str) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint(checkpoint, end_to_end=True)
    output = tmp_path / f"detector-{branch}.onnx"

    export_onnx(
        checkpoint,
        output,
        dynamic_batch=True,
        check=True,
        branch=branch,
    )

    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["architecture_version"] == 8
    assert metadata["branch"] == branch
    assert metadata["requires_nms"] is (branch == "o2m")
    assert check_parity(
        checkpoint,
        output,
        num_tests=1,
        batch_size=2,
        tolerance=1e-4,
    )
