import json

import numpy as np
import pytest
import torch
from PIL import Image

from complexity.deploy.onnx_detector.metadata import OnnxDetectorMetadata
from complexity.deploy.onnx_detector.pipeline import OnnxDetectorPipeline
from complexity.deploy.onnx_detector.preprocess import ImageGeometry, restore_boxes
from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.generative.detection.exporting import RawDetectorExport
from complexity.generative.detection.hub import (
    preprocess_detector_image,
    restore_detector_boxes,
)


class FakeSession:
    provider_used = "FakeExecutionProvider"

    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = predictions
        self.input_shape: tuple[int, ...] | None = None

    def run(self, pixel_values: np.ndarray) -> np.ndarray:
        self.input_shape = pixel_values.shape
        return self.predictions


def _metadata() -> OnnxDetectorMetadata:
    return OnnxDetectorMetadata(
        architecture_version=8,
        image_size=16,
        num_classes=3,
        num_cells=16,
        regression_width=12,
        reg_max=2,
        scale_factors=(1,),
        grid_sizes=(4,),
        p2_head=False,
        branch="nms-free",
        requires_nms=False,
        output_semantics="raw_ltrb_dfl_and_quality_class_logits",
        confidence_threshold=0.9,
        max_detections=1,
    )


def _tiny_detector_config() -> TRHashDetectorConfig:
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
        end_to_end=True,
    )


def _export_metadata(config: TRHashDetectorConfig, branch: str) -> dict[str, object]:
    return {
        "architecture_version": config.architecture_version,
        "image_size": config.image_size,
        "num_classes": config.num_classes,
        "num_cells": config.num_cells,
        "regression_width": config.regression_width,
        "reg_max": config.reg_max,
        "scale_factors": list(config.scale_factors),
        "grid_sizes": list(config.grid_sizes),
        "p2_head": config.p2_head,
        "branch": branch,
        "requires_nms": branch == "o2m",
        "output_semantics": "raw_ltrb_dfl_and_quality_class_logits",
        "confidence_threshold": 0.0,
        "iou_threshold": 0.45,
        "max_detections": 5,
    }


def test_restore_boxes_undoes_letterbox_for_non_square_image() -> None:
    boxes = np.array([[2.0, 2.0, 6.0, 6.0]], dtype=np.float32)
    geometry = ImageGeometry(
        original_width=16,
        original_height=8,
        image_size=16,
        scale=1.0,
        left=0,
        top=4,
    )

    restored = restore_boxes(boxes, geometry)

    np.testing.assert_allclose(restored, np.array([[2.0, 0.0, 6.0, 2.0]]))


def test_pipeline_predict_runs_fake_session_and_returns_stable_schema() -> None:
    metadata = _metadata()
    predictions = np.full(
        (1, metadata.num_cells, metadata.prediction_width),
        -20.0,
        dtype=np.float32,
    )
    # First cell center is (2, 2), stride is 4, DFL distances are [0, 0, 1, 1].
    predictions[0, 0, 0] = 40.0
    predictions[0, 0, 3] = 40.0
    predictions[0, 0, 7] = 40.0
    predictions[0, 0, 10] = 40.0
    predictions[0, 0, metadata.regression_width + 2] = 20.0
    session = FakeSession(predictions)
    pipeline = OnnxDetectorPipeline(metadata=metadata, session=session)  # type: ignore[arg-type]
    image = np.zeros((8, 16, 3), dtype=np.uint8)

    result = pipeline.predict(image)

    assert session.input_shape == (1, 3, 16, 16)
    assert result.provider_used == "FakeExecutionProvider"
    assert result.branch_type == "nms-free"
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.class_id == 2
    assert detection.score > 0.99
    np.testing.assert_allclose(detection.box_norm, (0.125, 0.125, 0.375, 0.375))
    np.testing.assert_allclose(detection.box_pixel, (2.0, 0.0, 6.0, 2.0))
    assert result.timing.preprocess_ms >= 0.0
    assert result.timing.inference_ms >= 0.0
    assert result.timing.postprocess_ms >= 0.0


@pytest.mark.parametrize("branch", ("o2m", "nms-free"))
def test_onnx_pipeline_matches_pytorch_postprocessing_on_fixed_input(
    tmp_path,
    branch: str,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(13)
    model = TRHashObjectDetector(_tiny_detector_config()).eval()
    export_model = RawDetectorExport(model, branch).eval()
    onnx_path = tmp_path / f"detector-{branch}.onnx"
    metadata_path = tmp_path / f"detector-{branch}.json"
    metadata_path.write_text(json.dumps(_export_metadata(model.config, branch)))
    dummy_input = torch.randn(1, 3, model.config.image_size, model.config.image_size)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(onnx_path),
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["predictions"],
        do_constant_folding=True,
        dynamo=False,
    )

    pixels = np.arange(20 * 28 * 3, dtype=np.uint8).reshape(20, 28, 3)
    image = Image.fromarray(pixels)
    pipeline = OnnxDetectorPipeline.from_files(
        onnx_path,
        metadata_path,
        providers=("CPUExecutionProvider",),
    )
    onnx_result = pipeline.predict(image)

    pytorch_pixels, geometry = preprocess_detector_image(image, model.config.image_size)
    with torch.no_grad():
        if branch == "nms-free":
            reference = model.predict_end_to_end(
                pytorch_pixels.unsqueeze(0),
                confidence_threshold=0.0,
                max_detections=5,
            )[0]
        else:
            reference = model.predict(
                pytorch_pixels.unsqueeze(0),
                confidence_threshold=0.0,
                iou_threshold=0.45,
                max_detections=5,
                nms_free=False,
            )[0]
        reference_boxes_pixel = restore_detector_boxes(reference["boxes"], geometry)

    assert len(onnx_result.detections) == len(reference["scores"])
    np.testing.assert_array_equal(
        np.array([detection.class_id for detection in onnx_result.detections]),
        reference["labels"].cpu().numpy(),
    )
    np.testing.assert_allclose(
        np.array([detection.score for detection in onnx_result.detections]),
        reference["scores"].cpu().numpy(),
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.array([detection.box_norm for detection in onnx_result.detections]),
        reference["boxes"].cpu().numpy(),
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.array([detection.box_pixel for detection in onnx_result.detections]),
        reference_boxes_pixel.cpu().numpy(),
        atol=1e-3,
        rtol=1e-3,
    )
