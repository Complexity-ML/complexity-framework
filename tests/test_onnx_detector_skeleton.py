from pathlib import Path

from complexity.deploy.onnx_detector import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    Detection,
    DetectionResult,
    OnnxDetectorMetadata,
    OnnxDetectorPipeline,
    TimingBreakdown,
)


def test_onnx_detector_package_imports_and_schema_constructs() -> None:
    metadata = OnnxDetectorMetadata(
        architecture_version=8,
        image_size=640,
        num_classes=80,
        num_cells=34000,
        regression_width=68,
        reg_max=16,
        scale_factors=(1, 2, 4),
        grid_sizes=(160, 80, 40, 20),
        p2_head=True,
        branch="o2m",
        requires_nms=True,
        output_semantics="raw_ltrb_dfl_and_quality_class_logits",
    )
    session = OnnxDetectorPipeline.create_session(
        Path("model.onnx"),
        providers=("CPUExecutionProvider",),
    )
    detection = Detection(
        box_norm=(0.1, 0.2, 0.3, 0.4),
        box_pixel=(64.0, 128.0, 192.0, 256.0),
        class_id=1,
        score=0.9,
    )
    result = DetectionResult(
        detections=(detection,),
        timing=TimingBreakdown(1.0, 2.0, 3.0),
        provider_used="CPUExecutionProvider",
        branch_type=metadata.branch,
    )

    assert metadata.dfl_bins == 17
    assert metadata.prediction_width == 148
    assert metadata.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert metadata.iou_threshold == DEFAULT_IOU_THRESHOLD
    assert metadata.max_detections == DEFAULT_MAX_DETECTIONS
    assert session.config.model_path == Path("model.onnx")
    assert result.detections[0].box_norm == (0.1, 0.2, 0.3, 0.4)
