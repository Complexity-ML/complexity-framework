from pathlib import Path

import numpy as np

from complexity.deploy.onnx_detector.dfl import decode_dfl_boxes
from complexity.deploy.onnx_detector.grid import generate_grid_geometry
from complexity.deploy.onnx_detector.metadata import BranchType, OnnxDetectorMetadata
from complexity.deploy.onnx_detector.pipeline import OnnxDetectorPipeline
from complexity.deploy.onnx_detector.postprocess import (
    class_aware_nms,
    filter_by_confidence,
)
from complexity.deploy.onnx_detector.session import (
    OnnxDetectorSession,
    OrtSessionConfig,
    _needs_cuda_dlls,
)


def _metadata(branch: BranchType = "o2m") -> OnnxDetectorMetadata:
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
        branch=branch,
        requires_nms=branch == "o2m",
        output_semantics="raw_ltrb_dfl_and_quality_class_logits",
        max_detections=2,
    )


def test_grid_mapping_is_row_major_and_counts_real_v8_cells() -> None:
    geometry = generate_grid_geometry(16, (4, 2))

    np.testing.assert_allclose(
        geometry.centers_xy[:5],
        np.array(
            [
                [2.0, 2.0],
                [6.0, 2.0],
                [10.0, 2.0],
                [14.0, 2.0],
                [2.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(geometry.strides[:4], np.full((4,), 4.0))
    np.testing.assert_allclose(geometry.centers_xy[16], np.array([4.0, 4.0]))

    real_geometry = generate_grid_geometry(640, (160, 80, 40, 20))
    assert real_geometry.centers_xy.shape == (34000, 2)
    assert real_geometry.strides.shape == (34000,)


def test_cuda_dll_preload_is_limited_to_nvidia_providers() -> None:
    assert _needs_cuda_dlls(("CUDAExecutionProvider",))
    assert _needs_cuda_dlls(("TensorrtExecutionProvider", "CPUExecutionProvider"))
    assert not _needs_cuda_dlls(("CPUExecutionProvider",))
    assert not _needs_cuda_dlls(("CoreMLExecutionProvider",))
    assert not _needs_cuda_dlls(("DmlExecutionProvider",))


def test_pipeline_session_factory_forwards_deterministic_thread_settings() -> None:
    session = OnnxDetectorPipeline.create_session(
        Path("model.onnx"),
        providers=("CPUExecutionProvider",),
        warmup_runs=0,
        intra_op_num_threads=1,
        inter_op_num_threads=1,
    )

    assert session.config.warmup_runs == 0
    assert session.config.intra_op_num_threads == 1
    assert session.config.inter_op_num_threads == 1


def test_dfl_decode_matches_hand_computed_expectation() -> None:
    metadata = _metadata()
    geometry = generate_grid_geometry(metadata.image_size, metadata.grid_sizes)
    logits = np.full((1, metadata.num_cells, metadata.regression_width), -40.0)
    # First cell center is (2, 2) and stride is 4. Distances are LTRB
    # [0, 0, 1, 1] bins, so the pixel box is [2, 2, 6, 6].
    logits[0, 0, 0] = 40.0
    logits[0, 0, 3] = 40.0
    logits[0, 0, 7] = 40.0
    logits[0, 0, 10] = 40.0

    boxes = decode_dfl_boxes(logits, metadata, geometry)

    np.testing.assert_allclose(boxes[0, 0], np.array([2.0, 2.0, 6.0, 6.0]), atol=1e-4)


def test_confidence_filter_includes_exact_threshold() -> None:
    boxes = np.zeros((3, 4), dtype=np.float32)
    scores = np.array([0.249, 0.25, 0.251], dtype=np.float32)
    classes = np.array([0, 1, 2])

    _, filtered_scores, filtered_classes, indices = filter_by_confidence(
        boxes,
        scores,
        classes,
        conf_threshold=0.25,
    )

    np.testing.assert_array_equal(indices, np.array([1, 2]))
    np.testing.assert_allclose(filtered_scores, np.array([0.25, 0.251], dtype=np.float32))
    np.testing.assert_array_equal(filtered_classes, np.array([1, 2]))


def test_o2m_nms_suppresses_same_class_only_and_caps_by_score() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 9.0, 9.0],
            [1.0, 1.0, 9.0, 9.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.95], dtype=np.float32)
    classes = np.array([1, 1, 2, 1])

    keep = class_aware_nms(
        boxes,
        scores,
        classes,
        iou_threshold=0.5,
        max_detections=2,
    )

    np.testing.assert_array_equal(keep, np.array([3, 0]))


def test_nms_free_pipeline_path_never_calls_nms(monkeypatch) -> None:
    metadata = _metadata(branch="nms-free")
    pipeline = OnnxDetectorPipeline(
        metadata=metadata,
        session=OnnxDetectorSession(OrtSessionConfig(model_path=Path("model.onnx"))),
    )
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 9.0, 9.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.8, 0.95, 0.7], dtype=np.float32)
    classes = np.array([1, 1, 2])

    def fail_nms(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("NMS-free path must not call class_aware_nms")

    monkeypatch.setattr(
        "complexity.deploy.onnx_detector.postprocess.class_aware_nms",
        fail_nms,
    )

    filtered_boxes, filtered_scores, filtered_classes = pipeline.postprocess_single_image(
        boxes,
        scores,
        classes,
    )

    np.testing.assert_allclose(filtered_scores, np.array([0.95, 0.8], dtype=np.float32))
    np.testing.assert_allclose(
        filtered_boxes,
        np.array([[1.0, 1.0, 9.0, 9.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(filtered_classes, np.array([1, 1]))
