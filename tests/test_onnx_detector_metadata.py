import json

import pytest

from complexity.deploy.onnx_detector.metadata import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    load_metadata,
    metadata_from_mapping,
    validate_output_shape,
)


def _sidecar(branch: str = "o2m") -> dict[str, object]:
    return {
        "architecture_version": 8,
        "image_size": 640,
        "num_classes": 80,
        "num_cells": 34000,
        "regression_width": 68,
        "reg_max": 16,
        "scale_factors": [1, 2, 4],
        "grid_sizes": [160, 80, 40, 20],
        "p2_head": True,
        "branch": branch,
        "requires_nms": branch == "o2m",
        "output_semantics": "raw_ltrb_dfl_and_quality_class_logits",
    }


def test_load_metadata_applies_defaults_and_derives_strides(tmp_path) -> None:
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(json.dumps(_sidecar()))

    metadata = load_metadata(metadata_path)

    assert metadata.branch == "o2m"
    assert metadata.requires_nms is True
    assert metadata.dfl_bins == 17
    assert metadata.prediction_width == 148
    assert metadata.strides == (4.0, 8.0, 16.0, 32.0)
    assert metadata.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert metadata.iou_threshold == DEFAULT_IOU_THRESHOLD


def test_metadata_normalizes_nms_free_branch_spelling() -> None:
    values = _sidecar("nms_free")
    values["requires_nms"] = False

    metadata = metadata_from_mapping(values)

    assert metadata.branch == "nms-free"
    assert metadata.requires_nms is False


def test_metadata_rejects_inconsistent_cell_count() -> None:
    values = _sidecar()
    values["num_cells"] = 123

    with pytest.raises(ValueError, match="num_cells"):
        metadata_from_mapping(values)


def test_validate_output_shape_accepts_dynamic_batch_and_rejects_width_mismatch() -> None:
    metadata = metadata_from_mapping(_sidecar())

    validate_output_shape(metadata, ("batch_size", 34000, 148))

    with pytest.raises(ValueError, match="prediction_width"):
        validate_output_shape(metadata, (1, 34000, 149))
