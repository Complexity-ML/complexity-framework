"""Metadata sidecar schema for TR-Hash Vision v8 ONNX detector exports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

BranchType = Literal["nms-free", "o2m"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 300


@dataclass(frozen=True)
class OnnxDetectorMetadata:
    """Validated deployment metadata loaded from an ONNX sidecar JSON file."""

    architecture_version: int
    image_size: int
    num_classes: int
    num_cells: int
    regression_width: int
    reg_max: int
    scale_factors: tuple[int, ...]
    grid_sizes: tuple[int, ...]
    p2_head: bool
    branch: BranchType
    requires_nms: bool
    output_semantics: str
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    max_detections: int = DEFAULT_MAX_DETECTIONS
    class_names: tuple[str, ...] | None = None

    @property
    def dfl_bins(self) -> int:
        return self.reg_max + 1 if self.reg_max else 1

    @property
    def prediction_width(self) -> int:
        return self.regression_width + self.num_classes

    @property
    def strides(self) -> tuple[float, ...]:
        return tuple(float(self.image_size) / float(grid) for grid in self.grid_sizes)

    def as_dict(self) -> dict[str, object]:
        return {
            "architecture_version": self.architecture_version,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "num_cells": self.num_cells,
            "regression_width": self.regression_width,
            "reg_max": self.reg_max,
            "scale_factors": list(self.scale_factors),
            "grid_sizes": list(self.grid_sizes),
            "p2_head": self.p2_head,
            "branch": self.branch,
            "requires_nms": self.requires_nms,
            "output_semantics": self.output_semantics,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "class_names": list(self.class_names) if self.class_names is not None else None,
        }


def load_metadata(path: str | Path) -> OnnxDetectorMetadata:
    """Load and validate an ONNX detector metadata sidecar."""

    metadata_path = Path(path)
    return metadata_from_mapping(json.loads(metadata_path.read_text()))


def metadata_from_mapping(values: Mapping[str, object]) -> OnnxDetectorMetadata:
    """Create validated metadata from a mapping."""

    branch = _branch(values.get("branch"))
    requires_nms = bool(values.get("requires_nms", branch == "o2m"))
    if requires_nms is not (branch == "o2m"):
        raise ValueError("requires_nms must be true only for the o2m branch")

    class_names_value = values.get("class_names")
    class_names = None
    if class_names_value is not None:
        if not isinstance(class_names_value, Sequence) or isinstance(
            class_names_value, (str, bytes)
        ):
            raise ValueError("class_names must be a sequence of strings")
        class_names = tuple(str(name) for name in class_names_value)

    metadata = OnnxDetectorMetadata(
        architecture_version=_positive_int(values, "architecture_version"),
        image_size=_positive_int(values, "image_size"),
        num_classes=_positive_int(values, "num_classes"),
        num_cells=_positive_int(values, "num_cells"),
        regression_width=_positive_int(values, "regression_width"),
        reg_max=_non_negative_int(values, "reg_max"),
        scale_factors=_positive_int_tuple(values, "scale_factors"),
        grid_sizes=_positive_int_tuple(values, "grid_sizes"),
        p2_head=bool(values.get("p2_head", False)),
        branch=branch,
        requires_nms=requires_nms,
        output_semantics=str(values.get("output_semantics", "")),
        confidence_threshold=float(
            values.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        ),
        iou_threshold=float(values.get("iou_threshold", DEFAULT_IOU_THRESHOLD)),
        max_detections=int(values.get("max_detections", DEFAULT_MAX_DETECTIONS)),
        class_names=class_names,
    )
    _validate_metadata(metadata)
    return metadata


def validate_output_shape(
    metadata: OnnxDetectorMetadata,
    output_shape: Sequence[int | str | None],
) -> None:
    """Validate an ONNX output shape against the sidecar metadata."""

    if len(output_shape) != 3:
        raise ValueError(f"ONNX output must be rank 3, got shape {tuple(output_shape)}")
    _validate_axis(output_shape[1], metadata.num_cells, "num_cells")
    _validate_axis(output_shape[2], metadata.prediction_width, "prediction_width")


def _validate_metadata(metadata: OnnxDetectorMetadata) -> None:
    if metadata.architecture_version != 8:
        raise ValueError("only TR-Hash detector architecture v8 metadata is supported")
    if metadata.num_cells != sum(grid * grid for grid in metadata.grid_sizes):
        raise ValueError("num_cells must equal sum(grid ** 2 for grid_sizes)")
    if metadata.regression_width != 4 * metadata.dfl_bins:
        raise ValueError("regression_width must equal 4 * DFL bin count")
    if metadata.max_detections <= 0:
        raise ValueError("max_detections must be positive")
    if not 0.0 <= metadata.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 <= metadata.iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if (
        metadata.class_names is not None
        and len(metadata.class_names) != metadata.num_classes
    ):
        raise ValueError("class_names length must match num_classes")


def _branch(value: object) -> BranchType:
    normalized = "nms-free" if value == "nms_free" else value
    if normalized not in {"nms-free", "o2m"}:
        raise ValueError(f"unsupported ONNX detector branch: {value!r}")
    return cast(BranchType, normalized)


def _positive_int(values: Mapping[str, object], key: str) -> int:
    value = int(values[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _non_negative_int(values: Mapping[str, object], key: str) -> int:
    value = int(values[key])
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _positive_int_tuple(values: Mapping[str, object], key: str) -> tuple[int, ...]:
    raw = values[key]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of positive integers")
    resolved = tuple(int(item) for item in raw)
    if not resolved or any(item <= 0 for item in resolved):
        raise ValueError(f"{key} must contain positive integers")
    return resolved


def _validate_axis(axis: int | str | None, expected: int, name: str) -> None:
    if isinstance(axis, int) and axis != expected:
        raise ValueError(f"ONNX output {name} mismatch: expected {expected}, got {axis}")
