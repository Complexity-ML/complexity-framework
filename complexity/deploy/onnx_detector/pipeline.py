"""High-level preprocessing, inference, decode, and postprocess pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .dfl import decode_dfl_boxes
from .grid import GridGeometry, generate_grid_geometry
from .metadata import OnnxDetectorMetadata
from . import postprocess
from .metadata import load_metadata, validate_output_shape
from .preprocess import preprocess_image, restore_boxes
from .session import OnnxDetectorSession, OrtSessionConfig
from .types import Detection, DetectionResult, TimingBreakdown


@dataclass
class OnnxDetectorPipeline:
    """Deployment pipeline configured by ONNX model and metadata sidecar."""

    metadata: OnnxDetectorMetadata
    session: OnnxDetectorSession
    geometry: GridGeometry | None = None

    def __post_init__(self) -> None:
        if self.geometry is None:
            self.geometry = generate_grid_geometry(
                self.metadata.image_size,
                self.metadata.grid_sizes,
            )

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        metadata_path: str | Path,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> "OnnxDetectorPipeline":
        metadata = load_metadata(metadata_path)
        session = cls.create_session(model_path, providers).open()
        validate_output_shape(metadata, session._require_session().get_outputs()[0].shape)
        if session.config.warmup_runs:
            session.warmup((1, 3, metadata.image_size, metadata.image_size))
        return cls(metadata=metadata, session=session)

    def predict(self, image: object) -> DetectionResult:
        preprocess_start = perf_counter()
        preprocessed = preprocess_image(image, self.metadata.image_size)
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        inference_start = perf_counter()
        predictions = self.session.run(preprocessed.pixel_values)
        inference_ms = (perf_counter() - inference_start) * 1000.0

        postprocess_start = perf_counter()
        detections = self._decode_and_postprocess(predictions, preprocessed.geometry)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0

        return DetectionResult(
            detections=tuple(detections),
            timing=TimingBreakdown(preprocess_ms, inference_ms, postprocess_ms),
            provider_used=self.session.provider_used,
            branch_type=self.metadata.branch,
            metadata=self.metadata.as_dict(),
        )

    def postprocess_single_image(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        filtered_boxes, filtered_scores, filtered_classes, _ = (
            postprocess.filter_by_confidence(
                boxes,
                scores,
                classes,
                self.metadata.confidence_threshold,
            )
        )
        if self.metadata.branch == "o2m":
            keep = postprocess.class_aware_nms(
                filtered_boxes,
                filtered_scores,
                filtered_classes,
                self.metadata.iou_threshold,
                self.metadata.max_detections,
            )
            return filtered_boxes[keep], filtered_scores[keep], filtered_classes[keep]

        order = np.argsort(filtered_scores)[::-1]
        keep = order[: self.metadata.max_detections]
        return filtered_boxes[keep], filtered_scores[keep], filtered_classes[keep]

    @staticmethod
    def create_session(
        model_path: str | Path,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> OnnxDetectorSession:
        return OnnxDetectorSession(
            OrtSessionConfig(Path(model_path), providers=tuple(providers))
        )

    def _decode_and_postprocess(
        self,
        predictions: np.ndarray,
        image_geometry: object,
    ) -> list[Detection]:
        if predictions.ndim != 3 or predictions.shape[0] != 1:
            raise ValueError("pipeline.predict currently expects ONNX output shape [1, N, C]")
        validate_output_shape(self.metadata, predictions.shape)

        regression = predictions[..., : self.metadata.regression_width]
        class_logits = predictions[..., self.metadata.regression_width :]
        boxes_input = decode_dfl_boxes(regression, self.metadata, self.geometry)[0]
        class_scores = _sigmoid(class_logits[0])
        scores = class_scores.max(axis=-1)
        classes = class_scores.argmax(axis=-1).astype(np.int64, copy=False)

        kept_boxes, kept_scores, kept_classes = self.postprocess_single_image(
            boxes_input,
            scores,
            classes,
        )
        boxes_pixel = restore_boxes(kept_boxes, image_geometry)
        boxes_norm = np.clip(kept_boxes / float(self.metadata.image_size), 0.0, 1.0)
        return [
            Detection(
                box_norm=tuple(float(value) for value in norm_box),
                box_pixel=tuple(float(value) for value in pixel_box),
                class_id=int(class_id),
                score=float(score),
            )
            for norm_box, pixel_box, class_id, score in zip(
                boxes_norm,
                boxes_pixel,
                kept_classes,
                kept_scores,
            )
        ]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))
