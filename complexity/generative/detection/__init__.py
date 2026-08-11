"""TR-Hash single-stage object detector (YOLO-style, anchor-free).

Built on ``TRHashVisionTower`` (real multi-expert MoE routing by patch
position — see ``complexity.generative.vision_language``), with a detection
head predicting local LTRB box distributions and joint quality-class scores
directly per grid cell. No anchors or region-proposal stage.

Usage:
    from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector

    model = TRHashObjectDetector(TRHashDetectorConfig(num_classes=80))
    raw = model(pixel_values)  # [batch, num_cells, 4 * DFL bins + num_classes]

    # targets: list of [N_i, 5] (cx, cy, w, h, class_id), normalized [0, 1]
    losses = model.compute_loss(raw, targets)
    losses["loss"].backward()

    detections = model.predict(pixel_values, confidence_threshold=0.25)
"""

from .config import TRHashDetectorConfig
from .data import (
    CocoDetectionDataset,
    SyntheticShapesDataset,
    YoloDetectionDataset,
    collate_detection,
)
from .hub import (
    COCO_CLASS_NAMES,
    VOC_CLASS_NAMES,
    DetectionImageMetadata,
    export_detector_for_hub,
    load_detector_checkpoint,
    load_detector_from_hub,
    preprocess_detector_image,
    restore_detector_boxes,
    upload_detector_to_hub,
)
from .losses import distribution_focal_loss, quality_focal_loss
from .model import (
    TRHashObjectDetector,
    box_iou,
    class_aware_nms,
    complete_iou_loss,
)
from .ops import greedy_nms

__all__ = [
    "TRHashDetectorConfig",
    "TRHashObjectDetector",
    "box_iou",
    "class_aware_nms",
    "complete_iou_loss",
    "greedy_nms",
    "distribution_focal_loss",
    "quality_focal_loss",
    "CocoDetectionDataset",
    "SyntheticShapesDataset",
    "YoloDetectionDataset",
    "collate_detection",
    "COCO_CLASS_NAMES",
    "VOC_CLASS_NAMES",
    "DetectionImageMetadata",
    "export_detector_for_hub",
    "load_detector_checkpoint",
    "load_detector_from_hub",
    "preprocess_detector_image",
    "restore_detector_boxes",
    "upload_detector_to_hub",
]
