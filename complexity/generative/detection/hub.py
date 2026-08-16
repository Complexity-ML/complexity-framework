"""Hugging Face packaging and loading helpers for TR-Hash detectors."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from .config import TRHashDetectorConfig
from .model import TRHashObjectDetector
from .provenance import (
    NATIVE_COCO_DATASET,
    read_detector_provenance,
    validate_native_random_init_provenance,
)

VOC_CLASS_NAMES: Tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

COCO_CLASS_NAMES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

RELEASE_METRICS = frozenset(
    {
        "map50",
        "map50_95",
        "ap_small",
        "ap_medium",
        "ap_large",
        "precision",
        "recall",
        "best_f1",
        "best_confidence",
        "coco_map50",
        "coco_map50_95",
        "coco_ap_small",
        "coco_ap_medium",
        "coco_ap_large",
        "coco_ar100",
        "official_coco",
        "evaluator_backend",
        "checkpoint_selection_metric",
    }
)
HUB_ARTIFACT_NAMES = (
    "README.md",
    "class_names.json",
    "config.json",
    "ema.safetensors",
    "ema_tower.safetensors",
    "model.safetensors",
    "preprocessor_config.json",
    "provenance.json",
    "tower.safetensors",
    "validation.json",
    "validation_nms_free.json",
)


@dataclass(frozen=True)
class DetectionImageMetadata:
    """Geometry required to restore letterboxed predictions to source pixels."""

    original_width: int
    original_height: int
    image_size: int
    scale: float
    left: int
    top: int


def preprocess_detector_image(
    image: Image.Image, image_size: int
) -> tuple[torch.Tensor, DetectionImageMetadata]:
    """Apply the exact RGB letterbox and normalization used during training."""

    image = image.convert("RGB")
    original_width, original_height = image.size
    scale = min(image_size / original_width, image_size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    pixels = torch.from_numpy(np.array(canvas)).float().permute(2, 0, 1) / 255.0
    pixels = (pixels - 0.5) / 0.5
    metadata = DetectionImageMetadata(
        original_width=original_width,
        original_height=original_height,
        image_size=image_size,
        scale=scale,
        left=left,
        top=top,
    )
    return pixels, metadata


def restore_detector_boxes(boxes: torch.Tensor, metadata: DetectionImageMetadata) -> torch.Tensor:
    """Map normalized letterbox ``xyxy`` boxes back to source-image pixels."""

    restored = boxes.clone()
    restored[:, (0, 2)] = (
        restored[:, (0, 2)] * metadata.image_size - metadata.left
    ) / metadata.scale
    restored[:, (1, 3)] = (
        restored[:, (1, 3)] * metadata.image_size - metadata.top
    ) / metadata.scale
    restored[:, (0, 2)] = restored[:, (0, 2)].clamp(0, metadata.original_width)
    restored[:, (1, 3)] = restored[:, (1, 3)].clamp(0, metadata.original_height)
    return restored


def load_detector_checkpoint(
    checkpoint: Path | str, *, device: str | torch.device = "cpu"
) -> TRHashObjectDetector:
    """Load a framework or Hub snapshot directory with strict state validation."""

    checkpoint = Path(checkpoint)
    config = TRHashDetectorConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    model = TRHashObjectDetector(config)
    weights = (
        checkpoint / "ema.safetensors"
        if (checkpoint / "ema.safetensors").is_file()
        else checkpoint / "model.safetensors"
    )
    state = load_file(str(weights), device=str(device))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_detector_from_hub(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    device: str | torch.device = "cpu",
    token: Optional[str] = None,
) -> TRHashObjectDetector:
    """Download a TR-Hash detector snapshot and load it strictly."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("install huggingface_hub to load a Hub checkpoint") from error
    checkpoint = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        allow_patterns=("config.json", "model.safetensors", "ema.safetensors"),
    )
    return load_detector_checkpoint(checkpoint, device=device)


def _dataset_metadata(
    class_names: Sequence[str], dataset: Optional[str] = None
) -> tuple[str, str, str, str]:
    if dataset is not None:
        normalized = dataset.lower().replace("-", "")
        explicit = {
            "coco": ("COCO 2017", "coco", "coco", "coco"),
            "voc": ("Pascal VOC 2007+2012", "pascal_voc", "pascal-voc", "voc"),
            "pascalvoc": (
                "Pascal VOC 2007+2012",
                "pascal_voc",
                "pascal-voc",
                "voc",
            ),
        }
        if normalized not in explicit:
            raise ValueError(f"unsupported detection dataset: {dataset}")
        return explicit[normalized]
    names = tuple(class_names)
    if names == COCO_CLASS_NAMES:
        return "COCO 2017", "coco", "coco", "coco"
    if names == VOC_CLASS_NAMES:
        return "Pascal VOC 2007+2012", "pascal_voc", "pascal-voc", "voc"
    return "Custom detection dataset", "custom", "custom-dataset", "custom"


def _model_card(
    repo_id: str,
    *,
    config: Optional[TRHashDetectorConfig],
    metrics: Optional[Dict[str, float]],
    nms_free_metrics: Optional[Dict[str, float]],
    class_names: Sequence[str],
    training: bool,
    dataset: Optional[str] = None,
) -> str:
    model_name = repo_id.split("/")[-1]
    image_size = config.image_size if config is not None else 640
    architecture_version = config.architecture_version if config is not None else 6
    dataset_name, dataset_type, dataset_tag, dataset_key = _dataset_metadata(class_names, dataset)
    if config is None:
        parameter_text = "approximately 1.12M"
    else:
        parameter_text = f"{sum(parameter.numel() for parameter in TRHashObjectDetector(config).parameters()) / 1e6:.2f}M"
    metrics_yaml = ""
    metrics_table = "Training is currently in progress; validated metrics will be added here."
    reported_metrics = metrics or nms_free_metrics
    if reported_metrics:
        map50_95 = reported_metrics.get("map50_95", reported_metrics["map50"])
        task_name = "Object Detection" if metrics is not None else "Object Detection (NMS-free)"
        metrics_yaml = f"""
model-index:
- name: {model_name}
  results:
  - task:
      type: object-detection
      name: {task_name}
    dataset:
      name: {dataset_name}
      type: {dataset_type}
    metrics:
    - name: mAP50
      type: map
      value: {reported_metrics["map50"]:.6f}
    - name: mAP50-95
      type: map
      value: {map50_95:.6f}
""".rstrip()
        rows = []
        for branch, values in (
            ("O2M + NMS", metrics),
            ("NMS-free", nms_free_metrics),
        ):
            if values is None:
                continue
            rows.append(
                f"| {branch} | {values['map50']:.4f} | "
                f"{values.get('map50_95', values['map50']):.4f} | "
                f"{values.get('ap_small', 0.0):.4f} | "
                f"{values.get('ap_medium', 0.0):.4f} | "
                f"{values.get('ap_large', 0.0):.4f} | "
                f"{values['precision']:.4f} | {values['recall']:.4f} | "
                f"{values['best_f1']:.4f} | {values['best_confidence']:.3f} |"
            )
        metrics_table = (
            "| Inference branch | mAP50 | mAP50-95 | AP small | AP medium | "
            "AP large | Precision | Recall | Best F1 | Best confidence |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
        )
    status = (
        "> **Training in progress.** This private draft intentionally contains no final "
        "weights or release metrics yet."
        if training
        else "This repository contains the validated TR-Hash detector checkpoint."
    )
    inference = (
        "Inference instructions will be added when validated v6 weights are uploaded."
        if training
        else f"""```python
from PIL import Image
import torch

from complexity.generative.detection import (
    load_detector_from_hub,
    preprocess_detector_image,
    restore_detector_boxes,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_detector_from_hub("{repo_id}", device=device)
pixels, metadata = preprocess_detector_image(
    Image.open("image.jpg"), model.config.image_size
)

with torch.inference_mode():
    prediction = model.predict(pixels[None].to(device))[0]
prediction["boxes"] = restore_detector_boxes(
    prediction["boxes"].cpu(), metadata
)
```

Class IDs are listed in `class_names.json`. Boxes returned by `predict` are
normalized `xyxy` coordinates until `restore_detector_boxes` maps them to source
pixels."""
    )
    if dataset_key == "coco":
        initialization = "Random initialization; no external detector or classification backbone"
        training_role = "Full-detector COCO 2017 training"
        augmentation = "EMA, multi-resolution, Mosaic, MixUp, Copy-Paste and random erasing"
    else:
        initialization = "Random initialization or an explicitly documented task checkpoint"
        training_role = f"{dataset_name} detector training"
        augmentation = "EMA and task-specific detection augmentation"
    return f"""---
license: cc-by-nc-4.0
library_name: complexity-framework
pipeline_tag: object-detection
tags:
- object-detection
- tr-hash
- mixture-of-experts
- hierarchical-vision-transformer
- pytorch
- triton
- {dataset_tag}
{metrics_yaml}
---

# {model_name}

{status}

TR-Hash Vision v{architecture_version} is a compact anchor-free detector built on an
ID-hash-routed MoE vision tower. Spatial-token IDs select expert parameter
subspaces while shared attention preserves contextual feature mixing. The detector combines native
hierarchical P3/P4/P5 features, shifted-window attention, a lightweight PAN,
an optional P2 small-object path, dynamic one-to-many assignment with STAL,
decoupled LTRB/DFL regression, unified QFL quality-class scores, and an optional
one-to-one NMS-free inference branch. The realized model has
**{parameter_text} parameters** and uses **{image_size} px** inputs on {dataset_name}.

The specialized detector path adds residual adapters at every pyramid level, an
ID-hash-routed class x level gate, class x object-size x scene-density weighting,
level-balancing and gate-calibration objectives, and supervised object contrastive
learning. Motion features are not applied to static images; they are enabled only
for an explicit temporal video input branch.

## Evaluation

{metrics_table}

## Inference

{inference}

## Training

- Dataset: {dataset_name}
- Optimizer: MuSGD
- Routed-expert LR multiplier: 1.5x
- Initialization: {initialization}
- Role: {training_role}
- Training: {augmentation}
- Framework: [Complexity Framework](https://github.com/Complexity-ML/complexity-framework)

## Limitations

This is a research checkpoint under CC BY-NC 4.0. Validate accuracy, calibration,
latency, and failure modes on your own target domain. A training configuration or
active run is not evidence of accuracy; release claims require a realized checkpoint
and an explicit evaluation protocol.
"""


def _validate_release_metrics(values: object, *, branch: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{branch} validation metrics must be a JSON object")
    missing = sorted(RELEASE_METRICS.difference(values))
    if missing:
        raise ValueError(f"{branch} validation is missing release metrics: " + ", ".join(missing))
    if values["official_coco"] is not True:
        raise ValueError(f"{branch} validation must use official COCOeval")
    if values["checkpoint_selection_metric"] != "coco_map50_95":
        raise ValueError(f"{branch} checkpoint must be selected by official COCO mAP50-95")
    if values["map50"] != values["coco_map50"] or values["map50_95"] != values["coco_map50_95"]:
        raise ValueError(f"{branch} public mAP fields must match official COCOeval")


def _validate_native_coco_release(checkpoint: Path, dataset: Optional[str]) -> None:
    if dataset != "coco":
        raise ValueError("native release validation currently requires dataset='coco'")
    if checkpoint.name != "best":
        raise ValueError("only the validated best checkpoint can be released")
    provenance = read_detector_provenance(checkpoint)
    validate_native_random_init_provenance(provenance, dataset=NATIVE_COCO_DATASET)
    if not any((checkpoint / name).is_file() for name in ("ema.safetensors", "model.safetensors")):
        raise ValueError("validated checkpoint does not contain detector weights")
    validation_path = checkpoint / "validation.json"
    if not validation_path.is_file():
        raise ValueError("validated best checkpoint is missing validation.json")
    _validate_release_metrics(json.loads(validation_path.read_text()), branch="O2M + NMS")
    nms_free_path = checkpoint.parent / "best_nms_free" / "validation.json"
    if not nms_free_path.is_file():
        raise ValueError("validated release is missing NMS-free evaluation metrics")
    _validate_release_metrics(json.loads(nms_free_path.read_text()), branch="NMS-free")


def export_detector_for_hub(
    output: Path | str,
    repo_id: str,
    *,
    checkpoint: Optional[Path | str] = None,
    class_names: Sequence[str] = VOC_CLASS_NAMES,
    training: bool = False,
    dataset: Optional[str] = None,
    require_native_random_init: bool = False,
) -> Path:
    """Build a complete Hub upload folder, or a card-only training draft."""

    output = Path(output)
    config = None
    metrics = None
    nms_free_metrics = None
    if require_native_random_init and checkpoint is None:
        raise ValueError("native release validation requires a checkpoint")
    if checkpoint is not None and require_native_random_init:
        _validate_native_coco_release(Path(checkpoint), dataset)

    output.mkdir(parents=True, exist_ok=True)
    for name in HUB_ARTIFACT_NAMES:
        stale = output / name
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        config = TRHashDetectorConfig.from_dict(
            json.loads((checkpoint / "config.json").read_text())
        )
        if len(class_names) != config.num_classes:
            raise ValueError("class_names length must match detector num_classes")
        for name in (
            "config.json",
            "model.safetensors",
            "tower.safetensors",
            "ema.safetensors",
            "ema_tower.safetensors",
            "provenance.json",
        ):
            source = checkpoint / name
            if source.exists():
                shutil.copy2(source, output / name)
        validation_path = checkpoint / "validation.json"
        if validation_path.exists():
            metrics = json.loads(validation_path.read_text())
            shutil.copy2(validation_path, output / "validation.json")
        nms_free_validation = checkpoint.parent / "best_nms_free" / "validation.json"
        if nms_free_validation.is_file():
            nms_free_metrics = json.loads(nms_free_validation.read_text())
            shutil.copy2(nms_free_validation, output / "validation_nms_free.json")
        (output / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "do_resize": True,
                    "size": {"height": config.image_size, "width": config.image_size},
                    "do_rescale": True,
                    "rescale_factor": 1.0 / 255.0,
                    "do_normalize": True,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "letterbox": True,
                    "letterbox_color": [114, 114, 114],
                },
                indent=2,
            )
            + "\n"
        )
    (output / "class_names.json").write_text(json.dumps(list(class_names), indent=2) + "\n")
    (output / "README.md").write_text(
        _model_card(
            repo_id,
            config=config,
            metrics=metrics if not training else None,
            nms_free_metrics=nms_free_metrics if not training else None,
            class_names=class_names,
            training=training,
            dataset=dataset,
        )
    )
    return output


def upload_detector_to_hub(
    folder: Path | str,
    repo_id: str,
    *,
    private: bool = True,
    token: Optional[str] = None,
    commit_message: str = "Publish TR-Hash Vision detector",
) -> str:
    """Create/update a Hub model repository from an exported folder."""

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError("install huggingface_hub to publish a Hub checkpoint") from error
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    ).oid
