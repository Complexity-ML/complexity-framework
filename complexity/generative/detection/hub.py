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
    state = load_file(str(checkpoint / "model.safetensors"), device=str(device))
    state = {
        name: value for name, value in state.items() if not name.startswith("one_to_one_heads.")
    }
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
        allow_patterns=("config.json", "model.safetensors"),
    )
    return load_detector_checkpoint(checkpoint, device=device)


def _model_card(
    repo_id: str,
    *,
    config: Optional[TRHashDetectorConfig],
    metrics: Optional[Dict[str, float]],
    class_names: Sequence[str],
    training: bool,
) -> str:
    model_name = repo_id.split("/")[-1]
    image_size = config.image_size if config is not None else 224
    parameter_text = "0.83M" if config is None else "compact"
    metrics_yaml = ""
    metrics_table = "Training is currently in progress; validated metrics will be added here."
    if metrics:
        metrics_yaml = f"""
model-index:
- name: {model_name}
  results:
  - task:
      type: object-detection
      name: Object Detection
    dataset:
      name: Pascal VOC 2007+2012
      type: pascal_voc
    metrics:
    - name: mAP50
      type: map
      value: {metrics['map50']:.6f}
""".rstrip()
        metrics_table = (
            "| mAP50 | Precision | Recall | Best F1 | Best confidence |\n"
            "|---:|---:|---:|---:|---:|\n"
            f"| {metrics['map50']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['best_f1']:.4f} | "
            f"{metrics['best_confidence']:.3f} |"
        )
    status = (
        "> **Training in progress.** This private draft intentionally contains no final "
        "weights or release metrics yet."
        if training
        else "This repository contains the validated TR-Hash detector checkpoint."
    )
    return f"""---
license: cc-by-nc-4.0
library_name: complexity-framework
pipeline_tag: object-detection
tags:
- object-detection
- tr-hash
- mixture-of-experts
- pytorch
- pascal-voc
{metrics_yaml}
---

# {model_name}

{status}

TR-Hash Vision is a compact anchor-free detector built on a deterministic
token-routed MoE vision tower. The architecture combines dynamic one-to-many
assignment, class-aware batched NMS, a P2 small-object scale, STAL-style
assignment, and progressive loss balancing. The target release is
approximately **{parameter_text} parameters**, trained at **{image_size} px** on
Pascal VOC 2007+2012.

## Evaluation

{metrics_table}

## Inference

```python
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
pixels.

## Training

- Dataset: Pascal VOC 2007+2012, 16,551 train and 4,952 validation images
- Optimizer: SGD with Nesterov momentum
- Routed-expert LR multiplier: 1.5x
- Backbone initialization: TR-Hash ImageNet-100 vision pretraining
- Framework: [Complexity Framework](https://github.com/Complexity-ML/complexity-framework)

## Limitations

This is a research checkpoint under CC BY-NC 4.0. Validate accuracy, calibration,
latency, and failure modes on your own target domain. Pascal VOC is small and does
not represent the diversity of modern production detection datasets.
"""


def export_detector_for_hub(
    output: Path | str,
    repo_id: str,
    *,
    checkpoint: Optional[Path | str] = None,
    class_names: Sequence[str] = VOC_CLASS_NAMES,
    training: bool = False,
) -> Path:
    """Build a complete Hub upload folder, or a card-only training draft."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = None
    metrics = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        config = TRHashDetectorConfig.from_dict(
            json.loads((checkpoint / "config.json").read_text())
        )
        if len(class_names) != config.num_classes:
            raise ValueError("class_names length must match detector num_classes")
        for name in ("config.json", "model.safetensors", "tower.safetensors"):
            source = checkpoint / name
            if source.exists():
                shutil.copy2(source, output / name)
        validation_path = checkpoint / "validation.json"
        if validation_path.exists():
            metrics = json.loads(validation_path.read_text())
            shutil.copy2(validation_path, output / "validation.json")
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
            class_names=class_names,
            training=training,
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
