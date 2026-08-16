"""Lazy, box-aware Albumentations adapter for detection training."""

from __future__ import annotations

import inspect
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
from PIL import Image


def _sanitize_yolo_targets(targets: torch.Tensor) -> torch.Tensor:
    """Clip normalized boxes and remove invalid boxes before Albumentations."""

    if not targets.numel():
        return torch.empty(0, 5, dtype=torch.float32)
    cleaned = targets.to(dtype=torch.float32).clone()
    finite = torch.isfinite(cleaned).all(dim=1)
    cleaned = cleaned[finite]
    if not cleaned.numel():
        return torch.empty(0, 5, dtype=torch.float32)

    centers = cleaned[:, :2]
    sizes = cleaned[:, 2:4]
    xy_min = (centers - sizes / 2).clamp(0.0, 1.0)
    xy_max = (centers + sizes / 2).clamp(0.0, 1.0)
    keep = (xy_max[:, 0] > xy_min[:, 0]) & (xy_max[:, 1] > xy_min[:, 1])
    cleaned = cleaned[keep]
    xy_min = xy_min[keep]
    xy_max = xy_max[keep]
    if not cleaned.numel():
        return torch.empty(0, 5, dtype=torch.float32)

    cleaned[:, :2] = (xy_min + xy_max) / 2
    cleaned[:, 2:4] = xy_max - xy_min
    return cleaned


@contextmanager
def _temporary_seed(seed: int) -> Iterator[None]:
    """Make Albumentations deterministic without leaking global RNG changes."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _affine_fill_arguments(affine: object) -> dict[str, object]:
    parameters = inspect.signature(affine).parameters
    if "fill" in parameters:
        return {"fill": (114, 114, 114)}
    return {"cval": (114, 114, 114)}


def _bbox_parameters(module: object) -> object:
    parameters = inspect.signature(module.BboxParams).parameters
    arguments: dict[str, object] = {
        "format": "yolo",
        "label_fields": ["class_labels"],
        "min_area": 4.0,
        "min_visibility": 0.05,
    }
    if "clip" in parameters:
        arguments["clip"] = True
    return module.BboxParams(**arguments)


def build_transform(mode: str) -> object:
    try:
        import albumentations as A
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Albumentations requires the detection extra: pip install -e '.[detection]'"
        ) from error

    transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.20 if mode == "light" else 0.35,
            contrast_limit=0.20 if mode == "light" else 0.35,
            p=0.8,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8 if mode == "light" else 15,
            sat_shift_limit=20 if mode == "light" else 35,
            val_shift_limit=15 if mode == "light" else 25,
            p=0.7,
        ),
    ]
    if mode == "strong":
        transforms.extend(
            [
                A.Affine(
                    scale=(0.75, 1.25),
                    translate_percent=(-0.15, 0.15),
                    rotate=(-10, 10),
                    shear=(-3, 3),
                    p=0.8,
                    **_affine_fill_arguments(A.Affine),
                ),
                A.ToGray(p=0.10),
                A.GaussianBlur(blur_limit=(3, 5), p=0.10),
            ]
        )
    return A.Compose(transforms, bbox_params=_bbox_parameters(A))


def apply_albumentations(
    transform: object,
    image: Image.Image,
    targets: torch.Tensor,
    *,
    seed: int,
) -> tuple[Image.Image, torch.Tensor]:
    """Apply a transform to normalized YOLO boxes and preserve tensor layout."""

    targets = _sanitize_yolo_targets(targets)
    boxes = targets[:, :4].tolist() if targets.numel() else []
    labels = targets[:, 4].tolist() if targets.numel() else []
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)
    with _temporary_seed(seed):
        result = transform(
            image=np.asarray(image),
            bboxes=boxes,
            class_labels=labels,
        )
    output_boxes = result["bboxes"]
    output_labels = result["class_labels"]
    if output_boxes:
        boxes_tensor = torch.tensor(output_boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(output_labels, dtype=torch.float32).unsqueeze(1)
        remapped = torch.cat((boxes_tensor, labels_tensor), dim=1)
        remapped = _sanitize_yolo_targets(remapped)
    else:
        remapped = torch.empty(0, 5, dtype=torch.float32)
    return Image.fromarray(result["image"].astype(np.uint8)), remapped
