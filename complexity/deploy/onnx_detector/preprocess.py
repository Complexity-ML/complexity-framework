"""Image preprocessing and box restoration for ONNX detector inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageGeometry:
    """Geometry required to undo square letterbox preprocessing."""

    original_width: int
    original_height: int
    image_size: int
    scale: float
    left: int
    top: int


@dataclass(frozen=True)
class PreprocessResult:
    """Preprocessed model input and the geometry needed for restoration."""

    pixel_values: np.ndarray
    geometry: ImageGeometry


def preprocess_image(image: Any, image_size: int) -> PreprocessResult:
    """Mirror PyTorch RGB letterbox preprocessing for detector exports."""

    from PIL import Image

    if isinstance(image, (str, Path)):
        pil_image = Image.open(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.fromarray(np.asarray(image))

    pil_image = pil_image.convert("RGB")
    original_width, original_height = pil_image.size
    scale = min(image_size / original_width, image_size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2

    resized = pil_image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    canvas.paste(resized, (left, top))

    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    pixels = (pixels - 0.5) / 0.5
    pixel_values = np.expand_dims(pixels, axis=0).astype(np.float32, copy=False)
    geometry = ImageGeometry(
        original_width=original_width,
        original_height=original_height,
        image_size=image_size,
        scale=scale,
        left=left,
        top=top,
    )
    return PreprocessResult(pixel_values=pixel_values, geometry=geometry)


def restore_boxes(boxes: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
    """Map input-square pixel xyxy boxes back to source-image pixel xyxy boxes."""

    restored = np.asarray(boxes, dtype=np.float32).copy()
    restored[:, (0, 2)] = (restored[:, (0, 2)] - geometry.left) / geometry.scale
    restored[:, (1, 3)] = (restored[:, (1, 3)] - geometry.top) / geometry.scale
    restored[:, (0, 2)] = np.clip(restored[:, (0, 2)], 0.0, geometry.original_width)
    restored[:, (1, 3)] = np.clip(restored[:, (1, 3)], 0.0, geometry.original_height)
    return restored

