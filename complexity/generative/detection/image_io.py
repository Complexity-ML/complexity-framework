"""Image decoding backends for detection datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_BACKENDS = frozenset({"pillow", "opencv"})


def load_rgb_image(path: Path, backend: str = "pillow") -> Image.Image:
    """Decode an image as RGB while preserving the dataset's PIL contract."""

    if backend not in IMAGE_BACKENDS:
        raise ValueError(f"unsupported image backend: {backend}")
    if backend == "pillow":
        with Image.open(path) as source:
            return source.convert("RGB").copy()

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "OpenCV decoding requires the detection extra: pip install -e '.[detection]'"
        ) from error
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb))
