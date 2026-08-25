"""Anchor-center generation for TR-Hash Vision v8 ONNX detector exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GridGeometry:
    """Concatenated row-major feature-grid geometry."""

    centers_xy: np.ndarray
    strides: np.ndarray
    grid_sizes: tuple[int, ...]


def generate_grid_geometry(image_size: int, grid_sizes: Sequence[int]) -> GridGeometry:
    """Generate concatenated anchor centers and strides for each grid level."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")

    resolved_grid_sizes = tuple(int(grid) for grid in grid_sizes)
    if not resolved_grid_sizes:
        raise ValueError("grid_sizes must not be empty")
    if any(grid <= 0 for grid in resolved_grid_sizes):
        raise ValueError("grid_sizes must contain only positive values")

    centers: list[np.ndarray] = []
    strides: list[np.ndarray] = []
    for grid in resolved_grid_sizes:
        stride = float(image_size) / float(grid)
        coords = (np.arange(grid, dtype=np.float32) + 0.5) * np.float32(stride)
        x_centers, y_centers = np.meshgrid(coords, coords, indexing="xy")
        centers.append(
            np.stack((x_centers.reshape(-1), y_centers.reshape(-1)), axis=1)
        )
        strides.append(np.full((grid * grid,), stride, dtype=np.float32))

    return GridGeometry(
        centers_xy=np.concatenate(centers, axis=0).astype(np.float32, copy=False),
        strides=np.concatenate(strides, axis=0).astype(np.float32, copy=False),
        grid_sizes=resolved_grid_sizes,
    )
