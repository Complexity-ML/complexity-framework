"""DFL box decode for TR-Hash Vision v8 ONNX detector exports."""

from __future__ import annotations

import numpy as np

from .grid import GridGeometry
from .metadata import OnnxDetectorMetadata


def decode_dfl_boxes(
    regression_logits: np.ndarray,
    metadata: OnnxDetectorMetadata,
    geometry: GridGeometry,
) -> np.ndarray:
    """Decode raw LTRB DFL logits into input-pixel xyxy boxes."""

    logits = np.asarray(regression_logits, dtype=np.float32)
    if logits.ndim not in {2, 3}:
        raise ValueError("regression_logits must have shape [N, C] or [B, N, C]")
    if logits.shape[-1] != metadata.regression_width:
        raise ValueError(
            "regression_logits last dimension does not match metadata.regression_width"
        )
    if logits.shape[-2] != geometry.centers_xy.shape[0]:
        raise ValueError("regression_logits cell count does not match grid geometry")

    bins = metadata.dfl_bins
    distances = logits.reshape(*logits.shape[:-1], 4, bins)
    if metadata.reg_max:
        shifted = distances - distances.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        bucket_indices = np.arange(bins, dtype=np.float32)
        distances = (probabilities * bucket_indices).sum(axis=-1)
    else:
        # DFL disabled mirrors PyTorch's single-bin regression width contract.
        distances = np.log1p(np.exp(distances[..., 0]))

    distances_px = distances * geometry.strides.reshape(1, -1, 1)
    centers = geometry.centers_xy.reshape(1, -1, 2)
    top_left = centers - distances_px[..., (0, 1)]
    bottom_right = centers + distances_px[..., (2, 3)]
    boxes = np.concatenate((top_left, bottom_right), axis=-1)
    boxes = np.clip(boxes, 0.0, float(metadata.image_size)).astype(np.float32)

    if logits.ndim == 2:
        return boxes[0]
    return boxes
