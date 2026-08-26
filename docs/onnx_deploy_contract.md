# TR-Hash Vision v8 ONNX Deployment Contract

This document records the PyTorch detector contract that deployment runtimes must
preserve when using exported TR-Hash Vision v8 ONNX models. The ONNX graph exports
one raw prediction branch only; preprocessing, decode, confidence filtering, and
optional NMS are runtime responsibilities.

## Source Of Truth

- Export wrapper: `complexity/generative/detection/exporting.py`
- Export CLI and sidecar metadata: `scripts/export_onnx.py`
- Detector geometry and decode: `complexity/generative/detection/model.py`
- Class-aware NMS: `complexity/generative/detection/ops.py`
- Image preprocessing and box restoration: `complexity/generative/detection/hub.py`
- Detector shape properties: `complexity/generative/detection/config.py`

## ONNX Tensor Contract

- Input name: `pixel_values`
- Input dtype and shape: `float32[B, 3, image_size, image_size]`
- Output name: `predictions`
- Output dtype and shape: `float32[B, num_cells, regression_width + num_classes]`
- Default opset: `17`, unless `scripts/export_onnx.py --opset` overrides it.
- Batch dimension is static `1` unless exported with `--dynamic-batch`.

For the current v8 640 px COCO exports:

- `image_size`: `640`
- `num_classes`: `80`
- `reg_max`: `16`
- `dfl_bins`: `17`
- `regression_width`: `68`
- `grid_sizes`: `[160, 80, 40, 20]`
- `num_cells`: `34000`
- `prediction_width`: `148`
- `predictions` shape: `[B, 34000, 148]`

The first `regression_width` channels are LTRB distribution-focal-loss logits.
The remaining `num_classes` channels are unified quality-class logits.

## Sidecar Metadata

The exporter writes a JSON sidecar next to each `.onnx` file with:

- `architecture_version`
- `image_size`
- `num_classes`
- `num_cells`
- `regression_width`
- `reg_max`
- `scale_factors`
- `grid_sizes`
- `p2_head`
- `branch`
- `requires_nms`
- `output_semantics`

Deployment code should validate on first inference that:

- `architecture_version == 8`
- `num_cells == sum(grid ** 2 for grid in grid_sizes)`
- `regression_width == 4 * (reg_max + 1 if reg_max else 1)`
- `prediction_width == regression_width + num_classes`
- ONNX output shape matches `[B, num_cells, prediction_width]`

The sidecar currently does not encode confidence threshold, IoU threshold, max
detections, class names, preprocessing constants, original-image geometry, input
and output names, or dynamic axes. Those values must be supplied by the runtime
from this contract or future expanded metadata.

## Preprocessing Contract

Preprocessing must match `preprocess_detector_image`:

1. Convert the source image to RGB.
2. Compute `scale = min(image_size / original_width, image_size / original_height)`.
3. Resize with bilinear filtering to `round(original_width * scale)` by
   `round(original_height * scale)`, clamped to at least `1` pixel per dimension.
4. Center the resized image on an `image_size x image_size` RGB canvas filled with
   `(114, 114, 114)`.
5. Convert to CHW float tensor in `[0, 1]`.
6. Normalize with `(pixel - 0.5) / 0.5`, yielding the training-time `[-1, 1]`
   scale.
7. Add batch dimension before ONNX inference.

The runtime must retain `original_width`, `original_height`, `image_size`,
`scale`, `left`, and `top` so decoded normalized boxes can be restored to source
pixels.

## Grid And Decode Contract

Prediction cells are ordered by feature level, then row-major within each level.
For every `grid` in `grid_sizes`:

- `row = 0..grid-1`
- `col = 0..grid-1`
- normalized center x is `(col + 0.5) / grid`
- normalized center y is `(row + 0.5) / grid`

For the current 640 px v8 model with `p2_head=true`, grids `[160, 80, 40, 20]`
correspond to input-pixel strides `[4, 8, 16, 32]`. Do not assume a fixed
`[8, 16, 32]` pyramid; derive it from sidecar `grid_sizes`.

Decode raw predictions as:

1. `regression = predictions[..., :regression_width]`
2. `class_logits = predictions[..., regression_width:]`
3. If `reg_max > 0`, reshape regression to `[B, N, 4, reg_max + 1]`, softmax over
   the final bin axis, and compute the expected value against bins
   `0..reg_max`.
4. Divide each expected LTRB distance by that cell's `grid` size to get
   normalized distances.
5. Convert to normalized `xyxy` with
   `[center_x - left, center_y - top, center_x + right, center_y + bottom]`.
6. Clamp normalized boxes to `[0, 1]`.
7. Compute `class_scores = sigmoid(class_logits)`.
8. For each cell, choose `score, label = max(class_scores, dim=-1)`.

`reg_max` is the maximum DFL bin index, not the number of bins. With
`reg_max=16`, there are `17` bins and `4 * 17 = 68` regression channels.

## Branch Postprocessing

The export wrapper exposes exactly one raw branch:

- `branch == "nms-free"` returns the one-to-one end-to-end head.
- `branch == "o2m"` returns the one-to-many dense head.
- `branch == "auto"` resolves to `nms-free` when the checkpoint has a one-to-one
  head, otherwise `o2m`.

Both branches use the same decode contract. Their postprocessing differs after
scores and labels are computed:

- `nms-free`: keep predictions with `score >= confidence_threshold`, then keep
  the top `max_detections` by score. PyTorch defaults are
  `confidence_threshold=0.25` and `max_detections=300`.
- `o2m`: keep predictions with `score >= confidence_threshold`, then run
  class-aware NMS with `iou_threshold=0.45` and `max_detections=300`.

The score threshold is inclusive. Empty inputs must return empty boxes, scores,
and labels without error.

## Box Restoration Contract

Decoded boxes are normalized `xyxy` coordinates in the square letterboxed image.
Map them back to original source pixels as:

- `x = (x * image_size - left) / scale`
- `y = (y * image_size - top) / scale`
- clamp x coordinates to `[0, original_width]`
- clamp y coordinates to `[0, original_height]`

## Runtime Defaults To Preserve

Until these are encoded in sidecar metadata, deployment runtimes should preserve
the PyTorch defaults:

- confidence threshold: `0.25`
- O2M IoU threshold: `0.45`
- max detections: `300`
- letterbox fill: RGB `(114, 114, 114)`
- resize filter: bilinear
- normalization: `(pixel / 255.0 - 0.5) / 0.5`

