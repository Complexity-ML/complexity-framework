# TR-Hash Vision v8 ONNX Deployment

This guide runs exported TR-Hash Vision v8 detector models with ONNX Runtime.
The ONNX file returns raw predictions; the deployment pipeline performs the
same preprocessing, DFL decode, confidence filtering, and branch-specific
postprocessing documented in [the contract](onnx_deploy_contract.md).

## Export Inputs

Each deployed model needs two files:

- the `.onnx` model exported by `scripts/export_onnx.py`;
- the JSON sidecar written next to it by the exporter.

Do not rename one without passing both paths to the CLI.

## CPU Example

```bash
python scripts/onnx_detect.py \
  --model tr_hash_v8_o2m.onnx \
  --metadata tr_hash_v8_o2m.json \
  --image sample.jpg \
  --provider cpu \
  --pretty
```

The output is JSON:

```json
{
  "provider_used": "CPUExecutionProvider",
  "branch_type": "o2m",
  "timing": {
    "preprocess_ms": 0.0,
    "inference_ms": 0.0,
    "postprocess_ms": 0.0
  },
  "detections": []
}
```

Actual timings and detections depend on the image and hardware.

## CUDA Example

```bash
python scripts/onnx_detect.py \
  --model tr_hash_v8_nms_free.onnx \
  --metadata tr_hash_v8_nms_free.json \
  --image sample.jpg \
  --provider cuda \
  --pretty
```

`--provider cuda` requests `CUDAExecutionProvider` first and falls back to
`CPUExecutionProvider`. The JSON `provider_used` field reports what ONNX Runtime
actually selected after session creation.

TensorRT can be requested with:

```bash
python scripts/onnx_detect.py --model model.onnx --metadata model.json --image sample.jpg --provider tensorrt
```

That expands to TensorRT, CUDA, then CPU fallback.

## Branch Behavior

- `o2m` exports run confidence filtering followed by class-aware NMS.
- `nms-free` exports run confidence filtering and top-k score selection only.
- `--iou-threshold` only affects `o2m`; the CLI warns if it is passed for an
  NMS-free export.
- `--conf-threshold` overrides the default confidence threshold for either
  branch.

The default thresholds match the PyTorch detector path:

- confidence threshold: `0.25`;
- O2M IoU threshold: `0.45`;
- max detections: `300`.

## Output Schema

Each detection contains:

- `box_norm`: normalized `xyxy` relative to the square model input;
- `box_pixel`: restored `xyxy` in original source-image pixels;
- `class_id`: integer class index;
- `score`: sigmoid quality-class score.

See [the deployment contract](onnx_deploy_contract.md) for the exact tensor
layout, grid mapping, DFL decode formula, preprocessing, and sidecar validation
rules.

