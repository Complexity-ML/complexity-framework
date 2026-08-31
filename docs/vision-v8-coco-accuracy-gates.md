# Vision v8 COCO Accuracy Gates

Vision v8 checkpoint publication must be gated by an end-to-end COCO accuracy
report, not only raw-logit ONNX parity. The gate validates that the checkpoint
is still useful as a detector and that the report was produced from pinned,
reproducible inputs.

## Evaluation Contract

Release reports target COCO 2017 `val2017` with all 5,000 validation images.
Reports must record both the annotation JSON SHA-256 and a sorted image-list
manifest SHA-256. The dataset name, split, image count, and hashes are checked
before any metric threshold is trusted.

Both branches are evaluated:

- `o2m-nms`: decode, confidence prefilter, class-aware NMS.
- `nms-free`: decode, confidence prefilter, top detections without NMS.

Both backends must use the shared deployment preprocessing and decoded-output
contract:

- RGB conversion;
- aspect-preserving letterbox with `(114, 114, 114)` fill;
- `[-1, 1]` normalization;
- DFL LTRB box decode;
- original-pixel box restoration.

## Determinism

Evaluation commands must seed `random`, `numpy`, and `torch`, set deterministic
PyTorch flags where supported, use a stable sorted image iteration order, and
fix ONNX Runtime intra-op and inter-op thread counts for CPU evaluation. CI
fixtures should run the same evaluation twice and diff the JSON metrics. CPU
reports should be identical within `1e-12`; CUDA and TensorRT may use wider
documented tolerances because provider kernels can change floating-point
accumulation order.

## Report Metadata

JSON reports must include:

- framework commit;
- checkpoint revision or artifact hash;
- backend (`pytorch` or `onnx`);
- requested and actual provider for ONNX Runtime;
- Python, OS, PyTorch, ONNX Runtime, CUDA, driver, and TensorRT versions when
  available;
- evaluated image count;
- confidence threshold, NMS IoU threshold, and max detections;
- AP, AP50, AP75, APs, APm, APl, AR100, precision, and recall.

## Gate Policy

The gate uses `configs/vision_v8_coco_accuracy_gate.json` and fails reports for
two separate metric reasons:

- absolute floor failure: the metric is below the minimum acceptable value;
- baseline regression failure: the metric dropped more than the configured
  delta from the known-good baseline.

Reports also fail if required metadata or dataset hashes are missing. Gate
configuration must be changed explicitly in source review; CI must not relax
tolerances or thresholds at runtime.

The `Vision v8 COCO accuracy` workflow runs lightweight gate tests on pull
requests. Its manual full-COCO job runs the selected evaluator twice with the
same seed, rejects malformed/regressed/non-deterministic reports, uploads the
JSON and Markdown reports as workflow artifacts, and can attach those reports to
an existing GitHub Release when `release_tag` is provided.

## Reproduction Commands

Native PyTorch full COCO evaluation:

```bash
python scripts/evaluate_tr_hash_coco.py \
  models/TR-HASH-Vision-v8-2M-COCO-SFT \
  --annotations artifacts/COCO/annotations/instances_val2017.json \
  --images artifacts/COCO/images/val2017 \
  --output artifacts/vision_v8_coco_eval/pytorch \
  --branch both \
  --device cuda \
  --precision bf16
```

ONNX Runtime full COCO evaluation for one exported branch:

```bash
python scripts/evaluate_onnx_coco.py \
  --model artifacts/onnx/tr_hash_v8_o2m.onnx \
  --metadata artifacts/onnx/tr_hash_v8_o2m.json \
  --annotations artifacts/COCO/annotations/instances_val2017.json \
  --images artifacts/COCO/images/val2017 \
  --output artifacts/vision_v8_coco_eval/onnx_o2m \
  --branch o2m-nms \
  --provider cuda
```

Each ONNX sidecar describes one branch. To evaluate both ONNX branches, run the
command once for the O2M export and once for the NMS-free export, then gate both
reports.

Report gate:

```bash
python scripts/check_vision_v8_coco_report.py \
  artifacts/vision_v8_coco_eval/pytorch/evaluation.json \
  --config configs/vision_v8_coco_accuracy_gate.json
```

The ONNX Runtime evaluator emits the same report schema with `backend=onnx`,
`requested_provider`, and `actual_provider` populated from ONNX Runtime after
session creation.
