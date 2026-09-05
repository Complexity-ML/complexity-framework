# Vision v8 ONNX Quantization

Vision v8 ONNX quantization produces smaller deployment artifacts while keeping
FP32 as the accuracy and behavior reference. Quantized artifacts are not
accepted only because they run; they must pass raw-output, decoded-detection,
COCO accuracy, provider, dtype, benchmark, checksum, and release metadata gates.

## Precision Contract

- `fp32`: the exported reference model and source of truth.
- `fp16`: smaller floating-point model for CUDA/TensorRT-style deployment.
- `int8`: post-training quantized model calibrated from pinned inputs.

Each branch is quantized separately:

- `o2m`: decode, confidence filtering, and class-aware NMS.
- `nms-free`: decode and confidence filtering only; NMS must not run.

Unsupported provider and precision pairs fail explicitly using
`configs/vision_v8_quantization_thresholds.json`. Provider fallback is separate
from FP16 partial fallback: the release check must verify both that ONNX Runtime
used the requested provider and that unexpected graph nodes did not remain FP32.

## Calibration Contract

INT8 calibration is pinned by
`configs/vision_v8_quantization_calibration.json`. The manifest records the
dataset, image-ID manifest hash, annotation hash, calibration method,
per-channel/per-tensor mode, symmetric/asymmetric choices, activation type,
weight type, and batch size. The ORT symmetry options and batch size are passed
into quantization directly; unsupported settings are not recorded in release
metadata.
Placeholder hashes are rejected by `load_calibration_manifest`; before a release
run, replace them with the approved calibration subset and real SHA-256 values.

Calibration images must be disjoint from the COCO evaluation images used by the
accuracy gate. This prevents tuning the INT8 ranges on the same images used to
claim final AP.

## Reproduction Commands

Create an FP16 artifact:

```bash
python scripts/quantize_onnx.py \
  --fp32-model artifacts/onnx/tr_hash_v8_o2m.onnx \
  --metadata artifacts/onnx/tr_hash_v8_o2m.json \
  --precision fp16 \
  --output artifacts/onnx/tr_hash_v8_o2m_fp16.onnx \
  --repeat-output artifacts/onnx/tr_hash_v8_o2m_fp16_repeat.onnx \
  --require-identical-hash \
  --checkpoint-revision AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO-SFT@REVISION
```

By default the CLI writes two JSON files for a quantized artifact:
`tr_hash_v8_o2m_fp16.json` is a detector metadata copy used by the inference
pipeline, while `tr_hash_v8_o2m_fp16.quantization.json` is the quantization
provenance sidecar used by release verification.

Create an INT8 artifact:

```bash
python scripts/quantize_onnx.py \
  --fp32-model artifacts/onnx/tr_hash_v8_o2m.onnx \
  --metadata artifacts/onnx/tr_hash_v8_o2m.json \
  --precision int8 \
  --calibration-manifest configs/vision_v8_quantization_calibration.json \
  --output artifacts/onnx/tr_hash_v8_o2m_int8.onnx \
  --repeat-output artifacts/onnx/tr_hash_v8_o2m_int8_repeat.onnx \
  --require-identical-hash \
  --checkpoint-revision AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO-SFT@REVISION
```

Evaluate a quantized ONNX artifact on COCO:

```bash
python scripts/evaluate_onnx_coco.py \
  --model artifacts/onnx/tr_hash_v8_o2m_fp16.onnx \
  --metadata artifacts/onnx/tr_hash_v8_o2m_fp16.json \
  --annotations artifacts/COCO/annotations/instances_val2017.json \
  --images artifacts/COCO/images/val2017 \
  --output artifacts/vision_v8_quantized_eval/o2m_fp16 \
  --branch o2m-nms \
  --provider cuda
```

Before publishing, merge the FP32 reference and quantized branch reports into
`artifacts/vision_v8_quantized_eval/accuracy.json` and
`artifacts/vision_v8_quantized_eval/accuracy.md`. The release builder validates
that JSON against `configs/vision_v8_quantization_thresholds.json`; missing or
regressed reports block publication.

Benchmark an artifact:

```bash
python scripts/benchmark_onnx_artifacts.py \
  --model artifacts/onnx/tr_hash_v8_o2m_fp16.onnx \
  --metadata artifacts/onnx/tr_hash_v8_o2m_fp16.json \
  --output artifacts/vision_v8_quantized_eval/o2m_fp16/benchmark.json \
  --provider cuda \
  --warmup-iterations 25 \
  --measured-iterations 100
```

## Release Policy

The quantized release workflow blocks the release when any required artifact
fails. Do not publish a partial release, such as FP32 plus FP16 only, unless the
checked-in release policy explicitly marks the failed precision/provider as
optional.

Release assets must include:

- FP32, FP16, and INT8 ONNX files for both O2M and NMS-free branches.
- Detector metadata JSON sidecars and quantization provenance JSON sidecars.
- FP32-vs-quantized parity reports for raw logits, decoded boxes, scores, and
  class/count stability.
- Accuracy reports in JSON and Markdown.
- Benchmark reports in JSON and Markdown.
- A manifest binding every artifact to framework commit, checkpoint revision,
  SHA-256, file size, provider, precision, and quantization settings.

ONNX binaries remain excluded from normal source commits.
