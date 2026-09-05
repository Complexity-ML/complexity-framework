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
The default release contract runs FP32 and INT8 on CPU, but FP16 on CUDA because
the CPU ONNX Runtime build does not support executing FP16 detector graphs.

## Calibration Contract

INT8 calibration is pinned by
the release evidence artifact. `configs/vision_v8_quantization_calibration.example.json`
documents the expected shape. The manifest records the
dataset, image-ID manifest hash, annotation hash, calibration method,
per-channel/per-tensor mode, symmetric/asymmetric choices, activation type,
weight type, and batch size. The ORT symmetry options and batch size are passed
into quantization directly; unsupported settings are not recorded in release
metadata.
The default release export uses a fixed batch of 1, so the default calibration
manifest must also use `batch_size: 1`. If the export is changed to dynamic
batching, the calibration batch size can be raised in the same release contract.
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
  --calibration-manifest artifacts/vision_v8_quantized_eval/calibration.json \
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
regressed reports block publication. The report must include the evaluated image
IDs so the release gate can prove the calibration set and evaluation set are
actually disjoint.

The merge report is precision-nested by branch. For example, each branch must
contain `fp32`, `fp16`, and `int8` entries so the gate can compare every
quantized candidate against the FP32 reference:

```bash
python scripts/merge_vision_v8_coco_reports.py \
  artifacts/vision_v8_coco_eval/run_a_o2m/evaluation.json \
  artifacts/vision_v8_coco_eval/run_a_nms_free/evaluation.json \
  artifacts/vision_v8_quantized_eval/o2m_fp16/evaluation.json \
  artifacts/vision_v8_quantized_eval/nms_free_fp16/evaluation.json \
  artifacts/vision_v8_quantized_eval/o2m_int8/evaluation.json \
  artifacts/vision_v8_quantized_eval/nms_free_int8/evaluation.json \
  --output artifacts/vision_v8_quantized_eval
cp artifacts/vision_v8_quantized_eval/evaluation.json \
  artifacts/vision_v8_quantized_eval/accuracy.json
cp artifacts/vision_v8_quantized_eval/evaluation.md \
  artifacts/vision_v8_quantized_eval/accuracy.md
```

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

The release build also generates `quantized_benchmarks.json` and
`quantized_benchmarks.md` for every FP32, FP16, and INT8 branch artifact using
the benchmark methodology from the checked-in threshold config.

The GitHub release workflow downloads a prior Actions artifact named
`vision-v8-quantized-release-inputs`. The manual Vision v8 COCO accuracy
workflow produces that artifact when `backend=onnx`, `calibration_manifest` is
set, and FP32/FP16/INT8 model plus metadata paths are provided for both
branches. The artifact must provide `calibration.json`, `calibration_images/`,
`accuracy.json`, and `accuracy.md` under
`artifacts/vision_v8_quantized_eval/` before `build_onnx_release.py` runs. The
COCO workflow copies the pinned calibration images into `calibration_images/`
and rewrites the manifest paths so the fresh release runner can open them.
Pass the source run as workflow input `evidence_run_id`, or set repository
variable `ONNX_RELEASE_EVIDENCE_RUN_ID` for tag-triggered releases.
Run the ONNX release workflow on a CUDA-capable self-hosted runner so the FP16
parity and benchmark gates use `onnxruntime-gpu` instead of CPU fallback.

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
