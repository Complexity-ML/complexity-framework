# Vision v8 ONNX Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible FP16 and INT8 ONNX artifacts for both Vision v8 detector branches and gate them against FP32 accuracy, decoded detections, performance, and release metadata.

**Architecture:** Quantization is implemented as a post-export pipeline layered on top of the existing Vision v8 ONNX export, ONNX detector runtime, COCO evaluation, and release workflow. FP32 remains the reference; FP16 and INT8 artifacts are generated deterministically, validated for provider support and node dtypes, evaluated through the shared ONNX COCO evaluator, and published only after release gates pass.

**Tech Stack:** Python, ONNX, ONNX Runtime quantization tools, ONNX Runtime providers, PyTorch, COCO evaluation helpers, GitHub Actions.

**Spec:** User issue: “The published Vision v8 ONNX artifacts are FP32 only. We need smaller and faster deployment artifacts without trading away decoded detections or COCO accuracy silently.”

## Global Constraints

- Quantization commands must be deterministic from a pinned checkpoint, FP32 ONNX model, framework commit, calibration manifest, and quantization settings.
- Calibration inputs must be pinned and disjoint from the COCO evaluation inputs used by the accuracy gate.
- Unsupported provider/precision combinations must fail clearly instead of falling back silently.
- FP16 conversion must report remaining FP32 nodes and fail unless every remaining FP32 node is covered by a checked-in allowlist.
- INT8 settings must pin calibration method, per-channel/per-tensor mode, symmetric/asymmetric mode, activation type, weight type, and calibration batch size.
- FP32, FP16, and INT8 reports must compare artifact size, latency, throughput, peak memory, raw-logit parity, decoded-output parity, and COCO metrics.
- Tolerances and floors must live in checked-in config, not inside scripts.
- If any required quantized artifact fails its gate, the release blocks; do not publish a partial FP32+FP16-only release unless the release config explicitly marks INT8 optional for that provider.
- ONNX binaries must remain excluded from normal source commits and be published as release assets.

---

## File Structure

- Create `configs/vision_v8_quantization_thresholds.json` for FP16/INT8 raw-logit, decoded-output, COCO, performance, dtype, and provider thresholds.
- Create `configs/vision_v8_quantization_calibration.example.json` for pinned calibration input metadata and quantization settings.
- Create `scripts/quantize_onnx.py` to generate FP16 and INT8 ONNX files plus JSON sidecars.
- Create `scripts/check_onnx_quantized_artifacts.py` to validate metadata, checksums, node dtypes, provider support, calibration/eval disjointness, parity reports, COCO reports, and performance reports.
- Create `scripts/benchmark_onnx_artifacts.py` to measure artifact latency, throughput, and peak memory with stable warm-up and measured iteration counts.
- Modify `scripts/check_onnx_parity.py` only if needed to expose reusable raw-logit and decoded-output comparison helpers.
- Reuse `scripts/evaluate_onnx_coco.py` from the COCO accuracy gate work for all ONNX COCO AP metrics.
- Modify `.github/workflows/detector-export.yml` or add `.github/workflows/vision-v8-onnx-quantization-release.yml` to generate, gate, and upload FP32/FP16/INT8 release assets.
- Create `docs/vision-v8-onnx-quantization.md` for reproduction commands, supported providers, precision limitations, and release policy.
- Add tests under `tests/test_onnx_quantization_*.py`.

---

### Task 1: Quantization Config Contract

**Files:**
- Create: `configs/vision_v8_quantization_thresholds.json`
- Create: `configs/vision_v8_quantization_calibration.example.json`
- Test: `tests/test_onnx_quantization_config.py`

**Interfaces:**
- Produces: `load_quantization_thresholds(path: Path) -> dict[str, Any]`
- Produces: `load_calibration_manifest(path: Path) -> dict[str, Any]`

- [ ] **Step 1: Write config validation tests**

```python
from pathlib import Path

import pytest

from scripts.check_onnx_quantized_artifacts import (
    load_calibration_manifest,
    load_quantization_thresholds,
)


def test_threshold_config_requires_explicit_precision_policy(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text('{"schema_version": 1, "precisions": {"fp16": {}, "int8": {}}}')

    with pytest.raises(ValueError, match="release_policy"):
        load_quantization_thresholds(path)


def test_calibration_manifest_pins_int8_settings(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "dataset": {"name": "coco-2017-train", "image_ids_sha256": "abc"},
          "quantization": {
            "calibration_method": "minmax",
            "per_channel": true,
            "symmetric_activations": false,
            "symmetric_weights": true,
            "activation_type": "quint8",
            "weight_type": "qint8",
            "batch_size": 8
          }
        }
        """
    )

    manifest = load_calibration_manifest(path)

    assert manifest["quantization"]["calibration_method"] == "minmax"
    assert manifest["quantization"]["batch_size"] == 8
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest -q tests/test_onnx_quantization_config.py`

Expected: imports fail because `scripts/check_onnx_quantized_artifacts.py` does not exist yet.

- [ ] **Step 3: Add config files**

Create `configs/vision_v8_quantization_thresholds.json`:

```json
{
  "schema_version": 1,
  "release_policy": {
    "required_precisions": ["fp32", "fp16", "int8"],
    "partial_release": "block",
    "optional_provider_precisions": []
  },
  "precisions": {
    "fp16": {
      "max_raw_logit_abs_error": 0.01,
      "max_decoded_box_px_error": 1.0,
      "max_score_abs_error": 0.01,
      "max_map50_95_drop": 0.005,
      "max_map50_drop": 0.01,
      "unexpected_fp32_nodes": "fail"
    },
    "int8": {
      "max_raw_logit_abs_error": 0.12,
      "max_decoded_box_px_error": 4.0,
      "max_score_abs_error": 0.05,
      "max_map50_95_drop": 0.02,
      "max_map50_drop": 0.03,
      "unexpected_fp32_nodes": "allow"
    }
  },
  "providers": {
    "CPUExecutionProvider": ["fp32", "int8"],
    "CUDAExecutionProvider": ["fp32", "fp16"],
    "TensorrtExecutionProvider": ["fp32", "fp16", "int8"]
  },
  "benchmark": {
    "warmup_iterations": 25,
    "measured_iterations": 100,
    "report": ["median_ms", "mean_ms", "stddev_ms", "p95_ms", "throughput_images_per_second", "peak_memory_mb"]
  }
}
```

Create `configs/vision_v8_quantization_calibration.example.json`:

```json
{
  "schema_version": 1,
  "dataset": {
    "name": "coco-2017-train-calibration",
    "split": "train2017-calibration",
    "image_ids_sha256": "replace-with-calibration-image-id-manifest-sha256",
    "annotations_sha256": "replace-with-annotations-sha256",
    "disjoint_from": "coco-2017-val2017"
  },
  "quantization": {
    "calibration_method": "minmax",
    "per_channel": true,
    "symmetric_activations": false,
    "symmetric_weights": true,
    "activation_type": "quint8",
    "weight_type": "qint8",
    "batch_size": 8
  }
}
```

- [ ] **Step 4: Add minimal config loaders**

Create `scripts/check_onnx_quantized_artifacts.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_quantization_thresholds(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if "release_policy" not in config:
        raise ValueError("threshold config missing release_policy")
    precisions = config.get("precisions")
    if not isinstance(precisions, dict) or "fp16" not in precisions or "int8" not in precisions:
        raise ValueError("threshold config must define fp16 and int8 precisions")
    return config


def load_calibration_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("calibration manifest missing quantization settings")
    required = {
        "calibration_method",
        "per_channel",
        "symmetric_activations",
        "symmetric_weights",
        "activation_type",
        "weight_type",
        "batch_size",
    }
    missing = sorted(required - set(quantization))
    if missing:
        raise ValueError(f"calibration manifest missing settings: {missing}")
    return manifest
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_onnx_quantization_config.py`

Commit:

```bash
git add configs/vision_v8_quantization_thresholds.json configs/vision_v8_quantization_calibration.example.json scripts/check_onnx_quantized_artifacts.py tests/test_onnx_quantization_config.py
git commit -m "Add Vision v8 quantization gate config"
```

---

### Task 2: Calibration and Evaluation Disjointness

**Files:**
- Modify: `scripts/check_onnx_quantized_artifacts.py`
- Test: `tests/test_onnx_quantization_config.py`

**Interfaces:**
- Produces: `assert_disjoint_image_ids(calibration_ids: set[int], evaluation_ids: set[int]) -> None`
- Produces: `image_id_manifest_sha256(image_ids: Sequence[int]) -> str`

- [ ] **Step 1: Write failing overlap tests**

```python
import pytest

from scripts.check_onnx_quantized_artifacts import (
    assert_disjoint_image_ids,
    image_id_manifest_sha256,
)


def test_calibration_and_eval_image_ids_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_image_ids({1, 2, 3}, {3, 4, 5})


def test_image_id_manifest_hash_is_order_stable() -> None:
    assert image_id_manifest_sha256([3, 1, 2]) == image_id_manifest_sha256([1, 2, 3])
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest -q tests/test_onnx_quantization_config.py`

- [ ] **Step 3: Implement helpers**

Add:

```python
import hashlib
from collections.abc import Sequence


def image_id_manifest_sha256(image_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for image_id in sorted(map(int, image_ids)):
        digest.update(f"{image_id}\n".encode("utf-8"))
    return digest.hexdigest()


def assert_disjoint_image_ids(calibration_ids: set[int], evaluation_ids: set[int]) -> None:
    overlap = calibration_ids & evaluation_ids
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(f"calibration/evaluation image ID overlap: {preview}")
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest -q tests/test_onnx_quantization_config.py`

Commit:

```bash
git add scripts/check_onnx_quantized_artifacts.py tests/test_onnx_quantization_config.py
git commit -m "Reject quantization calibration eval leakage"
```

---

### Task 3: FP16 and INT8 Quantization CLI

**Files:**
- Create: `scripts/quantize_onnx.py`
- Test: `tests/test_onnx_quantization_cli.py`

**Interfaces:**
- Produces: `quantize_fp16(input_model: Path, output_model: Path, keep_fp32_op_types: Sequence[str]) -> None`
- Produces: `quantize_int8(input_model: Path, output_model: Path, calibration_manifest: Mapping[str, Any]) -> None`
- Produces: sidecar JSON with precision, source SHA-256, output SHA-256, toolchain versions, and quantization settings.

- [ ] **Step 1: Write failing CLI metadata tests**

```python
import json
from pathlib import Path

from scripts.quantize_onnx import write_quantization_sidecar


def test_quantization_sidecar_binds_artifact_to_source_and_settings(tmp_path: Path) -> None:
    sidecar = tmp_path / "model.fp16.json"

    write_quantization_sidecar(
        sidecar,
        precision="fp16",
        source_model_sha256="source",
        output_model_sha256="output",
        framework_commit="commit",
        checkpoint_revision="checkpoint",
        settings={"keep_fp32_op_types": ["ReduceSum"]},
    )

    data = json.loads(sidecar.read_text())
    assert data["precision"] == "fp16"
    assert data["source_model_sha256"] == "source"
    assert data["output_model_sha256"] == "output"
    assert data["framework_commit"] == "commit"
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest -q tests/test_onnx_quantization_cli.py`

- [ ] **Step 3: Implement CLI skeleton and sidecar writer**

Use `onnxconverter-common` or ONNX Runtime quantization APIs where available. If the dependency is unavailable, fail with:

```text
FP16 quantization requires onnxconverter-common; install the export/quantization extra.
```

For INT8, fail with:

```text
INT8 quantization requires onnxruntime.quantization and a calibration manifest.
```

- [ ] **Step 4: Implement FP16 conversion**

Call the FP16 converter with explicit settings:

```python
keep_fp32_op_types = tuple(settings.get("keep_fp32_op_types", ()))
disable_shape_infer = bool(settings.get("disable_shape_infer", False))
```

Write output ONNX and sidecar.

- [ ] **Step 5: Implement INT8 static quantization**

Use pinned calibration settings:

```python
calibration_method = manifest["quantization"]["calibration_method"]
per_channel = manifest["quantization"]["per_channel"]
activation_type = manifest["quantization"]["activation_type"]
weight_type = manifest["quantization"]["weight_type"]
```

Set calibration reader iteration order from the sorted calibration manifest.

- [ ] **Step 6: Run quantization twice and assert same hash**

Add CLI option:

```bash
python scripts/quantize_onnx.py \
  --fp32-model tr_hash_v8_o2m.onnx \
  --metadata tr_hash_v8_o2m.json \
  --precision fp16 \
  --output artifacts/a.onnx \
  --repeat-output artifacts/b.onnx \
  --require-identical-hash
```

If hashes differ, fail with:

```text
quantization is not deterministic: first SHA-256 ... differs from repeat SHA-256 ...
```

- [ ] **Step 7: Run tests and commit**

Run: `python -m pytest -q tests/test_onnx_quantization_cli.py`

Commit:

```bash
git add scripts/quantize_onnx.py tests/test_onnx_quantization_cli.py
git commit -m "Add deterministic Vision v8 ONNX quantization CLI"
```

---

### Task 4: Node Dtype and Provider Support Validation

**Files:**
- Modify: `scripts/check_onnx_quantized_artifacts.py`
- Test: `tests/test_onnx_quantization_artifact_checks.py`

**Interfaces:**
- Produces: `inspect_onnx_node_dtypes(model_path: Path) -> dict[str, Any]`
- Produces: `check_provider_precision_supported(provider: str, precision: str, thresholds: Mapping[str, Any]) -> None`
- Produces: `check_unexpected_fp32_nodes(dtype_report: Mapping[str, Any], allowlist: Sequence[str]) -> list[str]`

- [ ] **Step 1: Write provider support tests**

```python
import pytest

from scripts.check_onnx_quantized_artifacts import check_provider_precision_supported


def test_unsupported_provider_precision_fails_clearly() -> None:
    thresholds = {"providers": {"CPUExecutionProvider": ["fp32", "int8"]}}

    with pytest.raises(ValueError, match="does not support fp16"):
        check_provider_precision_supported("CPUExecutionProvider", "fp16", thresholds)
```

- [ ] **Step 2: Write FP32 node allowlist tests**

```python
from scripts.check_onnx_quantized_artifacts import check_unexpected_fp32_nodes


def test_unexpected_fp32_nodes_are_reported() -> None:
    report = {"fp32_nodes": [{"name": "Conv_1", "op_type": "Conv"}, {"name": "ReduceSum_1", "op_type": "ReduceSum"}]}

    unexpected = check_unexpected_fp32_nodes(report, allowlist=["ReduceSum"])

    assert unexpected == ["Conv_1:Conv"]
```

- [ ] **Step 3: Implement provider support check**

```python
def check_provider_precision_supported(provider: str, precision: str, thresholds: Mapping[str, Any]) -> None:
    supported = thresholds.get("providers", {}).get(provider)
    if precision not in supported:
        raise ValueError(f"{provider} does not support {precision} in quantization release config")
```

- [ ] **Step 4: Implement dtype allowlist check**

```python
def check_unexpected_fp32_nodes(dtype_report: Mapping[str, Any], allowlist: Sequence[str]) -> list[str]:
    allowed = set(allowlist)
    unexpected = []
    for node in dtype_report.get("fp32_nodes", []):
        if node["op_type"] not in allowed:
            unexpected.append(f"{node['name']}:{node['op_type']}")
    return unexpected
```

- [ ] **Step 5: Implement ONNX dtype inspection**

Use ONNX graph initializers and value info to identify FP32 tensors that remain after FP16 conversion. Report:

```json
{
  "fp32_nodes": [{"name": "node_name", "op_type": "Conv"}],
  "fp16_nodes": 123,
  "int8_nodes": 45
}
```

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest -q tests/test_onnx_quantization_artifact_checks.py`

Commit:

```bash
git add scripts/check_onnx_quantized_artifacts.py tests/test_onnx_quantization_artifact_checks.py
git commit -m "Validate quantized ONNX providers and node dtypes"
```

---

### Task 5: Quantized Parity and COCO Accuracy Gate

**Files:**
- Modify: `scripts/check_onnx_quantized_artifacts.py`
- Modify: `scripts/check_onnx_parity.py` if helper reuse is needed
- Reuse: `scripts/evaluate_onnx_coco.py`
- Test: `tests/test_onnx_quantization_accuracy_gate.py`

**Interfaces:**
- Produces: `check_quantized_accuracy_report(report: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]`
- Consumes: COCO report schema emitted by `scripts/evaluate_onnx_coco.py`

- [ ] **Step 1: Write FP32 comparison tests**

```python
from scripts.check_onnx_quantized_artifacts import check_quantized_accuracy_report


def test_quantized_accuracy_fails_when_map_drop_exceeds_precision_threshold() -> None:
    report = {
        "reference": {"precision": "fp32", "metrics": {"map50_95": 0.2, "map50": 0.32}},
        "candidate": {"precision": "int8", "metrics": {"map50_95": 0.17, "map50": 0.31}},
    }
    thresholds = {"precisions": {"int8": {"max_map50_95_drop": 0.02, "max_map50_drop": 0.03}}}

    failures = check_quantized_accuracy_report(report, thresholds)

    assert any("map50_95" in failure for failure in failures)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest -q tests/test_onnx_quantization_accuracy_gate.py`

- [ ] **Step 3: Implement accuracy gate helper**

Compare candidate metrics against FP32 reference metrics for each branch:

```python
drop = float(reference_metric) - float(candidate_metric)
if drop > allowed_drop:
    failures.append(f"{precision} {branch} {metric} dropped by {drop:.6f}")
```

- [ ] **Step 4: Wire to existing ONNX COCO evaluator**

Do not create a new COCO eval path. The release workflow must call:

```bash
python scripts/evaluate_onnx_coco.py \
  --model "$MODEL" \
  --metadata "$METADATA" \
  --annotations "$COCO_ANNOTATIONS" \
  --images "$COCO_IMAGES" \
  --output "$REPORT_DIR" \
  --provider "$PROVIDER"
```

- [ ] **Step 5: Add decoded-output parity checks**

Reuse the ONNX detector pipeline to compare:

```json
{
  "max_box_pixel_error": 0.7,
  "max_score_abs_error": 0.006,
  "class_id_mismatches": 0
}
```

Fail if the precision-specific threshold is exceeded.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
python -m pytest -q tests/test_onnx_quantization_accuracy_gate.py tests/test_onnx_detector_core.py tests/test_vision_v8_coco_accuracy_gate.py
```

Commit:

```bash
git add scripts/check_onnx_quantized_artifacts.py tests/test_onnx_quantization_accuracy_gate.py
git commit -m "Gate quantized ONNX accuracy against FP32"
```

---

### Task 6: Benchmark Methodology and Reports

**Files:**
- Create: `scripts/benchmark_onnx_artifacts.py`
- Test: `tests/test_onnx_quantization_benchmark.py`

**Interfaces:**
- Produces: `summarize_latency_ms(values: Sequence[float]) -> dict[str, float]`
- Produces: benchmark JSON with median, mean, stddev, p95, throughput, peak memory, provider requested, provider used, warmup count, measured count.

- [ ] **Step 1: Write benchmark summary tests**

```python
from scripts.benchmark_onnx_artifacts import summarize_latency_ms


def test_benchmark_report_uses_distribution_not_single_shot() -> None:
    summary = summarize_latency_ms([10.0, 12.0, 14.0])

    assert summary["median_ms"] == 12.0
    assert summary["mean_ms"] == 12.0
    assert summary["stddev_ms"] > 0.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest -q tests/test_onnx_quantization_benchmark.py`

- [ ] **Step 3: Implement benchmark script**

Use fixed methodology:

```text
warmup_iterations = 25
measured_iterations = 100
batch_size = 1 unless explicitly configured
report median + mean + stddev + p95
benchmark tolerance is separate from accuracy tolerance
```

- [ ] **Step 4: Add peak memory capture**

For CUDA providers, use CUDA memory APIs where available. For CPU, record process RSS before/after and document it as approximate.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_onnx_quantization_benchmark.py`

Commit:

```bash
git add scripts/benchmark_onnx_artifacts.py tests/test_onnx_quantization_benchmark.py
git commit -m "Add quantized ONNX benchmark reports"
```

---

### Task 7: Release Workflow Integration

**Files:**
- Create or modify: `.github/workflows/vision-v8-onnx-quantization-release.yml`
- Modify: `.github/workflows/detector-export.yml` only if the repo prefers one detector workflow
- Test: workflow YAML lint through `ruff` only for Python files and local shell dry-run where possible

**Interfaces:**
- Consumes: `scripts/quantize_onnx.py`
- Consumes: `scripts/check_onnx_quantized_artifacts.py`
- Consumes: `scripts/evaluate_onnx_coco.py`
- Consumes: `scripts/benchmark_onnx_artifacts.py`

- [ ] **Step 1: Add manual/tag-triggered workflow**

Workflow triggers:

```yaml
on:
  workflow_dispatch:
  push:
    tags:
      - "vision-v8-onnx-*"
```

- [ ] **Step 2: Export FP32 artifacts**

Call the existing ONNX export path for:

```text
o2m fp32
nms-free fp32
```

- [ ] **Step 3: Quantize all required artifacts**

Generate:

```text
o2m fp16
o2m int8
nms-free fp16
nms-free int8
```

Run each quantization command twice with `--require-identical-hash`.

- [ ] **Step 4: Validate provider and node dtype policy**

Fail if:

```text
requested provider != actual provider
precision unsupported by provider config
FP16 output contains unexpected FP32 nodes outside allowlist
INT8 metadata does not match pinned calibration settings
```

- [ ] **Step 5: Run parity, COCO, and benchmark gates**

Call:

```bash
python scripts/evaluate_onnx_coco.py ...
python scripts/check_onnx_quantized_artifacts.py ...
python scripts/benchmark_onnx_artifacts.py ...
```

- [ ] **Step 6: Enforce partial-failure policy**

If any required artifact fails, stop the workflow before release upload:

```text
required quantized artifact failed gate; release blocked by release_policy.partial_release=block
```

- [ ] **Step 7: Publish release assets**

Upload:

```text
FP32 ONNX files
FP16 ONNX files
INT8 ONNX files
JSON sidecars
manifest JSON
accuracy reports JSON/Markdown
benchmark reports JSON/Markdown
```

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/vision-v8-onnx-quantization-release.yml
git commit -m "Publish gated quantized Vision v8 ONNX releases"
```

---

### Task 8: Documentation and Final Verification

**Files:**
- Create: `docs/vision-v8-onnx-quantization.md`
- Modify: `docs/index.md`

**Interfaces:**
- Documents exact commands for FP16, INT8, parity, COCO evaluation, benchmark, and release upload.

- [ ] **Step 1: Document beginner-friendly concepts**

Explain:

```text
FP32 = bigger, safest reference
FP16 = smaller/faster float model, may leave some nodes FP32 for stability
INT8 = smallest/fastest candidate, requires calibration data
Calibration = sample inputs used to estimate activation ranges
Provider fallback = ORT did not use requested runtime
Partial FP16 fallback = some graph nodes stayed FP32
```

- [ ] **Step 2: Document reproduction commands**

Include:

```bash
python scripts/quantize_onnx.py ...
python scripts/evaluate_onnx_coco.py ...
python scripts/check_onnx_quantized_artifacts.py ...
python scripts/benchmark_onnx_artifacts.py ...
```

- [ ] **Step 3: Run final local checks**

Run:

```bash
python -m ruff check scripts/quantize_onnx.py scripts/check_onnx_quantized_artifacts.py scripts/benchmark_onnx_artifacts.py tests/test_onnx_quantization_config.py tests/test_onnx_quantization_cli.py tests/test_onnx_quantization_artifact_checks.py tests/test_onnx_quantization_accuracy_gate.py tests/test_onnx_quantization_benchmark.py
python -m pytest -q tests/test_onnx_quantization_config.py tests/test_onnx_quantization_cli.py tests/test_onnx_quantization_artifact_checks.py tests/test_onnx_quantization_accuracy_gate.py tests/test_onnx_quantization_benchmark.py
```

- [ ] **Step 4: Record known untested release requirements**

Before opening PR, state clearly whether these have run:

```text
Full COCO FP32 vs FP16 vs INT8
CUDA provider execution
TensorRT provider execution
GitHub Release upload
Quantize-twice identical artifact hash
```

- [ ] **Step 5: Commit**

```bash
git add docs/vision-v8-onnx-quantization.md docs/index.md
git commit -m "Document Vision v8 ONNX quantization release process"
```

---

## Self-Review

- Spec coverage: FP16 and INT8 quantization, calibration pinning, toolchain/settings metadata, size/latency/throughput/memory reports, raw-logit/decoded/COCO gates, release artifact publishing, checksums, framework commit, checkpoint revision, provider failure policy, and no source-committed ONNX binaries are all mapped to tasks.
- Reviewer gaps covered: quantize-twice hash determinism is Task 3; FP16 partial dtype fallback is Task 4; calibration/eval leakage is Task 2; shared COCO evaluator reuse is Task 5; benchmark noise methodology is Task 6; checked-in thresholds are Task 1; partial release policy is Task 7.
- Type consistency: helper names used by later tasks are introduced before they are consumed.
