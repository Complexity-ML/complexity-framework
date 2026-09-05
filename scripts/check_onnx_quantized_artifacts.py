"""Validate Vision v8 quantized ONNX artifact metadata and reports."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_quantization_thresholds(path: Path) -> dict[str, Any]:
    """Load and validate the checked-in quantized artifact gate thresholds."""

    config = _load_json(path)
    if "release_policy" not in config:
        raise ValueError("threshold config missing release_policy")
    precisions = config.get("precisions")
    if not isinstance(precisions, dict) or "fp16" not in precisions or "int8" not in precisions:
        raise ValueError("threshold config must define fp16 and int8 precisions")
    return config


def load_calibration_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the pinned INT8 calibration manifest contract."""

    manifest = _load_json(path)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("calibration manifest missing dataset")
    for key in ("image_ids_sha256", "annotations_sha256"):
        if not _is_sha256(dataset.get(key)):
            raise ValueError(f"calibration manifest dataset.{key} must be a SHA-256")
    if "disjoint_from" not in dataset:
        raise ValueError("calibration manifest dataset.disjoint_from must be declared")

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
    image_ids = manifest.get("image_ids")
    if not isinstance(image_ids, Sequence) or isinstance(image_ids, (str, bytes)):
        raise ValueError("calibration manifest image_ids must be a sequence")
    if not image_ids:
        raise ValueError("calibration manifest image_ids must not be empty")
    actual_digest = image_id_manifest_sha256([int(image_id) for image_id in image_ids])
    expected_digest = str(dataset["image_ids_sha256"])
    if actual_digest != expected_digest:
        raise ValueError("calibration manifest dataset.image_ids_sha256 does not match image_ids")

    images = manifest.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise ValueError("calibration manifest images must be a sequence")
    if not images:
        raise ValueError("calibration manifest images must not be empty")
    manifest["images"] = [
        str(_resolve_manifest_path(path.parent, Path(str(image)))) for image in images
    ]
    return manifest


def materialize_calibration_manifest_images(
    manifest_path: Path,
    *,
    artifact_root: Path,
    output_manifest_path: Path | None = None,
    image_directory_name: str = "calibration_images",
) -> None:
    """Copy calibration images beside an evidence manifest and rewrite paths."""

    manifest = _load_json(manifest_path)
    images = manifest.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise ValueError("calibration manifest images must be a sequence")

    destination_dir = artifact_root / image_directory_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    rewritten: list[str] = []
    for index, image in enumerate(images):
        source = _resolve_manifest_path(manifest_path.parent, Path(str(image)))
        if not source.is_file():
            raise ValueError(f"calibration image does not exist: {source}")
        destination = destination_dir / f"{index:06d}_{source.name}"
        shutil.copyfile(source, destination)
        rewritten.append(destination.relative_to(artifact_root).as_posix())

    manifest["images"] = rewritten
    output_path = output_manifest_path or manifest_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def image_id_manifest_sha256(image_ids: Sequence[int]) -> str:
    """Hash a stable sorted image-ID manifest for calibration/eval pinning."""

    digest = hashlib.sha256()
    for image_id in sorted(map(int, image_ids)):
        digest.update(f"{image_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _resolve_manifest_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def assert_disjoint_image_ids(
    calibration_ids: set[int],
    evaluation_ids: set[int],
) -> None:
    """Fail when INT8 calibration inputs overlap accuracy-gate inputs."""

    overlap = calibration_ids & evaluation_ids
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(f"calibration/evaluation image ID overlap: {preview}")


def evaluation_image_ids_from_report(report: Mapping[str, Any]) -> set[int]:
    """Extract the actual evaluated image IDs from an accuracy report."""

    raw_ids = report.get("evaluation_image_ids")
    if raw_ids is None:
        raw_ids = _mapping(report.get("dataset")).get("image_ids")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError("accuracy report must include evaluation image IDs")
    if not raw_ids:
        raise ValueError("accuracy report evaluation image IDs must not be empty")
    return {int(image_id) for image_id in raw_ids}


def check_accuracy_artifact_bindings(
    report: Mapping[str, Any],
    generated_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> list[str]:
    """Verify accuracy evidence hashes match the artifacts being published."""

    branches = _mapping(report.get("branches"))
    failures: list[str] = []
    for branch, precision_hashes in generated_artifacts.items():
        branch_report = _mapping(branches.get(branch))
        if not branch_report:
            failures.append(f"accuracy report missing branch {branch}")
            continue
        for precision, expected_hashes in precision_hashes.items():
            precision_report = _mapping(branch_report.get(precision))
            if not precision_report:
                failures.append(f"accuracy report missing {branch} {precision}")
                continue
            for field, expected in expected_hashes.items():
                actual = precision_report.get(field)
                if actual != expected:
                    failures.append(
                        f"{branch} {precision} {field} {actual or 'missing'} "
                        f"does not match generated {expected}"
                    )
    return failures


def check_provider_precision_supported(
    provider: str,
    precision: str,
    thresholds: Mapping[str, Any],
) -> None:
    """Fail clearly when a provider/precision pair is not release-supported."""

    providers = thresholds.get("providers", {})
    if not isinstance(providers, Mapping) or provider not in providers:
        raise ValueError(f"{provider} is not configured in quantization release config")
    supported = providers[provider]
    if not isinstance(supported, Sequence) or isinstance(supported, (str, bytes)):
        raise ValueError(f"{provider} precision policy must be a sequence")
    if precision not in supported:
        raise ValueError(f"{provider} does not support {precision} in quantization release config")


def check_unexpected_fp32_nodes(
    dtype_report: Mapping[str, Any],
    allowlist: Sequence[str],
) -> list[str]:
    """Return FP32 nodes not covered by an explicit op-type allowlist."""

    allowed = set(allowlist)
    unexpected: list[str] = []
    fp32_nodes = dtype_report.get("fp32_nodes", [])
    if not isinstance(fp32_nodes, Sequence) or isinstance(fp32_nodes, (str, bytes)):
        raise ValueError("dtype report fp32_nodes must be a sequence")
    for node in fp32_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("dtype report fp32_nodes entries must be objects")
        name = str(node.get("name", ""))
        op_type = str(node.get("op_type", ""))
        if op_type not in allowed:
            unexpected.append(f"{name}:{op_type}")
    return unexpected


def inspect_onnx_node_dtypes(model_path: Path) -> dict[str, Any]:
    """Inventory node-level dtype hints in an ONNX graph.

    ONNX does not assign one dtype directly to every node, so this inspector
    maps typed graph values back to producer nodes and reports nodes producing
    FP32, FP16, or INT8/UINT8 tensors.
    """

    try:
        import onnx
        from onnx import TensorProto
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("ONNX dtype inspection requires the onnx package") from error

    model = onnx.load(str(model_path))
    value_dtypes: dict[str, int] = {}
    for value_info in [
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
        *model.graph.initializer,
    ]:
        name = getattr(value_info, "name", "")
        data_type = None
        if hasattr(value_info, "data_type"):
            data_type = value_info.data_type
        elif getattr(value_info, "type", None) is not None:
            tensor_type = value_info.type.tensor_type
            data_type = tensor_type.elem_type
        if name and data_type:
            value_dtypes[name] = int(data_type)

    fp32_nodes: list[dict[str, str]] = []
    fp16_nodes = 0
    int8_nodes = 0
    for node in model.graph.node:
        output_types = {value_dtypes[name] for name in node.output if name in value_dtypes}
        if TensorProto.FLOAT in output_types:
            fp32_nodes.append({"name": node.name or node.output[0], "op_type": node.op_type})
        if TensorProto.FLOAT16 in output_types:
            fp16_nodes += 1
        if TensorProto.INT8 in output_types or TensorProto.UINT8 in output_types:
            int8_nodes += 1

    return {
        "fp32_nodes": fp32_nodes,
        "fp16_nodes": fp16_nodes,
        "int8_nodes": int8_nodes,
    }


def check_quantized_accuracy_report(
    report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    required_branches: Sequence[str] = (),
    expected_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> list[str]:
    """Compare a quantized candidate COCO report against its FP32 reference."""

    if "branches" in report:
        failures = _check_branch_accuracy_report(
            report,
            thresholds,
            required_branches=required_branches,
        )
        release_policy = _mapping(thresholds.get("release_policy"))
        if release_policy.get("require_artifact_bindings") is True:
            if expected_artifacts is None:
                failures.append("generated artifact bindings are required by release policy")
            else:
                failures.extend(check_accuracy_artifact_bindings(report, expected_artifacts))
        return failures

    reference = _mapping(report.get("reference"))
    candidate = _mapping(report.get("candidate"))
    return _check_accuracy_pair(reference, candidate, thresholds)


def check_quantized_parity_report(
    report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[str]:
    """Gate raw-logit and decoded-output drift against checked-in thresholds."""

    precision = str(report.get("precision", ""))
    branch = str(report.get("branch", ""))
    precision_thresholds = _mapping(_mapping(thresholds.get("precisions")).get(precision))
    if not precision_thresholds:
        return [f"candidate precision {precision} has no quantization thresholds"]

    failures: list[str] = []
    for metric, threshold_name in (
        ("max_raw_logit_abs_error", "max_raw_logit_abs_error"),
        ("max_decoded_box_px_error", "max_decoded_box_px_error"),
        ("max_score_abs_error", "max_score_abs_error"),
    ):
        if metric not in report:
            failures.append(f"missing parity metric {metric}")
            continue
        value = _finite_float(report[metric])
        if value is None:
            failures.append(f"non-finite parity metric {metric}")
            continue
        allowed = float(precision_thresholds[threshold_name])
        if value > allowed:
            failures.append(f"{precision} {branch} {metric} {value:.6f} exceeds {allowed:.6f}")
    return failures


def check_quantized_benchmark_report(
    report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    required_branches: Sequence[str],
) -> list[str]:
    """Validate benchmark evidence covers every release branch and precision."""

    branches = _mapping(report.get("branches"))
    if not branches:
        return ["benchmark report must contain branches"]
    required_precisions = _required_precisions(thresholds)
    required_fields = tuple(_mapping(thresholds.get("benchmark")).get("report", ()))
    failures: list[str] = []
    for branch in required_branches:
        branch_report = _mapping(branches.get(branch))
        if not branch_report:
            failures.append(f"benchmark report missing branch {branch}")
            continue
        for precision in required_precisions:
            precision_report = _mapping(branch_report.get(precision))
            if not precision_report:
                failures.append(f"benchmark report missing {branch} {precision}")
                continue
            for field in required_fields:
                value = _benchmark_field(precision_report, str(field))
                if _finite_float(value) is None:
                    failures.append(f"benchmark report has invalid {branch} {precision} {field}")
    return failures


def _check_branch_accuracy_report(
    report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    required_branches: Sequence[str],
) -> list[str]:
    branches = _mapping(report.get("branches"))
    if not branches:
        return ["quantized COCO report must contain branch comparisons"]
    reference_precision = str(report.get("reference_precision", "fp32"))
    threshold_precisions = set(_mapping(thresholds.get("precisions")))
    candidate_precisions = _candidate_precisions(report, branches, threshold_precisions)
    required_candidates = tuple(
        precision
        for precision in _required_precisions(thresholds)
        if precision != reference_precision and precision in threshold_precisions
    )
    if required_candidates:
        candidate_precisions = required_candidates

    failures: list[str] = []
    if not candidate_precisions:
        return ["candidate precision missing from quantized COCO report"]
    for branch in required_branches:
        if branch not in branches:
            failures.append(f"quantized COCO report missing branch {branch}")
    for branch, branch_report in branches.items():
        branch_data = _mapping(branch_report)
        for candidate_precision in candidate_precisions:
            reference = _mapping(branch_data.get(reference_precision, branch_data.get("reference")))
            candidate = _mapping(branch_data.get(candidate_precision, branch_data.get("candidate")))
            if not reference or not candidate:
                failures.append(
                    f"{branch} missing {reference_precision} or {candidate_precision} metrics"
                )
                continue
            failures.extend(_check_accuracy_pair(reference, candidate, thresholds))
    return failures


def _required_precisions(thresholds: Mapping[str, Any]) -> tuple[str, ...]:
    release_policy = _mapping(thresholds.get("release_policy"))
    raw_precisions = release_policy.get("required_precisions", ())
    if not isinstance(raw_precisions, Sequence) or isinstance(
        raw_precisions,
        (str, bytes),
    ):
        return ()
    return tuple(str(precision) for precision in raw_precisions)


def _benchmark_field(report: Mapping[str, Any], field: str) -> object:
    if field in report:
        return report[field]
    latency = _mapping(report.get("latency"))
    return latency.get(field)


def _candidate_precisions(
    report: Mapping[str, Any],
    branches: Mapping[str, Any],
    threshold_precisions: set[str],
) -> tuple[str, ...]:
    raw_candidate_precisions = report.get("candidate_precisions")
    if isinstance(raw_candidate_precisions, Sequence) and not isinstance(
        raw_candidate_precisions,
        (str, bytes),
    ):
        candidates = {str(precision) for precision in raw_candidate_precisions}
    else:
        candidates = set()
    raw_candidate_precision = report.get("candidate_precision", report.get("precision"))
    if raw_candidate_precision is not None:
        candidates.add(str(raw_candidate_precision))
    for branch_report in branches.values():
        branch_data = _mapping(branch_report)
        candidates.update(str(key) for key in branch_data if str(key) in threshold_precisions)
    return tuple(sorted(candidates))


def _check_accuracy_pair(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[str]:
    reference_branch = str(reference.get("branch", ""))
    candidate_branch = str(candidate.get("branch", ""))
    if reference_branch != candidate_branch:
        return [
            "candidate branch "
            f"{candidate_branch} does not match FP32 reference branch "
            f"{reference_branch}"
        ]

    precision = str(candidate.get("precision", ""))
    precision_thresholds = _mapping(_mapping(thresholds.get("precisions")).get(precision))
    if not precision_thresholds:
        return [f"candidate precision {precision} has no quantization thresholds"]

    reference_metrics = _mapping(reference.get("metrics"))
    candidate_metrics = _mapping(candidate.get("metrics"))
    failures: list[str] = []
    for metric, threshold_name in (
        ("map50_95", "max_map50_95_drop"),
        ("map50", "max_map50_drop"),
    ):
        if metric not in reference_metrics or metric not in candidate_metrics:
            failures.append(f"missing metric {metric} in FP32 or candidate report")
            continue
        reference_value = _finite_float(reference_metrics[metric])
        candidate_value = _finite_float(candidate_metrics[metric])
        if reference_value is None or candidate_value is None:
            failures.append(f"non-finite metric {metric} in FP32 or candidate report")
            continue
        allowed_drop = float(precision_thresholds[threshold_name])
        drop = reference_value - candidate_value
        if drop > allowed_drop:
            failures.append(
                f"{precision} {candidate_branch} {metric} dropped by {drop:.6f}; "
                f"allowed drop {allowed_drop:.6f}"
            )
    return failures


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in string.hexdigits for character in value)
