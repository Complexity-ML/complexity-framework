"""Validate Vision v8 quantized ONNX artifact metadata and reports."""

from __future__ import annotations

import hashlib
import json
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
        "num_threads",
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
        raise ValueError(
            "calibration manifest dataset.image_ids_sha256 does not match image_ids"
        )

    images = manifest.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise ValueError("calibration manifest images must be a sequence")
    if not images:
        raise ValueError("calibration manifest images must not be empty")
    return manifest


def image_id_manifest_sha256(image_ids: Sequence[int]) -> str:
    """Hash a stable sorted image-ID manifest for calibration/eval pinning."""

    digest = hashlib.sha256()
    for image_id in sorted(map(int, image_ids)):
        digest.update(f"{image_id}\n".encode("utf-8"))
    return digest.hexdigest()


def assert_disjoint_image_ids(
    calibration_ids: set[int],
    evaluation_ids: set[int],
) -> None:
    """Fail when INT8 calibration inputs overlap accuracy-gate inputs."""

    overlap = calibration_ids & evaluation_ids
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(f"calibration/evaluation image ID overlap: {preview}")


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
) -> list[str]:
    """Compare a quantized candidate COCO report against its FP32 reference."""

    reference = _mapping(report.get("reference"))
    candidate = _mapping(report.get("candidate"))
    reference_branch = str(reference.get("branch", ""))
    candidate_branch = str(candidate.get("branch", ""))
    if reference_branch != candidate_branch:
        return [
            "candidate branch "
            f"{candidate_branch} does not match FP32 reference branch {reference_branch}"
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
        allowed_drop = float(precision_thresholds[threshold_name])
        drop = float(reference_metrics[metric]) - float(candidate_metrics[metric])
        if drop > allowed_drop:
            failures.append(
                f"{precision} {candidate_branch} {metric} dropped by {drop:.6f}; "
                f"allowed drop {allowed_drop:.6f}"
            )
    return failures


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in string.hexdigits for character in value)
