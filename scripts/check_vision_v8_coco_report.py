"""Validate Vision v8 COCO accuracy reports before checkpoint publication."""

from __future__ import annotations

import argparse
import json
import math
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REQUIRED_METRICS = (
    "map50_95",
    "map50",
    "map75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_100",
)
REQUIRED_ENVIRONMENT_KEYS = (
    "python",
    "os",
    "torch",
    "onnxruntime",
)
REQUIRED_DATASET_HASHES = ("annotations_sha256", "image_list_sha256")
REQUIRED_PROTOCOL_KEYS = ("seed", "release_eligible")
VALID_BACKENDS = {"pytorch", "onnx"}


@dataclass(frozen=True)
class GateFailure:
    """One explicit accuracy-gate failure."""

    kind: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--repeat-report",
        type=Path,
        help="optional second same-seed report to compare for deterministic metrics",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vision_v8_coco_accuracy_gate.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"malformed JSON file: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def check_report(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> list[GateFailure]:
    failures: list[GateFailure] = _check_config(config)
    branches = _branches(report)
    if not branches:
        failures.append(GateFailure("malformed_report", "report has no branch results"))
        return failures

    required_branches = _required_branches(config)
    for branch in required_branches:
        if branch not in branches:
            failures.append(
                GateFailure("malformed_report", f"missing required branch: {branch}")
            )

    for branch, branch_report in branches.items():
        branch_config = _mapping(config.get("branches", {})).get(branch)
        if not isinstance(branch_config, Mapping):
            failures.append(
                GateFailure("malformed_report", f"branch {branch!r} has no gate config")
            )
            continue
        failures.extend(_check_metadata(report, branch, branch_report, config))
        failures.extend(_check_metrics(branch, branch_report, branch_config))
    return failures


def compare_repeated_reports(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    tolerance: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[GateFailure]:
    """Compare repeated same-seed evaluation reports for deterministic metrics."""

    failures: list[GateFailure] = []
    tolerance = (
        determinism_tolerance(first, config)
        if config is not None
        else float(0.0 if tolerance is None else tolerance)
    )
    first_branches = _branches(first)
    second_branches = _branches(second)
    if first_branches.keys() != second_branches.keys():
        return [
            GateFailure(
                "determinism",
                "repeated reports contain different branch sets",
            )
        ]
    for branch, first_branch in first_branches.items():
        first_metrics = _mapping(first_branch.get("metrics", first_branch))
        second_metrics = _mapping(
            second_branches[branch].get("metrics", second_branches[branch])
        )
        for metric in REQUIRED_METRICS:
            if metric not in first_metrics or metric not in second_metrics:
                continue
            first_value = _metric(
                first_metrics,
                metric,
                branch=branch,
                failures=failures,
                kind="determinism",
            )
            second_value = _metric(
                second_metrics,
                metric,
                branch=branch,
                failures=failures,
                kind="determinism",
            )
            if first_value is None or second_value is None:
                continue
            delta = abs(first_value - second_value)
            if delta > tolerance:
                failures.append(
                    GateFailure(
                        "determinism",
                        (
                            f"{branch} {metric} changed by {delta:.12f}, "
                            f"tolerance {tolerance:.12f}"
                        ),
                    )
                )
    return failures


def determinism_tolerance(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> float:
    """Select the same-seed metric tolerance for the report backend/provider."""

    determinism = _mapping(config.get("determinism", {}))
    default = _float_config(determinism.get("metric_tolerance", 0.0), default=0.0)
    environment = _mapping(report.get("environment", {}))

    if report.get("backend") == "onnx":
        provider = str(environment.get("actual_provider", "")).lower()
        if "tensorrt" in provider:
            return _float_config(
                determinism.get("tensorrt_metric_tolerance", default), default=default
            )
        if "cuda" in provider:
            return _float_config(
                determinism.get("cuda_metric_tolerance", default), default=default
            )
        if "cpu" in provider:
            return _float_config(
                determinism.get("cpu_metric_tolerance", default), default=default
            )

    if environment.get("cuda_available") is True:
        return _float_config(
            determinism.get("cuda_metric_tolerance", default), default=default
        )
    return _float_config(
        determinism.get("cpu_metric_tolerance", default), default=default
    )


def _branches(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if isinstance(report.get("branches"), Mapping):
        return {
            str(branch): _mapping(branch_report)
            for branch, branch_report in _mapping(report["branches"]).items()
        }
    branch = report.get("branch")
    if branch is not None:
        return {str(branch): report}
    return {}


def _check_metadata(
    report: Mapping[str, Any],
    branch: str,
    branch_report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[GateFailure]:
    failures: list[GateFailure] = []
    dataset = _mapping(report.get("dataset", branch_report.get("dataset", {})))
    expected_dataset = _mapping(config.get("dataset", {}))
    if dataset.get("name") != expected_dataset.get("name"):
        failures.append(GateFailure("malformed_report", "dataset name mismatch"))
    if dataset.get("split") != expected_dataset.get("split"):
        failures.append(GateFailure("malformed_report", "dataset split mismatch"))

    expected_count = expected_dataset.get("required_image_count")
    actual_count = dataset.get("evaluated_images", dataset.get("images"))
    if expected_count is not None and actual_count != expected_count:
        failures.append(
            GateFailure(
                "malformed_report",
                (
                    f"{branch} evaluated image count mismatch: "
                    f"expected {expected_count}, got {actual_count}"
                ),
            )
        )

    for key in REQUIRED_DATASET_HASHES:
        expected_hash = expected_dataset.get(key)
        actual_hash = dataset.get(key)
        if not _valid_sha256(actual_hash):
            failures.append(
                GateFailure(
                    "malformed_report",
                    f"missing or invalid dataset hash: {key}",
                )
            )
        elif actual_hash != expected_hash:
            failures.append(
                GateFailure("malformed_report", f"dataset hash mismatch: {key}")
            )

    environment = _mapping(
        report.get("environment", branch_report.get("environment", {}))
    )
    for key in REQUIRED_ENVIRONMENT_KEYS:
        if key not in environment:
            failures.append(
                GateFailure("malformed_report", f"missing environment.{key}")
            )
    backend = report.get("backend", branch_report.get("backend"))
    if backend not in VALID_BACKENDS:
        failures.append(GateFailure("malformed_report", "missing or invalid backend"))
    if backend == "onnx":
        failures.extend(_check_onnx_provider(environment))
        for key in ("model", "metadata"):
            if _field(report, branch_report, key) is None:
                failures.append(GateFailure("malformed_report", f"missing {key}"))
        metadata_hash = _field(report, branch_report, "metadata_sha256")
        if not _valid_sha256(metadata_hash):
            failures.append(
                GateFailure("malformed_report", "missing or invalid metadata_sha256")
            )

    protocol = _mapping(report.get("protocol", branch_report.get("protocol", {})))
    for key in REQUIRED_PROTOCOL_KEYS:
        if key not in protocol:
            failures.append(GateFailure("malformed_report", f"missing protocol.{key}"))
    expected_seed = _mapping(config.get("determinism", {})).get("seed")
    if expected_seed is not None and protocol.get("seed") != expected_seed:
        failures.append(
            GateFailure(
                "malformed_report",
                (
                    f"protocol.seed mismatch: expected {expected_seed}, "
                    f"got {protocol.get('seed')}"
                ),
            )
        )
    if protocol.get("release_eligible") is not True:
        failures.append(
            GateFailure("malformed_report", "protocol.release_eligible must be true")
        )

    if (
        report.get("framework_commit") is None
        and branch_report.get("framework_commit") is None
    ):
        failures.append(GateFailure("malformed_report", "missing framework_commit"))
    checkpoint_hash = _field(report, branch_report, "checkpoint_sha256")
    if _field(report, branch_report, "checkpoint") is None:
        failures.append(GateFailure("malformed_report", "missing checkpoint"))
    if not _valid_sha256(checkpoint_hash):
        failures.append(
            GateFailure("malformed_report", "missing or invalid checkpoint_sha256")
        )
    return failures


def _check_metrics(
    branch: str,
    branch_report: Mapping[str, Any],
    branch_config: Mapping[str, Any],
) -> list[GateFailure]:
    failures: list[GateFailure] = []
    metrics = _mapping(branch_report.get("metrics", branch_report))
    metric_values: dict[str, float] = {}
    for metric in REQUIRED_METRICS:
        if metric not in metrics:
            failures.append(
                GateFailure("malformed_report", f"{branch} missing metric {metric}")
            )
            continue
        value = _metric(metrics, metric, branch=branch, failures=failures)
        if value is not None:
            metric_values[metric] = value

    floors = _mapping(branch_config.get("absolute_floors", {}))
    for metric, floor in floors.items():
        value = metric_values.get(metric)
        if value is None:
            continue
        if value < float(floor):
            failures.append(
                GateFailure(
                    "absolute_floor",
                    f"{branch} {metric}={value:.6f} below floor {float(floor):.6f}",
                )
            )

    baselines = _mapping(branch_config.get("baseline_metrics", {}))
    max_regressions = _mapping(branch_config.get("max_regressions", {}))
    for metric, baseline in baselines.items():
        value = metric_values.get(metric)
        if value is None:
            continue
        allowed_drop = float(max_regressions.get(metric, 0.0))
        minimum = float(baseline) - allowed_drop
        if value < minimum:
            failures.append(
                GateFailure(
                    "baseline_regression",
                    (
                        f"{branch} {metric}={value:.6f} regressed beyond baseline "
                        f"{float(baseline):.6f} minus allowed drop {allowed_drop:.6f}"
                    ),
                )
            )
    return failures


def _metric(
    metrics: Mapping[str, Any],
    name: str,
    *,
    branch: str,
    failures: list[GateFailure],
    kind: str = "malformed_report",
) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        failures.append(GateFailure(kind, f"{branch} {name} is not numeric"))
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        failures.append(GateFailure(kind, f"{branch} {name} is not numeric"))
        return None
    if not math.isfinite(numeric):
        failures.append(GateFailure(kind, f"{branch} {name} is non-finite"))
        return None
    if numeric < 0.0 or numeric > 1.0:
        failures.append(
            GateFailure(kind, f"{branch} {name}={numeric:.6f} outside [0, 1]")
        )
        return None
    return numeric


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _field(
    report: Mapping[str, Any], branch_report: Mapping[str, Any], key: str
) -> object:
    return branch_report.get(key, report.get(key))


def _check_config(config: Mapping[str, Any]) -> list[GateFailure]:
    failures: list[GateFailure] = []
    dataset = _mapping(config.get("dataset", {}))
    for key in REQUIRED_DATASET_HASHES:
        if not _valid_sha256(dataset.get(key)):
            failures.append(
                GateFailure(
                    "config",
                    f"dataset.{key} must be pinned to a 64-character SHA-256",
                )
            )

    branches = _mapping(config.get("branches", {}))
    if not branches:
        failures.append(
            GateFailure("config", "branches must configure at least one branch")
        )
    for branch in _required_branches(config):
        if branch not in branches:
            failures.append(
                GateFailure(
                    "config",
                    f"required branch {branch!r} has no branch config",
                )
            )
    if "seed" not in _mapping(config.get("determinism", {})):
        failures.append(GateFailure("config", "determinism.seed must be pinned"))
    return failures


def _required_branches(config: Mapping[str, Any]) -> list[str]:
    configured = list(_mapping(config.get("branches", {})).keys())
    required = config.get("required_branches", configured)
    if not isinstance(required, list):
        return configured
    return [str(branch) for branch in required]


def _check_onnx_provider(environment: Mapping[str, Any]) -> list[GateFailure]:
    failures: list[GateFailure] = []
    requested = environment.get("requested_provider")
    actual = environment.get("actual_provider")
    if not isinstance(requested, list) or not all(
        isinstance(provider, str) and provider for provider in requested
    ):
        failures.append(
            GateFailure(
                "malformed_report",
                "missing or invalid environment.requested_provider",
            )
        )
    if not isinstance(actual, str) or not actual:
        failures.append(
            GateFailure(
                "malformed_report",
                "missing or invalid environment.actual_provider",
            )
        )
    elif isinstance(requested, list) and requested and actual != requested[0]:
        failures.append(
            GateFailure(
                "malformed_report",
                (
                    f"environment.actual_provider {actual!r} does not match "
                    f"requested provider {requested[0]!r}"
                ),
            )
        )
    return failures


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in string.hexdigits for character in value)


def _float_config(value: object, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def main() -> None:
    args = parse_args()
    report = load_json(args.report)
    config = load_json(args.config)
    failures = check_report(report, config)
    if args.repeat_report is not None:
        repeat_report = load_json(args.repeat_report)
        failures.extend(check_report(repeat_report, config))
        failures.extend(
            compare_repeated_reports(report, repeat_report, config=config)
        )
    if failures:
        for failure in failures:
            print(f"{failure.kind}: {failure.message}")
        raise SystemExit(1)
    print("Vision v8 COCO accuracy report passed")


if __name__ == "__main__":
    main()
