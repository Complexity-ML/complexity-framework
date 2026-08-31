"""Validate Vision v8 COCO accuracy reports before checkpoint publication."""

from __future__ import annotations

import argparse
import json
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


def check_report(report: Mapping[str, Any], config: Mapping[str, Any]) -> list[GateFailure]:
    failures: list[GateFailure] = []
    branches = _branches(report)
    if not branches:
        failures.append(GateFailure("malformed_report", "report has no branch results"))
        return failures

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
    tolerance: float,
) -> list[GateFailure]:
    """Compare repeated same-seed evaluation reports for deterministic metrics."""

    failures: list[GateFailure] = []
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
        second_metrics = _mapping(second_branches[branch].get("metrics", second_branches[branch]))
        for metric in REQUIRED_METRICS:
            if metric not in first_metrics or metric not in second_metrics:
                continue
            delta = abs(float(first_metrics[metric]) - float(second_metrics[metric]))
            if delta > tolerance:
                failures.append(
                    GateFailure(
                        "determinism",
                        f"{branch} {metric} changed by {delta:.12f}, tolerance {tolerance:.12f}",
                    )
                )
    return failures


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
                f"{branch} evaluated image count mismatch: expected {expected_count}, got {actual_count}",
            )
        )

    for key in ("annotations_sha256", "image_list_sha256"):
        expected_hash = expected_dataset.get(key)
        actual_hash = dataset.get(key)
        if expected_hash is None and not actual_hash:
            failures.append(GateFailure("malformed_report", f"missing dataset hash: {key}"))
        elif expected_hash is not None and actual_hash != expected_hash:
            failures.append(GateFailure("malformed_report", f"dataset hash mismatch: {key}"))

    environment = _mapping(report.get("environment", branch_report.get("environment", {})))
    for key in REQUIRED_ENVIRONMENT_KEYS:
        if key not in environment:
            failures.append(GateFailure("malformed_report", f"missing environment.{key}"))
    if report.get("framework_commit") is None and branch_report.get("framework_commit") is None:
        failures.append(GateFailure("malformed_report", "missing framework_commit"))
    return failures


def _check_metrics(
    branch: str,
    branch_report: Mapping[str, Any],
    branch_config: Mapping[str, Any],
) -> list[GateFailure]:
    failures: list[GateFailure] = []
    metrics = _mapping(branch_report.get("metrics", branch_report))
    for metric in REQUIRED_METRICS:
        if metric not in metrics:
            failures.append(GateFailure("malformed_report", f"{branch} missing metric {metric}"))

    floors = _mapping(branch_config.get("absolute_floors", {}))
    for metric, floor in floors.items():
        value = _metric(metrics, metric)
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
        value = _metric(metrics, metric)
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


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main() -> None:
    args = parse_args()
    report = load_json(args.report)
    config = load_json(args.config)
    failures = check_report(report, config)
    if args.repeat_report is not None:
        tolerance = float(
            _mapping(config.get("determinism", {})).get("metric_tolerance", 0.0)
        )
        failures.extend(
            compare_repeated_reports(report, load_json(args.repeat_report), tolerance)
        )
    if failures:
        for failure in failures:
            print(f"{failure.kind}: {failure.message}")
        raise SystemExit(1)
    print("Vision v8 COCO accuracy report passed")


if __name__ == "__main__":
    main()
