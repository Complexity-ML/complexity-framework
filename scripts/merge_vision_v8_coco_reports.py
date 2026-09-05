"""Merge per-branch Vision v8 COCO reports into one gated release report."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_tr_hash_coco import _write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {path}")
    return payload


def merge_reports(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")

    merged = deepcopy(dict(reports[0]))
    merged["checkpoint"] = "multiple ONNX branch artifacts"
    merged["model"] = "multiple ONNX branch artifacts"
    merged["metadata"] = "multiple ONNX branch sidecars"
    merged["branches"] = {}
    shared_keys = ("backend", "framework_commit", "dataset", "protocol")
    precision_reports = [_report_precision(report) for report in reports]
    nest_by_precision = any(precision is not None for precision in precision_reports)
    if nest_by_precision and not all(precision_reports):
        raise ValueError("cannot merge mixed precision and non-precision reports")
    for report in reports:
        for key in shared_keys:
            if report.get(key) != reports[0].get(key):
                raise ValueError(f"cannot merge reports with different {key}")

        branches = report.get("branches")
        if not isinstance(branches, Mapping):
            raise ValueError("each report must contain a branches object")
        for branch, branch_report in branches.items():
            if not isinstance(branch_report, Mapping):
                raise ValueError(f"branch report must be an object: {branch}")
            merged_branch = deepcopy(dict(branch_report))
            for key in (
                "checkpoint",
                "checkpoint_sha256",
                "model",
                "metadata",
                "metadata_sha256",
            ):
                merged_branch.setdefault(key, report.get(key))
            precision = _report_precision(report)
            if nest_by_precision:
                assert precision is not None
                merged_branch.setdefault("precision", precision)
                branch_precisions = merged["branches"].setdefault(branch, {})
                if precision in branch_precisions:
                    raise ValueError(f"duplicate precision in merged reports: {branch} {precision}")
                branch_precisions[precision] = merged_branch
            else:
                if branch in merged["branches"]:
                    raise ValueError(f"duplicate branch in merged reports: {branch}")
                merged["branches"][branch] = merged_branch

    if nest_by_precision:
        precisions = tuple(dict.fromkeys(str(precision) for precision in precision_reports))
        if "fp32" in precisions:
            merged["reference_precision"] = "fp32"
            merged["candidate_precisions"] = [
                precision for precision in precisions if precision != "fp32"
            ]
        else:
            merged["candidate_precisions"] = list(precisions)

    return merged


def _report_precision(report: Mapping[str, Any]) -> str | None:
    precision = report.get("precision")
    if precision is None:
        protocol = report.get("protocol")
        if isinstance(protocol, Mapping):
            precision = protocol.get("precision")
    return str(precision) if precision is not None else None


def main() -> None:
    args = parse_args()
    report = merge_reports([load_json(path) for path in args.reports])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(report, args.output / "evaluation.md")


if __name__ == "__main__":
    main()
