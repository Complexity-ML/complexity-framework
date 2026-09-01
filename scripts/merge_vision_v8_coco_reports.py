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
    for report in reports:
        for key in shared_keys:
            if report.get(key) != reports[0].get(key):
                raise ValueError(f"cannot merge reports with different {key}")

        branches = report.get("branches")
        if not isinstance(branches, Mapping):
            raise ValueError("each report must contain a branches object")
        for branch, branch_report in branches.items():
            if branch in merged["branches"]:
                raise ValueError(f"duplicate branch in merged reports: {branch}")
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
            merged["branches"][branch] = merged_branch

    return merged


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
