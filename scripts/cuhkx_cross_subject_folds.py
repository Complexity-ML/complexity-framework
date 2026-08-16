#!/usr/bin/env python3
"""Inspect and validate the fixed CUHK-X cross-subject protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from complexity.generative.sensor_fusion import (
    load_cuhkx_manifest,
    resolve_cross_subject_fold,
    validate_cross_subject_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default=None, help="fold_a/fold_b/fold_c or a/b/c")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--print-users", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold is not None and args.print_users:
        fold = resolve_cross_subject_fold(args.fold)
        print(" ".join(map(str, fold.validation_users)))
        return
    records = load_cuhkx_manifest(args.manifest) if args.manifest is not None else None
    report = validate_cross_subject_folds(records=records)
    if args.fold is not None:
        name = resolve_cross_subject_fold(args.fold).name
        report["folds"] = [item for item in report["folds"] if item["name"] == name]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
