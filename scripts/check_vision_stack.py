#!/usr/bin/env python3
"""Verify the backend-aware TR-HASH Vision dependency stack."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata

from packaging.requirements import Requirement

CHECKS = (
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "filelock>=3.16.1",
    "numpy>=1.23.0",
    "matplotlib>=3.3.0",
    "opencv-python>=4.7.0,!=4.13.0.90",
    "Pillow>=10.0.0",
    "PyYAML>=5.3.1",
    "requests>=2.23.0",
    "psutil>=5.8.0",
    "polars>=0.20.0",
    "albumentations>=1.4.6",
    "faster-coco-eval>=1.6.7",
    "pycocotools>=2.0.7",
    "onnxslim>=0.1.82",
)


def check_requirement(raw: str) -> tuple[bool, str]:
    requirement = Requirement(raw)
    try:
        version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        return False, "missing"
    return version in requirement.specifier, version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero on failure")
    args = parser.parse_args()

    failures = 0
    print(f"platform={platform.platform()} python={platform.python_version()}")
    checks = list(CHECKS)
    if platform.system() == "Darwin" and sys.version_info >= (3, 13):
        checks.append("onnx>=1.20.0")
    elif platform.system() == "Darwin":
        checks.append("onnx>=1.12.0,<1.18.0")
    else:
        checks.append("onnx>=1.12.0")
    checks.append("onnxruntime>=1.20.0" if sys.version_info >= (3, 11) else "onnxruntime<1.20.0")
    if platform.system() == "Linux":
        checks.append("nvidia-ml-py>=12.0.0")
    for raw in checks:
        valid, installed = check_requirement(raw)
        failures += int(not valid)
        print(f"{'OK' if valid else 'FAIL':4} {raw:42} installed={installed}")

    if args.strict and failures:
        raise SystemExit(f"vision dependency check failed: {failures} requirement(s)")


if __name__ == "__main__":
    main()
