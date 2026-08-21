"""Verify numerical parity between a detector checkpoint and its ONNX export.

Usage:
    python scripts/check_onnx_parity.py CHECKPOINT model.onnx
    python scripts/check_onnx_parity.py CHECKPOINT model.onnx --num-tests 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ["COMPLEXITY_DISABLE_KERNELS"] = "1"

from complexity.generative.detection.exporting import ExportBranch, RawDetectorExport
from complexity.generative.detection.hub import load_detector_checkpoint

DEFAULT_PARITY_TOLERANCE = 1e-4
V8_PARITY_TOLERANCES = {
    "o2m": 2e-3,
    "nms-free": 3.5e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="PyTorch checkpoint directory")
    parser.add_argument("onnx_model", type=Path, help="Exported ONNX model path")
    parser.add_argument(
        "--branch",
        choices=("auto", "nms-free", "o2m"),
        default="auto",
        help="Prediction branch; auto reads the ONNX sidecar when available",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Max allowed absolute difference. Defaults to calibrated v8 branch "
            "thresholds when ONNX metadata is available, otherwise 1e-4."
        ),
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of random inputs to test (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size, including for dynamic-batch exports",
    )
    return parser.parse_args()


def sidecar_metadata(onnx_path: Path) -> dict:
    metadata_path = onnx_path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text())


def branch_from_sidecar(metadata: dict, requested: ExportBranch) -> ExportBranch:
    if requested != "auto":
        return requested
    if "branch" not in metadata:
        return "auto"
    branch = metadata.get("branch", "auto")
    if branch not in {"nms-free", "o2m"}:
        raise ValueError(f"invalid export branch in ONNX metadata: {branch}")
    return branch


def calibrated_parity_tolerance(metadata: dict, branch: ExportBranch) -> float:
    """Return the default raw-logit parity tolerance for an exported model."""

    if metadata.get("architecture_version") == 8 and branch in V8_PARITY_TOLERANCES:
        return V8_PARITY_TOLERANCES[branch]
    return DEFAULT_PARITY_TOLERANCE


def check_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    branch: ExportBranch = "auto",
    tolerance: float | None = None,
    num_tests: int = 5,
    batch_size: int = 1,
) -> bool:
    """Compare PyTorch and ONNX raw outputs on random inputs."""

    if num_tests <= 0 or batch_size <= 0:
        raise ValueError("num_tests and batch_size must be positive")
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        return False

    print(f"Loading PyTorch model: {checkpoint_path}")
    detector = load_detector_checkpoint(checkpoint_path, device="cpu")
    metadata = sidecar_metadata(onnx_path)
    resolved_branch = branch_from_sidecar(metadata, branch)
    effective_tolerance = (
        calibrated_parity_tolerance(metadata, resolved_branch)
        if tolerance is None
        else tolerance
    )
    export_model = RawDetectorExport(detector, resolved_branch).eval()
    image_size = detector.config.image_size
    print(f"Prediction branch: {export_model.branch}")

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    all_passed = True
    for test_index in range(num_tests):
        torch.manual_seed(test_index)
        values = torch.randn(batch_size, 3, image_size, image_size)
        with torch.no_grad():
            pytorch_output = export_model(values).numpy()
        onnx_output = session.run([output_name], {input_name: values.numpy()})[0]

        absolute_difference = np.abs(pytorch_output - onnx_output)
        max_difference = float(absolute_difference.max())
        mean_difference = float(absolute_difference.mean())
        passed = max_difference <= effective_tolerance
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(
            f"  Test {test_index + 1}/{num_tests}: "
            f"max_diff={max_difference:.2e}, mean_diff={mean_difference:.2e} [{status}]"
        )

    summary = "PASSED" if all_passed else "FAILED"
    print(
        f"\nParity {summary}: "
        f"branch={export_model.branch}, tolerance={effective_tolerance}"
    )
    return all_passed


def main() -> None:
    args = parse_args()
    success = check_parity(
        args.checkpoint,
        args.onnx_model,
        branch=args.branch,
        tolerance=args.tolerance,
        num_tests=args.num_tests,
        batch_size=args.batch_size,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
