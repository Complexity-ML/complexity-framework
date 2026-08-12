"""Verify ONNX model produces identical outputs to PyTorch.

Runs both models on the same random inputs and compares outputs
element-wise. Passes if max absolute difference is below tolerance.

Usage:
    python test_onnx_parity.py /path/to/checkpoint tr_hash_detector.onnx
    python test_onnx_parity.py /path/to/checkpoint tr_hash_detector.onnx --num-tests 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ["COMPLEXITY_DISABLE_KERNELS"] = "1"

from complexity.generative.detection.hub import load_detector_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="PyTorch checkpoint directory")
    parser.add_argument("onnx_model", type=Path, help="Exported ONNX model path")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max allowed absolute difference (default: %(default)s)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of random inputs to test (default: %(default)s)",
    )
    return parser.parse_args()


def test_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    tolerance: float = 1e-4,
    num_tests: int = 5,
) -> bool:
    """Compare PyTorch and ONNX outputs on random inputs."""

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        return False

    # Load PyTorch model
    print(f"Loading PyTorch model: {checkpoint_path}")
    pytorch_model = load_detector_checkpoint(checkpoint_path, device="cpu")
    pytorch_model.eval()
    image_size = pytorch_model.config.image_size

    # Load ONNX model
    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    all_passed = True

    for test_index in range(num_tests):
        # Generate random input
        torch.manual_seed(test_index)
        dummy = torch.randn(1, 3, image_size, image_size)

        # PyTorch forward
        with torch.no_grad():
            pytorch_output = pytorch_model.forward_predictions(dummy).numpy()

        # ONNX forward
        onnx_output = session.run(
            ["predictions"],
            {"pixel_values": dummy.numpy()},
        )[0]

        # Compare
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        passed = max_diff < tolerance

        status = "PASS" if passed else "FAIL"
        print(
            f"  Test {test_index + 1}/{num_tests}: "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} [{status}]"
        )

        if not passed:
            all_passed = False

    if all_passed:
        print(f"\nAll {num_tests} tests PASSED (tolerance={tolerance})")
    else:
        print(f"\nSome tests FAILED (tolerance={tolerance})")

    return all_passed


def main() -> None:
    args = parse_args()
    success = test_parity(
        args.checkpoint,
        args.onnx_model,
        tolerance=args.tolerance,
        num_tests=args.num_tests,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
