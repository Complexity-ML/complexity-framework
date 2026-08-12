"""Export a TR-Hash Vision detector to ONNX format.

Exports the backbone + detection head (forward_predictions). Post-processing
(decode + NMS) stays in Python — this is the standard approach for detection
models (same as Ultralytics YOLO, DETR, etc.).

Usage:
    python export_onnx.py /path/to/checkpoint --output model.onnx
    python export_onnx.py /path/to/checkpoint --output model.onnx --dynamic-batch
    python export_onnx.py /path/to/checkpoint --output model.onnx --simplify
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

# Force PyTorch-only backend (no Triton/CUDA kernels — they can't be traced)
os.environ["COMPLEXITY_DISABLE_KERNELS"] = "1"

from complexity.generative.detection.config import TRHashDetectorConfig
from complexity.generative.detection.hub import load_detector_checkpoint


class DetectorForExport(torch.nn.Module):
    """Thin wrapper that exposes only forward_predictions for tracing."""

    def __init__(self, detector: torch.nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.detector.forward_predictions(pixel_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint directory with config.json + model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tr_hash_detector.onnx"),
        help="Output ONNX file path (default: %(default)s)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: %(default)s)",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow variable batch size at inference",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnx-simplifier on the exported model (requires onnxsim)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the exported model loads correctly",
    )
    return parser.parse_args()


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset_version: int = 17,
    dynamic_batch: bool = False,
    simplify: bool = False,
    check: bool = False,
) -> Path:
    """Export a TR-Hash detector checkpoint to ONNX."""

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_detector_checkpoint(checkpoint_path, device="cpu")
    config = model.config
    print(
        f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"{config.image_size}px, {config.num_classes} classes"
    )

    # Wrap for export
    export_model = DetectorForExport(model)
    export_model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)

    # Verify forward works before export
    with torch.no_grad():
        reference_output = export_model(dummy_input)
    print(f"Forward OK: output shape {reference_output.shape}")

    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        }

    # Export
    print(f"Exporting to ONNX (opset {opset_version})...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["pixel_values"],
        output_names=["predictions"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {output_path} ({file_size_mb:.1f} MB)")

    # Verify
    if check or simplify:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")

    # Simplify
    if simplify:
        try:
            import onnxsim

            print("Simplifying...")
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(output_path))
                new_size = output_path.stat().st_size / (1024 * 1024)
                print(f"Simplified: {output_path} ({new_size:.1f} MB)")
            else:
                print("Warning: simplification failed, keeping original")
        except ImportError:
            print("Warning: onnxsim not installed, skipping simplification")

    # Export metadata alongside for inference
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "image_size": config.image_size,
        "num_classes": config.num_classes,
        "num_cells": config.num_cells,
        "regression_width": config.regression_width,
        "reg_max": config.reg_max,
        "scale_factors": list(config.scale_factors),
        "grid_sizes": list(config.grid_sizes),
        "p2_head": config.p2_head,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Metadata: {metadata_path}")

    return output_path


def main() -> None:
    args = parse_args()
    export_onnx(
        args.checkpoint,
        args.output,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
        simplify=args.simplify,
        check=args.check,
    )


if __name__ == "__main__":
    main()
