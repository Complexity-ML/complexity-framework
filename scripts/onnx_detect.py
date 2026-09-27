"""Run TR-Hash Vision v8 ONNX detector inference on one image.

Example:
    python scripts/onnx_detect.py --model tr_hash_v8_o2m.onnx --metadata tr_hash_v8_o2m.json --image sample.jpg --provider cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON sidecar path")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help=(
            "Provider alias or ORT provider name. May be repeated or comma-separated. "
            "Aliases: cpu, cuda, tensorrt."
        ),
    )
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def provider_names(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        values = ["cpu"]

    resolved: list[str] = []
    for value in values:
        for token in value.split(","):
            provider = token.strip()
            if not provider:
                continue
            lowered = provider.lower()
            if lowered == "cpu":
                candidates = ("CPUExecutionProvider",)
            elif lowered == "cuda":
                candidates = ("CUDAExecutionProvider", "CPUExecutionProvider")
            elif lowered == "tensorrt":
                candidates = (
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                )
            else:
                candidates = (provider,)
            for candidate in candidates:
                if candidate not in resolved:
                    resolved.append(candidate)
    return tuple(resolved)


def main() -> None:
    args = parse_args()

    from complexity.deploy.onnx_detector import OnnxDetectorPipeline
    from complexity.deploy.onnx_detector.metadata import load_metadata

    metadata = load_metadata(args.metadata)
    if args.conf_threshold is not None:
        metadata = replace(metadata, confidence_threshold=args.conf_threshold)
    if args.iou_threshold is not None:
        if metadata.branch == "nms-free":
            print(
                "warning: --iou-threshold is ignored for nms-free exports",
                file=sys.stderr,
            )
        metadata = replace(metadata, iou_threshold=args.iou_threshold)

    session = OnnxDetectorPipeline.create_session(
        args.model,
        providers=provider_names(args.provider),
    ).open()
    pipeline = OnnxDetectorPipeline(metadata=metadata, session=session)
    result = pipeline.predict(args.image)
    payload = {
        "provider_used": result.provider_used,
        "branch_type": result.branch_type,
        "timing": result.timing.as_dict(),
        "detections": [
            {
                "box_norm": detection.box_norm,
                "box_pixel": detection.box_pixel,
                "class_id": detection.class_id,
                "score": detection.score,
            }
            for detection in result.detections
        ],
    }
    print(json.dumps(payload, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
