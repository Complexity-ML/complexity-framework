"""Canonical visual operations used by the CLI and compatibility launchers."""

import sys
from importlib import import_module

WORKFLOWS = {
    "train": {
        "detector": "complexity.vision.detection.training",
        "image": "complexity.vision.image.training",
        "image-edit": "complexity.vision.image.edit_training",
        "image-text": "complexity.vision.vision_language.training",
    },
    "pretrain": {"backbone": "complexity.vision.vision_language.pretraining"},
    "evaluate": {
        "detector": "complexity.vision.detection.evaluation",
        "onnx": "complexity.vision.detection.onnx_evaluation",
    },
    "export": {
        "detector": "complexity.vision.detection.onnx_export",
        "tensorrt": "complexity.vision.detection.tensorrt_export",
    },
    "check": {
        "coco": "complexity.vision.detection.accuracy_gate",
        "onnx-parity": "complexity.vision.detection.onnx_parity",
        "quantized-parity": "complexity.vision.detection.quantized_parity",
    },
    "merge": {"coco": "complexity.vision.detection.report_merging"},
    "infer": {"onnx": "complexity.vision.detection.onnx_inference"},
    "benchmark": {"onnx": "complexity.vision.detection.onnx_benchmark"},
    "quantize": {"onnx": "complexity.vision.detection.onnx_quantization"},
    "release": {"onnx": "complexity.vision.detection.onnx_release"},
    "generate": {"image": "complexity.vision.image.generation"},
    "edit": {"image": "complexity.vision.image.edit_inference"},
    "serve": {"detector": "complexity.vision.detection.service"},
}


def run_workflow(operation: str, task: str, arguments: list[str]) -> None:
    """Delegate to the existing runtime, restoring argv even on failure.

    No runtime is imported until an operation is explicitly selected. Argument
    parsing, validation and distributed behavior remain owned by that runtime.
    """
    try:
        module_name = WORKFLOWS[operation][task]
    except KeyError as exc:
        available = ", ".join(WORKFLOWS.get(operation, {}))
        raise ValueError(f"Unknown vision {operation} task {task!r}. Choose: {available}") from exc
    previous_argv = sys.argv
    sys.argv = [f"complexity vision {operation} {task}", *arguments]
    try:
        import_module(module_name).main()
    finally:
        sys.argv = previous_argv
