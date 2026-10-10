"""Compatibility launcher; use complexity.vision.detection.onnx_export."""
import sys
from importlib import import_module

_module = import_module("complexity.vision.detection.onnx_export")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
