"""Compatibility launcher; use complexity.vision.image.generation."""
from importlib import import_module
import sys

_module = import_module("complexity.vision.image.generation")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
