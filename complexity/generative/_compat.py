"""Lazy import aliases for visual modules moved to ``complexity.vision``.

Aliases return the canonical module object. This keeps old pickle paths,
class identities and monkeypatches working without executing a second copy of
the implementation. Old ``python -m`` launchers delegate to the new module.
"""

import importlib
import importlib.abc
import importlib.util
import sys

VISION_AREAS = ("detection", "image", "video", "vision_language", "vision_tasks")


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.original_spec = module.__spec__
        self.original_loader = module.__loader__
        return module

    def exec_module(self, module):
        module.__spec__ = self.original_spec
        module.__loader__ = self.original_loader

    def get_code(self, fullname):
        source = (
            "from runpy import run_module\n"
            f"run_module({self.target!r}, run_name='__main__')\n"
        )
        return compile(source, f"<compatibility: {fullname}>", "exec")


class _VisionAliases(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for area in VISION_AREAS:
            prefix = f"complexity.generative.{area}"
            if fullname == prefix or fullname.startswith(prefix + "."):
                canonical = "complexity.vision." + fullname.removeprefix("complexity.generative.")
                canonical_spec = importlib.util.find_spec(canonical)
                if canonical_spec is None:
                    return None
                return importlib.util.spec_from_loader(
                    fullname, _AliasLoader(canonical),
                    origin=canonical_spec.origin,
                    is_package=canonical_spec.submodule_search_locations is not None,
                )
        return None


def install_aliases():
    if not any(isinstance(finder, _VisionAliases) for finder in sys.meta_path):
        sys.meta_path.insert(0, _VisionAliases())
