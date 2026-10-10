"""Operation-level guards for integrations outside the native framework core."""

from importlib import import_module
from types import ModuleType


class MissingOptionalDependencyError(ImportError):
    """A requested integration is absent, rather than installed but broken."""


def require_optional(module: str, extra: str, *, feature: str | None = None) -> ModuleType:
    """Import a requested integration and explain how to install it if absent.

    Missing transitive dependencies and errors inside an installed integration
    retain their original exception. They must not be reported as an absent
    framework extra or silently disable an explicitly requested operation.
    """
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in {module, module.split(".", 1)[0]}:
            raise
        operation = feature or module
        raise MissingOptionalDependencyError(
            f'{operation} requires {module}. Install with: '
            f'pip install "complexity-framework[{extra}]"'
        ) from exc
