"""Compatibility shim for experimental alternative architectures."""

import warnings

warnings.warn(
    "complexity.core.architectures is experimental; import from "
    "complexity.experimental.architectures for new code.",
    DeprecationWarning,
    stacklevel=2,
)

from complexity.experimental.architectures import *  # noqa: F401,F403
