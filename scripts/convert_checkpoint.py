"""Checkpoint conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def detensor(value):
    """Convert DTensor-like values to local tensors inside nested structures."""
    if hasattr(value, "to_local"):
        return value.to_local()
    if isinstance(value, Mapping):
        return type(value)((k, detensor(v)) for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(detensor(v) for v in value)
    if isinstance(value, list):
        return [detensor(v) for v in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(detensor(v) for v in value)
    return value
