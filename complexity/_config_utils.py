"""Shared type checks for authored dataclass configurations."""

import math
import types
from dataclasses import fields
from typing import Any, Union, get_args, get_origin


def validate_field_types(config_type, data):
    """Reject values that silently change meaning in JSON/YAML authoring."""
    def matches(value, annotation):
        if annotation is Any:
            return True
        origin = get_origin(annotation)
        if origin in (Union, types.UnionType):
            return any(matches(value, member) for member in get_args(annotation))
        if annotation is int:
            return type(value) is int
        if annotation is float:
            return type(value) in (int, float) and math.isfinite(value)
        if annotation is bool:
            return type(value) is bool
        if annotation is type(None):
            return value is None
        if origin is not None:
            return isinstance(value, origin)
        return isinstance(value, annotation)

    for field in fields(config_type):
        if field.name in data and not matches(data[field.name], field.type):
            raise ValueError(f"{field.name} has an invalid type or non-finite value: {data[field.name]!r}")
