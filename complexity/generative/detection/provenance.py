"""Checkpoint provenance contract for native TR-Hash detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import json


PROVENANCE_FORMAT_VERSION = 1
NATIVE_DETECTOR_IMPLEMENTATION = (
    "complexity.generative.detection.TRHashObjectDetector"
)
NATIVE_COCO_DATASET = "coco-2017"


def read_detector_provenance(checkpoint: Path | str) -> dict[str, object]:
    """Read checkpoint provenance, failing closed when it is absent or malformed."""

    path = Path(checkpoint) / "provenance.json"
    if not path.is_file():
        raise ValueError(f"checkpoint is missing provenance.json: {path.parent}")
    try:
        provenance = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid checkpoint provenance: {path}") from error
    if not isinstance(provenance, dict):
        raise ValueError(f"checkpoint provenance must be a JSON object: {path}")
    return provenance


def validate_native_random_init_provenance(
    provenance: Mapping[str, object], *, dataset: str
) -> None:
    """Reject checkpoints that do not descend from native random initialization."""

    if provenance.get("format_version") != PROVENANCE_FORMAT_VERSION:
        raise ValueError("unsupported detector provenance format")
    if provenance.get("implementation") != NATIVE_DETECTOR_IMPLEMENTATION:
        raise ValueError("checkpoint was not produced by the native TR-Hash detector")
    if provenance.get("initialization") != "random":
        raise ValueError("checkpoint does not originate from random initialization")
    if provenance.get("external_checkpoint") is not None:
        raise ValueError("native checkpoint provenance references external weights")
    if provenance.get("dataset") != dataset:
        raise ValueError(
            f"checkpoint provenance dataset must be {dataset!r}, got "
            f"{provenance.get('dataset')!r}"
        )
