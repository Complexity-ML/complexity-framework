"""Small, task-agnostic TensorBoard metric writer."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

from .._optional import MissingOptionalDependencyError, require_optional

LOGGER = logging.getLogger(__name__)

CSV_SCALARS = {
    "train_loss": "Loss/train",
    "train_ppl": "PPL/train",
    "matched_eval_loss": "Loss/eval",
    "matched_eval_ppl": "PPL/eval",
    "lr": "LR",
    "expert_lr": "LR/expert",
    "tok_s": "Throughput/tokens_per_second",
    "supervised_tokens": "Tokens/supervised_per_batch",
    "matched_eval_tokens": "Tokens/eval",
    "natural_eval_loss": "Loss/natural_eval",
    "natural_eval_ppl": "PPL/natural_eval",
    "natural_eval_tokens": "Tokens/natural_eval",
    "loss": "Loss/train",
    "gradient_norm": "Gradient/norm",
    "seconds": "Time/update_seconds",
}


def finite_float(value: str | int | float | None) -> float | None:
    """Ignore empty, partial and non-finite scalar cells in a live CSV."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mirror_once(
    csv_path: str | Path,
    writer: Any,
    seen: set[tuple[int, str]],
    *,
    scalar_tags: Mapping[str, str] | None = None,
) -> int:
    """Mirror unseen CSV/JSONL scalars; callers retain ``seen`` between polls.

    A writer implements add_scalar/flush, so this helper does not import Torch
    or require TensorBoard. Partially written rows can be retried on a later
    poll because invalid or empty cells are never marked as seen.
    """
    written = 0
    tags = CSV_SCALARS if scalar_tags is None else scalar_tags
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            rows = _jsonl_rows(handle)
        else:
            rows = csv.DictReader(handle)
            if rows.fieldnames is None or "step" not in rows.fieldnames:
                raise ValueError("Metrics CSV requires a step column")
        for row in rows:
            raw_step = row.get("step")
            if isinstance(raw_step, bool) or (
                isinstance(raw_step, float) and not raw_step.is_integer()
            ):
                continue
            try:
                step = int(raw_step if raw_step is not None else "")
            except (TypeError, ValueError):
                continue
            if step < 0:
                continue
            for column, tag in tags.items():
                value = finite_float(row.get(column))
                key = (step, tag)
                if value is None or key in seen:
                    continue
                writer.add_scalar(tag, value, step)
                seen.add(key)
                written += 1
    if written:
        writer.flush()
    return written


def _jsonl_rows(handle):
    for number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if not line.endswith("\n"):
                # The active writer has not committed its final line yet.
                break
            raise ValueError(f"Malformed metrics JSON at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Metrics JSON must contain an object at line {number}")
        yield row


def export_recorded_metrics(metrics_path: str | Path, log_dir: str | Path) -> int:
    """Export recorded CSV/JSONL scalars once, requiring optional TensorBoard."""
    metrics_path = Path(metrics_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")
    require_optional("tensorboard", "tracking", feature="Metrics export")
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        return mirror_once(metrics_path, writer, set())


class TensorBoardMetricWriter:
    """Write scalar dictionaries without making TensorBoard a core dependency."""

    def __init__(self, log_dir: str | Path, *, enabled: bool = True) -> None:
        self.writer = None
        self.log_dir = Path(log_dir)
        if not enabled:
            return
        try:
            require_optional("tensorboard", "tracking", feature="TensorBoard logging")
            from torch.utils.tensorboard import SummaryWriter
        except MissingOptionalDependencyError as exc:
            LOGGER.warning("TensorBoard logging disabled: %s", exc)
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def add_scalars(
        self,
        prefix: str,
        metrics: Mapping[str, object],
        step: int,
    ) -> None:
        if self.writer is None:
            return
        clean_prefix = prefix.strip("/")
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            scalar = float(value)
            if not math.isfinite(scalar):
                continue
            self.writer.add_scalar(f"{clean_prefix}/{name}", scalar, step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
