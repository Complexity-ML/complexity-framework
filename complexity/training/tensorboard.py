"""Small, task-agnostic TensorBoard metric writer."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Mapping

LOGGER = logging.getLogger(__name__)


class TensorBoardMetricWriter:
    """Write scalar dictionaries without making TensorBoard a core dependency."""

    def __init__(self, log_dir: str | Path, *, enabled: bool = True) -> None:
        self.writer = None
        self.log_dir = Path(log_dir)
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            LOGGER.warning(
                "TensorBoard logging disabled: install the task extra containing tensorboard"
            )
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
