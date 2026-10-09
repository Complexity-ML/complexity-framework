"""Raw detector wrappers shared by ONNX and runtime parity tools."""

from __future__ import annotations

from typing import Literal

import torch

from .model import TRHashObjectDetector

ExportBranch = Literal["auto", "nms-free", "o2m"]
ResolvedExportBranch = Literal["nms-free", "o2m"]


def resolve_export_branch(
    detector: TRHashObjectDetector,
    branch: ExportBranch,
) -> ResolvedExportBranch:
    """Resolve ``auto`` to the checkpoint's production inference branch."""

    if branch == "auto":
        return "nms-free" if detector.one_to_one_head is not None else "o2m"
    if branch == "nms-free" and detector.one_to_one_head is None:
        raise ValueError("NMS-free export requires a checkpoint with end_to_end=True")
    if branch not in {"nms-free", "o2m"}:
        raise ValueError(f"unsupported detector export branch: {branch}")
    return branch


class RawDetectorExport(torch.nn.Module):
    """Expose one raw prediction branch without decode or post-processing."""

    def __init__(
        self,
        detector: TRHashObjectDetector,
        branch: ExportBranch = "auto",
    ) -> None:
        super().__init__()
        self.detector = detector
        self.branch = resolve_export_branch(detector, branch)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.branch == "o2m":
            return self.detector.forward_predictions(pixel_values)
        _, one_to_one = self.detector.forward_branches(pixel_values)
        if one_to_one is None:  # Guard eager execution; branch validity is checked at init.
            raise RuntimeError("NMS-free prediction head is unavailable")
        return one_to_one
