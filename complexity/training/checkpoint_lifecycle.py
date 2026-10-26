"""Explicit cleanup of numbered research checkpoints after final export."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_NUMBERED = re.compile(r"(?:token_pack_\d+|step_\d+|interrupted_\d+|final_\d+|best_\d+)")


def cleanup_numbered_checkpoints(
    checkpoint_dir: str | Path, *, dry_run: bool = False
) -> tuple[list[str], int]:
    """Preserve final/assets/logs; reject missing or unreadable SafeTensors.

    Export readability permits storage cleanup, not a claim about model
    quality. ``dry_run=True`` returns the same inventory without deleting it.
    Every candidate is inspected before the first directory is removed.
    """
    from safetensors import safe_open

    root = Path(checkpoint_dir).resolve()
    final = root / "final" / "model.safetensors"
    if (
        final.parent.is_symlink()
        or final.is_symlink()
        or not final.is_file()
        or not final.stat().st_size
    ):
        raise RuntimeError(f"refusing cleanup: final export is missing or unsafe: {final}")
    try:
        with safe_open(final, framework="pt", device="cpu") as handle:
            if not handle.keys():
                raise ValueError("empty tensor set")
            for name in handle.keys():
                handle.get_slice(name).get_shape()
    except Exception as exc:
        raise RuntimeError(f"refusing cleanup: final export is unreadable: {final}") from exc
    candidates, reclaimed = [], 0
    for child in sorted(root.iterdir()):
        if _NUMBERED.fullmatch(child.name) is None:
            continue
        if child.is_symlink():
            raise RuntimeError(f"refusing cleanup of a symlink checkpoint: {child}")
        if child.is_dir():
            reclaimed += sum(path.stat().st_size for path in child.rglob("*") if path.is_file())
            candidates.append(child)
    if not dry_run:
        for child in candidates:
            shutil.rmtree(child)
    return [child.name for child in candidates], reclaimed
