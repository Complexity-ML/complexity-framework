"""Collect best validation metrics from TR-Hash Vision v6 ablation arms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = (
    "map50",
    "map50_95",
    "ap_small",
    "ap_medium",
    "ap_large",
    "best_f1",
    "best_confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional full-model checkpoint directory to include as 'full'",
    )
    return parser.parse_args()


def load_metrics(checkpoint: Path) -> dict[str, float] | None:
    validation = checkpoint / "validation.json"
    if not validation.is_file():
        return None
    values = json.loads(validation.read_text())
    return {name: float(values[name]) for name in METRICS}


def collect(root: Path, reference: Path | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = []
    if reference is not None:
        candidates.append(("full", reference / "best"))
    if root.is_dir():
        candidates.extend(
            (path.name, path / "best")
            for path in sorted(root.iterdir())
            if path.is_dir()
        )
    for arm, checkpoint in candidates:
        metrics = load_metrics(checkpoint)
        if metrics is not None:
            rows.append({"arm": arm, "checkpoint": str(checkpoint), **metrics})
        nms_free = checkpoint.parent / "best_nms_free"
        if nms_free.is_dir():
            metrics = load_metrics(nms_free)
            if metrics is not None:
                rows.append(
                    {"arm": f"{arm}:nms-free", "checkpoint": str(nms_free), **metrics}
                )
    return rows


def write_reports(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    columns = ("arm", "checkpoint", *METRICS)
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# TR-Hash Vision v6 ablations",
        "",
        "| Arm | mAP50 | mAP50-95 | AP small | AP medium | AP large | Best F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['ap_small']:.4f} | {row['ap_medium']:.4f} | "
            f"{row['ap_large']:.4f} | {row['best_f1']:.4f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = collect(args.root, args.reference)
    write_reports(args.root, rows)
    print(f"collected {len(rows)} result rows in {args.root}")


if __name__ == "__main__":
    main()
