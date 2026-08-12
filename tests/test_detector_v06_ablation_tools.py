from __future__ import annotations

import json
from pathlib import Path

from scripts.collect_detector_v06_ablations import METRICS, collect, write_reports


def write_validation(path: Path, value: float) -> None:
    path.mkdir(parents=True)
    (path / "validation.json").write_text(
        json.dumps({name: value for name in METRICS})
    )


def test_collects_classic_and_nms_free_results(tmp_path: Path) -> None:
    root = tmp_path / "ablations"
    reference = tmp_path / "reference"
    write_validation(reference / "best", 0.5)
    write_validation(reference / "best_nms_free", 0.4)
    write_validation(root / "no-stal" / "best", 0.3)
    write_validation(root / "no-stal" / "best_nms_free", 0.2)

    rows = collect(root, reference)
    assert [row["arm"] for row in rows] == [
        "full",
        "full:nms-free",
        "no-stal",
        "no-stal:nms-free",
    ]

    write_reports(root, rows)
    assert "no-stal:nms-free" in (root / "summary.md").read_text()
    assert (root / "summary.csv").read_text().count("\n") == 5


def test_collects_reference_before_ablation_root_exists(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    write_validation(reference / "best", 0.5)

    rows = collect(tmp_path / "not-created-yet", reference)

    assert [row["arm"] for row in rows] == ["full"]
