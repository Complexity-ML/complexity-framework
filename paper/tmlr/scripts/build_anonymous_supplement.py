#!/usr/bin/env python3
"""Build the anonymous supplement from the standalone mini-framework only."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper/tmlr"
STANDALONE = PAPER / "standalone_artifact"
ARTIFACTS = PAPER / "artifacts/h200-controlled-replications"
OUT = PAPER / "submission"
STAGE = OUT / "anonymous-supplement"
ZIP = OUT / "anonymous-supplement.zip"

RUNS = (
    "h200-review-gqa-seed42",
    "h200-review-wrv-lexical-off-seed42",
    "h200-review-gqa-seed43",
    "h200-review-wrv-lexical-off-seed43",
    "h200-review-gqa-seed44",
    "h200-review-wrv-lexical-off-seed44",
    "h200-review-wrv-optimized-seed42",
    "h200-review-wrv-lexical-off-no-rmsnorm-seed42",
)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def scrub_json(source: Path, destination: Path) -> None:
    data = json.loads(source.read_text())

    def scrub(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if key != "checkpoint_sha256"
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            if value.startswith(("/root/", "/Users/")):
                return "[REDACTED]"
            return value.replace(
                "github.com/Complexity-ML/complexity-framework",
                "[REDACTED_REPOSITORY]",
            )
        return value

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(scrub(data), indent=2, sort_keys=True) + "\n")


def copy_standalone() -> None:
    excluded_parts = {"__pycache__", ".pytest_cache", ".venv", "build", "dist"}
    for source in STANDALONE.rglob("*"):
        relative = source.relative_to(STANDALONE)
        if not source.is_file():
            continue
        if any(part in excluded_parts or part.endswith(".egg-info") for part in relative.parts):
            continue
        if source.suffix in {".pyc", ".pyo"}:
            continue
        copy_file(source, STAGE / relative)


def copy_evidence() -> None:
    scrub_json(ARTIFACTS / "summary.json", STAGE / "evidence/summary.json")
    copy_file(
        ARTIFACTS / "exploratory_ablation_summary.json",
        STAGE / "evidence/exploratory_ablation_summary.json",
    )
    (STAGE / "evidence/DATASET_SHA256SUMS").write_text(
        "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871  "
        "sample/10BT/000_00000.parquet\n"
    )
    for run in RUNS:
        source = ARTIFACTS / run
        destination = STAGE / "evidence" / run
        copy_file(source / "metrics.csv", destination / "metrics.csv")
        scrub_json(source / "run_config.json", destination / "run_config.json")


def copy_paper() -> None:
    for name in (
        "main.pdf",
        "main_fr.pdf",
        "main.tex",
        "main_fr.tex",
        "references.bib",
        "tmlr.sty",
        "tmlr.bst",
        "math_commands.tex",
    ):
        copy_file(PAPER / name, STAGE / "paper" / name)
    for directory in ("sections", "sections_fr", "generated"):
        for source in (PAPER / directory).glob("*.tex"):
            copy_file(source, STAGE / "paper" / directory / source.name)
    copy_file(
        PAPER / "scripts/generate_controlled_tables.py",
        STAGE / "paper/scripts/generate_controlled_tables.py",
    )
    scrub_json(
        ARTIFACTS / "summary.json",
        STAGE / "paper/artifacts/h200-controlled-replications/summary.json",
    )


def main() -> None:
    shutil.rmtree(STAGE, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        ZIP.unlink()
    copy_standalone()
    copy_evidence()
    copy_paper()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(STAGE.rglob("*")):
            if source.is_file():
                archive.write(source, Path(STAGE.name) / source.relative_to(STAGE))


if __name__ == "__main__":
    main()