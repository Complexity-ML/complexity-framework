#!/usr/bin/env python3
"""Consolidate 100M-token ablation runs into paper-ready raw measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def collect(label: str, run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.csv"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"incomplete run {label}: {run_dir}")
    config = json.loads(config_path.read_text())
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics: {metrics_path}")
    eval_rows = [
        {"step": int(row["step"]), "eval_loss": value}
        for row in rows
        if (value := finite(row.get("eval_loss", ""))) is not None
    ]
    warmup_step = int(config["args"]["steps"] * 0.1)
    throughput = [
        value
        for row in rows
        if int(row["step"]) > warmup_step
        and (value := finite(row.get("tok_s", ""))) is not None
    ]
    profile_path = run_dir / "checkpoint_profile.json"
    return {
        "label": label,
        "run_dir": str(run_dir),
        "parameters": config["params"],
        "registered_total_tokens": config["total_tokens"],
        "attention_type": config["model_config"]["attention_type"],
        "mlp_type": config["model_config"]["mlp_type"],
        "final_step": int(rows[-1]["step"]),
        "final_train_loss": finite(rows[-1]["train_loss"]),
        "last_eval_loss": eval_rows[-1]["eval_loss"] if eval_rows else None,
        "best_eval_loss": min(row["eval_loss"] for row in eval_rows) if eval_rows else None,
        "median_post_warmup_tokens_per_second": statistics.median(throughput),
        "eval_history": eval_rows,
        "artifacts": {
            "metrics_csv_sha256": sha256(metrics_path),
            "run_config_json_sha256": sha256(config_path),
            "checkpoint_profile_json_sha256": (
                sha256(profile_path) if profile_path.exists() else None
            ),
        },
        "checkpoint_profile": (
            json.loads(profile_path.read_text()) if profile_path.exists() else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=RUN_DIRECTORY; may be repeated",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for specification in args.run:
        label, separator, directory = specification.partition("=")
        if not separator:
            raise ValueError(f"invalid --run value: {specification}")
        records.append(collect(label, Path(directory)))
    result = {"schema_version": 1, "runs": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    with args.output_csv.open("w", newline="") as handle:
        fields = [
            "label", "parameters", "registered_total_tokens", "attention_type",
            "mlp_type", "final_step", "final_train_loss", "last_eval_loss",
            "best_eval_loss", "median_post_warmup_tokens_per_second",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
    print(args.output_json)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
