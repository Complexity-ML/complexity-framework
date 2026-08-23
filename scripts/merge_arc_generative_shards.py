#!/usr/bin/env python3
"""Merge deterministic ARC generative shards into one auditable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval_arc_generative import summarize


def merge(shards: list[Path], output: Path, expected_examples: int) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    rows: dict[tuple[str, int], dict] = {}
    for report, path in zip(reports, shards, strict=True):
        traces = path.with_suffix(".jsonl")
        if not traces.is_file():
            raise FileNotFoundError(traces)
        for line in traces.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["task"], int(row["doc_id"]))
            if key in rows:
                raise ValueError(f"duplicate ARC example {key} in {path}")
            rows[key] = row
    if len(rows) != expected_examples:
        raise ValueError(f"expected {expected_examples} examples, found {len(rows)}")
    ordered = sorted(rows.values(), key=lambda row: (row["task"], int(row["doc_id"])))
    merged = {
        key: value
        for key, value in reports[0].items()
        if key not in {"combined", "benchmarks", "elapsed_seconds", "shard", "traces"}
    }
    merged["shards"] = len(shards)
    merged["combined"] = summarize(ordered)
    merged["benchmarks"] = {
        task: summarize([row for row in ordered if row["task"] == task])
        for task in ("arc_easy", "arc_challenge")
    }
    merged["elapsed_seconds"] = max(float(report["elapsed_seconds"]) for report in reports)
    traces_output = output.with_suffix(".jsonl")
    traces_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    merged["traces"] = str(traces_output.resolve())
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.shard, args.output, args.expected_examples), indent=2))


if __name__ == "__main__":
    main()
