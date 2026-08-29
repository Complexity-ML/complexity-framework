#!/usr/bin/env python3
"""Derive an exact one-pass lexical-refinement plan from a pretraining plan."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from complexity.training import validate_refinement_plan


def build_refinement_plan(pretrain_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return one clean pass over the pretraining plan's exact unique core."""

    phases = pretrain_plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("pretraining plan must contain a phases list")
    unique_core = [phase for phase in phases if phase.get("name") == "unique_core"]
    if len(unique_core) != 1:
        raise ValueError("pretraining plan must contain exactly one unique_core phase")

    core = copy.deepcopy(unique_core[0])
    core["passes"] = 1
    source_names = tuple(core.get("sources", ()))
    if not source_names:
        raise ValueError("pretraining unique_core must contain sources")

    unique_tokens = int(pretrain_plan["unique_tokens"])
    refinement = {
        key: copy.deepcopy(value)
        for key, value in pretrain_plan.items()
        if key not in {"phases", "trained_tokens", "source_passes"}
    }
    refinement.update(
        {
            "trained_tokens": unique_tokens,
            "source_passes": {name: 1 for name in source_names},
            "phases": [core],
        }
    )
    validate_refinement_plan(refinement, pretrain_plan)
    return refinement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-plan", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Path(args.pretrain_plan)
    pretrain_plan = json.loads(source.read_text(encoding="utf-8"))
    refinement_plan = build_refinement_plan(pretrain_plan)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(refinement_plan, indent=2) + "\n", encoding="utf-8")
    print(f"Refinement plan: {output}")
    print(f"Unique/trained tokens: {refinement_plan['trained_tokens']:,}")


if __name__ == "__main__":
    main()
