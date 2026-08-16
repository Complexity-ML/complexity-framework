"""Validate and report controlled COCO detector-specialization ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from scripts.detector_checkpoint_status import COMPLETE, checkpoint_status

METRICS = (
    "map50",
    "map50_95",
    "ap_small",
    "ap_medium",
    "ap_large",
    "best_f1",
    "best_confidence",
)

ARMS = ("baseline", "adapters", "hash-gate", "weighting", "auxiliary", "full")
CONTROLLED_CONFIG_FIELDS = {
    "level_adapters_enabled",
    "class_level_hash_gate_enabled",
    "object_weighting_enabled",
    "level_aux_loss_weight",
    "gate_calibration_loss_weight",
    "object_contrastive_loss_weight",
}
EXPECTED_FEATURES: dict[str, dict[str, bool | float]] = {
    "baseline": {},
    "adapters": {"level_adapters_enabled": True},
    "hash-gate": {
        "level_adapters_enabled": True,
        "class_level_hash_gate_enabled": True,
    },
    "weighting": {
        "level_adapters_enabled": True,
        "class_level_hash_gate_enabled": True,
        "object_weighting_enabled": True,
    },
    "auxiliary": {
        "level_adapters_enabled": True,
        "class_level_hash_gate_enabled": True,
        "object_weighting_enabled": True,
        "level_aux_loss_weight": 0.10,
        "gate_calibration_loss_weight": 0.10,
    },
    "full": {
        "level_adapters_enabled": True,
        "class_level_hash_gate_enabled": True,
        "object_weighting_enabled": True,
        "level_aux_loss_weight": 0.10,
        "gate_calibration_loss_weight": 0.10,
        "object_contrastive_loss_weight": 0.05,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return values


def load_metrics(checkpoint: Path) -> dict[str, float] | None:
    validation = checkpoint / "validation.json"
    if not validation.is_file():
        return None
    values = _load_json(validation)
    return {name: float(values[name]) for name in METRICS}


def _final_state(root: Path, expected_epochs: int) -> tuple[Path, dict[str, Any]]:
    status, checkpoint = checkpoint_status(root, expected_epochs)
    if status != COMPLETE or checkpoint is None:
        raise ValueError(f"ablation arm is not complete: {root}")
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    return checkpoint, state


def _expected_feature_config(arm: str) -> dict[str, bool | float]:
    expected: dict[str, bool | float] = {
        "level_adapters_enabled": False,
        "class_level_hash_gate_enabled": False,
        "object_weighting_enabled": False,
        "level_aux_loss_weight": 0.0,
        "gate_calibration_loss_weight": 0.0,
        "object_contrastive_loss_weight": 0.0,
    }
    expected.update(EXPECTED_FEATURES[arm])
    return expected


def collect(root: Path, expected_epochs: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_arms: dict[str, Any] = {}
    baseline_config: dict[str, Any] | None = None
    baseline_options: dict[str, Any] | None = None
    baseline_metrics: dict[str, float] | None = None

    for arm in ARMS:
        arm_root = root / arm
        final_checkpoint, state = _final_state(arm_root, expected_epochs)
        best_checkpoint = arm_root / "best"
        metrics = load_metrics(best_checkpoint)
        if metrics is None:
            raise ValueError(f"missing COCO validation metrics: {best_checkpoint}")
        config = _load_json(best_checkpoint / "config.json")
        feature_config = _expected_feature_config(arm)
        mismatched_features = {
            name: (config.get(name), expected)
            for name, expected in feature_config.items()
            if config.get(name) != expected
        }
        if mismatched_features:
            raise ValueError(
                f"unexpected specialization config for {arm}: {mismatched_features}"
            )

        controlled_base = {
            name: value
            for name, value in config.items()
            if name not in CONTROLLED_CONFIG_FIELDS
        }
        options = state.get("training_options")
        if not isinstance(options, dict):
            raise ValueError(f"missing training options in {final_checkpoint}")
        if baseline_config is None:
            baseline_config = controlled_base
            baseline_options = options
            baseline_metrics = metrics
        else:
            if controlled_base != baseline_config:
                changed = sorted(
                    name
                    for name in set(controlled_base) | set(baseline_config)
                    if controlled_base.get(name) != baseline_config.get(name)
                )
                raise ValueError(
                    f"uncontrolled model-config drift in {arm}: {', '.join(changed)}"
                )
            if options != baseline_options:
                changed = sorted(
                    name
                    for name in set(options) | set(baseline_options or {})
                    if options.get(name) != (baseline_options or {}).get(name)
                )
                raise ValueError(
                    f"uncontrolled training-budget drift in {arm}: {', '.join(changed)}"
                )

        assert baseline_metrics is not None
        row: dict[str, Any] = {
            "arm": arm,
            "checkpoint": str(best_checkpoint),
            **metrics,
            "delta_map50_95": metrics["map50_95"] - baseline_metrics["map50_95"],
            "delta_ap_small": metrics["ap_small"] - baseline_metrics["ap_small"],
        }
        rows.append(row)
        manifest_arms[arm] = {
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "features": feature_config,
            "metrics": metrics,
        }

    manifest = {
        "protocol": "random -> COCO 2017 controlled ablations",
        "expected_epochs": expected_epochs,
        "arms": manifest_arms,
        "training_options": baseline_options,
        "common_model_config": baseline_config,
    }
    return rows, manifest


def write_reports(root: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    columns = (
        "arm",
        "checkpoint",
        *METRICS,
        "delta_map50_95",
        "delta_ap_small",
    )
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# TR-Hash COCO specialization ablations",
        "",
        "| Arm | mAP50-95 | Delta | AP small | Delta APs | AP medium | AP large |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['map50_95']:.4f} | "
            f"{row['delta_map50_95']:+.4f} | {row['ap_small']:.4f} | "
            f"{row['delta_ap_small']:+.4f} | {row['ap_medium']:.4f} | "
            f"{row['ap_large']:.4f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    (root / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    rows, manifest = collect(args.root, args.expected_epochs)
    write_reports(args.root, rows, manifest)
    print(f"validated and collected {len(rows)} COCO ablation arms in {args.root}")


if __name__ == "__main__":
    main()
