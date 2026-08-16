from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_collects_model_only_release_and_provenance(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = checkpoint_root / "step_000010"
    checkpoint_dir.mkdir(parents=True)
    tied = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    torch.save(
        {
            "step": 10,
            "model": {"embed.weight": tied, "lm_head.weight": tied},
            "optimizer": {"state": {}},
            "config": {"hidden_size": 4},
        },
        checkpoint_path,
    )
    (checkpoint_root / "latest").write_text("step_000010\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    _write_json(run_dir / "run_config.json", {"schema_version": 1})
    (run_dir / "metrics.csv").write_text("step,eval_loss\n10,1.0\n", encoding="utf-8")

    data_root = tmp_path / "data"
    _write_json(data_root / "dataset_manifest.json", {"schema_version": 1})
    for partition in ("train", "eval"):
        _write_json(
            data_root / partition / "tokens.idx.json",
            {"partition": partition, "sha256": partition},
        )

    tokenizer = tmp_path / "tokenizer"
    _write_json(tokenizer / "tokenizer.json", {"version": "1.0"})
    log_path = tmp_path / "training.log"
    log_path.write_text("finished\n", encoding="utf-8")
    output_dir = tmp_path / "release"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "collect_200m_run_artifacts.py"),
            "--run-name",
            "test-run",
            "--checkpoint-root",
            str(checkpoint_root / "latest"),
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(data_root),
            "--tokenizer",
            str(tokenizer),
            "--output-dir",
            str(output_dir),
            "--log",
            str(log_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    weights = load_file(output_dir / "model.safetensors")
    assert torch.equal(weights["embed.weight"], tied)
    assert torch.equal(weights["lm_head.weight"], tied)

    model_config = json.loads((output_dir / "model_config.json").read_text())
    assert model_config["checkpoint_step"] == 10
    assert model_config["model"]["hidden_size"] == 4

    artifact_manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text()
    )
    assert artifact_manifest["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    released_paths = {entry["path"] for entry in artifact_manifest["files"]}
    assert "model.safetensors" in released_paths
    assert "metrics.csv" in released_paths
    assert "dataset/train/tokens.idx.json" in released_paths
