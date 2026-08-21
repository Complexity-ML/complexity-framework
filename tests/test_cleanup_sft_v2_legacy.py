from __future__ import annotations

import pytest

from scripts.cleanup_sft_v2_legacy import cleanup_plan


def test_cleanup_plan_removes_only_legacy_sft_v1_paths() -> None:
    files = {
        "README.md",
        "config.json",
        "model.safetensors",
        "release_manifest.json",
        "tokenizer.json",
        "step_000463/checkpoint.pt",
        "step_000926/checkpoint.pt",
        "step_001389/checkpoint.pt",
        "reports/piqa/epoch-01.json",
        "reports/training/metrics.csv",
        "training/sft-v2-300k/checkpoints/step_005982/checkpoint.pt",
        "training/sft-v2-300k/evaluations/summary.json",
    }

    assert cleanup_plan(files, 5982) == [
        "reports/piqa/epoch-01.json",
        "reports/training/metrics.csv",
        "step_000463/checkpoint.pt",
        "step_000926/checkpoint.pt",
        "step_001389/checkpoint.pt",
    ]


def test_cleanup_plan_refuses_missing_promoted_root_or_checkpoint() -> None:
    with pytest.raises(ValueError, match="Refusing cleanup"):
        cleanup_plan({"README.md"}, 5982)
