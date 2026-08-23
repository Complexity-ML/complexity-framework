from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_tr_hash_sft_32004_dataset import (
    STALE_RELEASE_PATHS,
    validate_local_release,
)


def test_publish_validation_fails_closed_on_incomplete_release(tmp_path: Path) -> None:
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/release-audit.json").write_text(
        json.dumps({"status": "passed", "tokenizer_vocab_size": 32_004}),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"train_examples": 1, "eval_examples": 1}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="missing release files"):
        validate_local_release(tmp_path)


def test_publisher_removes_legacy_recipe_and_quality_audits() -> None:
    assert "metadata/recipe.json" in STALE_RELEASE_PATHS
    assert "metadata/quality-audit.json" in STALE_RELEASE_PATHS
