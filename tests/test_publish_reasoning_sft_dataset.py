from __future__ import annotations

import json

from scripts.publish_reasoning_sft_dataset import HASHED_FILES, TOKENIZED, validate_local


def test_validate_local_requires_passed_500m_raw_token_parity(tmp_path) -> None:
    for relative in HASHED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/release-audit.json").write_text(json.dumps({"status": "passed"}))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"actual_unique_formatted_tokens": 500_000_669})
    )
    (tmp_path / TOKENIZED / "manifest.json").write_text(
        json.dumps({"partitions": {"train": {"num_tokens": 500_000_669}}})
    )

    result = validate_local(tmp_path)

    assert result["audit"]["status"] == "passed"
    assert result["manifest"]["actual_unique_formatted_tokens"] == 500_000_669
