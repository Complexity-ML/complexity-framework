#!/usr/bin/env python3
"""Validate and pin the immutable Agentic 32K tokenizer into the 125B config."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from tokenizers import Tokenizer

from complexity.tokenizer import get_special_tokens
from scripts.build_agentic_pretraining_50b import sha256_file


def pin_tokenizer_contract(config_path: Path, tokenizer_dir: Path, revision: str) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("tokenizer revision must be an immutable 40-character git SHA")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = config.get("tokenizer_contract")
    if not isinstance(contract, dict):
        raise ValueError("config has no tokenizer_contract")

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    manifest_path = tokenizer_dir / str(
        contract.get("required_manifest", "agentic_tokenizer_manifest.json")
    )
    if not tokenizer_json.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("tokenizer.json and the agentic tokenizer manifest are required")

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    expected_vocab = int(contract["vocab_size"])
    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab != expected_vocab:
        raise ValueError(f"tokenizer has {actual_vocab} IDs, expected {expected_vocab}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("vocab_size", -1)) != expected_vocab:
        raise ValueError("tokenizer manifest vocab_size disagrees with the contract")
    expected_markers = get_special_tokens("tr_hash_agentic_reasoning")
    actual_marker_ids = {marker: tokenizer.token_to_id(marker) for marker in expected_markers}
    if any(value is None for value in actual_marker_ids.values()):
        raise ValueError("tokenizer is missing one or more Agentic 32K special markers")
    if manifest.get("special_token_ids") != actual_marker_ids:
        raise ValueError("special marker IDs disagree between tokenizer and manifest")

    contract.update(
        {
            "status": "validated",
            "revision": revision,
            "manifest_sha256": sha256_file(manifest_path),
            "tokenizer_sha256": sha256_file(tokenizer_json),
        }
    )
    temporary = config_path.with_suffix(config_path.suffix + ".partial")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, config_path)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agentic_pretraining/tr_hash_pretraining_125b.json")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    contract = pin_tokenizer_contract(Path(args.config), Path(args.tokenizer), args.revision)
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
