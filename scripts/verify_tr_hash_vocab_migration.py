#!/usr/bin/env python3
"""Verify tokenizer, state, and old-vocabulary logit parity after migration."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from integrations.transformers.tr_hash_moe.configuration_tr_hash_moe import TRHashConfig
from integrations.transformers.tr_hash_moe.modeling_tr_hash_moe import TRHashForCausalLM
from scripts.migrate_tr_hash_vocab_32004 import (
    NEW_VOCAB_SIZE,
    OLD_VOCAB_SIZE,
    REASONING_TOKENS,
)

PARITY_TEXTS = (
    "The capital of France is Paris.",
    "A glass falls from a table and breaks on the floor.",
    "Question: What is 17 times 23? Answer:",
)


def load_model(bundle: Path) -> TRHashForCausalLM:
    raw = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    model = TRHashForCausalLM(TRHashConfig(**raw))
    incompatible = model.load_state_dict(load_file(bundle / "model.safetensors"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible checkpoint: {incompatible}")
    return model.eval()


def score_old_logits(bundle: Path, encoded: list[list[int]]) -> list[torch.Tensor]:
    model = load_model(bundle)
    outputs = []
    with torch.inference_mode():
        for token_ids in encoded:
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            outputs.append(model(input_ids=input_ids).logits[..., :OLD_VOCAB_SIZE].clone())
    del model
    gc.collect()
    return outputs


def verify(source: Path, migrated: Path, *, atol: float) -> dict:
    source_tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    migrated_tokenizer = Tokenizer.from_file(str(migrated / "tokenizer.json"))
    encoded = []
    for text in PARITY_TEXTS:
        old_ids = source_tokenizer.encode(text, add_special_tokens=False).ids
        new_ids = migrated_tokenizer.encode(text, add_special_tokens=False).ids
        if old_ids != new_ids:
            raise AssertionError(f"old text tokenization changed for {text!r}")
        encoded.append(old_ids)
    for expected_id, token in zip(
        range(OLD_VOCAB_SIZE, NEW_VOCAB_SIZE), REASONING_TOKENS, strict=True
    ):
        if migrated_tokenizer.encode(token, add_special_tokens=False).ids != [expected_id]:
            raise AssertionError(f"reasoning token is not atomic: {token}")

    source_logits = score_old_logits(source, encoded)
    migrated_logits = score_old_logits(migrated, encoded)
    maximum_error = 0.0
    exact = True
    for old, new in zip(source_logits, migrated_logits, strict=True):
        maximum_error = max(maximum_error, float((old - new).abs().max()))
        exact = exact and torch.equal(old, new)
    if maximum_error > atol:
        raise AssertionError(f"old-vocabulary logit error {maximum_error} exceeds {atol}")
    return {
        "source": str(source.resolve()),
        "migrated": str(migrated.resolve()),
        "texts": len(PARITY_TEXTS),
        "old_tokenization_exact": True,
        "old_vocabulary_logits_exact": exact,
        "old_vocabulary_logits_max_abs_error": maximum_error,
        "tolerance": atol,
        "reasoning_tokens_atomic": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("migrated", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.source, args.migrated, atol=args.atol)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
