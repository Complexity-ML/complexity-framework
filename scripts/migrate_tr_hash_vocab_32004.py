#!/usr/bin/env python3
"""Append the four TR-HASH reasoning tokens without changing old semantics.

This is deliberately *not* implemented with ``resize_token_embeddings``.
TR-HASH persists one token-ID routing table and one fused route-code vector per
layer, so a safe vocabulary migration must resize those buffers together with
the tied token embedding matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from complexity.tr_hash import TRHashEngineConfig, TRHashStrategy
from complexity.tr_hash.routing import build_route_table, compile_top2_pair_metadata

REASONING_TOKENS = (
    "<|think_start|>",
    "<|think_end|>",
    "<|final_start|>",
    "<|final_end|>",
)
OLD_VOCAB_SIZE = 32_000
NEW_VOCAB_SIZE = OLD_VOCAB_SIZE + len(REASONING_TOKENS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tokenizer_extension(
    source_path: Path, target_path: Path
) -> tuple[Tokenizer, Tokenizer]:
    source_raw = json.loads(source_path.read_text(encoding="utf-8"))
    target_raw = json.loads(target_path.read_text(encoding="utf-8"))
    for key in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
        if source_raw.get(key) != target_raw.get(key):
            raise ValueError(f"tokenizer v2 changes the existing {key!r} contract")
    source_added = source_raw.get("added_tokens", [])
    target_added = target_raw.get("added_tokens", [])
    if target_added[: len(source_added)] != source_added:
        raise ValueError("tokenizer v2 changes existing added-token IDs")
    appended = [(entry["id"], entry["content"]) for entry in target_added[len(source_added) :]]
    expected = list(zip(range(OLD_VOCAB_SIZE, NEW_VOCAB_SIZE), REASONING_TOKENS, strict=True))
    if appended != expected:
        raise ValueError(f"unexpected tokenizer extension: {appended!r}")

    source = Tokenizer.from_file(str(source_path))
    target = Tokenizer.from_file(str(target_path))
    if source.get_vocab_size(with_added_tokens=True) != OLD_VOCAB_SIZE:
        raise ValueError("source tokenizer is not the expected 32,000-token tokenizer")
    if target.get_vocab_size(with_added_tokens=True) != NEW_VOCAB_SIZE:
        raise ValueError("target tokenizer is not the expected 32,004-token tokenizer")
    for token_id, token in zip(
        range(OLD_VOCAB_SIZE, NEW_VOCAB_SIZE), REASONING_TOKENS, strict=True
    ):
        encoded = target.encode(token, add_special_tokens=False).ids
        if encoded != [token_id]:
            raise ValueError(f"{token!r} is not atomic at ID {token_id}: {encoded}")
    return source, target


def expand_embeddings(
    embeddings: torch.Tensor,
    *,
    source_tokenizer: Tokenizer,
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    if tuple(embeddings.shape[:1]) != (OLD_VOCAB_SIZE,):
        raise ValueError(f"unexpected embedding shape: {tuple(embeddings.shape)}")
    expanded = embeddings.new_empty((NEW_VOCAB_SIZE, embeddings.size(1)))
    expanded[:OLD_VOCAB_SIZE].copy_(embeddings)
    initialization_ids: dict[str, list[int]] = {}
    for token_id, token in zip(
        range(OLD_VOCAB_SIZE, NEW_VOCAB_SIZE), REASONING_TOKENS, strict=True
    ):
        pieces = source_tokenizer.encode(token, add_special_tokens=False).ids
        if not pieces or any(piece >= OLD_VOCAB_SIZE for piece in pieces):
            raise ValueError(f"cannot initialize {token!r} from source subtokens: {pieces}")
        expanded[token_id].copy_(embeddings[pieces].mean(dim=0))
        initialization_ids[token] = pieces
    if not torch.equal(expanded[:OLD_VOCAB_SIZE], embeddings):
        raise AssertionError("old embedding rows changed during expansion")
    return expanded.contiguous(), initialization_ids


def _engine_config(
    raw_config: dict[str, Any], *, layer_index: int, vocab_size: int
) -> TRHashEngineConfig:
    return TRHashEngineConfig(
        hidden_size=int(raw_config["hidden_size"]),
        vocab_size=vocab_size,
        num_experts=int(raw_config["num_experts"]),
        top_k=int(raw_config["top_k"]),
        shared_width=int(raw_config["shared_intermediate_size"]),
        expert_width=int(raw_config["intermediate_size"]) // int(raw_config["num_experts"]),
        initializer_range=float(raw_config.get("initializer_range", 0.02)),
        routing_strategy=TRHashStrategy(raw_config["routing_strategy"]),
        layer_index=layer_index,
        route_hash_count=int(raw_config.get("route_hash_count", 2)),
        shared_output_scale=float(raw_config.get("shared_output_scale", 1.0)),
        routed_output_scale=float(raw_config.get("routed_output_scale", 1.0)),
    )


def expand_routing_state(
    state: dict[str, torch.Tensor],
    raw_config: dict[str, Any],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for layer_index in range(int(raw_config["num_hidden_layers"])):
        prefix = f"layers.{layer_index}.mlp.engine"
        route_key = f"{prefix}.route_table"
        code_key = f"{prefix}.fused_route_codes"
        pair_key = f"{prefix}.fused_expert_pairs"
        old_routes = state[route_key]
        old_codes = state[code_key]
        old_pairs = state[pair_key]
        candidate = build_route_table(
            _engine_config(raw_config, layer_index=layer_index, vocab_size=NEW_VOCAB_SIZE)
        )
        if not torch.equal(candidate[:, :OLD_VOCAB_SIZE], old_routes):
            raise ValueError(
                f"layer {layer_index}: current deterministic builder does not reproduce "
                "the persisted 32,000-token route prefix"
            )
        new_codes, new_pairs = compile_top2_pair_metadata(
            candidate,
            num_experts=int(raw_config["num_experts"]),
        )
        if not torch.equal(new_codes[:OLD_VOCAB_SIZE], old_codes):
            raise ValueError(f"layer {layer_index}: old fused route codes would change")
        if not torch.equal(new_pairs, old_pairs):
            raise ValueError(f"layer {layer_index}: fused expert-pair table would change")
        state[route_key] = candidate.contiguous()
        state[code_key] = new_codes.contiguous()
        state[pair_key] = new_pairs.contiguous()
        reports.append(
            {
                "layer": layer_index,
                "new_routes": candidate[:, OLD_VOCAB_SIZE:].T.tolist(),
                "old_route_prefix_exact": True,
                "old_fused_code_prefix_exact": True,
                "expert_pairs_exact": True,
            }
        )
    return reports


def update_tokenizer_metadata(source: Path, output: Path) -> None:
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config.setdefault("add_bos_token", True)
    config.setdefault("add_eos_token", True)
    config.setdefault("bos_token", "<s>")
    config.setdefault("eos_token", "</s>")
    config.setdefault("pad_token", "<pad>")
    config.setdefault("unk_token", "<unk>")
    config.setdefault("clean_up_tokenization_spaces", False)
    config.setdefault("model_max_length", 2048)
    config.setdefault("tokenizer_class", "PreTrainedTokenizerFast")
    decoder = config.setdefault("added_tokens_decoder", {})
    for token_id, token in enumerate(("</s>", "<pad>", "<s>", "<unk>")):
        decoder.setdefault(
            str(token_id),
            {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        )
    for token_id, token in zip(
        range(OLD_VOCAB_SIZE, NEW_VOCAB_SIZE), REASONING_TOKENS, strict=True
    ):
        decoder[str(token_id)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
    config["additional_special_tokens"] = list(REASONING_TOKENS)
    # ``extra_special_tokens`` replaces the complete special-token role
    # contract in Transformers 5.  These markers are additions; keeping them
    # only in ``additional_special_tokens`` preserves BOS/EOS/PAD/UNK.
    config.pop("extra_special_tokens", None)
    (output / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    special_path = source / "special_tokens_map.json"
    special = json.loads(special_path.read_text(encoding="utf-8")) if special_path.is_file() else {}
    special.setdefault("bos_token", "<s>")
    special.setdefault("eos_token", "</s>")
    special.setdefault("pad_token", "<pad>")
    special.setdefault("unk_token", "<unk>")
    special["additional_special_tokens"] = list(REASONING_TOKENS)
    (output / "special_tokens_map.json").write_text(
        json.dumps(special, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def migrate(source: Path, tokenizer_v2: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_weights = source / "model.safetensors"
    source_config = source / "config.json"
    source_tokenizer_path = source / "tokenizer.json"
    for required in (source_weights, source_config, source_tokenizer_path, tokenizer_v2):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_tokenizer, _target_tokenizer = validate_tokenizer_extension(
        source_tokenizer_path,
        tokenizer_v2,
    )
    raw_config = json.loads(source_config.read_text(encoding="utf-8"))
    if int(raw_config["vocab_size"]) != OLD_VOCAB_SIZE:
        raise ValueError("source config does not have vocab_size=32000")
    if not bool(raw_config.get("tie_word_embeddings", False)):
        raise ValueError("migration requires tied input/output embeddings")

    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    state = load_file(source_weights, device="cpu")
    if "lm_head.weight" in state:
        if not torch.equal(state["lm_head.weight"], state["embed_tokens.weight"]):
            raise ValueError("persisted lm_head is not tied to embed_tokens")
        del state["lm_head.weight"]
    state["embed_tokens.weight"], initialization_ids = expand_embeddings(
        state["embed_tokens.weight"],
        source_tokenizer=source_tokenizer,
    )
    routing = expand_routing_state(state, raw_config)

    raw_config["vocab_size"] = NEW_VOCAB_SIZE
    (output / "config.json").write_text(
        json.dumps(raw_config, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    model_yaml = source / "model_config.yaml"
    if model_yaml.is_file():
        yaml_config = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
        yaml_config["vocab_size"] = NEW_VOCAB_SIZE
        (output / "model_config.yaml").write_text(
            yaml.safe_dump(yaml_config, sort_keys=True),
            encoding="utf-8",
        )
    shutil.copy2(tokenizer_v2, output / "tokenizer.json")
    update_tokenizer_metadata(source, output)
    output_weights = output / "model.safetensors"
    save_file(state, output_weights, metadata=metadata)

    report = {
        "source": str(source.resolve()),
        "source_weights_sha256": sha256(source_weights),
        "output": str(output.resolve()),
        "output_weights_sha256": sha256(output_weights),
        "old_vocab_size": OLD_VOCAB_SIZE,
        "new_vocab_size": NEW_VOCAB_SIZE,
        "tied_embeddings": True,
        "new_embedding_initialization": "mean of the token's old-tokenizer subtoken rows",
        "initialization_token_ids": initialization_ids,
        "routing_layers": routing,
    }
    (output / "migration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("tokenizer_v2", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = migrate(args.source, args.tokenizer_v2, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
