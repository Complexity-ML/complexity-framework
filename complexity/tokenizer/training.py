"""Train the fixed 32K TR-HASH Agentic-Reasoning tokenizer from raw JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from .._optional import require_optional
from .config import get_special_tokens

AGENTIC_SPECIAL_TOKENS = get_special_tokens("tr_hash_agentic_reasoning")

CHAT_TEMPLATE = r"""{%- if tools %}
{{- '<|system|>Available tools:\n' }}{{ tools | tojson }}{{- '<|end_of_turn|>' }}
{%- endif %}
{%- for message in messages %}
{%- set role = message['role'] %}
{%- if role == 'tool' %}
{{- '<|tool_result_start|>' + message.get('content', '') + '<|tool_result_end|><|end_of_turn|>' }}
{%- elif role == 'assistant' %}
{{- '<|assistant|>' }}
{%- if message.get('reasoning') %}
{{- '<|think_start|>' + message['reasoning'] + '<|think_end|>' }}
{%- endif %}
{%- if message.get('tool_calls') %}
{{- '<|tool_call_start|>' }}{{ message['tool_calls'] | tojson }}{{- '<|tool_call_end|>' }}
{%- endif %}
{%- if message.get('content') %}
{{- '<|final_start|>' + message['content'] + '<|final_end|>' }}
{%- endif %}
{{- '<|end_of_turn|>' }}
{%- else %}
{{- '<|' + role + '|>' + message.get('content', '') + '<|end_of_turn|>' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|assistant|><|think_start|>' }}
{%- endif %}"""


def iter_corpus(corpus_dir: Path) -> Iterator[str]:
    corpus_dir = Path(corpus_dir)
    paths = sorted(path for path in corpus_dir.glob("*.jsonl") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no JSONL corpus shards in {corpus_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"missing text at {path}:{line_number}")
                yield text


def train_tokenizer_from_iterator(
    texts: Iterable[str],
    output_dir: Path,
    *,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    model_max_length: int = 2_048,
    corpus_manifest_sha256: str | None = None,
    manifest_extra: Mapping[str, object] | None = None,
) -> dict:
    require_optional("tokenizers", "tokenizers", feature="Agentic tokenizer training")
    require_optional("transformers", "hf", feature="Agentic tokenizer HF export")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    output_dir = Path(output_dir)
    if vocab_size <= len(AGENTIC_SPECIAL_TOKENS):
        raise ValueError(
            "vocab_size must leave room for ordinary tokens beyond the special markers"
        )
    if min_frequency <= 0 or model_max_length <= 0:
        raise ValueError("min_frequency and model_max_length must be positive")
    # Kept as a pure-data contract so CPU-only tokenizer workers do not import
    # the full training framework (and therefore do not require PyTorch).
    special_tokens = AGENTIC_SPECIAL_TOKENS
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        texts,
        trainer=BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        ),
    )
    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != vocab_size:
        raise RuntimeError(f"tokenizer produced {actual_vocab} entries, expected {vocab_size}")
    expected_ids = {token: index for index, token in enumerate(special_tokens)}
    actual_ids = {token: tokenizer.token_to_id(token) for token in special_tokens}
    if actual_ids != expected_ids:
        raise RuntimeError(f"special-token IDs changed: {actual_ids}")

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|begin|>",
        eos_token="<|end|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        additional_special_tokens=special_tokens[4:],
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
    )
    fast.chat_template = CHAT_TEMPLATE
    output_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output_dir)
    (output_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE + "\n", encoding="utf-8")

    manifest = {
        "schema": "tr-hash-agentic-tokenizer-v1",
        "vocab_size": vocab_size,
        "format": "tr_hash_agentic_reasoning",
        "special_token_ids": actual_ids,
        "corpus_manifest_sha256": corpus_manifest_sha256,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (output_dir / "agentic_tokenizer_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def train_tokenizer(
    corpus_dir: Path,
    output_dir: Path,
    *,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    model_max_length: int = 2_048,
) -> dict:
    corpus_dir = Path(corpus_dir)
    corpus_manifest = corpus_dir / "manifest.json"
    return train_tokenizer_from_iterator(
        iter_corpus(corpus_dir),
        output_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        model_max_length=model_max_length,
        corpus_manifest_sha256=(
            hashlib.sha256(corpus_manifest.read_bytes()).hexdigest()
            if corpus_manifest.is_file()
            else None
        ),
    )


def _contract_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tokenizer_contract(config: Mapping[str, object], tokenizer_path: Path) -> None:
    tokenizer_path = Path(tokenizer_path)
    contract = config.get("tokenizer_contract")
    if not contract:
        return
    if not isinstance(contract, Mapping):
        raise ValueError("tokenizer_contract must contain an object")
    if contract.get("status") != "validated":
        raise ValueError("corpus build is gated until tokenizer_contract.status is validated")
    revision = contract.get("revision")
    manifest_digest = contract.get("manifest_sha256")
    tokenizer_digest = contract.get("tokenizer_sha256")
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (manifest_digest, tokenizer_digest)
        )
    ):
        raise ValueError(
            "validated tokenizer contract requires revision, manifest_sha256, and tokenizer_sha256"
        )
    root = tokenizer_path if tokenizer_path.is_dir() else tokenizer_path.parent
    manifest = _contract_manifest(root, contract)
    if not manifest.is_file() or _contract_sha256(manifest) != manifest_digest:
        raise ValueError("local tokenizer manifest does not match the pinned contract")
    tokenizer_json = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    if (
        tokenizer_json.is_symlink()
        or not tokenizer_json.is_file()
        or _contract_sha256(tokenizer_json) != tokenizer_digest
    ):
        raise ValueError("local tokenizer.json does not match the pinned contract")
    Tokenizer = require_optional("tokenizers", "tokenizers", feature="Tokenizer contract validation").Tokenizer

    expected = contract.get("vocab_size")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if type(expected) is not int or expected <= 0 or manifest_data.get("vocab_size") != expected:
        raise ValueError("tokenizer manifest vocab_size disagrees with the contract")
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    if tokenizer.get_vocab_size(with_added_tokens=True) != expected:
        raise ValueError("tokenizer vocabulary differs from the pinned contract")


def _contract_manifest(root: Path, contract: Mapping[str, object]) -> Path:
    relative = contract.get("required_manifest", "agentic_tokenizer_manifest.json")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("tokenizer manifest must be a contained relative path")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("tokenizer manifest must remain inside its local directory")
    return path


def pin_tokenizer_contract(config_path: Path, tokenizer_dir: Path, revision: str) -> dict:
    Tokenizer = require_optional("tokenizers", "tokenizers", feature="Tokenizer contract pinning").Tokenizer

    config_path, tokenizer_dir = Path(config_path), Path(tokenizer_dir)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("tokenizer revision must be an immutable 40-character git SHA")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = config.get("tokenizer_contract")
    if not isinstance(contract, dict):
        raise ValueError("config has no tokenizer_contract")

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    manifest_path = _contract_manifest(tokenizer_dir, contract)
    if tokenizer_json.is_symlink() or not tokenizer_json.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("tokenizer.json and the agentic tokenizer manifest are required")

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    expected_vocab = contract.get("vocab_size")
    if type(expected_vocab) is not int or expected_vocab <= 0:
        raise ValueError("tokenizer contract vocab_size must be a positive integer")
    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab != expected_vocab:
        raise ValueError(f"tokenizer has {actual_vocab} IDs, expected {expected_vocab}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if type(manifest.get("vocab_size")) is not int or manifest["vocab_size"] != expected_vocab:
        raise ValueError("tokenizer manifest vocab_size disagrees with the contract")
    expected_markers = get_special_tokens("tr_hash_agentic_reasoning")
    actual_marker_ids = {marker: tokenizer.token_to_id(marker) for marker in expected_markers}
    if actual_marker_ids != {marker: index for index, marker in enumerate(expected_markers)}:
        raise ValueError("tokenizer is missing one or more Agentic 32K special markers")
    if manifest.get("special_token_ids") != actual_marker_ids:
        raise ValueError("special marker IDs disagree between tokenizer and manifest")

    contract.update(
        {
            "status": "validated",
            "revision": revision,
            "manifest_sha256": _contract_sha256(manifest_path),
            "tokenizer_sha256": _contract_sha256(tokenizer_json),
        }
    )
    temporary = config_path.with_suffix(config_path.suffix + ".partial")
    if config_path.is_symlink() or temporary.is_symlink():
        raise ValueError("tokenizer contract authoring paths must not be symlinks")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, config_path)
    return contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=2_048)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = train_tokenizer(
        Path(args.corpus_dir),
        Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        model_max_length=args.model_max_length,
    )
    print(f"Tokenizer ready: {args.output_dir} ({manifest['vocab_size']:,} IDs)")


if __name__ == "__main__":
    main()
