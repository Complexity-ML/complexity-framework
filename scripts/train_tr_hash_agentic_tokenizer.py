#!/usr/bin/env python3
"""Train the fixed 32K TR-HASH Agentic-Reasoning tokenizer from raw JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

AGENTIC_SPECIAL_TOKENS = [
    "<|begin|>",
    "<|end|>",
    "<|pad|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_of_turn|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|tool_result_start|>",
    "<|tool_result_end|>",
    "<|plan_start|>",
    "<|plan_end|>",
    "<|memory_start|>",
    "<|memory_end|>",
    "<|think_start|>",
    "<|think_end|>",
    "<|final_start|>",
    "<|final_end|>",
]

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
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

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
