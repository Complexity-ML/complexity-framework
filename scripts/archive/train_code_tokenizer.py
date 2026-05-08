"""
Train a code-aware BPE tokenizer on StarCoderData.

Produces a tokenizer compatible with PreTrainedTokenizerFast in
./tokenizer-code/, drop-in replacement for ./tokenizer/.

Usage:
    HF_TOKEN=... python3 scripts/train_code_tokenizer.py
    HF_TOKEN=... python3 scripts/train_code_tokenizer.py --vocab-size 32000 --num-samples 2000000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from itertools import islice
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.processors import ByteLevel as ByteLevelPost
from tokenizers.trainers import BpeTrainer

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("train_code_tokenizer")
for noisy in ("httpx", "httpcore", "huggingface_hub", "datasets",
              "filelock", "fsspec", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

DEFAULT_LANGS = [
    "python", "javascript", "typescript", "java", "c", "cpp", "go",
    "rust", "ruby", "php", "kotlin", "scala", "shell", "lua", "r",
    "julia", "haskell", "sql",
]


def build_dataset(languages):
    """Load and interleave StarCoderData language shards. Fails loud."""
    parts = []
    for lang in languages:
        try:
            ds = load_dataset("bigcode/starcoderdata", data_dir=lang,
                              split="train", streaming=True)
            ds = ds.select_columns(["content"])
            parts.append(ds)
            print(f"  Loaded {lang}", flush=True)
        except Exception as e:
            print(f"  FAILED {lang}: {type(e).__name__}: {e}", flush=True)
    if not parts:
        raise RuntimeError(
            "No StarCoderData languages could be loaded. "
            "Check HF_TOKEN env var and that you've accepted "
            "https://huggingface.co/datasets/bigcode/starcoderdata terms."
        )
    print(f"  → interleaving {len(parts)} languages", flush=True)
    return interleave_datasets(parts, stopping_strategy="all_exhausted")


def stream_code(merged, max_samples: int):
    """Yield code strings from a pre-built interleaved dataset."""
    n = 0
    t0 = time.time()
    for ex in merged:
        text = ex.get("content") or ""
        if len(text) < 50:
            continue
        yield text
        n += 1
        if n % 50000 == 0:
            print(f"  streamed {n:,} / {max_samples:,} "
                  f"({n/max(time.time()-t0,1):.0f} sample/s)", flush=True)
        if n >= max_samples:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--num-samples", type=int, default=2_000_000,
                        help="Number of code documents to train on (~5-10B chars)")
    parser.add_argument("--output", type=str, default="./tokenizer-code")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--languages", nargs="+", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Special tokens (mirror the existing tokenizer) ──────────────────
    specials = ["</s>", "<pad>", "<s>", "<unk>"]

    # ── Whitespace tokens to seed the merges (huge win on Python/YAML) ─
    # The trainer learns merges greedily; pre-seeding indent strings as
    # initial_alphabet ensures common indents become single tokens.
    indent_seeds = [" " * n for n in (2, 4, 8, 12, 16)]

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tok.decoder = ByteLevelDec()
    tok.post_processor = ByteLevelPost(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=specials,
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=True,
    )

    languages = args.languages or DEFAULT_LANGS
    print(f"Loading languages (HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))})", flush=True)
    merged = build_dataset(languages)

    print(f"Training BPE: vocab_size={args.vocab_size}, samples={args.num_samples:,}",
          flush=True)
    iterator = stream_code(merged, args.num_samples)

    # Pre-feed indent seeds so they appear early in the training stream
    def feed():
        for s in indent_seeds:
            yield s * 100  # repeat to boost frequency
        yield from iterator

    tok.train_from_iterator(feed(), trainer=trainer)

    # ── Save in HuggingFace format ──────────────────────────────────────
    tok.save(str(out_dir / "tokenizer.json"))

    # special_tokens_map.json
    (out_dir / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }, indent=2))

    # tokenizer_config.json — minimal config for PreTrainedTokenizerFast
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": True,
        "model_max_length": 4096,
    }, indent=2))

    # ── Quick quality check ─────────────────────────────────────────────
    sample = '''def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    encoded = tok.encode(sample)
    logger.info(f"Sample: {len(sample)} chars -> {len(encoded.ids)} tokens "
                f"({len(sample)/len(encoded.ids):.2f} chars/token)")

    indent4 = tok.encode("    ", add_special_tokens=False)
    indent8 = tok.encode("        ", add_special_tokens=False)
    logger.info(f"4 spaces -> {len(indent4.ids)} tokens (target: 1)")
    logger.info(f"8 spaces -> {len(indent8.ids)} tokens (target: 1)")

    logger.info(f"Tokenizer saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
