"""Side-by-side comparison of two MLX checkpoints with fixed seeds.

Usage:
    PYTHONPATH=/Users/boris/Dev/mlx-lm python mlx_compare.py <ckpt_a> <ckpt_b>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_model

sys.path.insert(0, str(Path(__file__).parent))
from mlx_generate import load_tiktoken  # noqa: E402


PROMPTS = [
    "Once upon a time",
    "The Eiffel Tower is located in",
    "The capital of France is",
    "Water boils at",
    "The largest planet in our solar system is",
    "Albert Einstein was a famous",
    "In 1969, Neil Armstrong",
    "The Pacific Ocean is",
]

SEEDS = [0, 1, 2]
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 48


def run_one(model, tokenizer, prompt: str, seed: int) -> str:
    mx.random.seed(seed)
    sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P)
    # Capture stdout from verbose=False — we re-tokenize after.
    text = ""
    from mlx_lm.generate import stream_generate
    for response in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        text += response.text
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a", type=Path)
    ap.add_argument("ckpt_b", type=Path)
    args = ap.parse_args()

    print(f"Loading A: {args.ckpt_a.name}")
    model_a, _ = load_model(args.ckpt_a)
    tok_a = load_tiktoken(args.ckpt_a)
    print(f"Loading B: {args.ckpt_b.name}")
    model_b, _ = load_model(args.ckpt_b)
    tok_b = load_tiktoken(args.ckpt_b)
    print()

    for prompt in PROMPTS:
        print("=" * 80)
        print(f"PROMPT: {prompt!r}")
        print("=" * 80)
        for seed in SEEDS:
            out_a = run_one(model_a, tok_a, prompt, seed)
            out_b = run_one(model_b, tok_b, prompt, seed)
            print(f"\n--- seed={seed} ---")
            print(f"[A {args.ckpt_a.name}]")
            print(f"  {prompt}{out_a}")
            print(f"[B {args.ckpt_b.name}]")
            print(f"  {prompt}{out_b}")
        print()


if __name__ == "__main__":
    main()
