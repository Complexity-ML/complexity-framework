"""Side-by-side comparison of two MLX checkpoints with fixed seeds.

Usage:
    PYTHONPATH=/Users/boris/Dev/mlx-lm python mlx_compare.py <ckpt_a> <ckpt_b>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load, load_model, load_tokenizer


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

def run_one(
    model,
    tokenizer,
    prompt: str,
    seed: int,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    # Capture stdout from verbose=False — we re-tokenize after.
    text = ""
    from mlx_lm.generate import stream_generate
    for response in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
    ):
        text += response.text
    return text.strip()


def load_bundle(path: Path):
    try:
        return load(path.as_posix())
    except ValueError as error:
        if "mlp.token_to_expert" not in str(error):
            raise
        model, _ = load_model(path, strict=False)
        return model, load_tokenizer(path)


def logit_parity(model_a, model_b, tokenizer, prompt: str) -> tuple[float, float, float]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = mx.array([ids], dtype=mx.int32)
    logits_a = model_a(tokens).astype(mx.float32)
    logits_b = model_b(tokens).astype(mx.float32)
    difference = mx.abs(logits_a - logits_b)
    agreement = mx.mean(mx.argmax(logits_a, axis=-1) == mx.argmax(logits_b, axis=-1))
    maximum = mx.max(difference)
    mean = mx.mean(difference)
    mx.eval(maximum, mean, agreement)
    return float(maximum.item()), float(mean.item()), float(agreement.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a", type=Path)
    ap.add_argument("ckpt_b", type=Path)
    ap.add_argument("--max-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--logit-parity", action="store_true")
    args = ap.parse_args()

    print(f"Loading A: {args.ckpt_a.name}")
    model_a, tok_a = load_bundle(args.ckpt_a)
    print(f"Loading B: {args.ckpt_b.name}")
    model_b, tok_b = load_bundle(args.ckpt_b)
    print()

    for prompt in PROMPTS[: args.max_prompts]:
        print("=" * 80)
        print(f"PROMPT: {prompt!r}")
        print("=" * 80)
        if args.logit_parity:
            maximum, mean, agreement = logit_parity(
                model_a, model_b, tok_a, prompt
            )
            print(
                f"logits: max_abs={maximum:.8g} mean_abs={mean:.8g} "
                f"top1_agreement={agreement:.6f}"
            )
        for seed in args.seeds:
            generation = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }
            out_a = run_one(model_a, tok_a, prompt, seed, **generation)
            out_b = run_one(model_b, tok_b, prompt, seed, **generation)
            print(f"\n--- seed={seed} ---")
            print(f"[A {args.ckpt_a.name}]")
            print(f"  {prompt}{out_a}")
            print(f"[B {args.ckpt_b.name}]")
            print(f"  {prompt}{out_b}")
        print()


if __name__ == "__main__":
    main()
