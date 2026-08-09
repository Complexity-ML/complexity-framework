"""Generate from a complexity MLX checkpoint via mlx_lm (KV cache, sampling, etc.).

Loads both the model and its local Hugging Face tokenizer through mlx-lm's
official loader.

Usage:
    python scripts/mlx_generate.py <model_dir> --prompt "..." --max-tokens 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_lm.generate import generate
from mlx_lm.utils import load
from mlx_lm.sample_utils import make_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    print(f"Loading model from {args.model_dir} ...")
    model, tokenizer = load(args.model_dir.as_posix())
    print(f"Ready. eos_token_id={tokenizer.eos_token_id}\n")
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=True,
    )


if __name__ == "__main__":
    main()
