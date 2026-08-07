#!/usr/bin/env python
"""Generate text with the converted MLX ComplexityModel + the repo tiktoken tokenizer.

mlx-lm's loader expects an HF tokenizer; this model uses tiktoken o200k, so we
build the MLX model from the converted dir directly and drive a small sampling
loop with mlx-lm's KV cache — no HF tokenizer needed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/Users/boris/Dev/mlx-lm")

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from mlx_lm.models.complexity import Model, ModelArgs
from complexity.tokenizer import Tokenizer


def load_model(mlx_dir: Path) -> Model:
    cfg = json.loads((mlx_dir / "config.json").read_text())
    model = Model(ModelArgs(**cfg))
    weights = mx.load(str(mlx_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())
    return model


def generate(
    model,
    prompt_ids,
    max_tokens,
    temp,
    eos_id,
    *,
    top_p=0.0,
    top_k=0,
    repetition_penalty=1.0,
    repetition_context_size=64,
):
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    processors = make_logits_processors(
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
    )
    out = []
    for token, _ in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
    ):
        nt = int(token)
        if eos_id is not None and nt == eos_id:
            break
        out.append(nt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-dir", required=True)
    ap.add_argument("--tokenizer", default="./tokenizer-o200k")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--repetition-context-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Diagnostic mode: continue generation through EOS until max-tokens.",
    )
    args = ap.parse_args()

    mx.random.seed(args.seed)
    model = load_model(Path(args.mlx_dir))
    tok = Tokenizer.load(args.tokenizer)
    eos = getattr(tok, "eos_token_id", None)

    ids = tok.encode(args.prompt)
    gen = generate(
        model,
        ids,
        args.max_tokens,
        args.temp,
        None if args.ignore_eos else eos,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
    )
    print(args.prompt + tok.decode(gen))


if __name__ == "__main__":
    main()
