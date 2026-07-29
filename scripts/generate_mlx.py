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
from mlx_lm.models.cache import make_prompt_cache

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


def generate(model, prompt_ids, max_tokens, temp, eos_id):
    cache = make_prompt_cache(model)
    y = mx.array(prompt_ids)[None]
    logits = model(y, cache=cache)[:, -1, :]
    out = []
    for _ in range(max_tokens):
        if temp <= 0.0:
            nt = int(mx.argmax(logits, axis=-1).item())
        else:
            nt = int(mx.random.categorical(logits * (1.0 / temp)).item())
        if eos_id is not None and nt == eos_id:
            break
        out.append(nt)
        logits = model(mx.array([[nt]]), cache=cache)[:, -1, :]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-dir", required=True)
    ap.add_argument("--tokenizer", default="./tokenizer-o200k")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    model = load_model(Path(args.mlx_dir))
    tok = Tokenizer.load(args.tokenizer)
    eos = getattr(tok, "eos_token_id", None)

    ids = tok.encode(args.prompt)
    gen = generate(model, ids, args.max_tokens, args.temp, eos)
    print(args.prompt + tok.decode(gen))


if __name__ == "__main__":
    main()
