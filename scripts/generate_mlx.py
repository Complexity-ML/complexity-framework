#!/usr/bin/env python
"""Generate text with the converted MLX ComplexityModel + the repo tiktoken tokenizer.

mlx-lm's loader expects an HF tokenizer; this model uses tiktoken o200k, so we
build the MLX model from the converted dir directly and drive a small sampling
loop with mlx-lm's KV cache — no HF tokenizer needed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/Users/boris/Dev/mlx-lm")

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from mlx_lm.models.complexity import Model, ModelArgs
from complexity.inference.chat_template import (
    load_chat_template,
    render_inference_prompt,
)
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
    top_p,
    top_k,
    repetition_penalty,
    repetition_context_size,
    eos_id,
):
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    logits_processors = make_logits_processors(
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
    )
    out = []
    for token, _ in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    ):
        nt = int(token.item())
        if eos_id is not None and nt == eos_id:
            break
        out.append(nt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-dir", required=True)
    ap.add_argument("--tokenizer", default="./tokenizer-o200k")
    ap.add_argument("--prompt", default="The")
    ap.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Bypass chat-template rendering for base-model completion tests.",
    )
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--repetition-context-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = load_model(Path(args.mlx_dir))
    tok = Tokenizer.load(args.tokenizer)
    eos = getattr(tok, "eos_token_id", None)

    mx.random.seed(args.seed)
    prompt = args.prompt
    template_path = Path(args.mlx_dir) / "chat_template.json"
    if template_path.exists() and not args.raw_prompt:
        prompt = render_inference_prompt(args.prompt, load_chat_template(template_path))
    ids = tok.encode(prompt)
    started = time.perf_counter()
    gen = generate(
        model,
        ids,
        args.max_tokens,
        args.temp,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
        args.repetition_context_size,
        eos,
    )
    elapsed = time.perf_counter() - started
    print(tok.decode(gen))
    print(
        f"\n[MLX] prompt={len(ids)} tok · generated={len(gen)} tok · "
        f"{len(gen) / max(elapsed, 1e-9):.1f} tok/s"
    )


if __name__ == "__main__":
    main()
