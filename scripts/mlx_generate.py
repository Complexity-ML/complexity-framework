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
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load, load_model, load_tokenizer

from complexity.inference.chat_template import (
    load_chat_template,
    load_chat_template_jinja,
    render_jinja_inference_prompt,
)


def render_bundle_prompt(
    model_dir: Path,
    user_message: str,
    *,
    eos_token: str,
) -> tuple[str, str]:
    template_path = model_dir / "chat_template.json"
    if not template_path.is_file():
        raise FileNotFoundError(f"MLX bundle has no chat template: {template_path}")
    template = load_chat_template(template_path)
    jinja_source = load_chat_template_jinja(model_dir)
    rendered = render_jinja_inference_prompt(
        user_message,
        jinja_source,
        eos_token=eos_token,
    )
    return rendered, str(template["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    prompt = ap.add_mutually_exclusive_group()
    prompt.add_argument(
        "--prompt",
        default="Hello!",
        help="User message rendered with the bundle's chat_template.json.",
    )
    prompt.add_argument(
        "--raw-prompt",
        help="Preformatted prompt passed directly to MLX without chat rendering.",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
        help="Sign-aware penalty for recently generated tokens; 1.0 disables it.",
    )
    ap.add_argument(
        "--repetition-context-size",
        type=int,
        default=64,
        help="Number of recent tokens considered by the repetition penalty.",
    )
    args = ap.parse_args()

    print(f"Loading model from {args.model_dir} ...")
    try:
        model, tokenizer = load(args.model_dir.as_posix())
    except ValueError as error:
        # Public TR-Hash exports may omit the primary token-to-expert table:
        # it is a hash-route runtime buffer, not a learned weight. Preserve
        # the table initialized by the MLX model while still surfacing every
        # unrelated checkpoint mismatch.
        if "mlp.token_to_expert" not in str(error):
            raise
        model, _ = load_model(args.model_dir, strict=False)
        tokenizer = load_tokenizer(args.model_dir)
    print(f"Ready. eos_token_id={tokenizer.eos_token_id}\n")
    if args.raw_prompt is None:
        rendered_prompt, template_id = render_bundle_prompt(
            args.model_dir,
            args.prompt,
            eos_token=tokenizer.eos_token,
        )
        print(f"Chat template: {template_id}")
    else:
        rendered_prompt = args.raw_prompt
        print("Chat template: bypassed (--raw-prompt)")
    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    logits_processors = make_logits_processors(
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
    )

    generate(
        model,
        tokenizer,
        prompt=rendered_prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=True,
    )


if __name__ == "__main__":
    main()
