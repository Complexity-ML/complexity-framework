"""Generate from a complexity MLX checkpoint via mlx_lm (KV cache, sampling, etc.).

Loads both the model and its local Hugging Face tokenizer through mlx-lm's
official loader.

Usage:
    python scripts/mlx_generate.py <model_dir> --prompt "..." --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load, load_model, load_tokenizer


def render_bundle_prompt(model_dir: Path, user_message: str) -> tuple[str, str]:
    template_path = model_dir / "chat_template.json"
    if not template_path.is_file():
        raise FileNotFoundError(f"MLX bundle has no chat template: {template_path}")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    required = ("id", "system_prompt", "system_format", "user_format", "assistant_prefix")
    missing = [name for name in required if name not in template]
    if missing:
        raise ValueError("MLX chat template is missing: " + ", ".join(missing))
    prompt = (
        template["system_format"].format(content=template["system_prompt"])
        + template["user_format"].format(content=user_message)
        + template["assistant_prefix"]
    )
    return prompt, str(template["id"])


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
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
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
        rendered_prompt, template_id = render_bundle_prompt(args.model_dir, args.prompt)
        print(f"Chat template: {template_id}")
    else:
        rendered_prompt = args.raw_prompt
        print("Chat template: bypassed (--raw-prompt)")
    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    generate(
        model,
        tokenizer,
        prompt=rendered_prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=True,
    )


if __name__ == "__main__":
    main()
