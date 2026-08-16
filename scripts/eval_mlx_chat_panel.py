"""Run a fixed chat panel against a TR-Hash MLX bundle.

The report records raw generations plus simple repetition indicators.  It is
not a replacement for human review; its purpose is to keep prompts and decoding
identical at every LoRA milestone.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_generate import render_bundle_prompt
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load, load_model, load_tokenizer


def load_bundle(path: Path):
    try:
        return load(path.as_posix())
    except ValueError as error:
        if "mlp.token_to_expert" not in str(error):
            raise
        model, _ = load_model(path, strict=False)
        return model, load_tokenizer(path)


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold())


def repetition_metrics(text: str) -> dict[str, float | int]:
    words = normalized_words(text)
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    repeated_trigrams = len(trigrams) - len(set(trigrams))
    return {
        "words": len(words),
        "distinct_2": round(len(set(bigrams)) / max(1, len(bigrams)), 6),
        "repeated_trigrams": repeated_trigrams,
        "repeated_trigram_ratio": round(repeated_trigrams / max(1, len(trigrams)), 6),
    }


def generate_one(
    model,
    tokenizer,
    prompt: str,
    generation: dict,
    *,
    raw_prompt: bool = False,
) -> str:
    rendered = prompt
    if not raw_prompt:
        rendered, _ = render_bundle_prompt(
            Path(generation["model_dir"]),
            prompt,
            eos_token=tokenizer.eos_token,
        )
    mx.random.seed(0)
    sampler = make_sampler(
        temp=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
    )
    processors = make_logits_processors(
        repetition_penalty=float(generation["repetition_penalty"]),
        repetition_context_size=int(generation["repetition_context_size"]),
    )
    pieces = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt=rendered,
        max_tokens=int(generation["max_tokens"]),
        sampler=sampler,
        logits_processors=processors,
    ):
        pieces.append(response.text)
    return "".join(pieces).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("configs/sft_500m_mlx_panel.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Bypass the bundle chat template and send each prompt directly.",
    )
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    generation = dict(panel["generation"])
    generation["model_dir"] = args.model_dir.as_posix()
    model, tokenizer = load_bundle(args.model_dir)

    results = []
    for item in panel["prompts"]:
        response = generate_one(
            model,
            tokenizer,
            item["prompt"],
            generation,
            raw_prompt=args.raw_prompt,
        )
        results.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "response": response,
                "repetition": repetition_metrics(response),
            }
        )
        print(f"\n[{item['id']}]\n{response}")

    report = {
        "panel_id": panel["id"],
        "model_dir": args.model_dir.as_posix(),
        "chat_template_applied": not args.raw_prompt,
        "generation": panel["generation"],
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
