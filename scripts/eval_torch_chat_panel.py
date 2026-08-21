#!/usr/bin/env python3
"""Run the fixed chat panel against a native PyTorch TR-HASH checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from complexity.inference.chat_template import render_messages_before_assistant
from scripts.chat_generate_local import build_prompt, generate_chat, load_model, pick_device


def repetition_metrics(text: str) -> dict[str, float | int]:
    words = re.findall(r"[\w']+", text.casefold())
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    repeated_trigrams = len(trigrams) - len(set(trigrams))
    return {
        "words": len(words),
        "distinct_2": round(len(set(bigrams)) / max(1, len(bigrams)), 6),
        "repeated_trigrams": repeated_trigrams,
        "repeated_trigram_ratio": round(
            repeated_trigrams / max(1, len(trigrams)), 6
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--repetition-context-size", type=int)
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    generation = dict(panel["generation"])
    for name in (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "repetition_context_size",
    ):
        value = getattr(args, name)
        if value is not None:
            generation[name] = value
    device = pick_device(args.device)
    model, tokenizer, chat_template = load_model(args.checkpoint, args.tokenizer, device)
    results = []
    for item in panel["prompts"]:
        torch.manual_seed(0)
        if "messages" in item:
            prompt = render_messages_before_assistant(item["messages"], chat_template)
            display_prompt = item["messages"]
        else:
            prompt = build_prompt(item["prompt"], False, chat_template)
            display_prompt = item["prompt"]
        response = generate_chat(
            model,
            tokenizer,
            prompt,
            device,
            int(generation["max_tokens"]),
            float(generation["temperature"]),
            float(generation["top_p"]),
            int(generation["top_k"]),
            float(generation["repetition_penalty"]),
            int(generation["repetition_context_size"]),
        )
        result = {
            "id": item["id"],
            "prompt": display_prompt,
            "response": response,
            "repetition": repetition_metrics(response),
        }
        results.append(result)
        print(f"\n[{item['id']}]\n{response}", flush=True)

    report = {
        "panel_id": panel["id"],
        "checkpoint": str(args.checkpoint.resolve()),
        "chat_template_applied": True,
        "runtime": "pytorch",
        "generation": generation,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
