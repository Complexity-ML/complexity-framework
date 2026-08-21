#!/usr/bin/env python3
"""Generate a chat response from a native TR-HASH checkpoint in eager PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from complexity.inference.chat_template import (
    default_chat_template,
    render_inference_prompt,
    validate_chat_template,
)
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils.device import configure_torch_acceleration
from scripts.sft_500m_32k_tr import (
    checkpoint_config,
    load_checkpoint_state,
    load_model_state_compat,
)


@torch.inference_mode()
def generate(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        logits = model(tokens)["logits"][:, -1].float()
        if repetition_penalty != 1.0:
            recent = tokens[:, -64:]
            for row in range(logits.shape[0]):
                ids = recent[row].unique()
                values = logits[row, ids]
                logits[row, ids] = torch.where(
                    values < 0,
                    values * repetition_penalty,
                    values / repetition_penalty,
                )
        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits /= temperature
            if 0 < top_k < logits.shape[-1]:
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            sampled = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
            next_token = sorted_indices.gather(-1, sampled)
        tokens = torch.cat((tokens, next_token), dim=1)
        if eos_id is not None and int(next_token.item()) == eos_id:
            break
    text = tokenizer.decode(tokens[0, len(prompt_ids) :], skip_special_tokens=True)
    for stop in ("<|endoftext|>", "\nUser:", "\n\nUser:", "\nAssistant:"):
        text = text.split(stop, 1)[0]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_torch_acceleration(kernel_policy=False, log=False)
    device = torch.device(args.device)
    _, state = load_checkpoint_state(args.checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = False
    model = ComplexityModel(config)
    load_model_state_compat(model, state["model"])
    model.to(device).eval()
    tokenizer = Tokenizer.load(str(args.tokenizer))
    template = validate_chat_template(
        state.get("chat_template", default_chat_template())
    )
    rendered = render_inference_prompt(args.prompt, template)
    torch.manual_seed(args.seed)
    print(
        generate(
            model,
            tokenizer,
            rendered,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
    )


if __name__ == "__main__":
    main()
