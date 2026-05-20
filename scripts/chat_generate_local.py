#!/usr/bin/env python3
"""Generate plain chat completions from a local Complexity checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils.device import configure_torch_acceleration
from scripts.sft_100m_o200k_tr_local import checkpoint_config, load_checkpoint_state


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, tokenizer_path: Path, device: torch.device) -> tuple[ComplexityModel, Tokenizer]:
    configure_torch_acceleration(kernel_policy=False, log=False)
    _, state = load_checkpoint_state(checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = False
    model = ComplexityModel(config).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, Tokenizer.load(str(tokenizer_path))


@torch.no_grad()
def generate_chat(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-5),
        top_k=0,
        top_p=top_p,
        do_sample=temperature > 0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )[0]
    text = tokenizer.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True)
    for stop in ("<|endoftext|>", "\nUser:", "\n\nUser:", "\nAssistant:"):
        text = text.split(stop, 1)[0]
    return text.strip()


def build_prompt(user_text: str, raw: bool) -> str:
    if raw:
        return user_text
    return f"User:\n{user_text}\n\nAssistant:\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plain local chat generation without tool orchestration")
    parser.add_argument("prompt", nargs="?", default="Hello")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer-o200k"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--raw", action="store_true", help="Use prompt exactly as provided")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    prompt = build_prompt(args.prompt, args.raw)
    if args.show_prompt:
        print("=== prompt ===")
        print(prompt)
        print("=== completion ===")
    print(generate_chat(model, tokenizer, prompt, device, args.max_new_tokens, args.temperature, args.top_p))


if __name__ == "__main__":
    main()
