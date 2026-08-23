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

from complexity.inference.chat_template import (  # noqa: E402
    default_chat_template,
    render_inference_prompt,
    render_thinking_inference_prompt,
    validate_chat_template,
)
from complexity.models import ComplexityModel  # noqa: E402
from complexity.tokenizer import Tokenizer  # noqa: E402
from complexity.utils.device import configure_torch_acceleration  # noqa: E402
from scripts.sft_500m_32k_tr import (  # noqa: E402
    checkpoint_config,
    load_checkpoint_state,
)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint: Path,
    tokenizer_path: Path,
    device: torch.device,
) -> tuple[ComplexityModel, Tokenizer, dict]:
    configure_torch_acceleration(kernel_policy=False, log=False)
    _, state = load_checkpoint_state(checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = False
    model = ComplexityModel(config).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    chat_template = validate_chat_template(state.get("chat_template", default_chat_template()))
    return model, Tokenizer.load(str(tokenizer_path)), chat_template


@torch.no_grad()
def generate_chat(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_context_size: int,
    stop_strings: tuple[str, ...] = (),
    stop_token_ids: tuple[int, ...] = (),
    skip_special_tokens: bool = True,
) -> str:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )
    output_ids = input_ids
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(max_new_tokens):
        logits = model(output_ids)["logits"][:, -1, :]
        if repetition_penalty != 1.0:
            recent = output_ids[:, -repetition_context_size:]
            for row in range(logits.shape[0]):
                token_ids = recent[row].unique()
                values = logits[row, token_ids]
                logits[row, token_ids] = torch.where(
                    values < 0,
                    values * repetition_penalty,
                    values / repetition_penalty,
                )
        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if 0 < top_k < logits.shape[-1]:
                threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            sampled = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
            next_token = sorted_indices.gather(-1, sampled)
        output_ids = torch.cat((output_ids, next_token), dim=1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break
        if stop_token_ids and all(
            int(token_id) in stop_token_ids for token_id in next_token.flatten().tolist()
        ):
            break
        if stop_strings:
            partial = tokenizer.decode(
                output_ids[0, input_ids.shape[1] :],
                skip_special_tokens=skip_special_tokens,
            )
            if any(stop in partial for stop in stop_strings):
                break
    output_ids = output_ids[0]
    text = tokenizer.decode(
        output_ids[input_ids.shape[1] :],
        skip_special_tokens=skip_special_tokens,
    )
    for stop in ("<|endoftext|>", "\nUser:", "\n\nUser:", "\nAssistant:"):
        text = text.split(stop, 1)[0]
    return text.strip()


def before_first(text: str, markers: tuple[str, ...]) -> str:
    positions = [text.find(marker) for marker in markers if marker in text]
    return text[: min(positions) if positions else len(text)].strip()


@torch.no_grad()
def generate_thinking_chat(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_context_size: int,
) -> str:
    """Decode reasoning and final answer separately under a valid envelope."""

    think_budget = max(1, max_new_tokens // 2)
    final_budget = max(1, max_new_tokens - think_budget)
    reasoning = generate_chat(
        model,
        tokenizer,
        prompt,
        device,
        think_budget,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        repetition_context_size,
        stop_strings=("<think>", "</think>", "<final>", "</final>"),
    )
    reasoning = before_first(reasoning, ("<think>", "</think>", "<final>", "</final>"))
    final_prompt = prompt + reasoning + "\n</think>\n<final>\n"
    final = generate_chat(
        model,
        tokenizer,
        final_prompt,
        device,
        final_budget,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        repetition_context_size,
        stop_strings=("<think>", "</think>", "<final>", "</final>"),
    )
    final = before_first(final, ("<think>", "</think>", "<final>", "</final>"))
    return f"<think>\n{reasoning}\n</think>\n<final>\n{final}\n</final>"


def build_prompt(
    user_text: str,
    raw: bool,
    chat_template: dict,
    thinking: bool = False,
) -> str:
    if raw:
        return user_text
    if thinking:
        return render_thinking_inference_prompt(user_text, chat_template)
    return render_inference_prompt(user_text, chat_template)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plain local chat generation without tool orchestration"
    )
    parser.add_argument("prompt", nargs="?", default="Hello")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer-o200k"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-context-size", type=int, default=64)
    parser.add_argument("--raw", action="store_true", help="Use prompt exactly as provided")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Prefill <think> after the assistant prefix and return the full envelope.",
    )
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, tokenizer, chat_template = load_model(
        args.checkpoint,
        args.tokenizer,
        device,
    )
    if args.raw and args.thinking:
        parser.error("--raw and --thinking cannot be combined")
    prompt = build_prompt(args.prompt, args.raw, chat_template, args.thinking)
    if args.show_prompt:
        print("=== prompt ===")
        print(prompt)
        print("=== completion ===")
    generator = generate_thinking_chat if args.thinking else generate_chat
    completion = generator(
        model,
        tokenizer,
        prompt,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
        args.repetition_context_size,
    )
    print(completion)


if __name__ == "__main__":
    main()
