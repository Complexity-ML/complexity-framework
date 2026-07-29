#!/usr/bin/env python3
"""GRPO RL on the reflect+calculator tool loop.

For each arithmetic question we know the ground-truth answer. We sample G
rollouts under the current policy, run each through the reflect-bootstrap
loop (gen reflect -> orchestrator extracts expr -> gen calculator -> execute),
score reward = 1.0 only if the executed calculator result matches the truth,
and do one GRPO update per group: advantage = (r - mean) / std, loss
= -mean(advantage_g * mean_logprob_generated_tokens_g).

Reward shaping:
    +1.0  : reflect parses, calculator parses, result == expected
     0.0  : tool calls parse but result wrong
    -0.5  : malformed JSON or wrong tool name
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complexity.config import ModelConfig
from complexity.inference.tool_rewards import CalculatorError, safe_calculator
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils.device import configure_torch_acceleration
from complexity.utils.mps import setup_mps

from scripts.agentic_calculator_demo import parse_tool_call, ToolCallParseError


OPS = ["+", "-", "*"]


@dataclass
class Rollout:
    prompt1_ids: torch.Tensor      # [1, P1]
    gen1_ids: torch.Tensor         # [1, R1]
    prompt2_ids: Optional[torch.Tensor]   # [1, P2] or None if stopped at step 1
    gen2_ids: Optional[torch.Tensor]      # [1, R2] or None
    reward: float
    trace: str                     # human-readable summary for logging


# --- prompt synthesis --------------------------------------------------------

def sample_problem(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    op = rng.choice(OPS)
    expr = f"{a}{op}{b}"
    expected = str(eval(expr))  # safe: literals + + - * only
    template = rng.choice([
        "What is {a} {op} {b}?",
        "Calculate {a} {op} {b}.",
        "Compute {a}{op}{b}.",
        "How much is {a} {op} {b}?",
    ])
    return template.format(a=a, b=b, op=op), expected


# --- sampling (no grad) ------------------------------------------------------

@torch.no_grad()
def sample_tokens(
    model: ComplexityModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Return only the newly generated ids, [1, T]. Stops at eos."""
    out_ids = []
    cur = input_ids
    past = None
    for _ in range(max_new_tokens):
        if past is None:
            out = model(cur, use_cache=True)
        else:
            out = model(cur[:, -1:], past_key_values=past, use_cache=True)
        past = out["past_key_values"]
        logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
        if 0 < top_k < logits.size(-1):
            cutoff = torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        out_ids.append(next_tok)
        cur = torch.cat([cur, next_tok], dim=1)
        if int(next_tok.item()) == eos_token_id:
            break
    if not out_ids:
        return cur[:, 0:0]
    return torch.cat(out_ids, dim=1)


# --- one rollout: reflect-bootstrap with scoring -----------------------------

def rollout(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    question: str,
    expected: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> Rollout:
    eos = getattr(tokenizer, "eos_token_id", 199999)

    def encode(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=device)

    def decode(ids: torch.Tensor) -> str:
        return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)

    prompt1 = f"User:\n{question}\n\nAssistant:\n"
    prompt1_ids = encode(prompt1)
    gen1_ids = sample_tokens(model, prompt1_ids, max_new_tokens, temperature, top_k, eos)
    gen1_text = decode(gen1_ids)

    # Try to parse the first tool call.
    try:
        tc1 = parse_tool_call(gen1_text)
    except ToolCallParseError:
        return Rollout(prompt1_ids, gen1_ids, None, None, -0.5, f"malformed_first: {gen1_text[:80]}")

    if tc1 is None:
        return Rollout(prompt1_ids, gen1_ids, None, None, -0.5, f"no_tool_call: {gen1_text[:80]}")

    # Direct calculator (skip reflect): allowed and scored on result.
    if tc1.name == "calculator":
        expr = str(tc1.arguments.get("expression", "")).strip()
        try:
            result = safe_calculator(expr)
        except CalculatorError:
            return Rollout(prompt1_ids, gen1_ids, None, None, -0.5, f"bad_calc: {expr}")
        reward = 1.0 if result == expected else 0.0
        return Rollout(prompt1_ids, gen1_ids, None, None, reward, f"direct_calc expr={expr} -> {result} (exp={expected})")

    if tc1.name != "reflect":
        return Rollout(prompt1_ids, gen1_ids, None, None, -0.5, f"wrong_tool1: {tc1.name}")

    # Reflect path: orchestrator extracts the expression from the embedded question.
    embedded = str(tc1.arguments.get("question", question))
    from scripts.agentic_calculator_demo import extract_arithmetic_expression
    extracted = extract_arithmetic_expression(embedded)
    if extracted is None:
        return Rollout(prompt1_ids, gen1_ids, None, None, -0.5, f"reflect_extract_fail: {embedded[:60]}")

    rresult = {"tool": "calculator", "arguments": {"expression": extracted}}
    prompt2_text = (
        prompt1
        + gen1_text
        + f"\n\nTool result from reflect: {json.dumps(rresult, separators=(',', ':'))}\n\nAssistant:\n"
    )
    prompt2_ids = encode(prompt2_text)
    gen2_ids = sample_tokens(model, prompt2_ids, max_new_tokens, temperature, top_k, eos)
    gen2_text = decode(gen2_ids)

    try:
        tc2 = parse_tool_call(gen2_text)
    except ToolCallParseError:
        return Rollout(prompt1_ids, gen1_ids, prompt2_ids, gen2_ids, -0.5, f"malformed_second: {gen2_text[:60]}")

    if tc2 is None or tc2.name != "calculator":
        return Rollout(prompt1_ids, gen1_ids, prompt2_ids, gen2_ids, -0.5,
                       f"wrong_tool2: {None if tc2 is None else tc2.name}")

    expr = str(tc2.arguments.get("expression", "")).strip()
    try:
        result = safe_calculator(expr)
    except CalculatorError:
        return Rollout(prompt1_ids, gen1_ids, prompt2_ids, gen2_ids, -0.5, f"bad_calc2: {expr}")

    reward = 1.0 if result == expected else 0.0
    return Rollout(prompt1_ids, gen1_ids, prompt2_ids, gen2_ids, reward,
                   f"reflect+calc expr={expr} -> {result} (exp={expected})")


# --- loss --------------------------------------------------------------------

def gen_logprob_mean(model: ComplexityModel, prompt_ids: torch.Tensor, gen_ids: torch.Tensor) -> torch.Tensor:
    """Mean log-prob of gen_ids tokens under the current policy, given prompt_ids.

    Forward on cat(prompt, gen)[:-1]; align so logits at position prompt_len-1+t
    predict gen[t].
    """
    input_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    out = model(input_ids[:, :-1])
    logits = out["logits"]
    log_probs = F.log_softmax(logits, dim=-1)
    P = prompt_ids.shape[1]
    R = gen_ids.shape[1]
    chosen = log_probs[:, P - 1 : P - 1 + R, :].gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    return chosen.mean()


def rollout_logprob(model: ComplexityModel, r: Rollout) -> torch.Tensor:
    lp = gen_logprob_mean(model, r.prompt1_ids, r.gen1_ids)
    if r.prompt2_ids is not None and r.gen2_ids is not None and r.gen2_ids.numel() > 0:
        lp = 0.5 * (lp + gen_logprob_mean(model, r.prompt2_ids, r.gen2_ids))
    return lp


# --- main loop ---------------------------------------------------------------

def load_model(ckpt_path: Path, tokenizer_path: Path, device: torch.device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(state["config"])
    cfg.use_custom_kernels = False
    model = ComplexityModel(cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    tokenizer = Tokenizer.load(str(tokenizer_path))
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "tokenizer-o200k")
    ap.add_argument("--output-dir", type=Path, default=Path("runs/rl_grpo_calc"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = setup_mps(seed=args.seed)
    configure_torch_acceleration(kernel_policy=False, log=False)

    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rng = random.Random(args.seed)
    metrics_path = args.output_dir / "grpo_metrics.jsonl"

    for step in range(1, args.steps + 1):
        question, expected = sample_problem(rng)
        rollouts: list[Rollout] = []
        model.eval()
        for _ in range(args.group_size):
            rollouts.append(rollout(
                model, tokenizer, question, expected, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ))

        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
        mean_r = rewards.mean().item()
        std_r = rewards.std(unbiased=False).item()

        if std_r < 1e-6:
            # No variance in this group -> skip update (GRPO advantage undefined).
            with metrics_path.open("a") as f:
                f.write(json.dumps({"step": step, "question": question, "expected": expected,
                                     "rewards": rewards.tolist(), "skipped": True}) + "\n")
            print(f"[{step:03d}] {question!r:<32} exp={expected:<6} rewards={rewards.tolist()} skip (no variance)")
            continue

        advantages = (rewards - mean_r) / (std_r + 1e-6)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for r, adv in zip(rollouts, advantages):
            lp = rollout_logprob(model, r)
            total_loss = total_loss + (-adv.to(device) * lp)
        loss = total_loss / len(rollouts)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        log_entry = {
            "step": step, "question": question, "expected": expected,
            "rewards": rewards.tolist(), "mean_reward": mean_r,
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm),
            "traces": [r.trace for r in rollouts],
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"[{step:03d}] {question!r:<32} exp={expected:<6} "
              f"r={rewards.tolist()} mean={mean_r:.2f} loss={float(loss.detach()):.3f} gn={float(grad_norm):.1f}")

        if step % args.save_every == 0:
            save_path = args.output_dir / f"checkpoint_grpo_{step:06d}.pt"
            torch.save({
                "model": model.state_dict(),
                "config": ModelConfig.from_dict(torch.load(args.checkpoint, map_location="cpu",
                                                          weights_only=False)["config"]).to_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, save_path)
            print(f"        saved {save_path}")


if __name__ == "__main__":
    main()
