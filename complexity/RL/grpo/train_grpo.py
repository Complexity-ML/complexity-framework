"""
GRPO (Group Relative Policy Optimization) for ComplexityModel.

From-scratch implementation — no TRL dependency.
Works natively with mu-guidance, token-routed MoE, deterministic Zipf routing.

Algorithm (DeepSeek-R1 style):
    For each prompt:
      1. Sample G completions from policy π_θ
      2. Score with reward function r(completion, answer)
      3. Advantage = (r - mean(r)) / (std(r) + eps)  [group-relative]
      4. Loss = -E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ] + β * KL(π_θ || π_ref)

Usage:
    python -m complexity.RL.grpo.train_grpo \
        --model_path checkpoints/400m-v1/final \
        --dataset cais/mmlu \
        --output_dir checkpoints/400m-grpo \
        --group_size 16 \
        --beta 0.04 \
        --lr 5e-6
"""

import argparse
import copy
import math
import os
import time
from typing import List, Dict, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset

from complexity.config import ModelConfig
from complexity.models.builder import ComplexityModel

from .rewards import mcq_exact_match_reward, combined_reward


# ── Dataset ──────────────────────────────────────────────────────────

CHOICES = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the following multiple choice question. "
    "Think step by step, then give your final answer as: The answer is X"
)


def format_mmlu_prompt(question: str, choices: List[str], subject: str = "") -> str:
    choices_text = "\n".join(f"{CHOICES[i]}. {c}" for i, c in enumerate(choices))
    subject_line = f" ({subject})" if subject else ""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question{subject_line}:\n{question}\n\n"
        f"{choices_text}\n\n"
        f"Answer:"
    )


def load_mmlu_dataset(
    dataset_name: str = "cais/mmlu",
    subset: str = "all",
    split: str = "all",
    max_samples: int = 0,
) -> List[Dict]:
    """Load MMLU and return list of {prompt, answer} dicts."""
    if subset != "all":
        ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        answer_idx = row["answer"]
        answer = CHOICES[answer_idx] if isinstance(answer_idx, int) else str(answer_idx).upper()
        examples.append({
            "prompt": format_mmlu_prompt(row["question"], row["choices"], row.get("subject", "")),
            "answer": answer,
        })
    return examples


# ── Tokenization helpers ─────────────────────────────────────────────

def encode_prompt(tokenizer, prompt: str, max_len: int) -> torch.Tensor:
    """Encode a prompt, truncating from left if needed."""
    ids = tokenizer.encode(prompt, add_bos=True)
    if len(ids) > max_len:
        ids = ids[-max_len:]
    return torch.tensor(ids, dtype=torch.long)


def decode_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    """Decode token IDs to string."""
    return tokenizer.decode(token_ids.tolist())


# ── Generation ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_group(
    model: ComplexityModel,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate G completions for a single prompt.

    Args:
        model: ComplexityModel (handles mu-guidance, MoE internally)
        prompt_ids: [prompt_len] token IDs
        group_size: Number of completions to generate (G)
        max_new_tokens: Max tokens per completion
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token_id: Stop token

    Returns:
        completions: [G, prompt_len + completion_len] padded token IDs
    """
    model.eval()
    device = next(model.parameters()).device
    prompt_len = prompt_ids.shape[0]

    # Expand prompt for all G completions: [G, prompt_len]
    input_ids = prompt_ids.unsqueeze(0).expand(group_size, -1).to(device)

    # Autoregressive generation with KV cache
    past_key_values = None
    for step in range(max_new_tokens):
        if past_key_values is None:
            outputs = model(input_ids, use_cache=True)
        else:
            outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs["past_key_values"]
        logits = outputs["logits"][:, -1, :]  # [G, vocab]

        # Temperature + nucleus sampling
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [G, 1]
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Check EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids  # [G, prompt_len + generated_len]


# ── Log-probability computation ──────────────────────────────────────

def compute_log_probs(
    model: ComplexityModel,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Compute per-token log-probs for the completion portion.

    Args:
        model: ComplexityModel in train mode
        input_ids: [G, seq_len] full sequences (prompt + completion)
        prompt_len: Length of the prompt (log-probs only for tokens after this)

    Returns:
        log_probs: [G] sum of log-probs over completion tokens
    """
    model.eval()
    with torch.no_grad():
        was_training = model.training
        model.eval()
        outputs = model(input_ids, use_cache=False)
        if was_training:
            model.train()

    logits = outputs["logits"]  # [G, seq_len, vocab]
    # Shift: logits[t] predicts input_ids[t+1]
    shift_logits = logits[:, prompt_len - 1:-1, :]  # [G, completion_len, vocab]
    shift_labels = input_ids[:, prompt_len:]         # [G, completion_len]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [G, completion_len]

    # Mask padding (0 tokens)
    mask = (shift_labels != 0).float()
    return (token_log_probs * mask).sum(dim=-1)  # [G]


def compute_log_probs_trainable(
    model: ComplexityModel,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Same as compute_log_probs but with gradients enabled for the policy model.
    Uses last_hidden_state + manual logit computation since ComplexityModel
    skips logits in training mode.
    """
    model.train()
    outputs = model(input_ids, use_cache=False)

    # In training mode, ComplexityModel returns last_hidden_state, not logits
    hidden = outputs["last_hidden_state"]  # [G, seq_len, hidden]
    if model.lm_head is not None:
        logits = model.lm_head(hidden)
    else:
        logits = F.linear(hidden, model.embed_tokens.weight)

    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = input_ids[:, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    mask = (shift_labels != 0).float()
    return (token_log_probs * mask).sum(dim=-1)  # [G]


# ── GRPO Loss ────────────────────────────────────────────────────────

def grpo_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """
    GRPO clipped policy gradient loss with KL penalty.

    Args:
        policy_log_probs: [G] current policy log-probs
        ref_log_probs: [G] reference (frozen) model log-probs
        old_log_probs: [G] log-probs from when completions were sampled
        advantages: [G] group-relative advantages
        clip_eps: PPO clipping epsilon
        beta: KL penalty coefficient

    Returns:
        Scalar loss
    """
    # Importance sampling ratio
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL penalty: KL(π_θ || π_ref) ≈ (ref_log_prob - policy_log_prob)
    kl = (ref_log_probs - policy_log_probs).mean()

    return policy_loss + beta * kl


# ── Main training loop ───────────────────────────────────────────────

def train_grpo(
    model_path: str,
    dataset_name: str = "cais/mmlu",
    dataset_subset: str = "all",
    output_dir: str = "checkpoints/grpo",
    # GRPO
    group_size: int = 16,
    max_completion_length: int = 512,
    max_prompt_length: int = 512,
    clip_eps: float = 0.2,
    beta: float = 0.04,
    temperature: float = 0.7,
    # Training
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    max_steps: int = 2000,
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    # Data
    max_samples: int = 0,
    reward_type: str = "exact_match",
    # System
    log_steps: int = 10,
    save_steps: int = 100,
    bf16: bool = True,
):
    """Run GRPO training on MMLU QCM with ComplexityModel."""

    # ── Distributed setup ──
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    def log(msg):
        if rank == 0:
            print(msg)

    # ── Load model ──
    log(f"[grpo] Loading model from {model_path}...")
    model = ComplexityModel.from_pretrained(model_path, device=str(device))
    if bf16:
        model = model.to(torch.bfloat16)
    log(f"[grpo] {model.num_parameters() / 1e6:.1f}M params, mu_guidance={model._has_mu}")

    # ── Reference model (frozen copy) ──
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    log("[grpo] Reference model frozen")

    # ── Tokenizer ──
    from complexity.tokenizer import Tokenizer
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer as HFTokenizer
        hf_tok = HFTokenizer.from_file(tokenizer_path)

        class TokWrapper:
            def __init__(self, tok):
                self._tok = tok
                self.eos_token_id = tok.token_to_id("</s>") or tok.token_to_id("<|endoftext|>") or 0
            def encode(self, text, add_bos=False):
                ids = self._tok.encode(text).ids
                return ids
            def decode(self, ids):
                return self._tok.decode(ids)

        tokenizer = TokWrapper(hf_tok)
    else:
        tokenizer = Tokenizer.from_pretrained(model_path)
    log(f"[grpo] Tokenizer loaded (eos={tokenizer.eos_token_id})")

    # ── Dataset ──
    log(f"[grpo] Loading {dataset_name} (subset={dataset_subset})...")
    examples = load_mmlu_dataset(dataset_name, dataset_subset, max_samples=max_samples)
    log(f"[grpo] {len(examples)} examples")

    # ── Optimizer (AdamW) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=weight_decay,
    )

    # ── LR schedule: linear warmup + cosine decay ──
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Reward function ──
    if reward_type == "exact_match":
        reward_fn = mcq_exact_match_reward
    elif reward_type == "combined":
        reward_fn = combined_reward
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    # ── Training loop ──
    log(f"[grpo] G={group_size}, beta={beta}, clip={clip_eps}, lr={lr}")
    log(f"[grpo] reward={reward_type}, max_completion={max_completion_length}")
    log(f"[grpo] Starting training for {max_steps} steps...")

    os.makedirs(output_dir, exist_ok=True)
    model.train()
    global_step = 0
    total_reward = 0.0
    total_correct = 0
    total_completions = 0
    t0 = time.time()

    while global_step < max_steps:
        # Sample a random prompt
        idx = torch.randint(len(examples), (1,)).item()
        example = examples[idx]
        prompt = example["prompt"]
        answer = example["answer"]

        # Encode prompt
        prompt_ids = encode_prompt(tokenizer, prompt, max_prompt_length)
        prompt_len = prompt_ids.shape[0]

        # ── Step 1: Generate G completions from current policy ──
        with torch.no_grad():
            sequences = generate_group(
                model, prompt_ids, group_size,
                max_new_tokens=max_completion_length,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )  # [G, seq_len]

        # Decode completions (only the generated part)
        completions_text = []
        for g in range(group_size):
            comp_ids = sequences[g, prompt_len:]
            completions_text.append(decode_tokens(tokenizer, comp_ids))

        # ── Step 2: Compute rewards ──
        rewards = reward_fn(completions_text, answer)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

        # Group-relative advantage
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        advantages = (rewards_t - mean_r) / (std_r + 1e-8)

        # Skip if all rewards are the same (no signal)
        if std_r < 1e-8:
            global_step += 1
            scheduler.step()
            continue

        # ── Step 3: Compute log-probs ──
        # Old log-probs (from sampling policy — no grad)
        old_log_probs = compute_log_probs(model, sequences, prompt_len)

        # Reference log-probs (frozen model)
        ref_log_probs = compute_log_probs(ref_model, sequences, prompt_len)

        # ── Step 4: Policy gradient step ──
        policy_log_probs = compute_log_probs_trainable(model, sequences, prompt_len)

        loss = grpo_loss(
            policy_log_probs, ref_log_probs, old_log_probs,
            advantages, clip_eps=clip_eps, beta=beta,
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Logging ──
        global_step += 1
        batch_reward = sum(rewards)
        batch_correct = sum(1 for r in rewards if r >= 1.0)
        total_reward += batch_reward
        total_correct += batch_correct
        total_completions += group_size

        if global_step % log_steps == 0:
            elapsed = time.time() - t0
            avg_reward = total_reward / total_completions
            accuracy = total_correct / total_completions
            current_lr = scheduler.get_last_lr()[0]
            kl = (ref_log_probs - old_log_probs).mean().item()
            log(
                f"step {global_step:5d} | loss {loss.item():.4f} | "
                f"reward {mean_r.item():.3f} | acc {accuracy:.3f} | "
                f"kl {kl:.4f} | lr {current_lr:.2e} | "
                f"{elapsed / global_step:.1f}s/step"
            )

        # ── Save checkpoint ──
        if global_step % save_steps == 0 and rank == 0:
            ckpt_dir = os.path.join(output_dir, f"step-{global_step}")
            model.save_pretrained(ckpt_dir)
            log(f"[grpo] Checkpoint saved → {ckpt_dir}")

    # ── Final save ──
    if rank == 0:
        final_dir = os.path.join(output_dir, "final")
        model.save_pretrained(final_dir)
        log(f"[grpo] Training done. Final model → {final_dir}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO for ComplexityModel")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cais/mmlu")
    parser.add_argument("--dataset_subset", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--reward_type", type=str, default="exact_match",
                        choices=["exact_match", "combined"])
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    train_grpo(
        model_path=args.model_path,
        dataset_name=args.dataset,
        dataset_subset=args.dataset_subset,
        output_dir=args.output_dir,
        group_size=args.group_size,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        clip_eps=args.clip_eps,
        beta=args.beta,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        max_samples=args.max_samples,
        reward_type=args.reward_type,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
    )


if __name__ == "__main__":
    main()
