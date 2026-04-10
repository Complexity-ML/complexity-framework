"""
DAPO (Decoupled Alignment from Policy Optimization) for ComplexityModel.

From-scratch implementation — no TRL dependency.
Works natively with mu-guidance, token-routed MoE, deterministic Zipf routing.

Improvements over GRPO:
  - No reference model (no KL penalty) → 2× less VRAM
  - Asymmetric clipping (clip_low < clip_high) → encourages exploration
  - Dynamic sampling: resample when all rewards are identical → no wasted steps
  - Overlong penalty: discourages excessively long completions

Algorithm (ByteDance, 2025):
    For each prompt:
      1. Sample G completions from policy π_θ
      2. Score with reward function r(completion, answer)
      3. If all rewards identical → resample (up to max_resample attempts)
      4. Advantage = (r - mean(r)) / (std(r) + eps)  [group-relative]
      5. Loss = -E[ min(ratio * A, clip_asym(ratio, ε_low, ε_high) * A) ]

Usage:
    python -m complexity.RL.grpo.train_grpo \
        --model_path checkpoints/400m-v1/final \
        --dataset cais/mmlu \
        --output_dir checkpoints/400m-dapo \
        --group_size 16 \
        --lr 5e-6
"""

import argparse
import csv
import logging
import math
import os
import time
from typing import List, Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset

from complexity.config import ModelConfig
from complexity.models.builder import ComplexityModel

from .rewards import mcq_exact_match_reward, combined_reward, language_quality_reward


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
    # cais/mmlu requires a config name; "all" loads every subject
    config = subset if subset != "all" else "all"
    ds = load_dataset(dataset_name, config, split=split)

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


def load_fineweb_edu_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    max_samples: int = 10000,
    prompt_words: int = 50,
) -> List[Dict]:
    """
    Load FineWeb-Edu and create prompts from document beginnings.
    Each example = first ~50 words as prompt, rest is what the model should continue.
    """
    ds = load_dataset(dataset_name, split=split, streaming=True)

    examples = []
    for row in ds:
        text = row.get("text", "")
        words = text.split()
        if len(words) < prompt_words + 20:
            continue
        prompt = " ".join(words[:prompt_words])
        examples.append({"prompt": prompt, "answer": ""})
        if len(examples) >= max_samples:
            break

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

def _init_vllm_engine(model_path: str, gpu_memory_utilization: float = 0.4):
    """Initialize vLLM LLM engine for fast generation."""
    from vllm import LLM
    return LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=False,  # Enable CUDA graphs
    )


def generate_group_vllm(
    vllm_engine,
    prompt_text: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> List[str]:
    """
    Generate G completions using vLLM (CUDA graphs, PagedAttention).
    Returns list of completion strings.
    """
    from vllm import SamplingParams
    params = SamplingParams(
        n=group_size,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    outputs = vllm_engine.generate([prompt_text], params)
    return [out.text for out in outputs[0].outputs]


@torch.no_grad()
def generate_group(
    model: ComplexityModel,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Fallback: generate G completions with native PyTorch (no vLLM).
    Used when --use_vllm is not set.
    """
    model.eval()
    device = next(model.parameters()).device
    prompt_len = prompt_ids.shape[0]

    input_ids = prompt_ids.unsqueeze(0).expand(group_size, -1).to(device)

    past_key_values = None
    for step in range(max_new_tokens):
        if past_key_values is None:
            outputs = model(input_ids, use_cache=True)
        else:
            outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs["past_key_values"]
        logits = outputs["logits"][:, -1, :]

        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids


# ── Log-probability computation ──────────────────────────────────────

@torch.no_grad()
def compute_log_probs(
    model: ComplexityModel,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token log-probs for completion portion (no grad)."""
    model.eval()
    outputs = model(input_ids, use_cache=False)

    logits = outputs["logits"]  # [G, seq_len, vocab]
    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = input_ids[:, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    mask = (shift_labels != 0).float()
    return (token_log_probs * mask).sum(dim=-1)  # [G]


def compute_log_probs_trainable(
    model: ComplexityModel,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute log-probs with gradients. Handles ComplexityModel training mode."""
    model.train()
    outputs = model(input_ids, use_cache=False)

    # In training mode, ComplexityModel returns last_hidden_state, not logits
    hidden = outputs["last_hidden_state"]
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


# ── DAPO Loss ────────────────────────────────────────────────────────

def dapo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.8,
    clip_high: float = 0.28,
    overlong_penalty: float = -0.5,
    completion_lengths: Optional[torch.Tensor] = None,
    max_completion_length: int = 256,
) -> torch.Tensor:
    """
    DAPO loss: asymmetric clipping, no reference model, overlong penalty.

    Args:
        policy_log_probs: [G] current policy log-probs
        old_log_probs: [G] log-probs from sampling
        advantages: [G] group-relative advantages
        clip_low: Lower clip bound (1 - clip_low). Default 0.8 → ratio clamped ≥ 0.2
        clip_high: Upper clip bound (1 + clip_high). Default 0.28 → ratio clamped ≤ 1.28
        overlong_penalty: Penalty added to advantage for completions exceeding max length
        completion_lengths: [G] actual completion lengths (for overlong penalty)
        max_completion_length: Max allowed length before penalty kicks in

    Returns:
        Scalar loss
    """
    # Importance sampling ratio
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # Asymmetric clipping — key DAPO innovation
    # Low clip is aggressive (allows ratio to drop to 0.2) → encourages exploration
    # High clip is moderate → prevents too-large updates
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Overlong penalty
    if completion_lengths is not None and overlong_penalty != 0:
        overlong_mask = (completion_lengths > max_completion_length).float()
        penalty = overlong_penalty * overlong_mask
        policy_loss = policy_loss - (ratio * penalty).mean()

    return policy_loss


# ── Main training loop ───────────────────────────────────────────────

def train_grpo(
    model_path: str,
    dataset_name: str = "cais/mmlu",
    dataset_subset: str = "all",
    output_dir: str = "checkpoints/dapo",
    # DAPO
    group_size: int = 16,
    max_completion_length: int = 256,
    max_prompt_length: int = 512,
    clip_low: float = 0.8,
    clip_high: float = 0.28,
    temperature: float = 1.0,
    max_resample: int = 3,
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
    # vLLM
    use_vllm: bool = False,
    vllm_gpu_memory: float = 0.4,
    # Compat (ignored, kept for CLI backwards compat)
    beta: float = 0.0,
    clip_eps: float = 0.0,
):
    """Run DAPO training on MMLU QCM with ComplexityModel."""

    # ── Distributed setup ──
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    def log(msg):
        if rank == 0:
            logger.info(msg)

    # ── Load model ──
    log(f"[dapo] Loading model from {model_path}...")
    model = ComplexityModel.from_pretrained(model_path, device=str(device))
    if bf16:
        model = model.to(torch.bfloat16)
    log(f"[dapo] {model.num_parameters() / 1e6:.1f}M params, mu_guidance={model._has_mu}")
    log(f"[dapo] No reference model (DAPO: clipping only, no KL)")

    # ── vLLM engine (for fast generation) ──
    vllm_engine = None
    if use_vllm:
        log(f"[dapo] Initializing vLLM engine (gpu_mem={vllm_gpu_memory})...")
        vllm_engine = _init_vllm_engine(model_path, gpu_memory_utilization=vllm_gpu_memory)
        log("[dapo] vLLM engine ready (CUDA graphs + PagedAttention)")
    else:
        log("[dapo] Using native PyTorch generation (use --use_vllm for faster sampling)")

    # ── Tokenizer ──
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
        from complexity.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(model_path)
    log(f"[dapo] Tokenizer loaded (eos={tokenizer.eos_token_id})")

    # ── Dataset ──
    log(f"[dapo] Loading {dataset_name} (subset={dataset_subset})...")
    if reward_type == "language_quality":
        examples = load_fineweb_edu_dataset(
            dataset_name=dataset_name, max_samples=max_samples or 10000,
        )
    else:
        examples = load_mmlu_dataset(dataset_name, dataset_subset, max_samples=max_samples)
    log(f"[dapo] {len(examples)} examples")

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
    elif reward_type == "language_quality":
        reward_fn = language_quality_reward
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    # ── Training loop ──
    log(f"[dapo] G={group_size}, clip_low={clip_low}, clip_high={clip_high}, lr={lr}")
    log(f"[dapo] reward={reward_type}, max_completion={max_completion_length}")
    log(f"[dapo] dynamic_sampling: max_resample={max_resample}")
    log(f"[dapo] Starting training for {max_steps} steps...")

    os.makedirs(output_dir, exist_ok=True)
    model.train()
    global_step = 0
    total_reward = 0.0
    total_correct = 0
    total_completions = 0
    skipped_resample = 0
    t0 = time.time()

    # ── CSV logging ──
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = None
    csv_writer = None
    if rank == 0:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss", "reward", "acc", "avg_reward", "avg_acc", "lr", "skipped", "s_per_step"])
        log(f"[dapo] Logging to {csv_path}")

    pbar = tqdm(total=max_steps, desc="DAPO", disable=(rank != 0))

    while global_step < max_steps:
        # Sample a random prompt
        idx = torch.randint(len(examples), (1,)).item()
        example = examples[idx]
        prompt = example["prompt"]
        answer = example["answer"]

        # Encode prompt
        prompt_ids = encode_prompt(tokenizer, prompt, max_prompt_length)
        prompt_len = prompt_ids.shape[0]

        # ── Step 1: Generate G completions — dynamic sampling ──
        # If all rewards are identical, resample with a new prompt (DAPO innovation)
        rewards = None
        sequences = None
        completions_text = None

        for resample_attempt in range(max_resample + 1):
            if vllm_engine is not None:
                # ── vLLM fast path ──
                completions_text = generate_group_vllm(
                    vllm_engine, prompt, group_size,
                    max_new_tokens=max_completion_length,
                    temperature=temperature,
                    top_p=0.9,
                )
                # Encode completions back to token IDs for log-prob computation
                all_ids = []
                for comp_text in completions_text:
                    comp_ids = tokenizer.encode(comp_text, add_bos=False)
                    full_ids = prompt_ids.tolist() + comp_ids
                    all_ids.append(full_ids)
                # Pad to same length
                max_len = max(len(ids) for ids in all_ids)
                for i in range(len(all_ids)):
                    all_ids[i] += [0] * (max_len - len(all_ids[i]))
                sequences = torch.tensor(all_ids, device=device, dtype=torch.long)
            else:
                # ── Native PyTorch fallback ──
                with torch.no_grad():
                    sequences = generate_group(
                        model, prompt_ids, group_size,
                        max_new_tokens=max_completion_length,
                        temperature=temperature,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                completions_text = []
                for g in range(group_size):
                    comp_ids = sequences[g, prompt_len:]
                    completions_text.append(decode_tokens(tokenizer, comp_ids))

            if reward_type == "language_quality":
                rewards = reward_fn(completions_text, prompt=prompt)
            else:
                rewards = reward_fn(completions_text, answer)
            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            std_r = rewards_t.std()

            if std_r > 1e-8:
                break  # Got variance in rewards — useful signal

            # No signal, try a different prompt
            if resample_attempt < max_resample:
                idx = torch.randint(len(examples), (1,)).item()
                example = examples[idx]
                prompt = example["prompt"]
                answer = example["answer"]
                prompt_ids = encode_prompt(tokenizer, prompt, max_prompt_length)
                prompt_len = prompt_ids.shape[0]

        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()

        # Still no signal after resampling — skip
        if std_r < 1e-8:
            skipped_resample += 1
            global_step += 1
            scheduler.step()
            continue

        # ── Step 2: Group-relative advantage ──
        advantages = (rewards_t - mean_r) / (std_r + 1e-8)

        # ── Step 3: Compute log-probs ──
        old_log_probs = compute_log_probs(model, sequences, prompt_len)

        # ── Step 4: DAPO policy gradient step ──
        policy_log_probs = compute_log_probs_trainable(model, sequences, prompt_len)

        # Completion lengths for overlong penalty
        comp_lengths = torch.tensor(
            [len(sequences[g, prompt_len:].nonzero()) for g in range(group_size)],
            device=device, dtype=torch.float32,
        )

        loss = dapo_loss(
            policy_log_probs, old_log_probs, advantages,
            clip_low=clip_low, clip_high=clip_high,
            completion_lengths=comp_lengths,
            max_completion_length=max_completion_length,
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Logging ──
        global_step += 1
        batch_correct = sum(1 for r in rewards if r >= 1.0)
        total_reward += sum(rewards)
        total_correct += batch_correct
        total_completions += group_size

        accuracy = total_correct / total_completions
        avg_reward = total_reward / total_completions
        current_lr = scheduler.get_last_lr()[0]

        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "reward": f"{mean_r.item():.3f}",
            "acc": f"{accuracy:.3f}",
            "lr": f"{current_lr:.2e}",
            "skip": skipped_resample,
        })

        # Write to CSV every step
        if csv_writer is not None:
            elapsed = time.time() - t0
            csv_writer.writerow([
                global_step,
                f"{loss.item():.6f}",
                f"{mean_r.item():.4f}",
                f"{batch_correct / group_size:.4f}",
                f"{avg_reward:.4f}",
                f"{accuracy:.4f}",
                f"{current_lr:.2e}",
                skipped_resample,
                f"{elapsed / global_step:.2f}",
            ])
            csv_file.flush()

        # ── Save checkpoint ──
        if global_step % save_steps == 0 and rank == 0:
            ckpt_dir = os.path.join(output_dir, f"step-{global_step}")
            model.save_pretrained(ckpt_dir)
            log(f"[dapo] Checkpoint saved → {ckpt_dir}")

    pbar.close()
    if csv_file is not None:
        csv_file.close()

    # ── Final save ──
    if rank == 0:
        final_dir = os.path.join(output_dir, "final")
        model.save_pretrained(final_dir)
        accuracy = total_correct / max(1, total_completions)
        log(f"[dapo] Training done. Final acc={accuracy:.3f}. Model → {final_dir}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="DAPO for ComplexityModel")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cais/mmlu")
    parser.add_argument("--dataset_subset", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="checkpoints/dapo")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--clip_low", type=float, default=0.8)
    parser.add_argument("--clip_high", type=float, default=0.28)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_resample", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--reward_type", type=str, default="exact_match",
                        choices=["exact_match", "combined", "language_quality"])
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true", default=True)
    # vLLM
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for fast generation")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.4)
    # Backwards compat (ignored by DAPO)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--clip_eps", type=float, default=0.0)
    args = parser.parse_args()

    train_grpo(
        model_path=args.model_path,
        dataset_name=args.dataset,
        dataset_subset=args.dataset_subset,
        output_dir=args.output_dir,
        group_size=args.group_size,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        temperature=args.temperature,
        max_resample=args.max_resample,
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
        use_vllm=args.use_vllm,
        vllm_gpu_memory=args.vllm_gpu_memory,
    )


if __name__ == "__main__":
    main()
