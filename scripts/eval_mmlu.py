"""
Quick MMLU evaluation for ComplexityModel checkpoints.

Measures log-likelihood of each choice (A/B/C/D), picks the best.
Standard 0-shot evaluation like lm-eval-harness.

Usage:
    python scripts/eval_mmlu.py --model_path checkpoints/dapo-step-1700
    python scripts/eval_mmlu.py --model_path checkpoints/400m  # baseline
"""

import argparse
import logging
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from complexity.config import ModelConfig
from complexity.models.builder import ComplexityModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHOICES = ["A", "B", "C", "D"]


def load_tokenizer(model_path):
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer as HFTokenizer
        hf_tok = HFTokenizer.from_file(tokenizer_path)

        class TokWrapper:
            def __init__(self, tok):
                self._tok = tok
            def encode(self, text):
                return self._tok.encode(text).ids
            def decode(self, ids):
                return self._tok.decode(ids)

        return TokWrapper(hf_tok)
    raise FileNotFoundError(f"No tokenizer found at {model_path}")


@torch.no_grad()
def compute_choice_logprob(model, tokenizer, prompt: str, choice: str, device) -> float:
    """Compute log-probability of choice given prompt."""
    prompt_ids = tokenizer.encode(prompt)
    choice_ids = tokenizer.encode(choice)

    full_ids = torch.tensor(prompt_ids + choice_ids, device=device).unsqueeze(0)

    model.eval()
    outputs = model(full_ids, use_cache=False)
    logits = outputs["logits"]  # [1, seq_len, vocab]

    # Log-probs for the choice tokens only
    start = len(prompt_ids) - 1  # shifted by 1
    end = start + len(choice_ids)
    shift_logits = logits[0, start:end, :]
    shift_labels = full_ids[0, len(prompt_ids):len(prompt_ids) + len(choice_ids)]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()


def format_question(question: str, choices: list, subject: str = "") -> str:
    choices_text = "\n".join(f"{CHOICES[i]}. {c}" for i, c in enumerate(choices))
    subject_str = f" ({subject})" if subject else ""
    return f"Question{subject_str}: {question}\n\n{choices_text}\n\nAnswer:"


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--subjects", type=str, default="all", help="'all' or comma-separated")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_per_subject", type=int, default=0, help="0=all")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = ComplexityModel.from_pretrained(args.model_path, device=str(device))
    if args.bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    logger.info(f"{model.num_parameters() / 1e6:.1f}M params")

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_path)

    # Load dataset
    logger.info("Loading MMLU...")
    ds = load_dataset("cais/mmlu", "all", split=args.split)
    logger.info(f"{len(ds)} questions")

    # Filter subjects if needed
    if args.subjects != "all":
        subjects = set(args.subjects.split(","))
        ds = ds.filter(lambda x: x.get("subject", "") in subjects)
        logger.info(f"Filtered to {len(ds)} questions")

    # Evaluate
    correct = 0
    total = 0
    per_subject = {}

    for row in tqdm(ds, desc="MMLU eval"):
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        subject = row.get("subject", "unknown")

        if isinstance(answer_idx, str):
            answer_idx = CHOICES.index(answer_idx.upper())

        if args.max_per_subject > 0:
            count = per_subject.get(subject, {}).get("total", 0)
            if count >= args.max_per_subject:
                continue

        prompt = format_question(question, choices, subject)

        # Score each choice
        log_probs = []
        for i in range(4):
            lp = compute_choice_logprob(model, tokenizer, prompt, f" {CHOICES[i]}", device)
            log_probs.append(lp)

        predicted = max(range(4), key=lambda i: log_probs[i])
        is_correct = predicted == answer_idx

        correct += int(is_correct)
        total += 1

        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["correct"] += int(is_correct)
        per_subject[subject]["total"] += 1

    # Results
    accuracy = correct / total if total > 0 else 0
    logger.info(f"\n{'='*50}")
    logger.info(f"MMLU Accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"{'='*50}")

    # Per-subject breakdown (top 10 and bottom 10)
    subject_accs = {
        s: d["correct"] / d["total"]
        for s, d in per_subject.items()
        if d["total"] > 0
    }
    sorted_subjects = sorted(subject_accs.items(), key=lambda x: x[1], reverse=True)

    logger.info("\nTop 10 subjects:")
    for s, a in sorted_subjects[:10]:
        n = per_subject[s]["total"]
        logger.info(f"  {s:40s} {a:.3f} ({per_subject[s]['correct']}/{n})")

    logger.info("\nBottom 10 subjects:")
    for s, a in sorted_subjects[-10:]:
        n = per_subject[s]["total"]
        logger.info(f"  {s:40s} {a:.3f} ({per_subject[s]['correct']}/{n})")


if __name__ == "__main__":
    main()
