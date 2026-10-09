"""
Core evaluation utilities.

Provides:
- Perplexity computation
- Accuracy metrics
- Loss computation
- Evaluation loops
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class EvalConfig:
    """Evaluation settings; explicit LM labels are pre-shifted by default.

    Set ``labels_are_shifted=False`` for full labels aligned with input_ids,
    as supplied by Hugging Face's usual causal-language-model collators.
    """

    batch_size: int = 8
    max_length: int = 2048
    stride: int = 512  # For sliding window perplexity
    num_samples: Optional[int] = None  # Limit evaluation samples
    verbose: bool = True
    dtype: torch.dtype = torch.float16
    labels_are_shifted: bool = True


def compute_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    stride: int = 512,
) -> float:
    """
    Compute perplexity using sliding window.

    Args:
        model: Language model
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask
        stride: Sliding window stride

    Returns:
        Perplexity value
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")

    model.eval()
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else input_ids.device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("inf")

    config = getattr(model, "config", None)
    context = (
        config.get("max_position_embeddings", seq_len)
        if isinstance(config, dict)
        else getattr(config, "max_position_embeddings", seq_len)
    )
    if not isinstance(context, int) or isinstance(context, bool) or context < 2:
        raise ValueError("Model context length must be an integer of at least two")
    stride = min(stride, context - 1)
    total_nll, total_tokens = 0.0, 0
    next_target = 1
    with torch.no_grad():
        while next_target < seq_len:
            end = min(next_target + stride, seq_len)
            begin = max(0, end - context)
            chunk_ids = input_ids[:, begin:end]
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask[:, begin:end]
            outputs = model(chunk_ids, **kwargs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            if hasattr(logits, "logits"):
                logits = logits.logits
            targets = chunk_ids[:, 1:].clone()
            # Retained context is not scored twice. Every new target has a
            # preceding input, including targets at window boundaries.
            targets[:, : max(0, next_target - begin - 1)] = -100
            if attention_mask is not None:
                mask = attention_mask[:, begin:end].bool()
                targets.masked_fill_(~(mask[:, :-1] & mask[:, 1:]), -100)
            count = int((targets != -100).sum().item())
            if count:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_nll += loss.item()
                total_tokens += count
            next_target = end
    return math.exp(total_nll / total_tokens) if total_tokens else float("inf")


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.

    Args:
        predictions: Model predictions [batch, seq, vocab] or [batch, seq]
        labels: Ground truth labels [batch, seq]
        ignore_index: Index to ignore in computation

    Returns:
        Accuracy value
    """
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)

    mask = labels != ignore_index
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def compute_f1(
    predictions: List[str],
    references: List[str],
    tokenize: bool = True,
) -> Dict[str, float]:
    """
    Compute F1 score between predictions and references.

    Args:
        predictions: List of predicted strings
        references: List of reference strings
        tokenize: Whether to tokenize strings

    Returns:
        Dictionary with precision, recall, F1
    """
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for pred, ref in zip(predictions, references):
        if tokenize:
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
        else:
            pred_tokens = set(pred)
            ref_tokens = set(ref)

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    n = len(predictions)
    return {
        "precision": total_precision / n if n > 0 else 0,
        "recall": total_recall / n if n > 0 else 0,
        "f1": total_f1 / n if n > 0 else 0,
    }


class Evaluator:
    """
    Comprehensive model evaluator.

    Supports:
    - Perplexity computation
    - Accuracy metrics
    - Loss tracking
    - Multiple choice evaluation

    Example:
        evaluator = Evaluator(model, tokenizer)

        # Perplexity on dataset
        ppl = evaluator.perplexity(test_dataloader)

        # Multiple choice accuracy
        acc = evaluator.multiple_choice_accuracy(questions)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        config: Optional[EvalConfig] = None,
    ):
        """
        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for encoding/decoding
            config: Evaluation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvalConfig()

        self.device = next(model.parameters()).device
        self.model.eval()

    def _align_lm_targets(self, batch, logits):
        labels = batch.get("labels")
        if labels is None:
            targets = batch["input_ids"].to(self.device)[..., 1:]
            aligned = logits[..., :-1, :]
        elif self.config.labels_are_shifted:
            targets = labels.to(self.device)
            if targets.shape[:-1] != logits.shape[:-2] or targets.size(-1) not in {
                logits.size(-2),
                logits.size(-2) - 1,
            }:
                raise ValueError("LM labels must be pre-shifted with sequence length S or S-1")
            aligned = logits[..., : targets.size(-1), :]
        else:
            if labels.shape != batch["input_ids"].shape:
                raise ValueError("Unshifted LM labels must have the same shape as input_ids")
            targets = labels.to(self.device)[..., 1:]
            aligned = logits[..., :-1, :]
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.shape != batch["input_ids"].shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            mask = attention_mask.to(self.device).bool()
            if labels is not None and self.config.labels_are_shifted:
                mask = mask[..., : targets.size(-1)]
            else:
                mask = mask[..., :-1] & mask[..., 1:]
            targets = targets.masked_fill(~mask, -100)
        return aligned.contiguous(), targets.contiguous()

    def perplexity(
        self,
        dataloader: Iterator,
        max_samples: Optional[int] = None,
    ) -> float:
        """
        Compute perplexity on a dataset.

        Args:
            dataloader: DataLoader yielding batches
            max_samples: Maximum samples to evaluate

        Returns:
            Perplexity value
        """
        total_loss = 0
        total_tokens = 0
        num_samples = 0

        max_samples = max_samples or self.config.num_samples

        iterator = (
            tqdm(dataloader, desc="Computing perplexity") if self.config.verbose else dataloader
        )

        with torch.no_grad():
            for batch in iterator:
                if max_samples and num_samples >= max_samples:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)

                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Compute loss
                shift_logits, shift_labels = self._align_lm_targets(batch, logits)

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                # Count valid tokens
                valid_tokens = (shift_labels != -100).sum()

                total_loss += loss.item()
                total_tokens += valid_tokens.item()
                num_samples += input_ids.size(0)

        ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        return ppl

    def accuracy(
        self,
        dataloader: Iterator,
        max_samples: Optional[int] = None,
    ) -> float:
        """
        Compute token-level accuracy on a dataset.

        Args:
            dataloader: DataLoader yielding batches
            max_samples: Maximum samples to evaluate

        Returns:
            Accuracy value
        """
        total_correct = 0
        total_tokens = 0
        num_samples = 0

        max_samples = max_samples or self.config.num_samples

        iterator = (
            tqdm(dataloader, desc="Computing accuracy") if self.config.verbose else dataloader
        )

        with torch.no_grad():
            for batch in iterator:
                if max_samples and num_samples >= max_samples:
                    break

                input_ids = batch["input_ids"].to(self.device)
                # Forward pass
                mask = batch.get("attention_mask")
                kwargs = {} if mask is None else {"attention_mask": mask.to(self.device)}
                outputs = self.model(input_ids, **kwargs)

                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Compute accuracy
                aligned_logits, targets = self._align_lm_targets(batch, logits)
                predictions = aligned_logits.argmax(dim=-1)

                mask = targets != -100
                correct = (predictions == targets) & mask

                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                num_samples += input_ids.size(0)

        return total_correct / total_tokens if total_tokens > 0 else 0

    def multiple_choice_accuracy(
        self,
        questions: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate multiple choice questions.

        Args:
            questions: List of questions with format:
                {
                    "context": str,
                    "choices": List[str],
                    "answer": int,  # Index of correct answer
                }

        Returns:
            Accuracy on multiple choice questions
        """
        correct = 0
        total = 0

        iterator = tqdm(questions, desc="Multiple choice") if self.config.verbose else questions

        with torch.no_grad():
            for q in iterator:
                context = q["context"]
                choices = q["choices"]
                answer = q["answer"]

                # Compute log probability for each choice
                log_probs = []

                for choice in choices:
                    full_text = context + choice

                    if self.tokenizer:
                        tokens = self.tokenizer.encode(full_text, return_tensors="pt")
                        context_tokens = self.tokenizer.encode(context, return_tensors="pt")
                    else:
                        # Assume pre-tokenized
                        tokens = torch.tensor([[ord(c) for c in full_text]])
                        context_tokens = torch.tensor([[ord(c) for c in context]])

                    tokens = tokens.to(self.device)
                    context_len = context_tokens.size(1)

                    # Forward pass
                    outputs = self.model(tokens)

                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs

                    # Compute log prob of choice tokens
                    choice_logits = logits[0, context_len - 1 : -1, :]
                    choice_tokens = tokens[0, context_len:]

                    log_probs_per_token = F.log_softmax(choice_logits, dim=-1)
                    choice_log_prob = (
                        log_probs_per_token.gather(1, choice_tokens.unsqueeze(1)).sum().item()
                    )

                    log_probs.append(choice_log_prob)

                # Select best choice
                predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])

                if predicted == answer:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0

    def generate_and_evaluate(
        self,
        prompts: List[str],
        references: List[str],
        max_new_tokens: int = 100,
        metrics: List[str] = ["f1", "exact_match"],
    ) -> Dict[str, float]:
        """
        Generate text and evaluate against references.

        Args:
            prompts: Input prompts
            references: Reference outputs
            max_new_tokens: Maximum tokens to generate
            metrics: Metrics to compute

        Returns:
            Dictionary of metric scores
        """
        generations = []

        iterator = tqdm(prompts, desc="Generating") if self.config.verbose else prompts

        with torch.no_grad():
            for prompt in iterator:
                if self.tokenizer:
                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                else:
                    input_ids = torch.tensor([[ord(c) for c in prompt]])

                input_ids = input_ids.to(self.device)

                # Simple greedy generation
                generated = input_ids.clone()

                for _ in range(max_new_tokens):
                    outputs = self.model(generated)

                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs

                    next_token = logits[:, -1, :].argmax(dim=-1)
                    generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

                # Decode
                if self.tokenizer:
                    text = self.tokenizer.decode(generated[0, input_ids.size(1) :])
                else:
                    text = "".join(chr(t) for t in generated[0, input_ids.size(1) :].tolist())

                generations.append(text)

        # Compute metrics
        results = {}

        if "f1" in metrics:
            f1_scores = compute_f1(generations, references)
            results.update(f1_scores)

        if "exact_match" in metrics:
            exact_matches = sum(
                1 for g, r in zip(generations, references) if g.strip().lower() == r.strip().lower()
            )
            results["exact_match"] = exact_matches / len(generations)

        return results

    def calibration_error(
        self,
        dataloader: Iterator,
        num_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Measures how well model confidence matches accuracy.

        Args:
            dataloader: DataLoader yielding batches
            num_bins: Number of confidence bins

        Returns:
            ECE value
        """
        confidences = []
        correct = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch.get("attention_mask")
                kwargs = {} if mask is None else {"attention_mask": mask.to(self.device)}
                outputs = self.model(input_ids, **kwargs)

                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Get predictions and confidences
                aligned_logits, targets = self._align_lm_targets(batch, logits)
                probs = F.softmax(aligned_logits, dim=-1)
                max_probs, preds = probs.max(dim=-1)

                # Flatten
                mask = targets != -100
                confidences.extend(max_probs[mask].cpu().tolist())
                correct.extend((preds[mask] == targets[mask]).cpu().tolist())

        # Compute ECE
        if not confidences:
            raise ValueError("Calibration requires at least one supervised target")
        confidences = torch.tensor(confidences)
        correct = torch.tensor(correct, dtype=torch.float)

        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0

        for i in range(num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = correct[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

        return ece.item()


def evaluate_on_dataset(
    model: nn.Module,
    dataloader: Iterator,
    tokenizer: Optional[Any] = None,
    metrics: List[str] = ["perplexity", "accuracy"],
) -> Dict[str, float]:
    """
    Convenience function to evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        tokenizer: Optional tokenizer
        metrics: Metrics to compute

    Returns:
        Dictionary of metric values
    """
    evaluator = Evaluator(model, tokenizer)
    results = {}

    if "perplexity" in metrics:
        results["perplexity"] = evaluator.perplexity(dataloader)

    if "accuracy" in metrics:
        results["accuracy"] = evaluator.accuracy(dataloader)

    return results
