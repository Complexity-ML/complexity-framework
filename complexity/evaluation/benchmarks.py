"""
Standard LLM benchmarks.

Implements evaluation on popular benchmarks:
- MMLU: Massive Multitask Language Understanding
- HellaSwag: Commonsense reasoning
- Winograde: Coreference resolution
- ARC: Science reasoning
- TruthfulQA: Truthfulness evaluation

Reference implementations aligned with:
- lm-evaluation-harness
- EleutherAI/lm-eval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""
    name: str
    accuracy: float
    num_correct: int
    num_total: int
    per_category: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    name: str = "base"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @abstractmethod
    def load_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load benchmark data."""
        pass

    @abstractmethod
    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a single sample."""
        pass

    def run(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run benchmark evaluation.

        Args:
            split: Data split to evaluate on
            num_samples: Maximum samples to evaluate
            verbose: Show progress bar

        Returns:
            BenchmarkResult with scores
        """
        data = self.load_data(split)

        if num_samples:
            data = data[:num_samples]

        correct = 0
        total = 0
        per_category = {}

        iterator = tqdm(data, desc=f"Evaluating {self.name}") if verbose else data

        with torch.no_grad():
            for sample in iterator:
                is_correct, info = self.evaluate_sample(sample)

                if is_correct:
                    correct += 1
                total += 1

                # Track per-category
                category = sample.get("category", "default")
                if category not in per_category:
                    per_category[category] = {"correct": 0, "total": 0}
                per_category[category]["total"] += 1
                if is_correct:
                    per_category[category]["correct"] += 1

        # Compute per-category accuracy
        category_acc = {
            cat: stats["correct"] / stats["total"]
            for cat, stats in per_category.items()
        }

        return BenchmarkResult(
            name=self.name,
            accuracy=correct / total if total > 0 else 0,
            num_correct=correct,
            num_total=total,
            per_category=category_acc,
        )

    def _compute_choice_logprob(
        self,
        context: str,
        choice: str,
    ) -> float:
        """Compute log probability of a choice given context."""
        full_text = context + choice

        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Get choice tokens
        choice_ids = full_ids[len(context_ids):]

        if len(choice_ids) == 0:
            return float('-inf')

        # Prepare input
        input_ids = torch.tensor([full_ids], device=self.device)

        # Forward pass
        outputs = self.model(input_ids)

        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        # Get log probs for choice tokens
        log_probs = F.log_softmax(logits[0], dim=-1)

        total_log_prob = 0
        for i, token_id in enumerate(choice_ids):
            pos = len(context_ids) + i - 1
            if pos >= 0:
                total_log_prob += log_probs[pos, token_id].item()

        return total_log_prob


class MMLUBenchmark(Benchmark):
    """
    MMLU: Massive Multitask Language Understanding.

    57 subjects across STEM, humanities, social sciences, etc.
    Tests knowledge and reasoning.
    """

    name = "mmlu"

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        data_path: Optional[str] = None,
        num_few_shot: int = 5,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.data_path = data_path
        self.num_few_shot = num_few_shot

    def load_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load MMLU data."""
        samples = []

        # Generate sample data if path not provided
        if self.data_path is None:
            logger.warning("No data path provided, using dummy data")
            for subject in self.SUBJECTS[:3]:  # Just 3 subjects for demo
                for i in range(10):
                    samples.append({
                        "question": f"Sample {subject} question {i}?",
                        "choices": ["A", "B", "C", "D"],
                        "answer": i % 4,
                        "category": subject,
                    })
            return samples

        # Load from files
        for subject in self.SUBJECTS:
            file_path = os.path.join(self.data_path, f"{subject}_{split}.jsonl")
            if os.path.exists(file_path):
                with open(file_path) as f:
                    for line in f:
                        data = json.loads(line)
                        data["category"] = subject
                        samples.append(data)

        return samples

    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a single MMLU sample."""
        question = sample["question"]
        choices = sample["choices"]
        answer = sample["answer"]

        # Format prompt
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"

        # Compute log prob for each choice
        log_probs = []
        for i in range(len(choices)):
            choice_text = f" {chr(65 + i)}"
            log_prob = self._compute_choice_logprob(prompt, choice_text)
            log_probs.append(log_prob)

        # Select best
        predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
        is_correct = predicted == answer

        return is_correct, {"predicted": predicted, "log_probs": log_probs}


class HellaSwagBenchmark(Benchmark):
    """
    HellaSwag: Commonsense reasoning benchmark.

    Complete sentences with most plausible continuation.
    """

    name = "hellaswag"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.data_path = data_path

    def load_data(self, split: str = "validation") -> List[Dict[str, Any]]:
        """Load HellaSwag data."""
        samples = []

        if self.data_path is None:
            logger.warning("No data path provided, using dummy data")
            for i in range(100):
                samples.append({
                    "context": f"A person is doing activity {i}. They",
                    "endings": [
                        f"do action A for {i}",
                        f"do action B for {i}",
                        f"do action C for {i}",
                        f"do action D for {i}",
                    ],
                    "label": i % 4,
                    "category": "default",
                })
            return samples

        # Load from file
        file_path = os.path.join(self.data_path, f"hellaswag_{split}.jsonl")
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f:
                    samples.append(json.loads(line))

        return samples

    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a single HellaSwag sample."""
        context = sample["context"]
        endings = sample["endings"]
        label = sample["label"]

        # Compute log prob for each ending
        log_probs = []
        for ending in endings:
            log_prob = self._compute_choice_logprob(context, " " + ending)
            log_probs.append(log_prob)

        predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
        is_correct = predicted == label

        return is_correct, {"predicted": predicted}


class WinogradeBenchmark(Benchmark):
    """
    Winograde: Coreference resolution benchmark.

    Resolve pronoun references in sentences.
    """

    name = "winograde"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.data_path = data_path

    def load_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load Winograde data."""
        samples = []

        if self.data_path is None:
            logger.warning("No data path provided, using dummy data")
            samples = [
                {
                    "sentence": "The trophy doesn't fit in the brown suitcase because it is too big.",
                    "pronoun": "it",
                    "choices": ["trophy", "suitcase"],
                    "answer": 0,
                },
                {
                    "sentence": "The trophy doesn't fit in the brown suitcase because it is too small.",
                    "pronoun": "it",
                    "choices": ["trophy", "suitcase"],
                    "answer": 1,
                },
            ] * 50
            return samples

        file_path = os.path.join(self.data_path, f"winograde_{split}.jsonl")
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f:
                    samples.append(json.loads(line))

        return samples

    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a single Winograde sample."""
        sentence = sample["sentence"]
        choices = sample["choices"]
        answer = sample["answer"]

        # Replace pronoun with each choice and compute probability
        pronoun = sample.get("pronoun", "it")

        log_probs = []
        for choice in choices:
            modified = sentence.replace(pronoun, choice, 1)
            # Compute sentence probability
            input_ids = self.tokenizer.encode(modified, return_tensors='pt')
            input_ids = input_ids.to(self.device)

            outputs = self.model(input_ids)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            log_probs_all = F.log_softmax(logits[0, :-1], dim=-1)
            targets = input_ids[0, 1:]
            total_log_prob = log_probs_all.gather(1, targets.unsqueeze(1)).sum().item()
            log_probs.append(total_log_prob)

        predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
        is_correct = predicted == answer

        return is_correct, {"predicted": predicted}


class ARCBenchmark(Benchmark):
    """
    ARC: AI2 Reasoning Challenge.

    Science exam questions requiring reasoning.
    Two splits: Easy and Challenge.
    """

    name = "arc"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        data_path: Optional[str] = None,
        challenge: bool = True,  # Use Challenge set
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.data_path = data_path
        self.challenge = challenge

    def load_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load ARC data."""
        samples = []

        if self.data_path is None:
            logger.warning("No data path provided, using dummy data")
            for i in range(100):
                samples.append({
                    "question": f"Science question {i}: What causes X?",
                    "choices": ["Answer A", "Answer B", "Answer C", "Answer D"],
                    "answer": i % 4,
                    "category": "science",
                })
            return samples

        subset = "Challenge" if self.challenge else "Easy"
        file_path = os.path.join(self.data_path, f"ARC-{subset}-{split}.jsonl")
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f:
                    samples.append(json.loads(line))

        return samples

    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a single ARC sample."""
        question = sample["question"]
        choices = sample["choices"]
        answer = sample["answer"]

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"

        log_probs = []
        for i in range(len(choices)):
            choice_text = f" {chr(65 + i)}"
            log_prob = self._compute_choice_logprob(prompt, choice_text)
            log_probs.append(log_prob)

        predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
        is_correct = predicted == answer

        return is_correct, {"predicted": predicted}


class TruthfulQABenchmark(Benchmark):
    """
    TruthfulQA: Truthfulness evaluation benchmark.

    Tests if models give truthful answers vs common misconceptions.
    """

    name = "truthfulqa"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.data_path = data_path

    def load_data(self, split: str = "validation") -> List[Dict[str, Any]]:
        """Load TruthfulQA data."""
        samples = []

        if self.data_path is None:
            logger.warning("No data path provided, using dummy data")
            samples = [
                {
                    "question": "What happens if you crack your knuckles?",
                    "correct_answers": ["Nothing harmful", "It makes a sound"],
                    "incorrect_answers": ["You get arthritis", "Your fingers swell"],
                    "category": "Health",
                },
            ] * 100
            return samples

        file_path = os.path.join(self.data_path, f"truthfulqa_{split}.jsonl")
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f:
                    samples.append(json.loads(line))

        return samples

    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a single TruthfulQA sample."""
        question = sample["question"]
        correct = sample["correct_answers"]
        incorrect = sample["incorrect_answers"]

        prompt = f"Q: {question}\nA:"

        # Compute log prob for correct and incorrect answers
        correct_probs = [self._compute_choice_logprob(prompt, " " + ans) for ans in correct]
        incorrect_probs = [self._compute_choice_logprob(prompt, " " + ans) for ans in incorrect]

        # Model is truthful if max correct > max incorrect
        max_correct = max(correct_probs) if correct_probs else float('-inf')
        max_incorrect = max(incorrect_probs) if incorrect_probs else float('-inf')

        is_truthful = max_correct > max_incorrect

        return is_truthful, {
            "max_correct_prob": max_correct,
            "max_incorrect_prob": max_incorrect,
        }


class BenchmarkRunner:
    """
    Runs multiple benchmarks and aggregates results.

    Example:
        runner = BenchmarkRunner(model, tokenizer)

        # Run all benchmarks
        results = runner.run_all()

        # Run specific benchmark
        mmlu_result = runner.run_benchmark("mmlu")
    """

    BENCHMARKS = {
        "mmlu": MMLUBenchmark,
        "hellaswag": HellaSwagBenchmark,
        "winograde": WinogradeBenchmark,
        "arc": ARCBenchmark,
        "truthfulqa": TruthfulQABenchmark,
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        data_paths: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            model: Language model
            tokenizer: Tokenizer
            data_paths: Mapping of benchmark name to data path
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_paths = data_paths or {}

    def run_benchmark(
        self,
        name: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Run a specific benchmark."""
        if name not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {name}. Available: {list(self.BENCHMARKS.keys())}")

        benchmark_cls = self.BENCHMARKS[name]
        data_path = self.data_paths.get(name)

        benchmark = benchmark_cls(
            self.model,
            self.tokenizer,
            data_path=data_path,
            **kwargs,
        )

        return benchmark.run(num_samples=num_samples)

    def run_all(
        self,
        benchmarks: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run all or specified benchmarks."""
        benchmarks = benchmarks or list(self.BENCHMARKS.keys())
        results = {}

        for name in benchmarks:
            logger.info(f"Running benchmark: {name}")
            results[name] = self.run_benchmark(name, num_samples=num_samples)
            logger.info(f"{name}: {results[name].accuracy:.4f}")

        return results

    def summarize(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate summary of benchmark results."""
        lines = ["=" * 50, "Benchmark Results Summary", "=" * 50]

        avg_acc = 0
        for name, result in results.items():
            lines.append(f"{name:20s}: {result.accuracy:.4f} ({result.num_correct}/{result.num_total})")
            avg_acc += result.accuracy

        avg_acc /= len(results) if results else 1
        lines.append("-" * 50)
        lines.append(f"{'Average':20s}: {avg_acc:.4f}")
        lines.append("=" * 50)

        return "\n".join(lines)
