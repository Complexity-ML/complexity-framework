#!/usr/bin/env python3
"""Evaluate ARC by generating reasoning traces and parsing a final answer.

This intentionally complements (and does not replace) the usual multiple-choice
log-likelihood evaluation.  Every model receives the same user instruction and
decoding parameters.  The complete completion is kept in JSONL so formatting
failures and answer extraction remain auditable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class ARCExample:
    task: str
    doc_id: int
    example_id: str
    question: str
    choices: tuple[str, ...]
    answer_index: int

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(chr(ord("A") + index) for index in range(len(self.choices)))

    @property
    def answer(self) -> str:
        return self.labels[self.answer_index]


class Generator(Protocol):
    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str: ...


NATIVE_RECIPES = {
    "nautile_torch": {
        "max_new_tokens": 512,
        "temperature": 0.15,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "tr_hash_mlx": {
        "max_new_tokens": 512,
        "temperature": 0.30,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.05,
    },
    "tr_hash_mlx_constrained": {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
    },
    "tr_hash_mlx_open": {
        "max_new_tokens": 64,
        "temperature": 0.30,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.05,
    },
    "tr_hash_torch": {
        "max_new_tokens": 256,
        # Checkpoint ranking must not depend on one lucky sampling seed.  The
        # released interactive sampling recipe is evaluated separately.
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.05,
    },
    "tr_hash_torch_direct": {
        "max_new_tokens": 64,
        "temperature": 0.30,
        "top_p": 0.90,
        "top_k": 30,
        "repetition_penalty": 1.05,
    },
    "tr_hash_torch_constrained": {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
    },
}


def apply_native_recipe(args: argparse.Namespace) -> None:
    """Fill unspecified decoding values with backend-specific public defaults."""

    recipe = NATIVE_RECIPES[args.backend]
    for name, value in recipe.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def load_lm_eval_samples(path: Path, task: str) -> list[ARCExample]:
    """Read the dataset documents preserved by ``lm_eval --log_samples``."""

    examples: list[ARCExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc = row["doc"]
            original_labels = [str(label) for label in doc["choices"]["label"]]
            answer_key = str(doc["answerKey"])
            try:
                answer_index = original_labels.index(answer_key)
            except ValueError as error:
                raise ValueError(
                    f"answer key {answer_key!r} is absent from {original_labels!r} "
                    f"in {path}:{row['doc_id']}"
                ) from error
            choices = tuple(str(choice) for choice in doc["choices"]["text"])
            if not 2 <= len(choices) <= 5:
                raise ValueError(f"unsupported ARC choice count: {len(choices)}")
            examples.append(
                ARCExample(
                    task=task,
                    doc_id=int(row["doc_id"]),
                    example_id=str(doc.get("id", row["doc_id"])),
                    question=str(doc["question"]),
                    choices=choices,
                    answer_index=answer_index,
                )
            )
    return examples


def evenly_spaced(rows: list[ARCExample], maximum: int | None) -> list[ARCExample]:
    """Select a deterministic whole-split probe instead of a biased prefix."""

    if maximum is None or maximum >= len(rows):
        return rows
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if maximum == 1:
        return [rows[0]]
    return [rows[round(index * (len(rows) - 1) / (maximum - 1))] for index in range(maximum)]


def build_prompt(
    example: ARCExample,
    *,
    reasoning: bool = True,
    prompt_style: str = "minimal",
) -> str:
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(example.labels, example.choices, strict=True)
    )
    question = f"Question: {example.question}\n\nChoices:\n{choices}"
    if prompt_style == "bare":
        return question
    if prompt_style != "minimal":
        raise ValueError(f"unsupported prompt style: {prompt_style}")
    allowed = ", ".join(example.labels)
    if reasoning:
        instruction = f"Think step by step. End with `Final answer: X` (X = {allowed})."
    else:
        instruction = f"Reply only with `Final answer: X` (X = {allowed})."
    return f"{instruction}\n\n{question}"


def build_open_answer_prompt(example: ARCExample) -> str:
    """Ask for an answer in natural language, without exposing option labels."""

    choices = "\n".join(f"- {choice}" for choice in example.choices)
    return (
        "Answer the question directly using the text of the correct answer. "
        "Do not mention option letters and do not explain.\n\n"
        f"Question: {example.question}\n\nPossible answers:\n{choices}"
    )


_EXPLICIT_FINAL = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*(?:is\s*)?[:=\-]?\s*"
    r"(?:option\s*)?[\[({`*\s]*([A-E])\b"
)
_THINK_END = ("<|think_end|>", "</think>")


def parse_strict_answer(text: str, allowed: tuple[str, ...]) -> str | None:
    """Accept one unambiguous explicit ``Final answer`` declaration."""

    matches = [match.upper() for match in _EXPLICIT_FINAL.findall(text)]
    matches = [match for match in matches if match in allowed]
    if not matches or len(set(matches)) != 1:
        return None
    return matches[-1]


def parse_flexible_answer(text: str, allowed: tuple[str, ...]) -> str | None:
    """Fallback parser for a single answer in the post-reasoning segment."""

    strict = parse_strict_answer(text, allowed)
    if strict is not None:
        return strict
    final_segment = text
    for marker in _THINK_END:
        if marker in final_segment:
            final_segment = final_segment.rsplit(marker, 1)[1]
    final_segment = final_segment.replace("<|im_end|>", "").strip()
    alternatives = "".join(allowed)
    candidates = re.findall(rf"(?<![A-Za-z])([{alternatives}])(?![A-Za-z])", final_segment)
    candidates = [candidate.upper() for candidate in candidates]
    if len(set(candidates)) != 1:
        return None
    return candidates[0]


def _normalize_answer_text(text: str) -> str:
    table = str.maketrans({character: " " for character in string.punctuation})
    return " ".join(text.lower().translate(table).split())


def parse_open_answer(text: str, example: ARCExample) -> str | None:
    """Map a free-text answer only when it unambiguously names one ARC choice."""

    completion = _normalize_answer_text(text.replace("<|im_end|>", ""))
    if not completion:
        return None
    normalized_choices = [_normalize_answer_text(choice) for choice in example.choices]
    exact = [index for index, choice in enumerate(normalized_choices) if completion == choice]
    if len(exact) == 1:
        return example.labels[exact[0]]
    contained = [
        index
        for index, choice in enumerate(normalized_choices)
        if choice and (completion.startswith(choice + " ") or f" {choice} " in f" {completion} ")
    ]
    if len(contained) == 1:
        return example.labels[contained[0]]
    return None


class TRHashMLXGenerator:
    def __init__(
        self,
        model_dir: Path,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> None:
        import mlx.core as mx
        from mlx_lm.utils import load, load_model, load_tokenizer

        from complexity.inference.chat_template import (
            load_chat_template_jinja,
            render_jinja_inference_prompt,
        )

        try:
            model, tokenizer = load(model_dir.as_posix())
        except ValueError as error:
            if "mlp.token_to_expert" not in str(error):
                raise
            model, _config = load_model(model_dir, strict=False)
            tokenizer = load_tokenizer(model_dir)
        model.eval()
        mx.eval(model.parameters())
        self.model = model
        self.tokenizer = tokenizer
        self.mx = mx
        self.jinja = load_chat_template_jinja(model_dir)
        self.render = render_jinja_inference_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    def render_prompt(self, prompt: str) -> str:
        return self.render(
            prompt,
            self.jinja,
            eos_token=self.tokenizer.eos_token,
        )

    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str:
        del allowed
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        self.mx.random.seed(seed)
        rendered = self.render_prompt(prompt)
        prompt_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        processors = make_logits_processors(
            repetition_penalty=self.repetition_penalty,
            repetition_context_size=64,
        )
        generated: list[int] = []
        for token, _probability in generate_step(
            self.mx.array(prompt_ids),
            self.model,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
            logits_processors=processors,
        ):
            token_id = int(token)
            if token_id == self.tokenizer.eos_token_id:
                break
            generated.append(token_id)
        return self.tokenizer.decode(generated, skip_special_tokens=False)


class TRHashMLXConstrainedGenerator(TRHashMLXGenerator):
    """Generate one answer letter after an explicit assistant-side prefix."""

    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str:
        del seed
        rendered = self.render_prompt(prompt) + "Final answer:"
        prompt_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        token_ids: list[int] = []
        for label in allowed:
            encoded = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"answer label {label!r} is not one token: {encoded}")
            token_ids.append(encoded[0])
        logits = self.model(self.mx.array([prompt_ids], dtype=self.mx.int32))[0, -1]
        allowed_logits = logits[self.mx.array(token_ids, dtype=self.mx.int32)]
        self.mx.eval(allowed_logits)
        answer = allowed[int(self.mx.argmax(allowed_logits).item())]
        return f"Final answer: {answer}"


class TRHashTorchGenerator:
    """Generate an auditable ARC answer from a native training checkpoint."""

    def __init__(
        self,
        model_dir: Path,
        *,
        tokenizer_path: Path,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> None:
        import torch

        from complexity.inference.chat_template import render_messages_before_assistant
        from scripts.chat_generate_local import generate_chat, load_model, pick_device

        resolved_device = pick_device(device)
        model, tokenizer, chat_template = load_model(model_dir, tokenizer_path, resolved_device)
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.render = render_messages_before_assistant
        self.generate_chat = generate_chat
        self.device = resolved_device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str:
        del allowed
        self.torch.manual_seed(seed)
        rendered = self.render([{"role": "user", "content": prompt}], self.chat_template)
        return self.generate_chat(
            self.model,
            self.tokenizer,
            rendered,
            self.device,
            self.max_new_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
            self.repetition_penalty,
            64,
        )


class TRHashTorchConstrainedGenerator(TRHashTorchGenerator):
    """Choose one ARC label from the next-token logits after a fixed prefix."""

    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str:
        del seed
        rendered = self.render([{"role": "user", "content": prompt}], self.chat_template)
        rendered += "Final answer:"
        prompt_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        allowed_ids = []
        for label in allowed:
            encoded = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"answer label {label!r} is not one token: {encoded}")
            allowed_ids.append(encoded[0])
        tokens = self.torch.tensor([prompt_ids], dtype=self.torch.long, device=self.device)
        with self.torch.inference_mode():
            logits = self.model(tokens)["logits"][0, -1].float()
        choice = int(logits[allowed_ids].argmax().item())
        return f"Final answer: {allowed[choice]}"


class NautileTorchGenerator:
    def __init__(
        self,
        model_dir: Path,
        *,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_thinking_tokens: int | None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        module_path = model_dir / "generation_utils.py"
        spec = importlib.util.spec_from_file_location("nautile_generation_utils", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        dtype = torch.float16 if device == "mps" else torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        ).to(device)
        model.eval()
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.generate_impl = module.generate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_thinking_tokens = max_thinking_tokens

    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str:
        del allowed
        self.torch.manual_seed(seed)
        return self.generate_impl(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            use_chat_template=True,
            use_triton=False,
            strip_thinking=False,
            max_thinking_tokens=self.max_thinking_tokens,
        )


def make_generator(args: argparse.Namespace) -> Generator:
    common = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.backend in {"tr_hash_mlx", "tr_hash_mlx_open"}:
        return TRHashMLXGenerator(args.model_dir, **common)
    if args.backend == "tr_hash_mlx_constrained":
        return TRHashMLXConstrainedGenerator(args.model_dir, **common)
    if args.backend in {"tr_hash_torch", "tr_hash_torch_direct", "tr_hash_torch_constrained"}:
        if args.tokenizer is None:
            raise ValueError("--tokenizer is required for tr_hash_torch")
        generator = (
            TRHashTorchConstrainedGenerator
            if args.backend == "tr_hash_torch_constrained"
            else TRHashTorchGenerator
        )
        return generator(
            args.model_dir,
            tokenizer_path=args.tokenizer,
            device=args.device,
            **common,
        )
    return NautileTorchGenerator(
        args.model_dir,
        device=args.device,
        max_thinking_tokens=args.max_thinking_tokens,
        **common,
    )


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    strict_answered = sum(row["strict_prediction"] is not None for row in rows)
    flexible_answered = sum(row["flexible_prediction"] is not None for row in rows)
    strict_correct = sum(row["strict_correct"] for row in rows)
    flexible_correct = sum(row["flexible_correct"] for row in rows)
    native_answered = sum(row["native_prediction"] is not None for row in rows)
    native_correct = sum(row["native_correct"] for row in rows)
    return {
        "examples": total,
        "strict_answered": strict_answered,
        "strict_format_rate": strict_answered / total if total else math.nan,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / total if total else math.nan,
        "flexible_answered": flexible_answered,
        "flexible_parse_rate": flexible_answered / total if total else math.nan,
        "flexible_correct": flexible_correct,
        "flexible_accuracy": flexible_correct / total if total else math.nan,
        "native_answered": native_answered,
        "native_parse_rate": native_answered / total if total else math.nan,
        "native_correct": native_correct,
        "native_accuracy": native_correct / total if total else math.nan,
    }


def attach_native_result(row: dict, backend: str) -> dict:
    """Select the answer convention actually taught to each released model."""

    if backend == "nautile_torch":
        field = "flexible_prediction"
    elif backend == "tr_hash_mlx_open":
        field = "open_prediction"
    else:
        field = "strict_prediction"
    row["native_prediction"] = row[field]
    row["native_correct"] = row[field] == row["answer"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend",
        choices=(
            "tr_hash_mlx",
            "tr_hash_mlx_constrained",
            "tr_hash_mlx_open",
            "tr_hash_torch",
            "tr_hash_torch_direct",
            "tr_hash_torch_constrained",
            "nautile_torch",
        ),
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--arc-easy-samples", type=Path, required=True)
    parser.add_argument("--arc-challenge-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples-per-task", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-thinking-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--prompt-style", choices=("minimal", "bare"), default="minimal")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    apply_native_recipe(args)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be within [0, num-shards)")

    examples: list[ARCExample] = []
    for task, path in (
        ("arc_easy", args.arc_easy_samples),
        ("arc_challenge", args.arc_challenge_samples),
    ):
        task_rows = load_lm_eval_samples(path, task)
        examples.extend(evenly_spaced(task_rows, args.max_samples_per_task))
    examples = [
        example
        for index, example in enumerate(examples)
        if index % args.num_shards == args.shard_index
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traces_path = args.output.with_suffix(".jsonl")
    completed: dict[tuple[str, int], dict] = {}
    if traces_path.exists():
        for line in traces_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[(row["task"], row["doc_id"])] = attach_native_result(row, args.backend)

    generator = make_generator(args)
    started = time.monotonic()
    for index, example in enumerate(examples, start=1):
        key = (example.task, example.doc_id)
        if key in completed:
            continue
        prompt = (
            build_open_answer_prompt(example)
            if args.backend == "tr_hash_mlx_open"
            else build_prompt(
                example,
                reasoning=args.backend in {"nautile_torch", "tr_hash_torch"},
                prompt_style=args.prompt_style,
            )
        )
        completion_started = time.monotonic()
        completion = generator.generate(
            prompt,
            seed=args.seed + index,
            allowed=example.labels,
        )
        strict_prediction = parse_strict_answer(completion, example.labels)
        flexible_prediction = parse_flexible_answer(completion, example.labels)
        open_prediction = parse_open_answer(completion, example)
        row = {
            **asdict(example),
            "labels": example.labels,
            "answer": example.answer,
            "prompt": prompt,
            "completion": completion,
            "strict_prediction": strict_prediction,
            "strict_correct": strict_prediction == example.answer,
            "flexible_prediction": flexible_prediction,
            "flexible_correct": flexible_prediction == example.answer,
            "open_prediction": open_prediction,
            "open_correct": open_prediction == example.answer,
            "elapsed_seconds": round(time.monotonic() - completion_started, 3),
        }
        attach_native_result(row, args.backend)
        with traces_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed[key] = row
        current = summarize(list(completed.values()))
        print(
            f"{index}/{len(examples)} {example.task}:{example.doc_id} "
            f"strict={strict_prediction or '-'} answer={example.answer} "
            f"acc={current['strict_accuracy']:.3f}",
            flush=True,
        )

    ordered = [completed[(example.task, example.doc_id)] for example in examples]
    by_task = {
        task: summarize([row for row in ordered if row["task"] == task])
        for task in ("arc_easy", "arc_challenge")
    }
    report = {
        "model": str(args.model_dir.resolve()),
        "backend": args.backend,
        "recipe": "backend_native",
        "protocol": "zero_shot_generative_reasoning_final_answer_extraction",
        "prompt_contract": (
            "bare_question_and_labeled_choices_without_instruction"
            if args.prompt_style == "bare"
            else (
                "native_reasoning_then_answer_letter"
                if args.backend in {"nautile_torch", "tr_hash_torch"}
                else (
                    "direct_constrained_single_answer_letter_without_reasoning"
                    if args.backend in {"tr_hash_mlx_constrained", "tr_hash_torch_constrained"}
                    else (
                        "direct_open_answer_text_without_option_letters_or_reasoning"
                        if args.backend == "tr_hash_mlx_open"
                        else "direct_explicit_final_answer_letter_without_reasoning"
                    )
                )
            )
        ),
        "chat_template_applied": True,
        "selection": "evenly_spaced_within_each_full_public_test_split",
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "max_thinking_tokens": args.max_thinking_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "combined": summarize(ordered),
        "benchmarks": by_task,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "traces": str(traces_path.resolve()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
