"""Auditable generative ARC prompts, parsers, reports and shard merging.

Generation is supplied by a callable or external serving client. Importing
these helpers never loads Torch, MLX, a model checkpoint or a dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ARCExample:
    task: str
    doc_id: int
    example_id: str
    question: str
    choices: tuple[str, ...]
    answer_index: int

    def __post_init__(self):
        if not 2 <= len(self.choices) <= 5 or not 0 <= self.answer_index < len(self.choices):
            raise ValueError("ARC requires 2–5 choices and a valid answer index")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(chr(ord("A") + index) for index in range(len(self.choices)))

    @property
    def answer(self) -> str:
        return self.labels[self.answer_index]


class Generator(Protocol):
    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str: ...


class ServingGenerator:
    """Adapter over the framework's external completion client.

    ``allowed`` documents ARC candidates for grading; this adapter does not
    claim constrained decoding. An explicit chat template renders raw
    completions, otherwise the provider receives the benchmark prompt as-is.
    """

    def __init__(self, client, config, *, chat_template=None):
        self.client = client
        self.config = config
        self.chat_template = chat_template

    def generate(self, prompt: str, *, seed: int, allowed: tuple[str, ...]) -> str:
        if self.chat_template is not None:
            from complexity.inference.chat_template import render_inference_prompt

            prompt = render_inference_prompt(prompt, self.chat_template)
        config = replace(self.config, extra={**self.config.extra, "seed": seed})
        return self.client.complete(prompt, config)


def serving_generator(*, backend, base_url, model, max_tokens, chat_template=None, stop=None):
    from complexity.inference.external import ExternalGenerationConfig, create_external_backend

    client = create_external_backend(backend, base_url=base_url, model=model)
    if stop is None and chat_template is not None:
        stop = list(
            dict.fromkeys(
                filter(
                    None, (chat_template.get("end_of_turn_token"), chat_template.get("eos_token"))
                )
            )
        )
    config = ExternalGenerationConfig(max_tokens=max_tokens, temperature=0.0, top_p=1.0, stop=stop)
    return ServingGenerator(client, config, chat_template=chat_template)


def generate_completion(generator, prompt, *, seed, allowed):
    """Accept a generation callable or the historical ``Generator`` protocol."""
    call = getattr(generator, "generate", generator)
    response = call(prompt, seed=seed, allowed=allowed)
    if not isinstance(response, str):
        raise TypeError("A benchmark generator must return completion text")
    return response


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


_FINAL_START = "<|final_start|>"


_FINAL_END = "<|final_end|>"


def final_answer_segment(text: str) -> str:
    """Return only the supervised v3 final envelope when it is present."""

    if _FINAL_START not in text:
        return text
    segment = text.split(_FINAL_START, 1)[1]
    if _FINAL_END in segment:
        segment = segment.split(_FINAL_END, 1)[0]
    return segment.strip()


def parse_strict_answer(text: str, allowed: tuple[str, ...]) -> str | None:
    """Accept one unambiguous explicit ``Final answer`` declaration."""

    matches = [match.upper() for match in _EXPLICIT_FINAL.findall(final_answer_segment(text))]
    matches = [match for match in matches if match in allowed]
    if not matches or len(set(matches)) != 1:
        return None
    return matches[-1]


def parse_flexible_answer(text: str, allowed: tuple[str, ...]) -> str | None:
    """Fallback parser for a single answer in the post-reasoning segment."""

    strict = parse_strict_answer(text, allowed)
    if strict is not None:
        return strict
    final_segment = final_answer_segment(text)
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

    completion = _normalize_answer_text(final_answer_segment(text).replace("<|im_end|>", ""))
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


_NEXT_QUESTION = re.compile(r"\n\s*(?:#{1,6}\s*)?Question\s*:", re.IGNORECASE)


_NATIVE_ANSWER_PATTERNS = (
    re.compile(
        r"(?i)\b(?:the\s+)?correct\s+(?:answer|choice|option)\s*(?:is|:|=|-)?\s*"
        r"(?:option|choice)?\s*[\[({`*$'\"\\\s]*(?:boxed\s*\{\s*)?([A-E])\b"
    ),
    re.compile(
        r"(?i)\banswer\s*(?:is|:|=|-)\s*(?:option|choice)?\s*"
        r"[\[({`*$'\"\\\s]*(?:boxed\s*\{\s*)?([A-E])\b"
    ),
    re.compile(r"(?i)\\boxed\s*\{\s*([A-E])\s*\}"),
    re.compile(r"(?i)^\s*\(?([A-E])\)?\s*[.):,-]"),
)


def first_response_segment(text: str) -> str:
    """Keep the answer before the model starts synthesizing another example."""

    return _NEXT_QUESTION.split(text, maxsplit=1)[0].strip()


def parse_native_first_answer(text: str, example: ARCExample) -> str | None:
    """Parse the first answer using formats present in the reasoning SFT data.

    The training targets use natural forms such as ``the correct choice is
    \\boxed{D}``, ``(B).`` and the literal answer text.  They do not enforce the
    synthetic ``Final answer: X`` contract used by the initial probe.
    """

    segment = first_response_segment(final_answer_segment(text))
    if not segment:
        return None
    candidates: list[tuple[int, str]] = []
    for pattern in _NATIVE_ANSWER_PATTERNS:
        candidates.extend(
            (match.start(), match.group(1).upper()) for match in pattern.finditer(segment)
        )
    candidates = [(offset, label) for offset, label in candidates if label in example.labels]
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return parse_open_answer(segment, example)


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
    elif backend == "tr_hash_torch":
        field = "native_first_prediction"
        if field not in row:
            example = ARCExample(
                task=row["task"],
                doc_id=int(row["doc_id"]),
                example_id=str(row["example_id"]),
                question=row["question"],
                choices=tuple(row["choices"]),
                answer_index=int(row["answer_index"]),
            )
            row[field] = parse_native_first_answer(row["completion"], example)
    elif backend == "tr_hash_mlx_open":
        field = "open_prediction"
    else:
        field = "strict_prediction"
    row["native_prediction"] = row[field]
    row["native_correct"] = row[field] == row["answer"]
    return row


def merge(shards: list[Path], output: Path, expected_examples: int) -> dict:
    if not shards or expected_examples < 1:
        raise ValueError("Provide nonempty shards and a positive expected example count")
    shards = [Path(path) for path in shards]
    output = Path(output)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    rows: dict[tuple[str, int], dict] = {}
    for report, path in zip(reports, shards, strict=True):
        traces = path.with_suffix(".jsonl")
        if not traces.is_file():
            raise FileNotFoundError(traces)
        if (
            report.get("traces_sha256") is not None
            and hashlib.sha256(traces.read_bytes()).hexdigest() != report["traces_sha256"]
        ):
            raise ValueError(f"ARC trace integrity differs from its report: {traces}")
        for line in traces.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["task"], int(row["doc_id"]))
            if key in rows:
                raise ValueError(f"duplicate ARC example {key} in {path}")
            rows[key] = row
    if len(rows) != expected_examples:
        raise ValueError(f"expected {expected_examples} examples, found {len(rows)}")
    varying = {"combined", "benchmarks", "elapsed_seconds", "shard", "traces", "traces_sha256"}
    identity = {key: value for key, value in reports[0].items() if key not in varying}
    for report in reports[1:]:
        if {key: value for key, value in report.items() if key not in varying} != identity:
            raise ValueError("ARC shards use different models, prompts or decoding provenance")
    ordered = sorted(rows.values(), key=lambda row: (row["task"], int(row["doc_id"])))
    merged = {key: value for key, value in reports[0].items() if key not in varying}
    merged["shards"] = len(shards)
    merged["combined"] = summarize(ordered)
    merged["benchmarks"] = {
        task: summarize([row for row in ordered if row["task"] == task])
        for task in sorted({row["task"] for row in ordered})
    }
    merged["elapsed_seconds"] = max(float(report["elapsed_seconds"]) for report in reports)
    write_report(output, merged, ordered)
    return merged


def write_report(output: str | Path, report: dict, rows: list[dict]) -> None:
    """Write complete traces and an atomic JSON report with trace integrity."""
    output = Path(output)
    traces = output.with_suffix(".jsonl")
    if output == traces:
        raise ValueError("Report output must have a JSON suffix distinct from JSONL traces")
    if output.exists() or traces.exists():
        raise FileExistsError("Benchmark output exists; choose fresh report paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    report["traces"] = str(traces.resolve())
    report["traces_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    pending_traces = traces.with_name(traces.name + ".tmp")
    pending_report = output.with_name(output.name + ".tmp")
    pending_traces.write_text(body, encoding="utf-8")
    pending_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pending_traces.replace(traces)
    pending_report.replace(output)


def evaluate(
    examples: list[ARCExample],
    generator: Generator,
    *,
    seed: int = 42,
    reasoning: bool = True,
    prompt_style: str = "minimal",
    answer_backend: str = "tr_hash_torch",
    output: str | Path | None = None,
    provenance: dict | None = None,
) -> dict:
    """Generate and grade ARC through a supplied generator, retaining every trace."""
    if not examples:
        raise ValueError("ARC selection is empty")
    if len({(example.task, example.doc_id) for example in examples}) != len(examples):
        raise ValueError("ARC selection contains duplicate example identities")
    if answer_backend not in NATIVE_RECIPES:
        raise ValueError("Unknown ARC answer convention")
    started, rows = time.monotonic(), []
    for index, example in enumerate(examples):
        prompt = (
            build_open_answer_prompt(example)
            if answer_backend == "tr_hash_mlx_open"
            else build_prompt(example, reasoning=reasoning, prompt_style=prompt_style)
        )
        completion = generate_completion(
            generator, prompt, seed=seed + index, allowed=example.labels
        )
        strict = parse_strict_answer(completion, example.labels)
        flexible = parse_flexible_answer(completion, example.labels)
        native = parse_native_first_answer(completion, example)
        opened = parse_open_answer(completion, example)
        row = {
            **asdict(example),
            "labels": example.labels,
            "answer": example.answer,
            "prompt": prompt,
            "completion": completion,
            "strict_prediction": strict,
            "strict_correct": strict == example.answer,
            "flexible_prediction": flexible,
            "flexible_correct": flexible == example.answer,
            "native_first_prediction": native,
            "native_first_correct": native == example.answer,
            "open_prediction": opened,
            "open_correct": opened == example.answer,
        }
        rows.append(attach_native_result(row, answer_backend))
    report = {
        "schema_version": 1,
        "benchmark": "allenai/ai2_arc",
        "complete": True,
        "protocol": "generative_arc_answer_extraction",
        "answer_convention": answer_backend,
        "prompt_style": prompt_style,
        "reasoning_instruction": reasoning,
        "seed": seed,
        "provenance": dict(provenance or {}),
        "combined": summarize(rows),
        "benchmarks": {
            task: summarize([row for row in rows if row["task"] == task])
            for task in sorted({row["task"] for row in rows})
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if output is not None:
        write_report(output, report, rows)
    else:
        report["records"] = rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merging = commands.add_parser("merge", help="Merge compatible complete ARC shards")
    merging.add_argument("--shard", type=Path, action="append", required=True)
    merging.add_argument("--output", type=Path, required=True)
    merging.add_argument("--expected-examples", type=int, required=True)
    grading = commands.add_parser(
        "evaluate", help="Evaluate local ARC using an external completion server"
    )
    grading.add_argument("samples", type=Path)
    grading.add_argument("--task", choices=("arc_easy", "arc_challenge"), required=True)
    grading.add_argument("--backend", choices=("vllm", "sglang"), default="vllm")
    grading.add_argument("--base-url", required=True)
    grading.add_argument("--model", required=True)
    grading.add_argument("--chat-template", type=Path)
    grading.add_argument("--max-new-tokens", type=int, default=256)
    grading.add_argument("--max-samples", type=int)
    grading.add_argument("--seed", type=int, default=42)
    grading.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "merge":
        report = merge(args.shard, args.output, args.expected_examples)
    else:
        if (
            args.max_new_tokens < 1
            or args.samples.resolve() == args.output.with_suffix(".jsonl").resolve()
        ):
            parser.error("Use positive generation length and output paths distinct from samples")
        examples = evenly_spaced(load_lm_eval_samples(args.samples, args.task), args.max_samples)
        template = json.loads(args.chat_template.read_text()) if args.chat_template else None
        generator = serving_generator(
            backend=args.backend,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_new_tokens,
            chat_template=template,
        )
        report = evaluate(
            examples,
            generator,
            seed=args.seed,
            output=args.output,
            provenance={
                "backend": args.backend,
                "model": args.model,
                "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                "chat_template_applied": template is not None,
                "chat_template_sha256": hashlib.sha256(args.chat_template.read_bytes()).hexdigest()
                if args.chat_template
                else None,
                "decoding": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            },
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
