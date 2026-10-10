#!/usr/bin/env python3
"""Zero-shot multiple-choice evaluation for native Torch or local MLX models.

Scores continuations by causal log-likelihood without a chat template. The
formatting and separate context/continuation tokenization preserve the archived
200M MLX evaluation contract. Benchmark loading may fetch public datasets;
choice encoding and report aggregation require no MLX or dataset runtime.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

from complexity._optional import require_optional


def load_dataset(*args, **kwargs):
    """Load benchmark rows on demand; formatting needs no dataset runtime."""
    require_optional("datasets", "data")
    from datasets import load_dataset as loader

    return loader(*args, **kwargs)


def _mlx():
    try:
        import mlx.core as mx
    except ImportError as error:
        raise ImportError("MLX scoring requires the optional mlx runtime") from error
    return mx


@dataclass(frozen=True)
class Example:
    benchmark: str
    example_id: str
    context: str
    continuations: tuple[str, ...]
    answer: int
    subject: str | None = None

    def __post_init__(self):
        if not self.continuations:
            raise ValueError("An evaluation example requires candidate continuations")
        if (
            isinstance(self.answer, bool)
            or not isinstance(self.answer, int)
            or not 0 <= self.answer < len(self.continuations)
        ):
            raise ValueError("The correct answer must index a candidate continuation")


@dataclass(frozen=True)
class EncodedChoice:
    example_index: int
    choice_index: int
    ids: tuple[int, ...]
    completion_start: int
    completion_length: int


def _load_bundle(model_dir: Path):
    """Load a complete local bundle; never ignore missing checkpoint weights."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"MLX bundle directory does not exist: {model_dir}")
    try:
        from mlx_lm.utils import load
    except ImportError as error:
        raise ImportError("Loading an MLX bundle requires the optional mlx-lm package") from error
    return load(model_dir.as_posix())


def load_torch_bundle(
    model_dir: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    kernel_policy: str = "checkpoint",
):
    """Load local native weights and tokenizer; missing learned weights fail.

    Device placement is explicit. CPU/MPS use float32; CUDA may use float16
    or bfloat16. ``portable`` uses reference PyTorch routing, eager attention
    and RMSNorm on these model instances. ``checkpoint`` preserves the saved
    execution options. No checkpoint or tokenizer is fetched or modified.
    """
    with redirect_stdout(sys.stderr):
        model = _load_torch_model(
            model_dir, device=device, dtype=dtype, kernel_policy=kernel_policy
        )
        tokenizer = _load_torch_tokenizer(tokenizer_path or model_dir)
    return model, tokenizer


def _portable_rmsnorm(self, hidden_states):
    """The registered RMSNorm's reference path, bound to this instance only."""
    import torch

    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + self.eps)
    return self.weight * normalized


def _load_torch_tokenizer(path):
    from complexity.tokenizer import Tokenizer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {path}")
    return Tokenizer.load(str(path))


def _torch_placement(device, dtype):
    import torch

    placement = torch.device(device)
    if placement.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("Torch evaluation requires CPU, MPS or CUDA")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Unknown Torch evaluation dtype")
    if placement.type != "cuda" and dtype != "float32":
        raise ValueError("CPU/MPS evaluation requires float32")
    if placement.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a silent device fallback")
    if placement.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing a silent device fallback")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The CUDA device does not support bfloat16")
    return placement


def _load_torch_model(model_dir, *, device, dtype, kernel_policy):
    import torch

    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    if kernel_policy not in {"checkpoint", "portable"}:
        raise ValueError("Kernel policy must be checkpoint or portable")
    placement = _torch_placement(device, dtype)
    model_dir = Path(model_dir)
    if kernel_policy == "checkpoint":
        model = ComplexityModel.from_pretrained(
            model_dir, device=str(placement), dtype=getattr(torch, dtype)
        )
    else:
        # A fresh config changes execution only; the source JSON and every
        # saved tensor retain their exact contents and names.
        config = ModelConfig.load(model_dir / "config.json")
        if config.attention_type not in {"gqa", "mha", "mqa"} or config.mlp_type not in {
            "tr_hash_engine",
            "tr_hash_moe",
        }:
            raise ValueError(
                "Portable evaluation supports native TR-Hash with GQA/MHA/MQA attention"
            )
        config.use_custom_kernels = False
        config.use_cggr = False
        config.use_sdpa = False
        model = ComplexityModel(config)
        model._load_pretrained_weights(model_dir)
        from complexity.core.normalization.norms import RMSNorm

        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.forward = MethodType(_portable_rmsnorm, module)
        model.to(device=placement, dtype=getattr(torch, dtype))
    return model.eval().requires_grad_(False)


def _validate_maximum(maximum: int | None) -> None:
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
    ):
        raise ValueError("maximum samples must be a positive integer or None")


def _limit(rows: Iterable[Example], maximum: int | None) -> list[Example]:
    _validate_maximum(maximum)
    if maximum is None:
        return list(rows)
    result: list[Example] = []
    for row in rows:
        if len(result) >= maximum:
            break
        result.append(row)
    return result


def load_hellaswag(maximum: int | None) -> list[Example]:
    _validate_maximum(maximum)
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    return _limit(
        (
            Example(
                benchmark="hellaswag",
                example_id=str(row["ind"]),
                context=str(row["ctx"]),
                continuations=tuple(" " + str(value).lstrip() for value in row["endings"]),
                answer=int(row["label"]),
                subject=str(row.get("activity_label", "")) or None,
            )
            for row in dataset
        ),
        maximum,
    )


def _load_arc(
    *,
    dataset_name: str,
    benchmark: str,
    maximum: int | None,
) -> list[Example]:
    """Load an AI2 ARC split using the lm-evaluation-harness contract."""

    _validate_maximum(maximum)
    dataset = load_dataset("allenai/ai2_arc", dataset_name, split="test")

    def examples() -> Iterable[Example]:
        for index, row in enumerate(dataset):
            labels = [str(value) for value in row["choices"]["label"]]
            answer_key = str(row["answerKey"])
            try:
                answer = labels.index(answer_key)
            except ValueError as error:
                raise ValueError(
                    f"{dataset_name} answer key {answer_key!r} is absent from "
                    f"choice labels {labels!r} for example {row.get('id', index)!r}"
                ) from error
            yield Example(
                benchmark=benchmark,
                example_id=str(row.get("id", index)),
                context=f"Question: {row['question']}\nAnswer:",
                continuations=tuple(" " + str(value).lstrip() for value in row["choices"]["text"]),
                answer=answer,
            )

    return _limit(examples(), maximum)


def load_arc_easy(maximum: int | None) -> list[Example]:
    return _load_arc(
        dataset_name="ARC-Easy",
        benchmark="arc_easy",
        maximum=maximum,
    )


def load_arc_challenge(maximum: int | None) -> list[Example]:
    return _load_arc(
        dataset_name="ARC-Challenge",
        benchmark="arc_challenge",
        maximum=maximum,
    )


def load_arc(path: str | Path, task: str, maximum: int | None = None) -> list[Example]:
    """Read local ARC documents, including archived lm-eval ``doc`` rows."""
    _validate_maximum(maximum)
    if task not in {"arc_easy", "arc_challenge"}:
        raise ValueError("Local ARC task must be arc_easy or arc_challenge")

    def examples():
        with Path(path).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                doc = row.get("doc", row)
                labels = [str(value) for value in doc["choices"]["label"]]
                answer_key = str(doc["answerKey"])
                if answer_key not in labels:
                    raise ValueError(
                        f"Answer key {answer_key!r} absent from choices in {path}:{index}"
                    )
                yield Example(
                    benchmark=task,
                    example_id=str(doc.get("id", row.get("doc_id", index))),
                    context=f"Question: {doc['question']}\nAnswer:",
                    continuations=tuple(
                        " " + str(value).lstrip() for value in doc["choices"]["text"]
                    ),
                    answer=labels.index(answer_key),
                )

    return _limit(examples(), maximum)


def load_mmlu(maximum: int | None) -> list[Example]:
    _validate_maximum(maximum)
    dataset = load_dataset("cais/mmlu", "all", split="test")
    letters = "ABCD"

    def examples() -> Iterable[Example]:
        for index, row in enumerate(dataset):
            choices = tuple(str(value) for value in row["choices"])
            context = f"Question: {row['question']}\n"
            context += "".join(
                f"{letter}. {choice}\n" for letter, choice in zip(letters, choices, strict=True)
            )
            context += "Answer:"
            yield Example(
                benchmark="mmlu",
                example_id=str(index),
                context=context,
                continuations=tuple(f" {letter}" for letter in letters),
                answer=int(row["answer"]),
                subject=str(row["subject"]),
            )

    return _limit(examples(), maximum)


def load_piqa(maximum: int | None, *, data_dir: str | Path | None = None) -> list[Example]:
    """Read official PIQA validation files, locally or via the dataset cache."""
    _validate_maximum(maximum)
    # datasets>=4 no longer executes the legacy ybisk/piqa loader. Read the
    # same official AI2 archive and validation files that loader references.
    if data_dir is None:
        require_optional("datasets", "data")
        from datasets import DownloadManager

        archive = DownloadManager().download_and_extract(
            "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
        )
        root = Path(archive) / "physicaliqa-train-dev"
    else:
        root = Path(data_dir)
    inputs = (root / "dev.jsonl").read_text().splitlines()
    labels = (root / "dev-labels.lst").read_text().splitlines()

    def examples() -> Iterable[Example]:
        for index, (encoded, label) in enumerate(zip(inputs, labels, strict=True)):
            row = json.loads(encoded)
            yield Example(
                benchmark="piqa",
                example_id=str(index),
                context=str(row["goal"]),
                continuations=(
                    " " + str(row["sol1"]).lstrip(),
                    " " + str(row["sol2"]).lstrip(),
                ),
                answer=int(label),
            )

    return _limit(examples(), maximum)


LOADERS = {
    "arc_challenge": load_arc_challenge,
    "arc_easy": load_arc_easy,
    "hellaswag": load_hellaswag,
    "mmlu": load_mmlu,
    "piqa": load_piqa,
}


def encode_choices(
    tokenizer,
    examples: list[Example],
    max_length: int,
) -> list[EncodedChoice]:
    """Preserve each complete continuation; trim only excess left context."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("maximum sequence length must allow context and continuation")
    choices: list[EncodedChoice] = []
    for example_index, example in enumerate(examples):
        context_ids = tokenizer.encode(example.context, add_special_tokens=False)
        for choice_index, continuation in enumerate(example.continuations):
            continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
            if not continuation_ids:
                raise ValueError(f"empty continuation in {example.benchmark}:{example.example_id}")
            ids = context_ids + continuation_ids
            if len(ids) > max_length:
                removed = len(ids) - max_length
                ids = ids[removed:]
                completion_start = len(context_ids) - removed
            else:
                completion_start = len(context_ids)
            if completion_start < 1:
                raise ValueError(
                    f"context truncated past scoring boundary in "
                    f"{example.benchmark}:{example.example_id}"
                )
            choices.append(
                EncodedChoice(
                    example_index=example_index,
                    choice_index=choice_index,
                    ids=tuple(ids),
                    completion_start=completion_start,
                    completion_length=len(ids) - completion_start,
                )
            )
    return choices


def score_examples(
    model,
    tokenizer,
    examples: list[Example],
    *,
    batch_size: int,
    max_length: int,
    verbose: bool = False,
    backend: str = "mlx",
    device: str | None = None,
) -> list[list[dict[str, float]]]:
    """Score continuation targets in a causal local Torch or MLX model.

    Both backends share formatting, tokenization, length truncation and score
    aggregation. Torch scoring uses inference mode and evaluation mode.
    """
    if backend not in {"torch", "mlx"}:
        raise ValueError("Evaluation backend must be torch or mlx")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    encoded = encode_choices(tokenizer, examples, max_length)
    return _score_encoded_examples(
        model,
        tokenizer,
        examples,
        encoded,
        batch_size=batch_size,
        verbose=verbose,
        backend=backend,
        device=device,
    )


def _score_encoded_examples(
    model, tokenizer, examples, encoded, *, batch_size, verbose, backend, device
):
    # Sorting must not mutate the cache shared by both checkpoints.
    encoded = list(encoded)
    if not encoded:
        return []
    if backend == "torch":
        return _score_torch_choices(
            model,
            tokenizer,
            examples,
            encoded,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
    mx = _mlx()
    # Similar lengths in one batch reduce padding and Metal memory traffic.
    encoded.sort(key=lambda row: len(row.ids))
    scores: list[list[dict[str, float] | None]] = [
        [None] * len(example.continuations) for example in examples
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    started = time.monotonic()
    for offset in range(0, len(encoded), batch_size):
        batch = encoded[offset : offset + batch_size]
        width = max(len(row.ids) for row in batch)
        token_rows = [list(row.ids) + [pad_id] * (width - len(row.ids)) for row in batch]
        tokens = mx.array(token_rows, dtype=mx.int32)
        logits = model(tokens)[:, :-1, :].astype(mx.float32)
        labels = tokens[:, 1:]
        selected = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
        log_probs = selected - mx.logsumexp(logits, axis=-1)
        positions = mx.arange(1, width)[None, :]
        starts = mx.array([row.completion_start for row in batch])[:, None]
        lengths = mx.array([len(row.ids) for row in batch])[:, None]
        mask = (positions >= starts) & (positions < lengths)
        totals = mx.sum(mx.where(mask, log_probs, 0.0), axis=1)
        counts = mx.sum(mask, axis=1)
        normalized = totals / counts
        mx.eval(totals, normalized)

        for row, total, norm in zip(batch, totals.tolist(), normalized.tolist(), strict=True):
            scores[row.example_index][row.choice_index] = {
                "loglikelihood": float(total),
                "loglikelihood_normalized": float(norm),
            }
        completed = min(offset + len(batch), len(encoded))
        if verbose and (completed == len(encoded) or completed % max(batch_size, 400) < batch_size):
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"  scored {completed:,}/{len(encoded):,} choices "
                f"({completed / elapsed:.1f} choices/s)",
                flush=True,
                file=sys.stderr,
            )

    if any(value is None for row in scores for value in row):
        raise RuntimeError("one or more benchmark choices were not scored")
    return [[value for value in row if value is not None] for row in scores]


def _score_torch_choices(model, tokenizer, examples, encoded, *, batch_size, device, verbose):
    import torch

    placement = torch.device(device) if device is not None else next(model.parameters()).device
    encoded.sort(key=lambda row: len(row.ids))
    scores = [[None] * len(example.continuations) for example in examples]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    started = time.monotonic()
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for offset in range(0, len(encoded), batch_size):
                batch = encoded[offset : offset + batch_size]
                width = max(len(choice.ids) for choice in batch)
                tokens = torch.tensor(
                    [list(choice.ids) + [pad_id] * (width - len(choice.ids)) for choice in batch],
                    dtype=torch.long,
                    device=placement,
                )
                logits = model(tokens)["logits"][:, :-1].float()
                labels = tokens[:, 1:]
                selected = torch.gather(logits, -1, labels[..., None]).squeeze(-1)
                log_probs = selected - torch.logsumexp(logits, dim=-1)
                positions = torch.arange(1, width, device=placement)[None, :]
                starts = torch.tensor(
                    [choice.completion_start for choice in batch], device=placement
                )[:, None]
                lengths = torch.tensor([len(choice.ids) for choice in batch], device=placement)[
                    :, None
                ]
                mask = (positions >= starts) & (positions < lengths)
                totals = torch.where(mask, log_probs, 0.0).sum(dim=1)
                normalized = totals / mask.sum(dim=1)
                for choice, total, norm in zip(
                    batch, totals.tolist(), normalized.tolist(), strict=True
                ):
                    scores[choice.example_index][choice.choice_index] = {
                        "loglikelihood": float(total),
                        "loglikelihood_normalized": float(norm),
                    }
                completed = min(offset + len(batch), len(encoded))
                if verbose and (
                    completed == len(encoded) or completed % max(batch_size, 400) < batch_size
                ):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        f"  scored {completed:,}/{len(encoded):,} choices ({completed / elapsed:.1f} choices/s)",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        model.train(was_training)
    if any(value is None for row in scores for value in row):
        raise RuntimeError("one or more benchmark choices were not scored")
    return scores


def summarize(examples: list[Example], scores: list[list[dict[str, float]]]) -> dict[str, Any]:
    """Aggregate total and token-normalized likelihood decisions independently."""
    if not examples:
        raise ValueError("Cannot summarize an empty benchmark")
    raw_correct = 0
    normalized_correct = 0
    by_subject: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for example, choice_scores in zip(examples, scores, strict=True):
        if len(choice_scores) != len(example.continuations):
            raise ValueError("Every candidate continuation requires exactly one score")
        if any(
            not math.isfinite(choice[key])
            for choice in choice_scores
            for key in ("loglikelihood", "loglikelihood_normalized")
        ):
            raise ValueError("Benchmark scores must be finite")
        raw_prediction = max(
            range(len(choice_scores)), key=lambda index: choice_scores[index]["loglikelihood"]
        )
        normalized_prediction = max(
            range(len(choice_scores)),
            key=lambda index: choice_scores[index]["loglikelihood_normalized"],
        )
        raw_hit = raw_prediction == example.answer
        normalized_hit = normalized_prediction == example.answer
        raw_correct += raw_hit
        normalized_correct += normalized_hit
        if example.subject:
            by_subject[example.subject].append((raw_hit, normalized_hit))

    total = len(examples)
    result: dict[str, Any] = {
        "examples": total,
        "correct": raw_correct,
        "acc": raw_correct / total,
        "correct_norm": normalized_correct,
        "acc_norm": normalized_correct / total,
    }
    if by_subject:
        result["subjects"] = {
            subject: {
                "examples": len(hits),
                "acc": sum(raw for raw, _norm in hits) / len(hits),
                "acc_norm": sum(norm for _raw, norm in hits) / len(hits),
            }
            for subject, hits in sorted(by_subject.items())
        }
        result["macro_acc"] = sum(item["acc"] for item in result["subjects"].values()) / len(
            result["subjects"]
        )
    return result


def _settings(tasks, batch_size, max_length, max_samples):
    tasks = [task.strip() for task in tasks.split(",")] if isinstance(tasks, str) else list(tasks)
    if not tasks or len(tasks) != len(set(tasks)) or any(not task for task in tasks):
        raise ValueError("Choose at least one benchmark, without duplicates")
    unknown = sorted(set(tasks) - LOADERS.keys())
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    _validate_maximum(max_samples)
    if type(max_length) is not int or max_length < 2:
        raise ValueError("maximum sequence length must allow context and continuation")
    return tasks


def _load_tasks(tasks, *, max_samples, arc_easy_samples, arc_challenge_samples, piqa_data):
    loaded = {}
    for task in tasks:
        print(f"Loading {task}...", flush=True, file=sys.stderr)
        arc_path = {"arc_easy": arc_easy_samples, "arc_challenge": arc_challenge_samples}.get(task)
        with redirect_stdout(sys.stderr):
            if arc_path is not None:
                examples = load_arc(arc_path, task, max_samples)
            elif task == "piqa" and piqa_data is not None:
                examples = load_piqa(max_samples, data_dir=piqa_data)
            else:
                examples = LOADERS[task](max_samples)
        if not examples:
            raise ValueError(f"Cannot evaluate an empty benchmark: {task}")
        loaded[task] = examples
    return loaded


def _protocol(*, backend, device, dtype, kernel_policy, batch_size, max_length, max_samples):
    return {
        "backend": backend,
        "device": device if backend == "torch" else "mlx",
        "dtype": dtype if backend == "torch" else "bundle",
        "kernel_policy": kernel_policy if backend == "torch" else "bundle",
        "kernel_overrides": {
            "use_custom_kernels": False,
            "use_cggr": False,
            "use_sdpa": False,
            "rmsnorm": "pytorch_reference",
        }
        if backend == "torch" and kernel_policy == "portable"
        else None,
        "zero_shot": True,
        "scoring": "causal_choice_loglikelihood",
        "chat_template_applied": False,
        "batch_size": batch_size,
        "max_length": max_length,
        "max_samples": max_samples,
        "continuation_tokenization": "separate_context_and_continuation",
    }


def _write_report(output, report):
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".tmp")
    pending.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pending.replace(output)


def _protect_output(output, *, model_paths, tokenizer_path, arc_paths, piqa_data):
    if output is None:
        return
    resolved = Path(output).resolve()
    protected = [
        (Path(path) / name).resolve()
        for path in model_paths
        for name in ("config.json", "model.safetensors", "pytorch_model.bin", "model.pt")
    ]
    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.is_dir():
            protected.extend(
                (tokenizer_path / name).resolve()
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "config.json",
                    "tiktoken_config.json",
                )
            )
        else:
            protected.append(tokenizer_path.resolve())
    if piqa_data is not None:
        protected.extend(
            (Path(piqa_data) / name).resolve() for name in ("dev.jsonl", "dev-labels.lst")
        )
    protected.extend(Path(path).resolve() for path in arc_paths if path is not None)
    if resolved in protected:
        raise ValueError("Evaluation output must not replace model/tokenizer/dataset inputs")


def _task_results(model, tokenizer, loaded, protocol, encoded=None):
    if protocol["backend"] == "torch":
        if protocol["max_length"] > model.config.max_position_embeddings:
            raise ValueError("Evaluation max_length exceeds the native model's context")
        if not model.config.is_causal:
            raise ValueError("Choice likelihood evaluation requires a causal native model")
    for task, examples in loaded.items():
        print(f"Evaluating {task}: {len(examples):,} examples", flush=True, file=sys.stderr)
        started = time.monotonic()
        choices = (
            encoded[task]
            if encoded is not None
            else encode_choices(tokenizer, examples, protocol["max_length"])
        )
        scores = _score_encoded_examples(
            model,
            tokenizer,
            examples,
            choices,
            batch_size=protocol["batch_size"],
            verbose=True,
            backend=protocol["backend"],
            device=protocol["device"] if protocol["backend"] == "torch" else None,
        )
        result = summarize(examples, scores)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(
            f"{task}: acc={result['acc']:.4f}, acc_norm={result['acc_norm']:.4f}",
            flush=True,
            file=sys.stderr,
        )
        yield task, result


def evaluate_choices(
    model_dir,
    *,
    backend="mlx",
    device="cpu",
    dtype="float32",
    tokenizer_path=None,
    tasks="hellaswag,mmlu,piqa",
    batch_size=16,
    max_length=2048,
    max_samples=None,
    kernel_policy="checkpoint",
    arc_easy_samples=None,
    arc_challenge_samples=None,
    piqa_data=None,
    output=None,
):
    """Evaluate a local bundle; public benchmark loaders may fill their cache."""
    tasks = _settings(tasks, batch_size, max_length, max_samples)
    if backend not in {"torch", "mlx"}:
        raise ValueError("Evaluation backend must be torch or mlx")
    if kernel_policy not in {"checkpoint", "portable"}:
        raise ValueError("Kernel policy must be checkpoint or portable")
    if backend == "mlx" and kernel_policy != "checkpoint":
        raise ValueError("Portable kernel policy applies only to native Torch bundles")
    model_dir = Path(model_dir)
    _protect_output(
        output,
        model_paths=[model_dir],
        tokenizer_path=tokenizer_path or model_dir,
        arc_paths=[arc_easy_samples, arc_challenge_samples],
        piqa_data=piqa_data,
    )
    print(f"Loading {backend} model: {model_dir}", flush=True, file=sys.stderr)
    if backend == "torch":
        model, tokenizer = load_torch_bundle(
            model_dir,
            tokenizer_path=tokenizer_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
        )
    else:
        with redirect_stdout(sys.stderr):
            mx = _mlx()
            model, tokenizer = _load_bundle(model_dir)
            model.eval()
            mx.eval(model.parameters())
    protocol = _protocol(
        backend=backend,
        device=device,
        dtype=dtype,
        kernel_policy=kernel_policy,
        batch_size=batch_size,
        max_length=max_length,
        max_samples=max_samples,
    )
    loaded = _load_tasks(
        tasks,
        max_samples=max_samples,
        arc_easy_samples=arc_easy_samples,
        arc_challenge_samples=arc_challenge_samples,
        piqa_data=piqa_data,
    )
    report = {"model": str(model_dir.resolve()), **protocol, "complete": False, "benchmarks": {}}
    for task, result in _task_results(model, tokenizer, loaded, protocol):
        report["benchmarks"][task] = result
        report["complete"] = len(report["benchmarks"]) == len(tasks)
        _write_report(output, report)
    return report


def compare_choices(
    model_a,
    model_b,
    *,
    tokenizer_path,
    device="cpu",
    dtype="float32",
    tasks="piqa,arc_easy,arc_challenge,hellaswag",
    batch_size=16,
    max_length=2048,
    max_samples=None,
    kernel_policy="portable",
    arc_easy_samples=None,
    arc_challenge_samples=None,
    piqa_data=None,
    output=None,
):
    """Compare two native bundles with one tokenizer and one encoded task cache.

    Models load sequentially on the same device. Delta values are B minus A.
    Portable execution overrides config options in memory for both bundles.
    It does not establish parity with archived MLX/custom-kernel results.
    """
    tasks = _settings(tasks, batch_size, max_length, max_samples)
    if kernel_policy != "portable":
        raise ValueError("Comparison requires portable kernel policy for both checkpoints")
    paths = {"A": Path(model_a), "B": Path(model_b)}
    _protect_output(
        output,
        model_paths=paths.values(),
        tokenizer_path=tokenizer_path,
        arc_paths=[arc_easy_samples, arc_challenge_samples],
        piqa_data=piqa_data,
    )
    for path in paths.values():
        if not (path / "config.json").is_file() or not any(
            (path / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin", "model.pt")
        ):
            raise FileNotFoundError(f"Expected a local native config and model weights in {path}")
    tokenizer_path = Path(tokenizer_path)
    tokenizer_file = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Comparison requires a local tokenizer.json: {tokenizer_file}")
    with redirect_stdout(sys.stderr):
        _torch_placement(device, dtype)
        tokenizer = _load_torch_tokenizer(tokenizer_path)
    loaded = _load_tasks(
        tasks,
        max_samples=max_samples,
        arc_easy_samples=arc_easy_samples,
        arc_challenge_samples=arc_challenge_samples,
        piqa_data=piqa_data,
    )
    encoded = {
        task: encode_choices(tokenizer, examples, max_length) for task, examples in loaded.items()
    }
    protocol = _protocol(
        backend="torch",
        device=device,
        dtype=dtype,
        kernel_policy=kernel_policy,
        batch_size=batch_size,
        max_length=max_length,
        max_samples=max_samples,
    )
    report = {
        **protocol,
        "tokenizer": {
            "path": str(tokenizer_path.resolve()),
            "sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        },
        "datasets": {
            task: {
                "examples": len(examples),
                "examples_sha256": hashlib.sha256(
                    json.dumps([asdict(row) for row in examples], sort_keys=True).encode()
                ).hexdigest(),
            }
            for task, examples in loaded.items()
        },
        "models": {},
        "delta_b_minus_a": {},
        "complete": False,
    }
    for name, path in paths.items():
        print(f"Loading model {name}: {path}", flush=True, file=sys.stderr)
        with redirect_stdout(sys.stderr):
            model = _load_torch_model(path, device=device, dtype=dtype, kernel_policy=kernel_policy)
        model_report = {"model": str(path.resolve()), "benchmarks": {}, "complete": False}
        report["models"][name] = model_report
        for task, result in _task_results(model, tokenizer, loaded, protocol, encoded):
            model_report["benchmarks"][task] = result
            model_report["complete"] = len(model_report["benchmarks"]) == len(tasks)
            _write_report(output, report)
        del model
        # Portable instance-bound forwards can form module reference cycles.
        # Release the first checkpoint before allocating the second one.
        gc.collect()
    for task in tasks:
        a = report["models"]["A"]["benchmarks"][task]
        b = report["models"]["B"]["benchmarks"][task]
        report["delta_b_minus_a"][task] = {
            "acc": b["acc"] - a["acc"],
            "acc_norm": b["acc_norm"] - a["acc_norm"],
            "acc_points": 100 * (b["acc"] - a["acc"]),
            "acc_norm_points": 100 * (b["acc_norm"] - a["acc_norm"]),
        }
    report["complete"] = True
    _write_report(output, report)
    return report


def _shared_arguments(parser, *, comparison=False):
    parser.add_argument("--device", default="cpu", help="Explicit Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=comparison,
        help="One local tokenizer for both models" if comparison else "Local Torch tokenizer path",
    )
    parser.add_argument(
        "--kernel-policy",
        choices=("portable",) if comparison else ("checkpoint", "portable"),
        default="portable" if comparison else "checkpoint",
        help="portable: reference routing, eager attention and RMSNorm",
    )
    parser.add_argument("--arc-easy-samples", type=Path, help="Local ARC-Easy JSONL documents")
    parser.add_argument(
        "--arc-challenge-samples", type=Path, help="Local ARC-Challenge JSONL documents"
    )
    parser.add_argument(
        "--tasks",
        default="piqa,arc_easy,arc_challenge,hellaswag" if comparison else "hellaswag,mmlu,piqa",
        help="Comma-separated subset of arc_challenge,arc_easy,hellaswag,mmlu,piqa",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--max-samples", type=int, help="Optional prefix subset; default is the complete split"
    )
    parser.add_argument("--piqa-data", type=Path, help="Local dev.jsonl / dev-labels.lst directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=not comparison,
        help="Optional comparison JSON" if comparison else "Evaluation JSON report",
    )


def compare_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="complexity eval compare",
        description="Compare two native models using identical zero-shot choices.",
    )
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    _shared_arguments(parser, comparison=True)
    options = vars(parser.parse_args(argv))
    options["tokenizer_path"] = options.pop("tokenizer")
    print(json.dumps(compare_choices(**options), indent=2, sort_keys=True))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="complexity eval choices" if argv is not None else None, description=__doc__
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx")
    _shared_arguments(parser)
    options = vars(parser.parse_args(argv))
    options["tokenizer_path"] = options.pop("tokenizer")
    print(json.dumps(evaluate_choices(**options), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
