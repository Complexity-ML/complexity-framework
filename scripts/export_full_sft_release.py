#!/usr/bin/env python3
"""Export the PIQA-selected full-parameter SFT checkpoint for Hugging Face."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from complexity.inference.chat_template import huggingface_chat_template, validate_chat_template


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_piqa_reports(directory: Path) -> list[dict]:
    reports = []
    for path in sorted(directory.glob("epoch_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_path"] = path
        reports.append(report)
    if len(reports) != 3:
        raise ValueError(f"Expected three PIQA reports, found {len(reports)} in {directory}")
    return reports


def eval_rows(metrics: Path) -> dict[int, dict[str, str]]:
    with metrics.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["step"]): row for row in csv.DictReader(handle)}
    selected = {step: rows[step] for step in (463, 926, 1389)}
    if any(not row["matched_eval_loss"] for row in selected.values()):
        raise ValueError("An epoch boundary is missing matched SFT evaluation metrics")
    return selected


def render_readme(reports: list[dict], rows: dict[int, dict[str, str]]) -> str:
    lines = []
    for epoch, report in enumerate(reports, start=1):
        step = int(report["checkpoint_step"])
        piqa = report["benchmarks"]["piqa"]
        lines.append(
            f"| {epoch} | {step:,} | {float(rows[step]['matched_eval_loss']):.6f} | "
            f"{float(rows[step]['matched_eval_ppl']):.2f} | {100 * float(piqa['acc']):.2f}% | "
            f"{100 * float(piqa['acc_norm']):.2f}% |"
        )
    table = "\n".join(lines)
    return f"""---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
library_name: pytorch
tags:
- safetensors
- tr-hash
- mixture-of-experts
- gqa
- supervised-finetuning
- full-parameter-finetuning
- custom-code
base_model: AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement
datasets:
- AETHORIA-AI/luciole-16way-sft-209k
---

# TR-HASH MoE 200M — 160B-source Full SFT

Full-parameter instruction SFT of
[`AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement`](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement)
on the audited 209k-example Luciole 16-way mixture. This is **not LoRA or QLoRA**:
all 201,194,368 parameters were trainable.

The root `model.safetensors` is the PIQA-selected **epoch 2 / step 926** model.
All three resumable training checkpoints remain in `step_000463`, `step_000926`,
and `step_001389`.

## Results

| Epoch | Step | Held-out SFT loss | SFT ppl | PIQA acc | PIQA acc_norm |
|---:|---:|---:|---:|---:|---:|
{table}

Epoch 2 was promoted because all epochs tie on PIQA `acc_norm`, while epoch 2
has the highest raw PIQA accuracy. Epoch 3 has the lowest held-out SFT loss but
was not silently substituted for the benchmark-selected release.

PIQA protocol: full 1,838-example validation split, zero-shot causal
continuation log-likelihood, no chat template, maximum sequence length 2,048,
FP16 eager PyTorch with the custom Triton kernels enabled.

## Training recipe

| Setting | Value |
|---|---|
| Method | Full-parameter supervised fine-tuning |
| Dataset | Luciole 16-way, 209,000 train / 2,100 held out |
| Supervision | Final assistant response only; prior assistant turns masked |
| Epochs | 3 |
| Sequence cap | 512 tokens |
| Tokenizer | TR-HASH 32k vocabulary |
| Optimizer | AdamW, betas 0.9 / 0.95, weight decay 0.1 |
| LR schedule | 2e-5 peak, 3% warmup, one continuous cosine decay over all epochs |
| Precision | BF16 training |
| Kernels | Liger required; custom Triton enabled |

The tokenized dataset view is published under
`tokenized/tr-hash-32k-v1/` in
[`AETHORIA-AI/luciole-16way-sft-209k`](https://huggingface.co/datasets/AETHORIA-AI/luciole-16way-sft-209k),
including hashes and the assistant-only label mask.

## Architecture

201.2M parameters, 16 transformer layers, GQA (14 query heads / 2 KV heads),
deterministic token-ID-routed TR-Hash MoE (4 stored experts, top-2 active), and
tied embeddings. Loading requires the Complexity Framework / TR-Hash runtime.

## License

CC BY-NC 4.0. Source datasets and individual documents retain their own
licenses and terms.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--piqa-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    reports = read_piqa_reports(args.piqa_dir)
    rows = eval_rows(args.metrics)
    selected = max(
        reports,
        key=lambda report: (
            float(report["benchmarks"]["piqa"]["acc_norm"]),
            float(report["benchmarks"]["piqa"]["acc"]),
        ),
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if int(state["step"]) != int(selected["checkpoint_step"]):
        raise ValueError("The supplied checkpoint is not the PIQA-selected checkpoint")

    args.output.mkdir(parents=True)
    weights = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state["model"].items()
    }
    weights_path = args.output / "model.safetensors"
    save_file(weights, str(weights_path))
    loaded = load_file(str(weights_path), device="cpu")
    if loaded.keys() != weights.keys() or any(
        not torch.equal(loaded[name], tensor) for name, tensor in weights.items()
    ):
        raise RuntimeError("SafeTensors round-trip verification failed")

    (args.output / "config.json").write_text(
        json.dumps(state["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    template = validate_chat_template(state["chat_template"])
    (args.output / "chat_template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "chat_template.jinja").write_text(
        huggingface_chat_template(template) + "\n",
        encoding="utf-8",
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        source = args.tokenizer / name
        if source.is_file():
            shutil.copy2(source, args.output / name)

    report_dir = args.output / "reports" / "piqa"
    report_dir.mkdir(parents=True)
    for epoch, report in enumerate(reports, start=1):
        shutil.copy2(report["_path"], report_dir / f"epoch-{epoch:02d}.json")
    training_dir = args.output / "reports" / "training"
    training_dir.mkdir(parents=True)
    shutil.copy2(args.metrics, training_dir / "metrics.csv")
    (args.output / "README.md").write_text(render_readme(reports, rows), encoding="utf-8")

    piqa = selected["benchmarks"]["piqa"]
    manifest = {
        "schema_version": 1,
        "release": "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT",
        "method": "full-parameter-sft",
        "source_model": "AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement",
        "dataset": "AETHORIA-AI/luciole-16way-sft-209k",
        "selected_epoch": 2,
        "selected_step": int(state["step"]),
        "selection": "max PIQA acc_norm, then max PIQA acc",
        "piqa_acc": float(piqa["acc"]),
        "piqa_acc_norm": float(piqa["acc_norm"]),
        "weights_sha256": sha256(weights_path),
    }
    (args.output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
