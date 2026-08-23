#!/usr/bin/env python3
"""Export the selected 500M-token reasoning-SFT checkpoint as a Hub bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from complexity.inference.chat_template import (
    huggingface_chat_template,
    validate_chat_template,
)
from scripts.export_tr_hash_transformers import (
    ADAPTER,
    TOKENIZER_FILES,
    build_transformers_config,
    tokenizer_special_token_ids,
)

REPO_ID = "AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-SFT"
BASE_MODEL = "AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement"
DATASET = "AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_readme(
    summary: dict[str, Any],
    arc: dict[str, Any] | None,
    source_arc_zero_shot: dict[str, Any] | None,
    selected_arc_zero_shot: dict[str, Any] | None,
) -> str:
    selected = summary["selected"]
    selection_basis = summary.get("release_selection_basis", "automatic-evaluation")
    if selection_basis == "free-generation-panel":
        selection_description = (
            "selected manually among the reasoning checkpoints for the strongest "
            "free-generation behavior on the fixed eight-prompt assistant panel"
        )
    else:
        selection_description = (
            "selected among the saved training checkpoints by the documented "
            "automatic evaluation policy"
        )
    learning_rate = float(summary.get("peak_learning_rate", 5e-6))
    rows = "\n".join(
        "| {step:,} | {loss:.6f} | {ppl:.2f} | {acc:.2f}% | {norm:.2f}% |".format(
            step=candidate["step"],
            loss=candidate["matched_eval_loss"],
            ppl=candidate["matched_eval_ppl"],
            acc=100 * candidate["piqa_acc"],
            norm=100 * candidate["piqa_acc_norm"],
        )
        for candidate in summary["candidates"]
    )
    arc_section = ""
    if arc is not None:
        combined = arc["combined"]
        arc_section = f"""
## Generative reasoning probe

The selected checkpoint was also evaluated on a deterministic 128-question
probe (64 evenly spaced ARC-Easy + 64 ARC-Challenge test questions). It
generated a reasoning trace and an explicit final answer; this is separate
from the full-split likelihood results.

| Probe | Strict accuracy | Flexible accuracy | Strict format rate |
|---|---:|---:|---:|
| ARC reasoning 128 | {100 * combined["strict_accuracy"]:.2f}% | {100 * combined["flexible_accuracy"]:.2f}% | {100 * combined["strict_format_rate"]:.2f}% |
"""
    arc_retention_section = ""
    if source_arc_zero_shot is not None and selected_arc_zero_shot is not None:
        labels = (
            ("ARC-Easy", "arc_easy"),
            ("ARC-Challenge", "arc_challenge"),
            ("Combined ARC", "combined"),
        )
        retention_rows = []
        for label, key in labels:
            source = (
                source_arc_zero_shot[key]
                if key == "combined"
                else source_arc_zero_shot["benchmarks"][key]
            )
            selected_arc = (
                selected_arc_zero_shot[key]
                if key == "combined"
                else selected_arc_zero_shot["benchmarks"][key]
            )
            retention_rows.append(
                "| {label} | {source_acc:.2f}% | {selected_acc:.2f}% | "
                "{source_norm:.2f}% | {selected_norm:.2f}% |".format(
                    label=label,
                    source_acc=100 * source["acc"],
                    selected_acc=100 * selected_arc["acc"],
                    source_norm=100 * source["acc_norm"],
                    selected_norm=100 * selected_arc["acc_norm"],
                )
            )
        arc_retention_section = """
## ARC zero-shot retention

The Refinement source and selected Reasoning-SFT checkpoint were evaluated
with the same full-split causal-continuation protocol: no demonstrations, no
chat template and no generated-answer parsing. This is the capability
retention control; it is distinct from the generative reasoning probe below.

| Benchmark | Source acc | Reasoning SFT acc | Source acc_norm | Reasoning SFT acc_norm |
|---|---:|---:|---:|---:|
{rows}
""".format(rows="\n".join(retention_rows))
    return f"""---
license: cc-by-nc-4.0
language:
- en
- fr
pipeline_tag: text-generation
library_name: transformers
tags:
- pytorch
- safetensors
- tr-hash
- mixture-of-experts
- gqa
- reasoning
- supervised-finetuning
- full-parameter-finetuning
- custom-code
base_model: {BASE_MODEL}
datasets:
- {DATASET}
---

# TR-HASH MoE 200M — 160B-source Reasoning SFT

Full-parameter reasoning and instruction SFT of
[`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) on the audited
[`{DATASET}`](https://huggingface.co/datasets/{DATASET}). All 201.2M model
parameters were trained; this release is **not LoRA or QLoRA**.

The root `model.safetensors` is step **{selected["step"]:,}**, {selection_description}.
Root weights are **F32 SafeTensors**.

This is an **experimental reasoning checkpoint**. On the matched free-generation
panel it was stronger than the other later reasoning candidates, but it did not
beat the released general-purpose Full SFT checkpoint. It must not be interpreted
as a replacement for that model.

## Checkpoint results

| Step | Held-out SFT loss | SFT ppl | PIQA acc | PIQA acc_norm |
|---:|---:|---:|---:|---:|
{rows}

PIQA uses all 1,838 validation examples, zero-shot causal-continuation
log-likelihood, no chat template, maximum length 2,048 and FP16 inference.
No hidden quality threshold is applied: every saved checkpoint is reported.
Reports and raw generative traces are under `evaluation/reasoning-sft-500m/`.
{arc_retention_section}
{arc_section}
## Training recipe

| Setting | Value |
|---|---|
| Method | Full-parameter supervised fine-tuning |
| Initialization | Refinement step 8,156 (about 162B prior token exposures) |
| Dataset | 882,408 train / 2,350 held-out conversations |
| Training corpus | 500,000,669 unique formatted tokens; no repeated epochs |
| Benchmark isolation | ARC, PIQA, GSM8K and HellaSwag prompts denied from training |
| Supervision | Final assistant turn; prior assistant history masked |
| Epochs | 1 |
| Context | 2,048 tokens, sequence packed |
| Optimizer | AdamW, betas 0.9 / 0.95, weight decay 0.1 |
| LR | {learning_rate:.1e} peak, 3% warmup, continuous cosine decay |
| Precision | BF16 training; F32 root release |
| Kernels | Liger required; custom Triton enabled |

Reasoning is learned from worked solutions in the corpus. The chat template
does not force or fabricate a hidden `<think>` block.

## Architecture and loading

201.2M parameters, 16 decoder layers, GQA (14 query heads / 2 KV heads), four
stored deterministic token-ID-routed experts with top-2 activation, an
always-on shared SwiGLU path and tied embeddings. Persisted multi-hash routing
tables are part of the checkpoint.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{REPO_ID}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

## License

The model follows the CC BY-NC 4.0 license of the Refinement source
checkpoint. Dataset records retain their individual upstream licenses; see
the dataset manifest.
"""


def export_release(
    *,
    summary_path: Path,
    metrics_path: Path,
    evaluation_root: Path,
    tokenizer_dir: Path,
    dataset_audit: Path,
    output: Path,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary.get("selected")
    if not summary.get("release_ready") or selected is None:
        raise ValueError("Selection summary does not authorize release")
    checkpoint = Path(selected["checkpoint"])
    state = torch.load(
        checkpoint / "checkpoint.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if int(state["step"]) != int(selected["step"]):
        raise ValueError("Selected checkpoint step does not match state")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    weights = {
        name: (
            tensor.detach().cpu().to(torch.float32).contiguous()
            if tensor.is_floating_point()
            else tensor.detach().cpu().contiguous()
        )
        for name, tensor in state["model"].items()
    }
    weights_path = output / "model.safetensors"
    save_file(weights, str(weights_path), metadata={"format": "pt"})
    round_trip = load_file(str(weights_path), device="cpu")
    if round_trip.keys() != weights.keys() or any(
        not torch.equal(round_trip[name], tensor) for name, tensor in weights.items()
    ):
        raise RuntimeError("SafeTensors round-trip verification failed")

    config = build_transformers_config(state["config"])
    config["torch_dtype"] = "float32"
    config.update(tokenizer_special_token_ids(tokenizer_dir))
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "model_config.yaml").write_text(
        yaml.safe_dump(state["config"], sort_keys=True), encoding="utf-8"
    )
    for filename in ("configuration_tr_hash_moe.py", "modeling_tr_hash_moe.py"):
        shutil.copy2(ADAPTER / filename, output / filename)

    template = validate_chat_template(state["chat_template"])
    (output / "chat_template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "chat_template.jinja").write_text(
        huggingface_chat_template(template) + "\n", encoding="utf-8"
    )
    for path in tokenizer_dir.iterdir():
        if path.is_file() and path.name in TOKENIZER_FILES:
            shutil.copy2(path, output / path.name)

    reports = output / "reports" / "reasoning-sft-500m"
    shutil.copytree(evaluation_root, reports / "evaluations")
    (reports / "evaluations" / ".evaluation_complete").unlink(missing_ok=True)
    shutil.copy2(metrics_path, reports / "metrics.csv")
    shutil.copy2(summary_path, reports / "selection_summary.json")
    shutil.copy2(dataset_audit, reports / "dataset-release-audit.json")

    def load_optional_report(filename: str) -> dict[str, Any] | None:
        path = evaluation_root / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    arc = load_optional_report("selected_arc_reasoning_64.json")
    source_arc_zero_shot = load_optional_report("source_arc_zero_shot_full.json")
    selected_arc_zero_shot = load_optional_report("selected_arc_zero_shot_full.json")
    if (source_arc_zero_shot is None) != (selected_arc_zero_shot is None):
        raise ValueError("ARC zero-shot source and selected reports must be paired")
    (output / "README.md").write_text(
        render_readme(summary, arc, source_arc_zero_shot, selected_arc_zero_shot),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "release": REPO_ID,
        "method": "full-parameter-reasoning-sft",
        "source_model": BASE_MODEL,
        "source_checkpoint_step": 8156,
        "dataset": DATASET,
        "training_unique_formatted_tokens": 500_000_669,
        "epochs": 1,
        "selected_step": int(selected["step"]),
        "selection_policy": summary["selection_policy"],
        "piqa_acc": float(selected["piqa_acc"]),
        "piqa_acc_norm": float(selected["piqa_acc_norm"]),
        "matched_eval_loss": float(selected["matched_eval_loss"]),
        "matched_eval_ppl": float(selected["matched_eval_ppl"]),
        "weights_sha256": sha256(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_dtype": "float32",
        "parameters": sum(tensor.numel() for tensor in weights.values()),
        "architecture": "tr_hash_moe",
        "num_experts": int(config["num_experts"]),
        "num_experts_per_tok": int(config["num_experts_per_tok"]),
    }
    if source_arc_zero_shot is not None and selected_arc_zero_shot is not None:
        manifest["arc_zero_shot"] = {
            "protocol": "full_split_causal_choice_loglikelihood_no_chat_template",
            "source": source_arc_zero_shot,
            "selected": selected_arc_zero_shot,
        }
    if not math.isfinite(manifest["matched_eval_loss"]):
        raise ValueError("Non-finite selected evaluation loss")
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_release(
        summary_path=args.summary,
        metrics_path=args.metrics,
        evaluation_root=args.evaluation_root,
        tokenizer_dir=args.tokenizer,
        dataset_audit=args.dataset_audit,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
