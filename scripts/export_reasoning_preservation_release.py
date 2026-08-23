#!/usr/bin/env python3
"""Export the promoted reasoning-preservation checkpoint as a Hub bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
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

REPO_ID = "AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-Preservation-50M"
BASE_MODEL = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
GENERAL_DATASET = "AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-300K"
REASONING_DATASET = "AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_readme(summary: dict[str, Any]) -> str:
    source = summary["source"]
    selected = summary["selected"]
    rows = []
    for item in summary["candidates"]:
        rows.append(
            "| {step:,} | {loss:.6f} | {ppl:.2f} | {piqa} | {arc} | "
            "{reasoning} | {behavior}/{total} | {eligible} |".format(
                step=item["step"],
                loss=item["matched_eval_loss"],
                ppl=item["matched_eval_ppl"],
                piqa=percent(item["piqa_acc_norm"]),
                arc=percent(item["arc_acc_norm"]),
                reasoning=percent(item["arc_reasoning_native_accuracy"]),
                behavior=item["behavior_passes"],
                total=item["behavior_total"],
                eligible="yes" if item["eligible"] else "no",
            )
        )
    return f"""---
license: apache-2.0
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
base_model: {BASE_MODEL}
datasets:
- {GENERAL_DATASET}
- {REASONING_DATASET}
---

# TR-HASH MoE 200M — Reasoning Preservation 50M

Conservative full-parameter reasoning extension of
[`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}). The stage mixes roughly
**150M replay tokens** from the released general SFT corpus with **50M audited
reasoning tokens**, for 200,002,668 formatted tokens in one epoch. It is not
LoRA or QLoRA.

The root `model.safetensors` is checkpoint **{selected['step']:,}**, selected
only after passing preservation controls against the source SFT v2 model. Root
weights are F32 SafeTensors.

## Selection results

| Step | Held-out loss | PPL | PIQA acc_norm | ARC acc_norm | ARC-64 native | Assistant panel | Eligible |
|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

Source SFT v2 controls: PIQA acc_norm **{percent(source['piqa_acc_norm'])}**,
combined ARC acc_norm **{percent(source['arc_acc_norm'])}**, ARC-64 native
accuracy **{percent(source['arc_reasoning_native_accuracy'])}**, and
**{source['behavior_passes']}/{source['behavior_total']}** assistant-panel
checks.

Selected step {selected['step']:,}: PIQA acc_norm
**{percent(selected['piqa_acc_norm'])}**, combined ARC acc_norm
**{percent(selected['arc_acc_norm'])}**, ARC-64 native accuracy
**{percent(selected['arc_reasoning_native_accuracy'])}**, and
**{selected['behavior_passes']}/{selected['behavior_total']}** assistant-panel
checks.

Selection policy: {summary['selection_policy']}.

## Evaluation protocols

- **PIQA:** complete 1,838-example validation split, zero-shot causal
  continuation likelihood, no chat template, FP16 inference.
- **ARC zero-shot:** complete 2,376 ARC-Easy + 1,172 ARC-Challenge public test
  splits, causal continuation likelihood, no demonstrations or chat template.
- **ARC reasoning probe:** 64 deterministic, evenly spaced questions (32 per
  ARC split), free greedy generation from the bare question and labeled choices;
  both parsing failures and answers are retained.
- **Assistant panel:** eight fixed chat-template prompts covering conversation,
  arithmetic, executable code, memory and formatting constraints.

The likelihood and generative ARC protocols measure different behavior and
must not be compared as if they were the same benchmark.

All reports, logs and generated traces are published under
`reports/reasoning-preservation-50m/`. All saved training checkpoints remain
under `training/reasoning-preservation-50m/checkpoints/`.

## Training recipe

| Setting | Value |
|---|---|
| Initialization | Released full-parameter SFT v2 checkpoint |
| Method | Full-parameter supervised fine-tuning |
| General replay | 150,000,130 formatted tokens |
| Reasoning addition | 50,002,538 formatted tokens |
| Total | 200,002,668 formatted tokens |
| Epochs | 1 |
| Context | 2,048 tokens, sequence packed |
| Peak learning rate | 1.5e-7 |
| Precision | BF16 training; F32 root release |
| Kernels | Liger fused linear CE required; custom Triton enabled |

## Architecture and loading

201.2M parameters, 16 decoder layers, GQA (14 query heads / 2 KV heads), four
stored deterministic token-ID-routed experts with top-2 activation, an
always-on shared SwiGLU path and tied embeddings. Persisted multi-hash routing
tables are included in the checkpoint.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{REPO_ID}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

## License

Apache-2.0 for this released checkpoint. Dataset records retain their upstream
licenses and terms.
"""


def export_release(
    *,
    summary_path: Path,
    metrics_path: Path,
    evaluation_root: Path,
    tokenizer_dir: Path,
    dataset_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary.get("selected")
    if not summary.get("release_ready") or selected is None:
        raise ValueError("Selection summary does not authorize release")
    checkpoint = Path(selected["checkpoint"])
    state = torch.load(
        checkpoint / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=False
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

    reports = output / "reports" / "reasoning-preservation-50m"
    shutil.copytree(evaluation_root, reports / "evaluations")
    (reports / "evaluations" / ".evaluation_complete").unlink(missing_ok=True)
    shutil.copy2(metrics_path, reports / "metrics.csv")
    shutil.copy2(dataset_manifest, reports / "dataset-manifest.json")
    (output / "README.md").write_text(render_readme(summary), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "release": REPO_ID,
        "method": "full-parameter-reasoning-preservation-sft",
        "source_model": BASE_MODEL,
        "training_formatted_tokens": 200_002_668,
        "general_replay_tokens": 150_000_130,
        "reasoning_tokens": 50_002_538,
        "epochs": 1,
        "selected_step": int(selected["step"]),
        "selection_policy": summary["selection_policy"],
        "piqa_acc": float(selected["piqa_acc"]),
        "piqa_acc_norm": float(selected["piqa_acc_norm"]),
        "arc_acc": float(selected["arc_acc"]),
        "arc_acc_norm": float(selected["arc_acc_norm"]),
        "arc_reasoning_native_accuracy": float(
            selected["arc_reasoning_native_accuracy"]
        ),
        "behavior_passes": int(selected["behavior_passes"]),
        "behavior_total": int(selected["behavior_total"]),
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
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_release(
        summary_path=args.summary,
        metrics_path=args.metrics,
        evaluation_root=args.evaluation_root,
        tokenizer_dir=args.tokenizer,
        dataset_manifest=args.dataset_manifest,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
