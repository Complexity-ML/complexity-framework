#!/usr/bin/env python3
"""Export the promotion-gated clean-SFT v2 checkpoint as a Hub root bundle."""

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

from complexity.inference.chat_template import huggingface_chat_template, validate_chat_template
from scripts.export_tr_hash_transformers import (
    ADAPTER,
    TOKENIZER_FILES,
    build_transformers_config,
    tokenizer_special_token_ids,
)

REPO_ID = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
BASE_MODEL = "AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement"
DATASET = "AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-300K"

_PUBLIC_FLOAT_DTYPES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_readme(summary: dict[str, Any]) -> str:
    selected = summary["selected"]
    rows = []
    for candidate in summary["candidates"]:
        rows.append(
            "| {epoch} | {step:,} | {loss:.6f} | {ppl:.2f} | {acc:.2f}% | "
            "{acc_norm:.2f}% | {gate} |".format(
                epoch=candidate["epoch"],
                step=candidate["step"],
                loss=candidate["matched_eval_loss"],
                ppl=candidate["matched_eval_ppl"],
                acc=100 * candidate["piqa_acc"],
                acc_norm=100 * candidate["piqa_acc_norm"],
                gate="pass" if candidate["promotion_passed"] else "fail",
            )
        )
    table = "\n".join(rows)
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
- supervised-finetuning
- full-parameter-finetuning
- custom-code
base_model: {BASE_MODEL}
datasets:
- {DATASET}
---

# TR-HASH MoE 200M — 160B-source Full SFT v2

Full-parameter instruction SFT of
[`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) on the audited
[`{DATASET}`](https://huggingface.co/datasets/{DATASET}). This release is
**not LoRA or QLoRA**: all 201.2M model parameters were trainable.

The root `model.safetensors` is the promotion-gated **epoch
{selected['epoch']} / step {selected['step']:,}** checkpoint. Selection first
requires the code, mathematics, multi-turn memory and instruction-following
regressions to pass, then ranks candidates by PIQA `acc_norm`, raw PIQA
accuracy and held-out SFT loss.

## Results

| Epoch | Step | Held-out SFT loss | SFT ppl | PIQA acc | PIQA acc_norm | Behavior gate |
|---:|---:|---:|---:|---:|---:|:---:|
{table}

PIQA uses the complete 1,838-example validation split, zero-shot causal
continuation log-likelihood, no chat template, maximum length 2,048 and FP16
PyTorch with the custom Triton path. Behavioral reports and exact prompts are
published under `training/sft-v2-300k/evaluations/`.

## Training recipe

| Setting | Value |
|---|---|
| Method | Full-parameter supervised fine-tuning |
| Source | Refinement step 8,156 (about 162B prior token exposures) |
| Dataset | 300,000 train / 3,000 held-out examples |
| Tokenized corpus | 202,948,693 train tokens; no truncation |
| Supervision | Final assistant turn only; prior assistant turns masked |
| Epochs | 3 |
| Context | 2,048 tokens |
| Tokenizer | TR-HASH 32,000-token vocabulary; EOS `</s>` (ID 0) |
| Optimizer | AdamW, betas 0.9 / 0.95, weight decay 0.1 |
| LR | 2e-5 peak, 3% warmup, continuous cosine decay |
| Precision | BF16 training |
| Root SafeTensors precision | {selected['weights_dtype']} |
| Kernels | Liger required; custom Triton enabled |

## Architecture and loading

201.2M parameters, 16 decoder layers, GQA (14 query heads / 2 KV heads),
four stored deterministic token-ID-routed experts with top-2 activation, an
always-on shared SwiGLU path and tied embeddings. The persisted multi-hash
routing tables are part of the checkpoint.

The repository includes an autonomous Transformers adapter. Load it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{REPO_ID}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

## License

The released SFT checkpoint is Apache-2.0. Source datasets retain their own
licenses and terms; see the dataset manifest for the per-source audit.
"""


def export_release(
    *,
    summary_path: Path,
    metrics_path: Path,
    evaluation_root: Path,
    tokenizer_dir: Path,
    output: Path,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("release_ready") or not summary.get("selected"):
        raise ValueError("Selection summary does not authorize release promotion")
    selected = summary["selected"]
    checkpoint = Path(selected["checkpoint"])
    state = torch.load(
        checkpoint / "checkpoint.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if int(state["step"]) != int(selected["step"]):
        raise ValueError("Selected checkpoint step does not match checkpoint state")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    weights = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state["model"].items()
    }
    floating_dtypes = {
        tensor.dtype for tensor in weights.values() if tensor.is_floating_point()
    }
    if len(floating_dtypes) != 1:
        raise ValueError(
            "Root release requires one unambiguous floating-point dtype, found "
            f"{sorted(map(str, floating_dtypes))}"
        )
    floating_dtype = next(iter(floating_dtypes))
    try:
        weights_dtype = _PUBLIC_FLOAT_DTYPES[floating_dtype]
    except KeyError as error:
        raise ValueError(
            "Root release weights must be float32 or bfloat16, found "
            f"{floating_dtype}"
        ) from error
    selected["weights_dtype"] = weights_dtype
    weights_path = output / "model.safetensors"
    save_file(weights, str(weights_path), metadata={"format": "pt"})
    round_trip = load_file(str(weights_path), device="cpu")
    if round_trip.keys() != weights.keys() or any(
        not torch.equal(round_trip[name], tensor) for name, tensor in weights.items()
    ):
        raise RuntimeError("SafeTensors round-trip verification failed")

    config = build_transformers_config(state["config"])
    config["torch_dtype"] = weights_dtype
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

    reports = output / "reports" / "sft-v2-300k"
    shutil.copytree(evaluation_root, reports / "evaluations")
    (reports / "evaluations" / ".evaluation_complete").unlink(missing_ok=True)
    shutil.copy2(metrics_path, reports / "metrics.csv")
    shutil.copy2(summary_path, reports / "selection_summary.json")
    (output / "README.md").write_text(render_readme(summary), encoding="utf-8")

    weights_hash = sha256(weights_path)
    manifest = {
        "schema_version": 2,
        "release": REPO_ID,
        "method": "full-parameter-sft",
        "source_model": BASE_MODEL,
        "source_checkpoint_step": 8156,
        "dataset": DATASET,
        "dataset_revision": "084a658ec47e4ee872f6d67fdbad3602f599424b",
        "selected_epoch": int(selected["epoch"]),
        "selected_step": int(selected["step"]),
        "selection_policy": summary["selection_policy"],
        "piqa_acc": float(selected["piqa_acc"]),
        "piqa_acc_norm": float(selected["piqa_acc_norm"]),
        "matched_eval_loss": float(selected["matched_eval_loss"]),
        "matched_eval_ppl": float(selected["matched_eval_ppl"]),
        "behavior_gate_passed": bool(selected["promotion_passed"]),
        "weights_sha256": weights_hash,
        "weights_bytes": weights_path.stat().st_size,
        "weights_dtype": weights_dtype,
        "floating_parameters": sum(
            tensor.numel() for tensor in weights.values() if tensor.is_floating_point()
        ),
        "parameters": sum(tensor.numel() for tensor in weights.values()),
        "architecture": "tr_hash_moe",
        "num_experts": int(config["num_experts"]),
        "num_experts_per_tok": int(config["num_experts_per_tok"]),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_release(
        summary_path=args.summary,
        metrics_path=args.metrics,
        evaluation_root=args.evaluation_root,
        tokenizer_dir=args.tokenizer,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
