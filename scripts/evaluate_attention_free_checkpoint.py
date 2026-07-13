#!/usr/bin/env python3
"""Measure the behavioral profile of an attention-free checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from complexity.config import ModelConfig
from complexity.evaluation.associative_recall import (
    build_associative_recall_batch,
    build_induction_batch,
    score_target_logits,
)
from complexity.models import ComplexityModel


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def target_logits(model: ComplexityModel, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids, return_logits=False)
    hidden = output["last_hidden_state"][:, -1]
    weight = model.embed_tokens.weight if model.lm_head is None else model.lm_head.weight
    return F.linear(hidden, weight)


def recall_profile(
    model: ComplexityModel,
    task: str,
    distances: list[int],
    batch_size: int,
    seeds: list[int],
    device: torch.device,
) -> list[dict[str, float | int]]:
    builder = build_associative_recall_batch if task == "associative_recall" else build_induction_batch
    rows = []
    for distance in distances:
        accumulators = {"accuracy": [], "mean_rank": [], "mean_margin": [], "nll": []}
        for seed in seeds:
            batch = builder(batch_size, distance, model.config.vocab_size, seed)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = target_logits(model, batch.input_ids.to(device))
            metrics = asdict(score_target_logits(logits.float(), batch.target_ids.to(device)))
            for name, value in metrics.items():
                accumulators[name].append(value)
        rows.append(
            {
                "distance": distance,
                "examples": batch_size * len(seeds),
                **{name: sum(values) / len(values) for name, values in accumulators.items()},
            }
        )
    return rows


def causality_check(model: ComplexityModel, device: torch.device) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(101)
    prefix = torch.randint(0, model.config.vocab_size, (1, 256), generator=generator)
    future_a = torch.randint(0, model.config.vocab_size, (1, 256), generator=generator)
    future_b = torch.randint(0, model.config.vocab_size, (1, 256), generator=generator)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden_a = model(torch.cat((prefix, future_a), dim=1).to(device), return_logits=False)[
            "last_hidden_state"
        ][:, : prefix.shape[1]]
        hidden_b = model(torch.cat((prefix, future_b), dim=1).to(device), return_logits=False)[
            "last_hidden_state"
        ][:, : prefix.shape[1]]
    difference = (hidden_a.float() - hidden_b.float()).abs()
    return {
        "prefix_tokens": prefix.shape[1],
        "future_tokens": future_a.shape[1],
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
    }


def incremental_check(model: ComplexityModel, device: torch.device, length: int) -> dict[str, object]:
    def tensors(value: object) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            return [tensor for item in value for tensor in tensors(item)]
        raise TypeError(f"unsupported state value: {type(value).__name__}")

    generator = torch.Generator().manual_seed(202)
    input_ids = torch.randint(0, model.config.vocab_size, (1, length), generator=generator).to(device)
    checkpoints = {1, 8, 32, 128, 256, 512, length}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_hidden = model(input_ids, return_logits=False)["last_hidden_state"]
        cache = None
        initial_pointers = None
        rows = []
        for index in range(length):
            output = model(
                input_ids[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
                return_logits=False,
            )
            cache = output["past_key_values"]
            assert cache is not None
            flat_cache = tensors(cache)
            pointers = [state.data_ptr() for state in flat_cache]
            if initial_pointers is None:
                initial_pointers = pointers
            if index + 1 in checkpoints:
                difference = (
                    output["last_hidden_state"][:, -1].float()
                    - full_hidden[:, index].float()
                ).abs()
                rows.append(
                    {
                        "length": index + 1,
                        "max_abs_difference": float(difference.max().item()),
                        "mean_abs_difference": float(difference.mean().item()),
                        "cache_addresses_stable": pointers == initial_pointers,
                    }
                )
    assert cache is not None
    flat_cache = tensors(cache)
    state_elements = sum(state.numel() for state in flat_cache)
    return {
        "comparisons": rows,
        "state_shapes": [list(state.shape) for state in flat_cache],
        "state_elements": state_elements,
        "state_bytes_bf16": state_elements * 2,
        "state_independent_of_context": all(row["cache_addresses_stable"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47, 71])
    parser.add_argument("--incremental-length", type=int, default=1024)
    args = parser.parse_args()

    run_config = json.loads(args.run_config.read_text())
    config = ModelConfig(**run_config["model_config"])
    model = ComplexityModel(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    incompatibility = model.load_state_dict(state, strict=False)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatibility.missing_keys}, "
            f"unexpected={incompatibility.unexpected_keys}"
        )
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16).eval()

    distances = args.distances
    seeds = args.seeds
    dilations = [int(getattr(layer.self_attn, "dilation")) for layer in model.layers]
    result = {
        "schema_version": 1,
        "checkpoint": {
            "sha256": checkpoint_sha256(args.checkpoint),
            "step": checkpoint.get("step"),
            "registered_total_tokens": run_config["total_tokens"],
            "parameters": run_config["params"],
        },
        "architecture": {
            "attention_type": config.attention_type,
            "mlp_type": config.mlp_type,
            "has_qkv_parameter_names": any(
                fragment in name.lower()
                for name, _ in model.named_parameters()
                for fragment in ("q_proj", "k_proj", "v_proj", "qkv_proj")
            ),
            "dilations": dilations,
            "receptive_field_tokens": 1
            + (config.causal_conv_kernel_size - 1)
            * sum(dilations),
        },
        "protocol": {
            "distances": distances,
            "batch_size": args.batch_size,
            "seeds": seeds,
            "precision": "bfloat16",
            "device": torch.cuda.get_device_name(0),
        },
        "associative_recall": recall_profile(
            model, "associative_recall", distances, args.batch_size, seeds, device
        ),
        "induction": recall_profile(model, "induction", distances, args.batch_size, seeds, device),
        "causality": causality_check(model, device),
        "incremental_equivalence": incremental_check(
            model, device, args.incremental_length
        ),
    }
    shared_context = getattr(model.layers[0].self_attn, "shared_context", None)
    if shared_context is not None:
        output_gate = getattr(shared_context, "output_gate")
        saved_gate = output_gate.detach().clone()
        with torch.no_grad():
            output_gate.zero_()
        result["context_ablation"] = {
            "intervention": "shared associative context output gate forced to zero",
            "associative_recall": recall_profile(
                model, "associative_recall", distances, args.batch_size, seeds, device
            ),
            "induction": recall_profile(
                model, "induction", distances, args.batch_size, seeds, device
            ),
        }
        with torch.no_grad():
            output_gate.copy_(saved_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
