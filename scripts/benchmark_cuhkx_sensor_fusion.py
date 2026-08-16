#!/usr/bin/env python3
"""Benchmark one CUHK-X training step without touching competition data."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext

import torch

from complexity.generative.sensor_fusion import (
    SENSOR_MODALITIES,
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--clip-frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--shared-width", type=int, default=128)
    parser.add_argument("--expert-width", type=int, default=64)
    parser.add_argument("--class-hash-expert-width", type=int, default=32)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast(device: torch.device, precision: str):
    if precision == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def inputs(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    batch = args.batch_size
    frames = args.clip_frames
    size = args.image_size
    return {
        "depth": torch.randn(batch, 3, frames, size, size, device=device),
        "ir": torch.randn(batch, 1, frames, size, size, device=device),
        "thermal": torch.randn(batch, 3, frames, size, size, device=device),
        "imu": torch.randn(batch, 64, 45, device=device),
        "radar": torch.randn(batch, 64, 16, device=device),
        "skeleton": torch.randn(batch, 64, 17, 3, device=device),
    }


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.image_size, args.clip_frames, args.iterations) <= 0:
        raise ValueError("batch, image, clip and iteration values must be positive")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    device = torch.device(args.device)
    model = TRHashSensorFusionClassifier(
        TRHashSensorFusionConfig(
            precision=args.precision,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.layers,
            num_attention_heads=args.heads,
            shared_width=args.shared_width,
            expert_width=args.expert_width,
            class_hash_expert_width=args.class_hash_expert_width,
        )
    ).to(device)
    model.train()
    batch = inputs(args, device)
    masks = {
        name: torch.ones(args.batch_size, dtype=torch.bool, device=device)
        for name in SENSOR_MODALITIES
    }
    labels = torch.arange(args.batch_size, device=device) % model.config.num_classes

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with autocast(device, args.precision):
            loss = model(batch, labels, modality_mask=masks)["loss"]
        loss.backward()

    for _ in range(args.warmup):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        step()
        synchronize(device)
        durations.append(time.perf_counter() - started)
    result = {
        "device": str(device),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "clip_frames": args.clip_frames,
        "class_hash_expert_width": args.class_hash_expert_width,
        "parameters": model.num_parameters(),
        "fp32_size_mb": model.fp32_size_bytes / 1_000_000,
        "mean_step_ms": statistics.mean(durations) * 1_000,
        "median_step_ms": statistics.median(durations) * 1_000,
        "samples_per_second": args.batch_size / statistics.mean(durations),
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1_000_000
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
