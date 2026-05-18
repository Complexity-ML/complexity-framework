#!/usr/bin/env python3
"""Smoke-test AMD ROCm support for Complexity training."""

from __future__ import annotations

import argparse
import sys

import torch

from complexity.config import ModelConfig
from complexity.core.losses import causal_lm_loss
from complexity.models import ComplexityModel
from complexity.utils.device import is_rocm_available, select_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ROCm/PyTorch support for Complexity.")
    parser.add_argument("--device", default="rocm", choices=["auto", "rocm", "cuda", "cpu", "mps"])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    print(f"torch={torch.__version__}")
    print(f"torch.version.hip={torch.version.hip}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")

    if args.device == "rocm" and not is_rocm_available():
        print("ERROR: ROCm requested, but this PyTorch install is not ROCm-enabled.", file=sys.stderr)
        return 1

    device = select_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"name={torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    x = torch.randn(512, 512, device=device, dtype=dtype)
    y = x @ x
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"matmul_ok mean={float(y.float().mean()):.6f}")

    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=args.seq_len,
        num_experts=4,
        mlp_type="token_routed",
        shared_expert=True,
        top_k=2,
        top_k_primary_weight=0.5,
    )
    model = ComplexityModel(config).to(device)
    model.train()
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )
    labels = input_ids.clone()
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type != "cpu"):
        out = model(input_ids)
        loss, _ = causal_lm_loss(out["logits"], labels, shift=True)
    loss.backward()
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"forward_backward_ok loss={float(loss.detach().cpu()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
