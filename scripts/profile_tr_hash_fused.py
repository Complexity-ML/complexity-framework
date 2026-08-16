"""Benchmark hash-native fused top-2 routing against the PyTorch reference."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
)


def synchronize() -> None:
    torch.cuda.synchronize()


def benchmark(
    engine: TRHashEngine,
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    engine.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            engine(hidden, token_ids)
        synchronize()
        timings = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            engine(hidden, token_ids)
            end.record()
            synchronize()
            timings.append(float(start.elapsed_time(end)))
    median_ms = statistics.median(timings)
    tokens = int(token_ids.numel())
    return {
        "median_ms": median_ms,
        "tokens_per_second": tokens * 1000.0 / median_ms,
        "tokens": tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=69)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--shared-width", type=int, default=512)
    parser.add_argument("--expert-width", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    common = dict(
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        num_experts=args.experts,
        top_k=2,
        shared_width=args.shared_width,
        expert_width=args.expert_width,
        precision=TRHashPrecision.BF16,
    )
    fused = TRHashEngine(
        TRHashEngineConfig(**common, backend=TRHashBackend.FUSED_CUDA)
    ).cuda().to(torch.bfloat16)
    reference = TRHashEngine(
        TRHashEngineConfig(**common, backend=TRHashBackend.PYTORCH)
    ).cuda().to(torch.bfloat16)
    reference.load_state_dict(fused.state_dict())
    hidden = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    token_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.sequence_length),
        device="cuda",
    )
    results = {
        "fused_cuda": benchmark(
            fused,
            hidden,
            token_ids,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "pytorch": benchmark(
            reference,
            hidden,
            token_ids,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    results["speedup"] = (
        results["fused_cuda"]["tokens_per_second"]
        / results["pytorch"]["tokens_per_second"]
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
