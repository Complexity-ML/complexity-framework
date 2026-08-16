"""M5 latency spike for a block-routed o200k language-model head."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn


class RoutedVocabHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_blocks: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.block_size = (vocab_size + num_blocks - 1) // num_blocks
        padded_vocab = self.block_size * num_blocks
        self.weight = nn.Parameter(torch.empty(padded_vocab, hidden_size))
        self.router = nn.Linear(hidden_size, num_blocks, bias=False)
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

    def dense(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight[: self.vocab_size].T

    def routed(self, hidden: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        block_ids = self.router(hidden).topk(top_k, dim=-1).indices
        offsets = torch.arange(self.block_size, device=hidden.device)
        token_ids = block_ids[..., None] * self.block_size + offsets
        selected = self.weight[token_ids]
        logits = torch.einsum("bd,bksd->bks", hidden, selected)
        valid = token_ids < self.vocab_size
        return logits.masked_fill(~valid, float("-inf")), token_ids


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(fn, device: torch.device, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1_000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--vocab-size", type=int, default=200_019)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32
    torch.manual_seed(7)
    head = RoutedVocabHead(args.hidden_size, args.vocab_size, args.num_blocks).to(
        device=device, dtype=dtype
    ).eval()
    hidden = torch.randn(1, args.hidden_size, device=device, dtype=dtype)

    dense_samples = benchmark(lambda: head.dense(hidden), device, 20, args.repeats)
    dense_median = statistics.median(dense_samples)
    dense_weights = args.vocab_size * args.hidden_size

    print("device,mode,top_k,candidates,weight_fraction,median_ms,p95_ms,speedup")
    print(
        f"{device},dense,{args.num_blocks},{args.vocab_size},1.000000,"
        f"{dense_median:.6f},{statistics.quantiles(dense_samples, n=20)[18]:.6f},1.000000"
    )
    for top_k in (1, 2, 4, 8, 16):
        samples = benchmark(
            lambda k=top_k: head.routed(hidden, k), device, 20, args.repeats
        )
        candidates = min(top_k * head.block_size, args.vocab_size)
        selected_weights = candidates * args.hidden_size + args.num_blocks * args.hidden_size
        print(
            f"{device},routed,{top_k},{candidates},"
            f"{selected_weights / dense_weights:.6f},"
            f"{statistics.median(samples):.6f},"
            f"{statistics.quantiles(samples, n=20)[18]:.6f},"
            f"{dense_median / statistics.median(samples):.6f}"
        )


if __name__ == "__main__":
    main()
