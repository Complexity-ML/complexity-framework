#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.token_routed import TokenRoutedMLP
from complexity.tokenizer import Tokenizer

TEXT = Path("data/local/fineweb_sample.txt")
TOKENIZER = "./tokenizer-o200k"
VOCAB_SIZE = 200_019
NUM_EXPERTS = 4
STRATEGIES = ["zipf", "random", "modulo", "round_robin"]


def main() -> None:
    tokenizer = Tokenizer.load(TOKENIZER)
    text = TEXT.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    ids = torch.tensor(tokens, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < VOCAB_SIZE)]
    freqs = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    seen = freqs > 0

    print(f"sample_tokens={int(freqs.sum().item())} seen_types={int(seen.sum().item())} vocab={VOCAB_SIZE}")
    print("strategy,mass_e0,mass_e1,mass_e2,mass_e3,mass_minmax_ratio,types_e0,types_e1,types_e2,types_e3,top_tokens")
    for strategy in STRATEGIES:
        mlp = TokenRoutedMLP(
            MLPConfig(
                hidden_size=8,
                intermediate_size=16,
                num_experts=NUM_EXPERTS,
                vocab_size=VOCAB_SIZE,
                routing_strategy=strategy,
                token_frequencies=freqs,
                shared_expert=False,
            )
        )
        mapping = mlp.token_to_expert.cpu()
        mass = torch.zeros(NUM_EXPERTS)
        types = torch.zeros(NUM_EXPERTS, dtype=torch.long)
        for e in range(NUM_EXPERTS):
            mask = mapping == e
            mass[e] = freqs[mask].sum()
            types[e] = (seen & mask).sum()
        top = []
        top_ids = torch.topk(freqs, k=16).indices.tolist()
        for tid in top_ids:
            token = tokenizer.decode([tid]).replace("\n", "\\n")
            top.append(f"{tid}:{mapping[tid].item()}:{token}")
        nonzero_mass = mass[mass > 0]
        ratio = float(nonzero_mass.max() / nonzero_mass.min()) if len(nonzero_mass) else float("nan")
        print(
            f"{strategy},"
            + ",".join(f"{float(x):.0f}" for x in mass.tolist())
            + f",{ratio:.4f},"
            + ",".join(str(int(x)) for x in types.tolist())
            + ","
            + " | ".join(top)
        )


if __name__ == "__main__":
    main()
