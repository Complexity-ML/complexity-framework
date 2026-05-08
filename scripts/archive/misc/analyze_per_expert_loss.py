"""
Per-expert loss analysis for a trained Token-Routed MoE checkpoint.

Loads the model, streams FineWeb-Edu eval chunks, computes per-token
cross-entropy loss, then groups by the token's Zipf-assigned expert.
Output: mean loss per expert + max gap (compare with paper's 187M result
max_gap=0.23 to see if experts collapse at larger scale).

Usage:
    python3 scripts/analyze_per_expert_loss.py \
        --checkpoint /Users/boris/Dev/checkpoints/abl-moe-adamw \
        --num-tokens 200000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from complexity.config import ModelConfig
from complexity.models import ComplexityModel

logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("analyze")


def make_400m_moe_config() -> ModelConfig:
    """Matches scripts/train_400m_v1.py"""
    return ModelConfig(
        hidden_size=1024, num_hidden_layers=20,
        num_attention_heads=16, num_key_value_heads=4,
        intermediate_size=2008, vocab_size=32000,
        max_position_embeddings=4096, attention_type="gqa",
        mlp_type="token_routed", num_experts=4, shared_expert=True,
        norm_type="rmsnorm", use_qk_norm=True, use_mu_guidance=True,
    )


@torch.no_grad()
def stream_eval_tokens(tokenizer, seq_len: int, target_tokens: int):
    """Yield (input_ids, labels) chunks from FineWeb-Edu until target_tokens."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf = []
    seen = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue
        buf.extend(tokenizer.encode(text))
        while len(buf) >= seq_len + 1:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len:]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y
            seen += seq_len
            if seen >= target_tokens:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Dir containing model.safetensors")
    ap.add_argument("--tokenizer", type=str, default="./tokenizer")
    ap.add_argument("--num-tokens", type=int, default=200_000,
                    help="Total eval tokens to process")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", type=str, default=None,
                    help="mps / cuda / cpu (auto-detect)")
    args = ap.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    log.info(f"device: {device}")

    # Load model
    config = make_400m_moe_config()
    # Build on CPU in fp32, load weights, cast to bf16, move to MPS.
    # Load-then-cast fits in ~3 GB peak (384M × 4 bytes), well under 24 GB.
    log.info("Building model on CPU...")
    model = ComplexityModel(config)
    log.info("Loading safetensors...")
    sd = load_file(f"{args.checkpoint}/model.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log.info(f"  {len(sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    del sd
    log.info(f"Casting bf16 and moving to {device}...")
    model = model.to(torch.bfloat16).to(device).eval()

    # Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Pull token_to_expert from the first MLP layer — should be identical across layers
    first_mapping = model.layers[0].mlp.token_to_expert.to(device)
    num_experts = config.num_experts

    # Accumulators
    expert_loss_sum = torch.zeros(num_experts, device=device, dtype=torch.float32)
    expert_count = torch.zeros(num_experts, device=device, dtype=torch.long)
    total_loss = 0.0
    total_tokens = 0

    log.info(f"Streaming {args.num_tokens:,} tokens from FineWeb-Edu...")

    for x, y in stream_eval_tokens(tokenizer, args.seq_len, args.num_tokens):
        x = x.unsqueeze(0).to(device)  # [1, S]
        y = y.unsqueeze(0).to(device)

        out = model(x)
        hidden = out["last_hidden_state"] if isinstance(out, dict) else out
        # Tied embeddings
        base = model
        while hasattr(base, "model") or hasattr(base, "module"):
            nxt = getattr(base, "model", None) or getattr(base, "module", None)
            if nxt is None or nxt is base:
                break
            base = nxt
        logits = F.linear(hidden, base.embed_tokens.weight)  # [1, S, V]

        # Per-token CE (no reduction)
        per_tok = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            y.view(-1),
            reduction="none",
        )  # [S]

        # Bucket by Zipf expert assignment on the TARGET token (y)
        y_flat = y.view(-1)
        expert_of_y = first_mapping[y_flat.clamp(0, config.vocab_size - 1)]

        for e in range(num_experts):
            mask = expert_of_y == e
            n = mask.sum().item()
            if n == 0:
                continue
            expert_loss_sum[e] += per_tok[mask].sum()
            expert_count[e] += n

        total_loss += per_tok.sum().item()
        total_tokens += per_tok.numel()

        if total_tokens % (args.seq_len * 20) == 0:
            log.info(f"  processed {total_tokens:,} / {args.num_tokens:,}")

    # Report
    print()
    print("=" * 60)
    print(f"Per-expert loss (total {total_tokens:,} eval tokens)")
    print("=" * 60)
    overall = total_loss / total_tokens
    print(f"Overall mean loss : {overall:.4f}  (PPL {torch.exp(torch.tensor(overall)).item():.2f})")
    print()
    print(f"{'expert':<8} {'tokens':>12} {'share':>8} {'mean loss':>12} {'delta vs avg':>15}")
    per_expert = []
    for e in range(num_experts):
        if expert_count[e] == 0:
            print(f"{e:<8} {0:>12} {'---':>8} {'n/a':>12}")
            continue
        m = (expert_loss_sum[e] / expert_count[e]).item()
        per_expert.append(m)
        share = expert_count[e].item() / total_tokens
        delta = m - overall
        print(f"{e:<8} {expert_count[e].item():>12,} {share:>8.3f} {m:>12.4f} {delta:>+15.4f}")
    print()
    if len(per_expert) == num_experts:
        gap = max(per_expert) - min(per_expert)
        var = torch.tensor(per_expert).std().item()
        print(f"max gap (worst − best) : {gap:.4f}")
        print(f"std across experts     : {var:.4f}")
        print()
        print("Paper reference (187M, training-time): max gap 0.23, avg diff 0.022")


if __name__ == "__main__":
    main()
