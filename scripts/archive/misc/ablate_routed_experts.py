"""
Offline ablation: how much do the routed experts actually contribute?

Runs the model 3× on the same eval slice:
  1. Full model (shared + routed)  → baseline
  2. Shared only   (routed zeroed) → lower bound of what "just dense part" gives
  3. Routed only   (shared zeroed) → what the specialists alone contribute

Compares losses. Interprets:
  - If (shared only) ≈ (full) → routed experts are decorative (≈0 contribution)
  - If (shared only) >> (full) → routed experts are load-bearing
  - If (routed only) ≈ (full) → shared expert is the decoration

Usage:
    python3 scripts/ablate_routed_experts.py --checkpoint /path/to/ckpt --num-tokens 30000
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from complexity.config import ModelConfig
from complexity.core.mlp.token_routed import TokenRoutedMLP
from complexity.models import ComplexityModel

logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("ablate")


def make_400m_moe_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=1024, num_hidden_layers=20,
        num_attention_heads=16, num_key_value_heads=4,
        intermediate_size=2008, vocab_size=32000,
        max_position_embeddings=4096, attention_type="gqa",
        mlp_type="token_routed", num_experts=4, shared_expert=True,
        norm_type="rmsnorm", use_qk_norm=True, use_mu_guidance=True,
    )


@torch.no_grad()
def stream_eval(tokenizer, seq_len: int, target: int):
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
            yield (
                torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long),
            )
            seen += seq_len
            if seen >= target:
                return


@torch.no_grad()
def eval_loss(model, tokenizer, device, seq_len: int, target: int) -> float:
    total_loss, total_tokens = 0.0, 0
    base = model
    while hasattr(base, "model") or hasattr(base, "module"):
        nxt = getattr(base, "model", None) or getattr(base, "module", None)
        if nxt is None or nxt is base:
            break
        base = nxt
    for x, y in stream_eval(tokenizer, seq_len, target):
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        out = model(x)
        hidden = out["last_hidden_state"] if isinstance(out, dict) else out
        logits = F.linear(hidden, base.embed_tokens.weight)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens


# Monkey-patch helpers: override TokenRoutedMLP.forward.
_ORIG_FORWARD = TokenRoutedMLP.forward


def _shared_only_forward(self, hidden_states, token_ids=None, **kwargs):
    """Keep shared expert, zero out routed output."""
    B, S, H = hidden_states.shape
    flat_x = hidden_states.view(-1, H)
    if self.use_shared_expert:
        from complexity.core.mlp.fused_activations import fused_silu_mul
        shared_out = self.shared_down(
            fused_silu_mul(self.shared_gate(flat_x), self.shared_up(flat_x))
        ).to(flat_x.dtype)
    else:
        shared_out = torch.zeros_like(flat_x)
    return shared_out.view(B, S, H)


def _routed_only_forward(self, hidden_states, token_ids=None, **kwargs):
    """Zero out shared, keep routed. Runs the original forward then subtracts the shared path."""
    full = _ORIG_FORWARD(self, hidden_states, token_ids=token_ids, **kwargs)
    shared = _shared_only_forward(self, hidden_states, token_ids=token_ids, **kwargs)
    return full - shared


def patch_mode(mode: str):
    if mode == "full":
        TokenRoutedMLP.forward = _ORIG_FORWARD
    elif mode == "shared_only":
        TokenRoutedMLP.forward = _shared_only_forward
    elif mode == "routed_only":
        TokenRoutedMLP.forward = _routed_only_forward
    else:
        raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="./tokenizer")
    ap.add_argument("--num-tokens", type=int, default=30_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = args.device
    log.info(f"device: {device}")

    config = make_400m_moe_config()
    log.info("Building model on CPU and loading weights...")
    model = ComplexityModel(config)
    sd = load_file(f"{args.checkpoint}/model.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log.info(f"  {len(sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    del sd
    model = model.to(torch.bfloat16).to(device).eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    results = {}
    for mode in ["full", "shared_only", "routed_only"]:
        patch_mode(mode)
        log.info(f"Evaluating mode: {mode}")
        loss = eval_loss(model, tokenizer, device, args.seq_len, args.num_tokens)
        results[mode] = loss
        log.info(f"  {mode}: loss {loss:.4f} (PPL {torch.exp(torch.tensor(loss)).item():.2f})")

    print()
    print("=" * 60)
    print("Routed expert contribution analysis")
    print("=" * 60)
    print(f"Full model       : {results['full']:.4f}")
    print(f"Shared-only      : {results['shared_only']:.4f}   Δ vs full: {results['shared_only']-results['full']:+.4f}")
    print(f"Routed-only      : {results['routed_only']:.4f}   Δ vs full: {results['routed_only']-results['full']:+.4f}")
    print()
    delta_shared = results['shared_only'] - results['full']
    if delta_shared < 0.02:
        print("VERDICT: routed experts contribute ≤0.02 loss — essentially decorative.")
    elif delta_shared < 0.1:
        print("VERDICT: routed experts add modest value (0.02-0.1 loss drop).")
    else:
        print("VERDICT: routed experts are load-bearing (>0.1 loss drop when removed).")


if __name__ == "__main__":
    main()
