#!/usr/bin/env python
"""Convert a Token-Routed/LSH ComplexityModel torch checkpoint to an mlx-lm dir.

Writes <out>/config.json (model_type=complexity) + <out>/model.safetensors with
weights renamed to the mlx_lm.models.complexity module tree. Tied embeddings →
no lm_head. RoPE inv_freq and non-param buffers are dropped (mlx recomputes).
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

# Buffers present in the torch state_dict that the MLX model does not hold.
DROP = (
    "rotary_emb.inv_freq",
    "expert_counts",
    "last_shared_rms",
    "last_routed_rms",
    "lsh_bit_values",
    "pair_hash_route_codes",
    "pair_hash_expert_pairs",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = d["model"]
    c = dict(d["config"])
    run_args = d.get("args", {}) or {}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = {}
    for k, v in sd.items():
        if any(s in k for s in DROP):
            continue
        t = v.detach().cpu()
        if t.dtype in (torch.int64, torch.int32, torch.bool):
            arr = mx.array(t.numpy().astype(np.int32))
        else:
            arr = mx.array(t.float().numpy())
        weights["model." + k] = arr
    mx.save_safetensors(str(out / "model.safetensors"), weights)

    # Final scheduled primary route weight (not persisted in the torch ckpt).
    pw = run_args.get("top_k_primary_weight_final") or c.get("top_k_primary_weight") or 0.85

    cfg = {
        "model_type": "complexity",
        "hidden_size": c["hidden_size"],
        "num_hidden_layers": c["num_hidden_layers"],
        "num_attention_heads": c["num_attention_heads"],
        "num_key_value_heads": c["num_key_value_heads"],
        "intermediate_size": c["intermediate_size"],
        "vocab_size": c["vocab_size"],
        "max_position_embeddings": c.get("max_position_embeddings", 2048),
        "rope_theta": c.get("rope_theta", 10000.0),
        "rope_traditional": False,
        "norm_eps": c.get("norm_eps", 1e-6),
        "num_experts": c.get("num_experts", 4),
        "shared_expert": c.get("shared_expert", True),
        "shared_intermediate_size": c.get("shared_intermediate_size"),
        "use_shared_routed_gates": c.get("use_shared_routed_gates", True),
        "use_mu_guidance": c.get("use_mu_guidance", False),
        "use_qk_norm": c.get("use_qk_norm", True),
        "tie_word_embeddings": c.get("tie_word_embeddings", True),
        "top_k": c.get("top_k", 2),
        "top_k_primary_weight": float(pw),
        "shared_output_scale": float(c.get("shared_output_scale", 1.0)),
        "routed_output_scale": float(c.get("routed_output_scale", 1.0)),
        "learn_hash_channel_modulation": bool(
            c.get("learn_hash_channel_modulation", False)
        ),
        "hash_channel_scale_init": float(c.get("hash_channel_scale_init", 0.0)),
        "mlp_type": c.get("mlp_type", "token_routed"),
        "routing_strategy": c.get("routing_strategy", "modulo_cyclic"),
        "lsh_routing": bool(c.get("lsh_routing", False)),
        "lsh_bits": int(c.get("lsh_bits", 0)),
        "lsh_from_layer": int(c.get("lsh_from_layer", 0)),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out} | {len(weights)} tensors | primary_weight={pw}")


if __name__ == "__main__":
    main()
