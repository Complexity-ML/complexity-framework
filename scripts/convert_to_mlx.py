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
    "fused_route_codes",
    "fused_expert_pairs",
)

ENGINE_PARAM_REMAP = {
    "expert_down": "down_proj_w",
    "expert_gate": "gate_proj_w",
    "expert_up": "up_proj_w",
    "shared_down.weight": "shared_down.weight",
    "shared_gate.weight": "shared_gate.weight",
    "shared_up.weight": "shared_up.weight",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = d["model"]
    c = dict(d["config"])
    run_args = d.get("args", {}) or {}
    canonical_engine = any(".mlp.engine." in key for key in sd)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = {}
    for k, v in sd.items():
        if any(s in k for s in DROP):
            continue
        t = v.detach().cpu()
        if ".mlp.engine." in k:
            prefix, engine_name = k.split(".mlp.engine.", maxsplit=1)
            if engine_name == "route_table":
                routes = t.to(torch.int32).contiguous()
                weights[f"model.{prefix}.mlp.topk_token_to_expert"] = mx.array(
                    np.asarray(routes, dtype=np.int32)
                )
                weights[f"model.{prefix}.mlp.token_to_expert"] = mx.array(
                    np.asarray(routes[0].contiguous(), dtype=np.int32)
                )
                continue
            if engine_name not in ENGINE_PARAM_REMAP:
                raise ValueError(f"Unsupported TRHashEngine tensor key: {k}")
            k = f"model.{prefix}.mlp.{ENGINE_PARAM_REMAP[engine_name]}"
        else:
            k = "model." + k

        if t.dtype in (torch.int64, torch.int32, torch.bool):
            arr = mx.array(t.numpy().astype(np.int32))
        else:
            arr = mx.array(t.float().numpy())
        weights[k] = arr

    raw_mlp_type = c.get("mlp_type", "token_routed")
    use_tr_token_mlp = c.get(
        "use_token_routed_mlp",
        raw_mlp_type in {"token_routed", "tr_hash", "tr_hash_engine", "tr_hash_moe"},
    )
    use_tr_token_mlp = bool(use_tr_token_mlp)
    # Canonical TRHashEngine combines both paths with fixed output scales. An
    # old run config may still contain the legacy gate flag even though those
    # parameters do not exist in the checkpoint.
    use_shared_routed_gates = (
        False if canonical_engine else bool(c.get("use_shared_routed_gates", False))
    )

    if use_tr_token_mlp and use_shared_routed_gates:
        for layer in range(int(c["num_hidden_layers"])):
            prefix = f"model.layers.{layer}.mlp"
            weights.setdefault(
                f"{prefix}.shared_output_gate",
                mx.array(np.array(0.0, dtype=np.float32)),
            )
            weights.setdefault(
                f"{prefix}.routed_output_gate",
                mx.array(np.array(0.0, dtype=np.float32)),
            )

    mx.save_safetensors(str(out / "model.safetensors"), weights)

    # Final scheduled primary route weight (not persisted in the torch ckpt).
    if canonical_engine:
        pw = c.get("top_k_primary_weight", 0.5)
    else:
        pw = (
            run_args.get("top_k_primary_weight_final")
            or c.get("top_k_primary_weight")
            or 0.85
        )

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
        "use_shared_routed_gates": use_shared_routed_gates,
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
        "mlp_type": "token_routed" if use_tr_token_mlp else raw_mlp_type,
        "use_token_routed_mlp": use_tr_token_mlp,
        "routing_strategy": c.get("routing_strategy", "modulo_cyclic"),
        "lsh_routing": bool(c.get("lsh_routing", False)),
        "lsh_bits": int(c.get("lsh_bits", 0)),
        "lsh_from_layer": int(c.get("lsh_from_layer", 0)),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out} | {len(weights)} tensors | primary_weight={pw}")


if __name__ == "__main__":
    main()
