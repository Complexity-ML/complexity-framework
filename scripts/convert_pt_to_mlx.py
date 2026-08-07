"""Convert a complexity-framework PyTorch checkpoint to an mlx-lm directory.

Usage:
    python convert_pt_to_mlx.py <checkpoint.pt> <output_dir> [--tokenizer DIR]

Produces:
    <output_dir>/config.json
    <output_dir>/model.safetensors
    <output_dir>/<tokenizer files>   (if --tokenizer is given)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from complexity.inference.chat_template import (
    default_chat_template,
    validate_chat_template,
)

_MLX_CONFIG_FIELDS = {
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "rope_traditional",
    "rope_scaling",
    "norm_eps",
    "num_experts",
    "shared_expert",
    "shared_intermediate_size",
    "use_shared_routed_gates",
    "use_mu_guidance",
    "use_qk_norm",
    "tie_word_embeddings",
    "top_k",
    "top_k_primary_weight",
    "shared_output_scale",
    "routed_output_scale",
    "routing_strategy",
    "learn_hash_channel_modulation",
    "hash_channel_scale_init",
    "mlp_type",
}

_TR_HASH_ENGINE_MLP_TYPES = {"tr_hash_engine", "tr_hash_moe"}

_ENGINE_PARAMETER_NAMES = {
    "expert_gate": "gate_proj_w",
    "expert_up": "up_proj_w",
    "expert_down": "down_proj_w",
    "shared_gate.weight": "shared_gate.weight",
    "shared_up.weight": "shared_up.weight",
    "shared_down.weight": "shared_down.weight",
}


def build_mlx_config(pt_config: dict) -> dict:
    cfg = {"model_type": "complexity"}
    for k in _MLX_CONFIG_FIELDS:
        if k in pt_config and pt_config[k] is not None:
            cfg[k] = pt_config[k]
    # MLX executes both the historical TokenRoutedMLP and the canonical
    # TRHashEngine checkpoint through its TokenRoutedMLP implementation.  The
    # state-dict remapper below preserves the engine's exact per-layer route
    # table, so this is an execution-layout translation rather than an
    # architecture change.
    if cfg.get("mlp_type") in _TR_HASH_ENGINE_MLP_TYPES:
        cfg["mlp_type"] = "token_routed"
        # TRHashEngineMLP combines shared and routed outputs with fixed scales;
        # it does not instantiate the legacy learned scalar output gates even
        # when an older run config still carries that flag.
        cfg["use_shared_routed_gates"] = False
    cfg.setdefault("rope_traditional", False)
    return cfg


def remap_state_dict(model_sd: dict, *, dtype: str) -> dict:
    """Translate PyTorch model weights into the MLX module tree.

    Canonical TRHashEngine blocks store parameters under ``mlp.engine`` and
    persist one exact ``[top_k, vocab]`` route table per layer.  MLX uses the
    historical TokenRoutedMLP parameter layout, so engine parameters and route
    tables are translated explicitly instead of being regenerated.
    """
    out = {}
    for k, v in model_sd.items():
        if k.endswith("rotary_emb.inv_freq") or k.endswith(
            (
                "pair_hash_route_codes",
                "pair_hash_expert_pairs",
                "fused_route_codes",
                "fused_expert_pairs",
            )
        ):
            continue

        if ".mlp.engine." in k:
            prefix, engine_name = k.split(".mlp.engine.", maxsplit=1)
            mlx_prefix = f"model.{prefix}.mlp."
            if engine_name == "route_table":
                routes = v.to(torch.int32).contiguous()
                out[f"{mlx_prefix}topk_token_to_expert"] = routes
                # Clone the primary route: safetensors intentionally rejects
                # separately named tensors that share the same backing
                # storage, even when one is only a view of the other.
                out[f"{mlx_prefix}token_to_expert"] = routes[0].clone()
                continue
            mapped_name = _ENGINE_PARAMETER_NAMES.get(engine_name)
            if mapped_name is None:
                raise ValueError(
                    f"Unsupported TRHashEngine checkpoint tensor: {k}"
                )
            new_key = f"{mlx_prefix}{mapped_name}"
        else:
            new_key = f"model.{k}"

        if new_key.endswith(
            ("token_to_expert", "topk_token_to_expert")
        ) and v.dtype != torch.int32:
            v = v.to(torch.int32)
        if dtype == "float16" and v.dtype == torch.float32:
            v = v.to(torch.float16)
        out[new_key] = v.contiguous()
    return out


def write_chat_template(checkpoint: dict, output: Path) -> dict:
    """Preserve the exact SFT prompt contract in the MLX model directory."""

    template = validate_chat_template(
        checkpoint.get("chat_template", default_chat_template())
    )
    (output / "chat_template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="Tokenizer directory to copy into output (e.g. ./tokenizer-o200k)")
    ap.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Floating-point storage dtype (float32 is useful for parity checks)",
    )
    args = ap.parse_args()

    print(f"Loading {args.checkpoint} ...")
    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    pt_config = ckpt["config"]
    model_sd = ckpt["model"]
    print(f"  step={ckpt.get('step')}  config keys={len(pt_config)}  tensors={len(model_sd)}")

    args.output.mkdir(parents=True, exist_ok=True)

    mlx_config = build_mlx_config(pt_config)
    with (args.output / "config.json").open("w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Wrote {args.output / 'config.json'}")

    template = write_chat_template(ckpt, args.output)
    print(
        f"Wrote {args.output / 'chat_template.json'} "
        f"({template['id']})"
    )

    remapped = remap_state_dict(model_sd, dtype=args.dtype)
    save_file(remapped, str(args.output / "model.safetensors"))
    n_bytes = (args.output / "model.safetensors").stat().st_size
    print(f"Wrote {args.output / 'model.safetensors'} ({n_bytes / 1e6:.1f} MB, {len(remapped)} tensors)")

    if args.tokenizer is not None:
        if not args.tokenizer.exists():
            raise SystemExit(f"Tokenizer dir not found: {args.tokenizer}")
        for f in args.tokenizer.iterdir():
            if f.is_file():
                shutil.copy2(f, args.output / f.name)
        print(f"Copied tokenizer files from {args.tokenizer}")


if __name__ == "__main__":
    main()
