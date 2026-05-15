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
}


def build_mlx_config(pt_config: dict) -> dict:
    cfg = {"model_type": "complexity"}
    for k in _MLX_CONFIG_FIELDS:
        if k in pt_config and pt_config[k] is not None:
            cfg[k] = pt_config[k]
    cfg.setdefault("rope_traditional", False)
    return cfg


def remap_state_dict(model_sd: dict) -> dict:
    """Add 'model.' prefix, drop rope buffers, cast token_to_expert to int32."""
    out = {}
    for k, v in model_sd.items():
        if k.endswith("rotary_emb.inv_freq"):
            continue
        new_key = f"model.{k}"
        if k.endswith("token_to_expert") and v.dtype != torch.int32:
            v = v.to(torch.int32)
        # Cast fp32 weights to fp16 for storage (MLX upcasts as needed).
        if v.dtype == torch.float32:
            v = v.to(torch.float16)
        out[new_key] = v.contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="Tokenizer directory to copy into output (e.g. ./tokenizer-o200k)")
    args = ap.parse_args()

    print(f"Loading {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    pt_config = ckpt["config"]
    model_sd = ckpt["model"]
    print(f"  step={ckpt.get('step')}  config keys={len(pt_config)}  tensors={len(model_sd)}")

    args.output.mkdir(parents=True, exist_ok=True)

    mlx_config = build_mlx_config(pt_config)
    with (args.output / "config.json").open("w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Wrote {args.output / 'config.json'}")

    remapped = remap_state_dict(model_sd)
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
