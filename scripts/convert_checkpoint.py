"""
Convert FSDP v2 checkpoint to plain safetensors (full model).

Single-process (CPU or GPU), no torchrun needed:
    python scripts/convert_checkpoint.py \
        --checkpoint checkpoints/1b-moe-v1/final \
        --output checkpoints/1b-moe-v1/final_hf

The checkpoint.pt saved by the trainer already contains the full model
(saved with full_state_dict=True), so no multi-GPU gather is needed.
"""

import argparse
import json
import torch
from pathlib import Path


CONFIG_JSON = {
    "model_type": "complexity",
    "hidden_size": 1792,
    "num_hidden_layers": 24,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "intermediate_size": 4608,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
    "attention_type": "gqa",
    "mlp_type": "token_routed",
    "num_experts": 4,
    "norm_type": "rmsnorm",
    "use_qk_norm": True,
    "use_mu_guidance": True,
    "torch_dtype": "bfloat16",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpt_file = Path(args.checkpoint) / "checkpoint.pt"
    print(f"Loading {ckpt_file} ...")
    raw = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    model_sd = raw["model"]

    total = sum(v.numel() for v in model_sd.values())
    print(f"  {total:,} params = {total * 2 / 1e9:.2f} GB bf16")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    weights_path = out_dir / "model.safetensors"
    # ensure all tensors are contiguous plain tensors
    clean_sd = {k: v.contiguous() for k, v in model_sd.items()}
    save_file(clean_sd, str(weights_path))
    print(f"Saved: {weights_path} ({weights_path.stat().st_size / 1e6:.0f} MB)")

    with open(out_dir / "config.json", "w") as f:
        json.dump(CONFIG_JSON, f, indent=2)

    print(f"Done. {out_dir}/")


if __name__ == "__main__":
    main()
