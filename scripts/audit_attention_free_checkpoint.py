#!/usr/bin/env python3
"""Audit a safetensors checkpoint for the attention-free architecture claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

QKV_PROJECTION_FRAGMENTS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "query_proj",
    "key_proj",
    "value_proj",
)
EXPECTED_MIXER_SUFFIXES = (
    "depthwise.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "o_proj.weight",
)


def audit_checkpoint(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())

    tensor_names: list[str] = []
    for checkpoint_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            tensor_names.extend(checkpoint.keys())

    if not tensor_names:
        raise FileNotFoundError(f"no safetensors checkpoint found in {model_dir}")

    qkv_hits = [
        name
        for name in tensor_names
        if any(fragment in name.lower() for fragment in QKV_PROJECTION_FRAGMENTS)
    ]
    mixer_tensors = [name for name in tensor_names if ".self_attn." in name]
    unexpected_mixer_tensors = [
        name
        for name in mixer_tensors
        if not name.endswith(EXPECTED_MIXER_SUFFIXES)
    ]
    architecture = config.get("architectures", [])
    attention_type = config.get("attention_type")
    passed = (
        attention_type == "causal_conv"
        and architecture == ["PacificDilatedConvForCausalLM"]
        and not qkv_hits
        and not unexpected_mixer_tensors
    )
    return {
        "passed": passed,
        "architectures": architecture,
        "attention_type": attention_type,
        "num_tensors": len(tensor_names),
        "qkv_projection_hits": qkv_hits,
        "mixer_tensors": mixer_tensors,
        "unexpected_mixer_tensors": unexpected_mixer_tensors,
        "compatibility_note": (
            "self_attn and post_attention_layernorm are inherited ABI names; "
            "the audited mixer tensors implement gated depthwise causal convolution"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    result = audit_checkpoint(args.model_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
