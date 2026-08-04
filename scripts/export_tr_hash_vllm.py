#!/usr/bin/env python3
"""Export a TR-Hash training checkpoint as a vLLM model directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from complexity.inference.chat_template import (
    default_chat_template,
    huggingface_chat_template,
    validate_chat_template,
)

DROP_SUFFIXES = (
    "rotary_emb.inv_freq",
    "pair_hash_route_codes",
    "pair_hash_expert_pairs",
)

TOKENIZER_FILENAMES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.tiktoken",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


def copy_tokenizer_files(source: Path, output: Path) -> list[str]:
    """Copy tokenizer assets without overwriting model export artifacts."""

    copied = []
    for path in source.iterdir():
        if path.is_file() and path.name in TOKENIZER_FILENAMES:
            shutil.copy2(path, output / path.name)
            copied.append(path.name)
    if not copied:
        raise FileNotFoundError(f"No recognized tokenizer files found in {source}")
    return sorted(copied)


def write_tokenizer_chat_template(output: Path, chat_template: dict) -> Path:
    """Synchronize AutoTokenizer chat rendering with the SFT contract."""

    path = output / "tokenizer_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer config not found after export: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    config["chat_template"] = huggingface_chat_template(chat_template)
    config["chat_template_id"] = chat_template["id"]
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_config(raw: dict, chat_template: dict | None = None) -> dict:
    """Translate the training configuration to the DeepConfig contract."""

    config = {
        "architectures": ["DeepForCausalLM"],
        "model_type": "deep",
        "hidden_size": raw["hidden_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "intermediate_size": raw["intermediate_size"],
        "vocab_size": raw["vocab_size"],
        "max_position_embeddings": raw.get("max_position_embeddings", 2048),
        "attention_type": raw.get("attention_type", "gqa"),
        "use_qk_norm": raw.get("use_qk_norm", True),
        "rope_theta": raw.get("rope_theta", 10000.0),
        "mlp_type": raw.get("mlp_type", "token_routed"),
        "num_experts": raw.get("num_experts", 4),
        "shared_expert": raw.get("shared_expert", True),
        "shared_intermediate_size": raw.get("shared_intermediate_size", 0),
        "use_shared_routed_gates": raw.get("use_shared_routed_gates", False),
        "shared_gate_init": raw.get("shared_gate_init", 1.0),
        "routed_gate_init": raw.get("routed_gate_init", 1.0),
        "shared_output_scale": raw.get("shared_output_scale", 1.0),
        "routed_output_scale": raw.get("routed_output_scale", 1.0),
        "top_k": raw.get("top_k", 2),
        "top_k_primary_weight": raw.get("top_k_primary_weight", 0.5),
        # Historical TR-MOE checkpoints predate the explicit strategy field.
        # Their persisted primary table uses the modulo/cyclic top-2 contract;
        # defaulting them to the newer compact hash engine changes inference.
        "routing_strategy": raw.get("routing_strategy", "modulo_cyclic"),
        "learn_hash_channel_modulation": raw.get(
            "learn_hash_channel_modulation", False
        ),
        "hash_channel_scale_init": raw.get("hash_channel_scale_init", 0.0),
        "use_mu_guidance": raw.get("use_mu_guidance", False),
        "disable_mu_guidance": not raw.get("use_mu_guidance", False),
        "norm_type": raw.get("norm_type", "rmsnorm"),
        "norm_eps": raw.get("norm_eps", 1e-6),
        "rms_norm_eps": raw.get("norm_eps", 1e-6),
        "tie_word_embeddings": raw.get("tie_word_embeddings", True),
        "torch_dtype": "bfloat16",
    }
    if chat_template is not None:
        config["chat_template_id"] = chat_template["id"]
        config["chat_template_file"] = "chat_template.json"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    target_dtype = getattr(torch, args.dtype)
    state = {}
    for name, tensor in checkpoint["model"].items():
        if name.endswith(DROP_SUFFIXES):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = tensor.to(target_dtype)
        state[name] = tensor.contiguous()
    save_file(state, str(output / "model.safetensors"))
    chat_template = validate_chat_template(
        checkpoint.get("chat_template", default_chat_template())
    )
    config = build_config(dict(checkpoint["config"]), chat_template)
    config["torch_dtype"] = args.dtype
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "max_new_tokens": 128}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "chat_template.json").write_text(
        json.dumps(chat_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.tokenizer:
        copy_tokenizer_files(args.tokenizer, output)
        write_tokenizer_chat_template(output, chat_template)

    print(
        f"exported step={checkpoint.get('step')} tensors={len(state)} "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
