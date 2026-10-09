"""Create an autonomous Hugging Face Transformers TR-HASH model bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

ADAPTER = Path(__file__).resolve().parent / "transformers_adapter"
TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
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


def validate_transformers_export_config(raw: dict) -> None:
    """Reject model options the self-contained public adapter does not implement.

    The export is scoped to the shared-plus-routed, tied-embedding GQA family
    used by the published TR-HASH top-2 checkpoints. It must not silently drop
    experimental runtime behavior while copying the tensor bytes unchanged.
    """
    required_values = {
        "attention_type": "gqa",
        "norm_type": "rmsnorm",
        "rope_type": "standard",
        "rope_fraction": 1.0,
        "tie_word_embeddings": True,
        "is_causal": True,
        "sliding_window": None,
        "lsh_routing": False,
        "use_shared_routed_gates": False,
        "learn_hash_pair_gates": False,
        "learn_hash_channel_modulation": False,
        "active_num_experts": None,
        "active_expert_width": None,
        "layer_mlp_controls": None,
        "shared_output_scale_first_layer": None,
        "shared_output_scale_last_layer": None,
        "routed_output_scale_first_layer": None,
        "routed_output_scale_last_layer": None,
        "lexical_zipf_mode": "uniform",
        "use_mup_attn_scale": False,
        "use_mup_output_mult": False,
    }
    unsupported = {
        name: raw[name]
        for name, expected in required_values.items()
        if name in raw and raw[name] != expected
    }
    if raw.get("mlp_type", "tr_hash_engine") not in {"tr_hash_engine", "tr_hash_moe"}:
        unsupported["mlp_type"] = raw["mlp_type"]
    top_k = raw.get("top_k", raw.get("num_experts_per_tok", 2))
    if top_k != 2:
        unsupported["top_k"] = top_k
    if raw.get("num_experts", 4) not in {4, 8, 16}:
        unsupported["num_experts"] = raw["num_experts"]
    if unsupported:
        fields = ", ".join(f"{name}={value!r}" for name, value in sorted(unsupported.items()))
        raise ValueError(f"Unsupported Transformers export configuration: {fields}")


def build_transformers_config(raw: dict) -> dict:
    validate_transformers_export_config(raw)
    config = dict(raw)
    if "top_k" in config:
        # Keep the native architectural key for TR-Hash-i64 while exposing the
        # unambiguous Transformers MoE name. TRHashConfig consumes the legacy
        # key on load, so it does not leak into GenerationConfig sampling.
        config["num_experts_per_tok"] = config["top_k"]
    config.update(
        {
            "model_type": "tr_hash_moe",
            "architectures": ["TRHashForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_tr_hash_moe.TRHashConfig",
                "AutoModelForCausalLM": "modeling_tr_hash_moe.TRHashForCausalLM",
            },
            "is_decoder": True,
            "is_encoder_decoder": False,
        }
    )
    config.setdefault("use_cache", True)
    config.setdefault("norm_eps", config.get("rms_norm_eps", 1e-6))
    config.setdefault("rope_theta", 10000.0)
    config.setdefault("attention_dropout", 0.0)
    return config


def tokenizer_special_token_ids(tokenizer_dir: Path | None) -> dict[str, int]:
    if tokenizer_dir is None:
        return {}
    path = tokenizer_dir / "tokenizer_config.json"
    if not path.exists():
        return {}
    tokenizer_config = json.loads(path.read_text(encoding="utf-8"))
    decoder = tokenizer_config.get("added_tokens_decoder", {})
    by_content = {
        value.get("content"): int(key)
        for key, value in decoder.items()
        if isinstance(value, dict) and value.get("content") is not None
    }
    result = {}
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token = tokenizer_config.get(name)
        if token in by_content:
            result[f"{name}_id"] = by_content[token]
    return result


def copy_safetensors_with_transformers_metadata(source: Path, target: Path) -> None:
    """Stream-copy SafeTensors while declaring its PyTorch framework.

    Transformers 4.30 rejects valid SafeTensors files whose header has no
    ``metadata.format`` value. Tensor offsets are relative to the data section,
    so replacing only the padded JSON header preserves every tensor byte and
    avoids deserializing an 800+ MB checkpoint into memory.
    """

    if source.suffix != ".safetensors":
        raise ValueError("--weights must point to a .safetensors file")
    with source.open("rb") as input_file:
        raw_length = input_file.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid SafeTensors header in {source}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = input_file.read(header_length)
        try:
            header = json.loads(raw_header.decode("utf-8").rstrip(" "))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid SafeTensors JSON header in {source}") from exc
        metadata = dict(header.get("__metadata__", {}))
        metadata["format"] = "pt"
        header["__metadata__"] = metadata
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)

        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("wb") as output_file:
            output_file.write(struct.pack("<Q", len(encoded)))
            output_file.write(encoded)
            shutil.copyfileobj(input_file, output_file, length=16 * 1024 * 1024)
        os.replace(temporary, target)


def export_bundle(
    *,
    config_path: Path,
    weights_path: Path,
    output: Path,
    tokenizer_dir: Path | None = None,
) -> None:
    config_path = Path(config_path)
    weights_path = Path(weights_path)
    output = Path(output)
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else None
    source_directories = {config_path.parent.resolve(), weights_path.parent.resolve()}
    if tokenizer_dir is not None:
        source_directories.add(tokenizer_dir.resolve())
    if output.resolve() in source_directories:
        raise ValueError(
            "Export output must be separate from source checkpoint/tokenizer directories"
        )
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Export requires an existing config and safetensors weight file")
    if tokenizer_dir is not None and not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = build_transformers_config(raw)
    config.update(tokenizer_special_token_ids(tokenizer_dir))
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copy_safetensors_with_transformers_metadata(weights_path, output / "model.safetensors")
    for filename in ("configuration_tr_hash_moe.py", "modeling_tr_hash_moe.py"):
        shutil.copy2(ADAPTER / filename, output / filename)
    if tokenizer_dir is not None:
        for path in tokenizer_dir.iterdir():
            if path.is_file() and path.name in TOKENIZER_FILES:
                shutil.copy2(path, output / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    export_bundle(
        config_path=args.config,
        weights_path=args.weights,
        output=args.output,
        tokenizer_dir=args.tokenizer_dir,
    )
    print(f"Transformers bundle written to {args.output}")


if __name__ == "__main__":
    main()
