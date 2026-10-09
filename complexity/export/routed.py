"""Tensor/config translations for the historical custom MLX and vLLM layouts.

These are offline data transformations, requiring Torch only for tensor
conversion. The matching custom inference runtimes are external to this
package; this module neither loads nor qualifies either runtime. For an
autonomous model bundle, use complexity.export.transformers instead.
"""

from __future__ import annotations

from .assets import copy_tokenizer_files

__all__ = [
    "build_mlx_config",
    "build_vllm_config",
    "copy_tokenizer_files",
    "remap_state_dict",
    "remap_hf_state_dict",
    "vllm_tensor_name",
]

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
    "use_qk_norm",
    "tie_word_embeddings",
    "top_k",
    "top_k_primary_weight",
    "shared_output_scale",
    "routed_output_scale",
    "routing_strategy",
    "route_hash_count",
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


def _validate_layout_config(raw):
    # Both archived layouts assume GQA, standard full RoPE and fixed branch
    # scales. Reject options whose inference behavior would be discarded.
    required = {
        "attention_type": "gqa",
        "norm_type": "rmsnorm",
        "rope_type": "standard",
        "rope_fraction": 1.0,
        "sliding_window": None,
        "learn_hash_pair_gates": False,
        "lsh_routing": False,
        "active_num_experts": None,
        "active_expert_width": None,
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
        for name, expected in required.items()
        if name in raw and raw[name] != expected
    }
    if raw.get("mlp_type", "token_routed") not in {"tr_hash_engine", "tr_hash_moe", "token_routed"}:
        unsupported["mlp_type"] = raw["mlp_type"]
    if unsupported:
        raise ValueError(f"Unsupported legacy routed layout options: {unsupported}")


def _target_dtype(dtype):
    import torch

    if dtype not in {"float16", "float32"}:
        raise ValueError("Legacy MLX storage conversion supports float16 or float32")
    return getattr(torch, dtype)


def _routes(value):
    import torch

    if (
        value.ndim != 2
        or min(value.shape) < 1
        or value.is_floating_point()
        or value.min().item() < 0
        or value.max().item() > torch.iinfo(torch.int32).max
    ):
        raise ValueError(
            "Route tables must be nonnegative integer [routes, vocabulary] arrays fitting int32"
        )
    return value.detach().cpu().to(torch.int32).contiguous().clone()


def build_mlx_config(pt_config: dict) -> dict:
    _validate_layout_config(pt_config)
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
    import torch

    target_dtype = _target_dtype(dtype)
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
                routes = _routes(v)
                out[f"{mlx_prefix}topk_token_to_expert"] = routes
                # Clone the primary route: safetensors intentionally rejects
                # separately named tensors that share the same backing
                # storage, even when one is only a view of the other.
                out[f"{mlx_prefix}token_to_expert"] = routes[0].clone()
                continue
            mapped_name = _ENGINE_PARAMETER_NAMES.get(engine_name)
            if mapped_name is None:
                raise ValueError(f"Unsupported TRHashEngine checkpoint tensor: {k}")
            new_key = f"{mlx_prefix}{mapped_name}"
        else:
            new_key = f"model.{k}"

        if new_key.endswith(("token_to_expert", "topk_token_to_expert")) and v.dtype != torch.int32:
            v = v.to(torch.int32)
        if v.is_floating_point():
            v = v.to(target_dtype)
        out[new_key] = v.detach().cpu().contiguous().clone()
    return out


def remap_hf_state_dict(model_sd: dict, *, dtype: str) -> dict:
    """Translate a public HF/vLLM TR-Hash bundle into the MLX module tree."""

    import torch

    target_dtype = _target_dtype(dtype)
    out = {}
    for key, value in model_sd.items():
        new_key = key if key.startswith("model.") else f"model.{key}"
        if new_key.endswith("topk_token_to_expert"):
            routes = _routes(value)
            out[new_key] = routes
            primary_key = new_key.removesuffix("topk_token_to_expert") + "token_to_expert"
            out[primary_key] = routes[0].clone()
            continue
        if new_key.endswith("token_to_expert") and value.dtype != torch.int32:
            value = value.to(torch.int32)
        if value.is_floating_point():
            value = value.to(target_dtype)
        out[new_key] = value.detach().cpu().contiguous().clone()
    return out


def build_vllm_config(raw: dict, chat_template: dict | None = None) -> dict:
    """Translate the training configuration to the DeepConfig contract."""

    _validate_layout_config(raw)
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
        "learn_hash_channel_modulation": raw.get("learn_hash_channel_modulation", False),
        "hash_channel_scale_init": raw.get("hash_channel_scale_init", 0.0),
        "norm_type": raw.get("norm_type", "rmsnorm"),
        "norm_eps": raw.get("norm_eps", 1e-6),
        "rms_norm_eps": raw.get("norm_eps", 1e-6),
        "tie_word_embeddings": raw.get("tie_word_embeddings", True),
        "torch_dtype": "bfloat16",
    }
    if raw.get("mlp_type") == "tr_hash_engine":
        # The public vLLM runtime represents the same fixed route table and
        # expert tensors through TokenRoutedMLP.  Serialize that runtime name
        # while retaining the training architecture as provenance.
        config["mlp_type"] = "token_routed"
        config["source_mlp_type"] = "tr_hash_engine"
        # TRHashEngine uses fixed branch scales and has no learned legacy
        # shared/routed scalar gates.
        config["use_shared_routed_gates"] = False
    if chat_template is not None:
        config["chat_template_id"] = chat_template["id"]
        config["chat_template_file"] = "chat_template.jinja"
    return config


def vllm_tensor_name(name: str) -> str:
    """Translate native TRHashEngine tensor names to the vLLM contract."""

    replacements = (
        (".mlp.engine.expert_gate", ".mlp.gate_proj_w"),
        (".mlp.engine.expert_up", ".mlp.up_proj_w"),
        (".mlp.engine.expert_down", ".mlp.down_proj_w"),
        (".mlp.engine.route_table", ".mlp.topk_token_to_expert"),
        (".mlp.engine.shared_gate", ".mlp.shared_gate"),
        (".mlp.engine.shared_up", ".mlp.shared_up"),
        (".mlp.engine.shared_down", ".mlp.shared_down"),
    )
    for source, target in replacements:
        if source in name:
            return name.replace(source, target, 1)
    return name
