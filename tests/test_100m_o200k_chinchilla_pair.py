"""Reproducibility guards for the matched 100M o200k / 2B-token B200 pair."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

CONFIG_ROOT = Path("configs/run_configs/100m_o200k_chinchilla")
DENSE_CONFIG = CONFIG_ROOT / "dense_gqa_seed42_2b_b200.yaml"
TR_CONFIG = CONFIG_ROOT / "tr_gqa_fixed_id_seed42_2b_b200.yaml"
FREQUENCY_BALANCED_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_frequency_balanced_seed42_2b_b200.yaml"
)


def _load_args(path: Path):
    from complexity.training.o200k.cli import build_parser
    from complexity.training.o200k.profiles import PROFILES
    from complexity.training.run_config import parse_args_with_yaml_config

    args = parse_args_with_yaml_config(build_parser(), ["--config", str(path)])
    profile = PROFILES[args.profile]
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "shared_intermediate_size",
        "run_name",
        "save_dir",
    ):
        if getattr(args, key) is None:
            setattr(args, key, profile[key])
    args.vocab_size = 200_019
    return args


def test_pair_is_exactly_parameter_matched():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    counts = []
    for path in (DENSE_CONFIG, TR_CONFIG):
        args = _load_args(path)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        counts.append(model.num_parameters())

    assert counts == [99_487_680, 99_487_680]


def test_frequency_balanced_pair_is_exactly_parameter_matched():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    counts = []
    for path in (DENSE_CONFIG, FREQUENCY_BALANCED_TR_CONFIG):
        args = _load_args(path)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        counts.append(model.num_parameters())

    assert counts == [99_487_680, 99_487_680]


def test_pair_shares_protocol_and_consumes_two_billion_tokens():
    dense = yaml.safe_load(DENSE_CONFIG.read_text())["run"]
    routed = yaml.safe_load(TR_CONFIG.read_text())["run"]
    matched = [
        "dataset",
        "tokens_path",
        "eval_tokens_path",
        "token_order",
        "tokenizer",
        "steps",
        "batch_size",
        "seq_len",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_type",
        "optimizer",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "bf16",
        "grad_ckpt",
        "loss_backend",
        "eval_steps",
        "eval_batches",
        "seed",
    ]
    assert {key: dense[key] for key in matched} == {
        key: routed[key] for key in matched
    }
    assert dense["intermediate_size"] == (
        routed["shared_intermediate_size"] + routed["intermediate_size"]
    )
    assert dense["steps"] * 4 * dense["batch_size"] * dense["seq_len"] == 1_999_896_576
    assert routed["routing_strategy"] == "modulo_cyclic"
    assert routed["top_k"] == 2
    assert routed["top_k_primary_weight"] == 0.5
    assert routed["learn_shared_routed_gates"] is False


def test_frequency_balanced_pair_shares_the_dense_b200_protocol():
    dense = yaml.safe_load(DENSE_CONFIG.read_text())["run"]
    routed = yaml.safe_load(FREQUENCY_BALANCED_TR_CONFIG.read_text())["run"]
    matched = [
        "dataset",
        "tokens_path",
        "eval_tokens_path",
        "token_order",
        "tokenizer",
        "steps",
        "batch_size",
        "seq_len",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_type",
        "optimizer",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "label_smoothing",
        "z_loss",
        "bf16",
        "grad_ckpt",
        "loss_backend",
        "loss_chunk_tokens",
        "loss_checkpoint_chunks",
        "use_custom_kernels",
        "cggr",
        "compile",
        "eval_steps",
        "eval_batches",
        "seed",
    ]

    assert {key: dense[key] for key in matched} == {
        key: routed[key] for key in matched
    }
    assert dense["intermediate_size"] == (
        routed["shared_intermediate_size"] + routed["intermediate_size"]
    )
    assert routed["routing_strategy"] == "modulo_frequency_balanced_secondary"
    assert routed["top_k"] == 2
    assert routed["top_k_primary_weight"] == 0.5
