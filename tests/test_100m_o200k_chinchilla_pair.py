"""Reproducibility guards for the matched 100M o200k / 2B-token B200 pair."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

CONFIG_ROOT = Path("configs/run_configs/100m_o200k_chinchilla")
DENSE_CONFIG = CONFIG_ROOT / "dense_gqa_seed42_2b_b200.yaml"
TR_CONFIG = CONFIG_ROOT / "tr_gqa_fixed_id_seed42_2b_b200.yaml"
PAPER_SCALED_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_paper_scaled_seed42_2b_b200.yaml"
)
BALANCED_SHARED_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_balanced_shared_seed42_2b_b200.yaml"
)
FREQUENCY_BALANCED_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_frequency_balanced_seed42_2b_b200.yaml"
)
LOCAL_WINNER_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_local_winner_seed42_2b_b200.yaml"
)
BALANCED_SECONDARY_MATCHED_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_balanced_secondary_matched_seed42_2b_b200.yaml"
)
BALANCED_SECONDARY_SCALE15_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_balanced_secondary_scale15_seed42_2b_b200.yaml"
)
DEPTH_SCALED_TR_CONFIG = (
    CONFIG_ROOT / "tr_gqa_depth_scaled_seed42_2b_b200.yaml"
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


def test_paper_scaled_pair_preserves_shared_residual_proportions():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    dense_args = _load_args(DENSE_CONFIG)
    routed_args = _load_args(PAPER_SCALED_TR_CONFIG)
    with torch.device("meta"):
        dense = ComplexityModel(make_config(dense_args))
        routed = ComplexityModel(make_config(routed_args))

    dense_run = yaml.safe_load(DENSE_CONFIG.read_text())["run"]
    routed_run = yaml.safe_load(PAPER_SCALED_TR_CONFIG.read_text())["run"]
    expert_width = routed_run["intermediate_size"] // 4

    assert routed_run["shared_intermediate_size"] == 1552
    assert expert_width == 24
    assert (
        routed_run["shared_intermediate_size"] + 4 * expert_width
        == dense_run["intermediate_size"]
    )
    assert (
        routed_run["shared_intermediate_size"] + 2 * expert_width
        == 1600
    )
    assert routed_run["learn_shared_routed_gates"] is True
    assert routed_run["shared_gate_init"] == 0.5
    assert routed_run["routed_gate_init"] == 0.5
    assert routed.num_parameters() - dense.num_parameters() == 20


def test_balanced_shared_pair_reproduces_the_winning_recipe_at_fixed_budget():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    dense_args = _load_args(DENSE_CONFIG)
    routed_args = _load_args(BALANCED_SHARED_TR_CONFIG)
    with torch.device("meta"):
        dense = ComplexityModel(make_config(dense_args))
        routed = ComplexityModel(make_config(routed_args))

    dense_run = yaml.safe_load(DENSE_CONFIG.read_text())["run"]
    routed_run = yaml.safe_load(BALANCED_SHARED_TR_CONFIG.read_text())["run"]

    assert routed_run["shared_intermediate_size"] == 816
    assert routed_run["intermediate_size"] == 832
    assert routed_run["intermediate_size"] // 4 == 208
    assert (
        routed_run["shared_intermediate_size"] + routed_run["intermediate_size"]
        == dense_run["intermediate_size"]
    )
    assert routed_run["routing_strategy"] == "modulo_balanced_secondary"
    assert routed_run["expert_initialization"] == "legacy_kaiming"
    assert routed_run["learn_shared_routed_gates"] is True
    assert routed_run["shared_gate_init"] == 1.0
    assert routed_run["routed_gate_init"] == 0.5
    assert routed.num_parameters() - dense.num_parameters() == 20


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

    routed = yaml.safe_load(FREQUENCY_BALANCED_TR_CONFIG.read_text())["run"]
    assert routed["shared_intermediate_size"] == 1392
    assert routed["intermediate_size"] == 256
    assert routed["routing_strategy"] == "modulo_frequency_balanced_secondary"
    assert routed["expert_initialization"] == "legacy_kaiming"
    assert routed["learn_shared_routed_gates"] is False
    assert routed["max_grad_norm"] == 1.0


def test_local_winner_pair_is_exactly_parameter_matched():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    counts = []
    for path in (DENSE_CONFIG, LOCAL_WINNER_TR_CONFIG):
        args = _load_args(path)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        counts.append(model.num_parameters())

    assert counts == [99_487_680, 99_487_680]

    routed = yaml.safe_load(LOCAL_WINNER_TR_CONFIG.read_text())["run"]
    assert routed["shared_intermediate_size"] == 1392
    assert routed["intermediate_size"] == 256
    assert routed["intermediate_size"] // 4 == 64
    assert routed["routing_strategy"] == "modulo_balanced_secondary"
    assert routed["expert_initialization"] == "legacy_kaiming"
    assert routed["learn_shared_routed_gates"] is False
    assert routed["max_grad_norm"] == 0.0


def test_balanced_secondary_matched_pair_changes_only_fixed_routing():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    counts = []
    for path in (DENSE_CONFIG, BALANCED_SECONDARY_MATCHED_TR_CONFIG):
        args = _load_args(path)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        counts.append(model.num_parameters())

    assert counts == [99_487_680, 99_487_680]

    routed = yaml.safe_load(
        BALANCED_SECONDARY_MATCHED_TR_CONFIG.read_text()
    )["run"]
    assert routed["shared_intermediate_size"] == 1392
    assert routed["intermediate_size"] == 256
    assert routed["routing_strategy"] == "modulo_balanced_secondary"
    assert routed["expert_initialization"] == "gpt_normal"
    assert routed["learn_shared_routed_gates"] is False
    assert routed["max_grad_norm"] == 1.0


def test_balanced_secondary_scale15_is_parameter_matched():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    counts = []
    for path in (DENSE_CONFIG, BALANCED_SECONDARY_SCALE15_TR_CONFIG):
        args = _load_args(path)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        counts.append(model.num_parameters())

    assert counts == [99_487_680, 99_487_680]

    routed = yaml.safe_load(
        BALANCED_SECONDARY_SCALE15_TR_CONFIG.read_text()
    )["run"]
    assert routed["routing_strategy"] == "modulo_balanced_secondary"
    assert routed["shared_output_scale"] == 1.0
    assert routed["routed_output_scale"] == 1.5
    assert routed["learn_shared_routed_gates"] is False


def test_depth_scaled_pair_is_parameter_matched_and_interpolates_by_layer():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    counts = []
    models = []
    for path in (DENSE_CONFIG, DEPTH_SCALED_TR_CONFIG):
        args = _load_args(path)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        counts.append(model.num_parameters())
        models.append(model)

    assert counts == [99_487_680, 99_487_680]

    routed_run = yaml.safe_load(DEPTH_SCALED_TR_CONFIG.read_text())["run"]
    assert routed_run["routing_strategy"] == "modulo_balanced_secondary"
    assert routed_run["shared_output_scale"] == 1.0
    assert routed_run["routed_output_scale_first_layer"] == 1.2
    assert routed_run["routed_output_scale_last_layer"] == 2.4
    assert routed_run["learn_shared_routed_gates"] is False

    routed_model = models[1]
    routed_scales = [
        layer.mlp.routed_output_scale for layer in routed_model.layers
    ]
    shared_scales = [
        layer.mlp.shared_output_scale for layer in routed_model.layers
    ]
    assert routed_scales[0] == 1.2
    assert routed_scales[-1] == 2.4
    assert routed_scales[4] == pytest.approx(1.2 + 4 * (1.2 / 9))
    assert shared_scales == [1.0] * 10


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
