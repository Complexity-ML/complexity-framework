from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


ABLATION_NAMES = [
    "100m_zipf_shared",
    "100m_zipf_no_shared",
    "100m_modulo_shared",
    "100m_random_shared",
    "100m_round_robin_shared",
    "100m_shared_only",
    "100m_dense_residual",
]


def test_token_routed_supports_explicit_lexical_routing_strategies():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    freqs = torch.tensor([100.0, 90.0, 80.0, 70.0, 4.0, 3.0, 2.0, 1.0])

    zipf = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="zipf",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    modulo = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="modulo",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    random_a = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    random_b = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    round_robin = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="round_robin",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()

    assert not torch.equal(zipf, modulo)
    assert torch.equal(random_a, random_b)
    assert not torch.equal(random_a, modulo)
    assert sorted(round_robin.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_topk_auxiliary_routes_preserve_control_strategy():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    freqs = torch.tensor([100.0, 90.0, 80.0, 70.0, 4.0, 3.0, 2.0, 1.0])

    modulo = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="modulo",
            token_frequencies=freqs,
            top_k=2,
            shared_expert=False,
        )
    ).topk_token_to_expert.cpu()
    random_a = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            top_k=2,
            shared_expert=False,
        )
    ).topk_token_to_expert.cpu()
    random_b = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            top_k=2,
            shared_expert=False,
        )
    ).topk_token_to_expert.cpu()

    assert torch.equal(modulo[1], (modulo[0] + 1) % 4)
    assert torch.equal(random_a, random_b)
    assert torch.all(random_a[0] != random_a[1])


def test_modulo_primary_balanced_secondary_is_distinct_and_balanced():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    freqs = torch.tensor(
        [100.0, 90.0, 80.0, 70.0, 40.0, 30.0, 20.0, 10.0]
    )
    mlp = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="modulo_balanced_secondary",
            token_frequencies=freqs,
            top_k=2,
            shared_expert=False,
        )
    )
    routes = mlp.topk_token_to_expert.cpu()

    assert torch.all(routes[0] != routes[1])
    secondary_load = torch.zeros(4)
    secondary_load.scatter_add_(0, routes[1], freqs)
    assert float(secondary_load.max() - secondary_load.min()) <= float(
        freqs.max()
    )


def test_model_config_and_o200k_parser_support_ablation_switches():
    from complexity.config import ModelConfig
    from complexity.training.o200k_pretrain import build_parser, make_config

    args = build_parser().parse_args([
        "--routing-strategy", "random",
        "--no-shared-expert",
    ])
    args.vocab_size = 200019
    profile = {
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "shared_intermediate_size": 1536,
    }
    for key, value in profile.items():
        setattr(args, key, value)

    config = make_config(args)

    assert ModelConfig(routing_strategy="random").routing_strategy == "random"
    assert config.routing_strategy == "random"
    assert config.shared_expert is False


def test_seven_100m_ablation_yaml_configs_are_4b_token_runs():
    root = Path("configs/run_configs/ablations_100m")
    expected = {f"{name}.yaml" for name in ABLATION_NAMES}

    found = {p.name for p in root.glob("*.yaml")}

    assert expected <= found
    for name in ABLATION_NAMES:
        data = yaml.safe_load((root / f"{name}.yaml").read_text())["run"]
        assert data["profile"] == "100m"
        assert data["dataset"] == "fineweb"
        assert data["steps"] == 954
        assert data["batch_size"] == 256
        assert data["seq_len"] == 2048
        assert data["run_name"].startswith(f"abl-4b-{name}")
        assert data["save_dir"].endswith(data["run_name"])


def test_seven_100m_ablation_entrypoints_reference_configs():
    root = Path("scripts/ablations_100m")
    expected = {f"train_{name}.sh" for name in ABLATION_NAMES}

    found = {p.name for p in root.glob("train_*.sh")}

    assert expected <= found
    for name in ABLATION_NAMES:
        script = (root / f"train_{name}.sh").read_text()
        assert "scripts/train_100m_o200k_tr_local.py" in script
        assert f"configs/run_configs/ablations_100m/{name}.yaml" in script


def test_best_mha_balanced_shared_pilot_keeps_the_matched_parameter_width():
    path = Path(
        "configs/run_configs/experiments_100m/"
        "100m_params_mha_modulo_balanced_shared_1296_mps.yaml"
    )
    run = yaml.safe_load(path.read_text())["run"]

    assert run["attention_type"] == "mha"
    assert run["num_attention_heads"] == run["num_key_value_heads"] == 8
    assert run["routing_strategy"] == "modulo_balanced_secondary"
    assert run["top_k"] == 2
    assert run["top_k_primary_weight"] == 0.5
    assert run["top_k_primary_weight_final"] == 0.5
    assert run["shared_intermediate_size"] == 1296
    assert run["intermediate_size"] == 160
    assert run["shared_intermediate_size"] + run["intermediate_size"] == 1456
    assert run["steps"] * run["batch_size"] * run["seq_len"] == 1_024_000


@pytest.mark.parametrize(
    ("filename", "fixed_name", "seed"),
    [
        (
            "100m_params_gqa_learned_top2_shared_256_mps.yaml",
            "100m_params_gqa_modulo_balanced_shared_256_mps.yaml",
            42,
        ),
        (
            "100m_params_gqa_learned_top2_shared_256_seed43_mps.yaml",
            "100m_params_gqa_modulo_balanced_shared_256_seed43_mps.yaml",
            43,
        ),
        (
            "100m_params_mha_learned_top2_shared_1296_mps.yaml",
            "100m_params_mha_modulo_balanced_shared_1296_mps.yaml",
            42,
        ),
        (
            "100m_params_mha_learned_top2_shared_1296_seed43_mps.yaml",
            "100m_params_mha_modulo_balanced_shared_1296_seed43_mps.yaml",
            43,
        ),
    ],
)
def test_learned_router_controls_match_fixed_protocol(
    filename, fixed_name, seed
):
    root = Path("configs/run_configs/experiments_100m")
    learned = yaml.safe_load((root / filename).read_text())["run"]
    fixed = yaml.safe_load((root / fixed_name).read_text())["run"]

    matched_fields = [
        "dataset",
        "text_file",
        "tokenizer",
        "steps",
        "batch_size",
        "seq_len",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_type",
        "intermediate_size",
        "shared_intermediate_size",
        "top_k",
        "shared_expert",
        "learn_shared_routed_gates",
        "optimizer",
        "lr",
        "weight_decay",
        "eval_steps",
        "eval_batches",
        "seed",
    ]
    assert {key: learned[key] for key in matched_fields} == {
        key: fixed[key] for key in matched_fields
    }
    assert learned["mlp_type"] == "learned_router"
    assert fixed["mlp_type"] == "token_routed"
    assert learned["router_aux_loss_weight"] == 0.01


@pytest.mark.parametrize("routed_width", [64, 128, 160, 256])
def test_gqa_balanced_shared_pilots_match_dense_width_and_protocol(routed_width):
    path = Path(
        "configs/run_configs/experiments_100m/"
        f"100m_params_gqa_modulo_balanced_shared_{routed_width}_mps.yaml"
    )
    run = yaml.safe_load(path.read_text())["run"]

    assert run["attention_type"] == "gqa"
    assert run["num_attention_heads"] == 8
    assert run["num_key_value_heads"] == 2
    assert run["routing_strategy"] == "modulo_balanced_secondary"
    assert run["top_k"] == 2
    assert run["top_k_primary_weight"] == 0.5
    assert run["top_k_primary_weight_final"] == 0.5
    assert run["learn_shared_routed_gates"] is False
    assert run["shared_intermediate_size"] + run["intermediate_size"] == 1648
    assert run["steps"] * run["batch_size"] * run["seq_len"] == 1_024_000
    assert run["seed"] == 42


def test_launcher_reports_the_real_tr_gqa_controls_only():
    from complexity.training.o200k_pretrain import (
        architecture_label,
        requires_routing_frequencies,
        token_routed_config_summary,
    )

    args = SimpleNamespace(
        attention_type="gqa",
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=384,
        num_hidden_layers=10,
        shared_intermediate_size=1520,
        intermediate_size=128,
        routing_strategy="modulo_balanced_secondary",
        top_k=2,
        top_k_primary_weight=0.5,
        top_k_primary_weight_final=0.5,
        grad_ckpt=False,
        learn_shared_routed_gates=False,
        expert_diversity_lambda=0.0,
        expert_diversity_target="down",
    )
    controls = SimpleNamespace(capabilities={"topk_primary_weight"})

    summary = token_routed_config_summary(args)

    assert architecture_label(args, controls) == "TR-GQA"
    assert "route=modulo_balanced_secondary" in summary
    assert "shared_width=1520" in summary
    assert "expert_width=32" in summary
    assert "Zipf" not in summary
    assert "lsh_threshold" not in summary
    assert "gates" not in summary
    assert not requires_routing_frequencies(
        SimpleNamespace(
            mlp_type="token_routed",
            routing_strategy="modulo_balanced_secondary",
        )
    )
    assert not requires_routing_frequencies(
        SimpleNamespace(
            mlp_type="token_routed",
            routing_strategy="modulo_cyclic",
        )
    )
    assert not requires_routing_frequencies(
        SimpleNamespace(
            mlp_type="swiglu",
            routing_strategy="modulo_balanced_secondary",
        )
    )


def test_gqa_seed43_confirmation_pair_matches_protocol():
    root = Path("configs/run_configs/experiments_100m")
    dense = yaml.safe_load(
        (root / "100m_params_gqa_dense_seed43_mps.yaml").read_text()
    )["run"]
    routed = yaml.safe_load(
        (
            root
            / "100m_params_gqa_modulo_balanced_shared_256_seed43_mps.yaml"
        ).read_text()
    )["run"]

    for run in (dense, routed):
        assert run["attention_type"] == "gqa"
        assert run["num_attention_heads"] == 8
        assert run["num_key_value_heads"] == 2
        assert run["steps"] * run["batch_size"] * run["seq_len"] == 1_024_000
        assert run["seed"] == 43
        assert run["save_steps"] == 0

    assert dense["intermediate_size"] == (
        routed["shared_intermediate_size"] + routed["intermediate_size"]
    )
    assert dense["mlp_type"] == "swiglu"
    assert routed["mlp_type"] == "token_routed"


def test_corrected_mha_pair_matches_gqa_pilot_protocol():
    root = Path("configs/run_configs/experiments_100m")
    dense = yaml.safe_load(
        (root / "100m_params_mha_dense_confirm_mps.yaml").read_text()
    )["run"]
    routed = yaml.safe_load(
        (
            root
            / "100m_params_mha_modulo_balanced_shared_1296_confirm_mps.yaml"
        ).read_text()
    )["run"]

    for run in (dense, routed):
        assert run["attention_type"] == "mha"
        assert run["num_attention_heads"] == 8
        assert run["num_key_value_heads"] == 8
        assert run["steps"] * run["batch_size"] * run["seq_len"] == 1_024_000
        assert run["seed"] == 42
        assert run["save_steps"] == 0

    assert dense["intermediate_size"] == (
        routed["shared_intermediate_size"] + routed["intermediate_size"]
    )
    assert dense["mlp_type"] == "swiglu"
    assert routed["mlp_type"] == "token_routed"


def test_mha_seed43_confirmation_pair_matches_protocol():
    root = Path("configs/run_configs/experiments_100m")
    dense = yaml.safe_load(
        (root / "100m_params_mha_dense_seed43_mps.yaml").read_text()
    )["run"]
    routed = yaml.safe_load(
        (
            root
            / "100m_params_mha_modulo_balanced_shared_1296_seed43_mps.yaml"
        ).read_text()
    )["run"]

    for run in (dense, routed):
        assert run["attention_type"] == "mha"
        assert run["num_attention_heads"] == 8
        assert run["num_key_value_heads"] == 8
        assert run["steps"] * run["batch_size"] * run["seq_len"] == 1_024_000
        assert run["seed"] == 43
        assert run["save_steps"] == 0

    assert dense["intermediate_size"] == (
        routed["shared_intermediate_size"] + routed["intermediate_size"]
    )
    assert dense["mlp_type"] == "swiglu"
    assert routed["mlp_type"] == "token_routed"
