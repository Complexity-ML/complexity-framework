from __future__ import annotations

from pathlib import Path

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
