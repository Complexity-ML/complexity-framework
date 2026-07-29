"""Reproducibility guards for the matched 200M o200k / 4B-token B200 pair."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

CONFIG_ROOT = Path("configs/run_configs/200m_o200k_chinchilla")
DENSE_CONFIG = CONFIG_ROOT / "dense_gqa_seed42_4b_b200.yaml"
TR_CONFIG = CONFIG_ROOT / "tr_gqa_fixed_id_seed42_4b_b200.yaml"


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


def test_dense_and_fixed_id_configs_are_exactly_parameter_matched():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.profiles import make_config

    counts = []
    for path in (DENSE_CONFIG, TR_CONFIG):
        args = _load_args(path)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        counts.append(model.num_parameters())

    assert counts == [200_081_920, 200_081_920]


def test_dense_and_fixed_id_configs_share_the_full_training_protocol():
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
    assert routed["routing_strategy"] == "modulo_cyclic"
    assert routed["top_k"] == 2
    assert routed["top_k_primary_weight"] == 0.5
    assert routed["learn_shared_routed_gates"] is False


def test_four_billion_token_budget_matches_frozen_shard_target():
    run = yaml.safe_load(DENSE_CONFIG.read_text())["run"]
    predicted_tokens = run["steps"] * 4 * run["batch_size"] * run["seq_len"]

    assert predicted_tokens == 3_999_793_152
    prepare_script = Path("scripts/prepare_fineweb_o200k_shards.py").read_text()
    assert "default=3_999_793_153" in prepare_script
    assert 'DTYPE = np.dtype("<u4")' in prepare_script


def test_sequential_token_shard_partitions_ddp_and_resumes_exactly(tmp_path):
    from complexity.data.token_shards import TokenShardDataset, write_token_shard

    write_token_shard(tmp_path, range(65), vocab_size=128, tokenizer="test")
    rank0 = iter(
        TokenShardDataset(
            tmp_path,
            seq_len=4,
            rank=0,
            world_size=2,
            split="all",
            eval_ratio=0.0,
            order="sequential",
        )
    )
    rank1 = iter(
        TokenShardDataset(
            tmp_path,
            seq_len=4,
            rank=1,
            world_size=2,
            split="all",
            eval_ratio=0.0,
            order="sequential",
        )
    )

    assert next(rank0)["input_ids"].tolist() == [0, 1, 2, 3]
    assert next(rank1)["input_ids"].tolist() == [4, 5, 6, 7]
    assert next(rank0)["input_ids"].tolist() == [8, 9, 10, 11]
    assert next(rank1)["input_ids"].tolist() == [12, 13, 14, 15]

    resumed = iter(
        TokenShardDataset(
            tmp_path,
            seq_len=4,
            rank=0,
            world_size=2,
            split="all",
            eval_ratio=0.0,
            order="sequential",
            start_sequence=4,
        )
    )
    assert next(resumed)["input_ids"].tolist() == [16, 17, 18, 19]
