"""CPU/meta contracts for the retained canonical FSDP recipe and corpus offsets.

The preparation script performs downloads at module scope, so it is never
imported here. No old schedule builder or training dataset is reconstructed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from scripts.pretrain_1b_fsdp.data import Corpus

ROOT = Path(__file__).parents[1] / "scripts/pretrain_1b_fsdp"


def test_fsdp_recipe_uses_canonical_tr_model_and_actual_meta_count():
    settings = json.loads((ROOT / "model.json").read_text())
    config = ModelConfig.from_dict(settings, strict=True)
    with torch.device("meta"):
        model = ComplexityModel(config)
    assert (config.num_attention_heads, config.num_key_value_heads) == (4, 1)
    assert (config.num_experts, config.top_k, config.top_k_primary_weight) == (4, 2, 0.5)
    assert (config.shared_output_scale, config.routed_output_scale) == (1.0, 2.0)
    assert config.routing_strategy == "token_id_composed_modulo"
    assert config.max_position_embeddings == 16384
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_011_838_464


def test_fsdp_recipe_model_pin_matches_the_retained_model():
    recipe = json.loads((ROOT / "recipe.json").read_text())
    assert recipe["model_sha256"] == hashlib.sha256((ROOT / "model.json").read_bytes()).hexdigest()


def test_fsdp_global_batch_matches_its_declared_seventy_billion_token_budget():
    recipe = json.loads((ROOT / "recipe.json").read_text())
    tokens_per_update = recipe["world_size"] * recipe["microbatch"] * recipe["context"]
    assert tokens_per_update == 131072
    assert recipe["max_updates"] == 70_000_000_000 // tokens_per_update
    assert recipe["max_updates"] * tokens_per_update == 69_999_919_104
    assert 0 <= 70_000_000_000 - recipe["max_updates"] * tokens_per_update < tokens_per_update


def synthetic_corpus():
    corpus = Corpus.__new__(Corpus)
    corpus.recipe = {"world_size": 4, "microbatch": 2}
    corpus.length = 4
    corpus.plan = {
        "updates": 4,
        "tokens_per_update": 32,
        "sources": [{"train_start_token": 32}, {"train_start_token": 96}],
    }
    corpus.schedule = np.array([0, 1, 0, 1], dtype=np.uint8)
    corpus.offsets = np.array([0, 0, 1, 1], dtype=np.int64)
    corpus.sequence = lambda source, offset: np.arange(
        source * 1000 + offset, source * 1000 + offset + 5
    )
    return corpus


def test_fsdp_rank_offsets_cover_each_update_once_and_resume_at_next_update():
    corpus = synthetic_corpus()
    observed = {0: [], 1: []}
    for update in range(4):
        source = int(corpus.schedule[update])
        beginning = (
            source * 1000
            + corpus.plan["sources"][source]["train_start_token"]
            + int(corpus.offsets[update]) * 32
        )
        inputs = np.concatenate(
            [corpus.batch(update, rank)[:, :-1].reshape(-1) for rank in range(4)]
        )
        assert sorted(inputs.tolist()) == list(range(beginning, beginning + 32))
        observed[source].extend(inputs.tolist())
    for inputs in observed.values():
        assert len(inputs) == len(set(inputs))
    for rank in range(4):
        uninterrupted = [corpus.batch(update, rank) for update in range(4)]
        resumed = [corpus.batch(update, rank) for update in range(2, 4)]
        assert len(resumed) == 2
        for actual, expected in zip(resumed, uninterrupted[2:], strict=True):
            np.testing.assert_array_equal(actual[:, :-1], expected[:, :-1])
            np.testing.assert_array_equal(actual[:, 1:], expected[:, 1:])


def test_fsdp_validation_reads_reserved_prefix_instead_of_training_offsets():
    corpus = synthetic_corpus()
    for source in range(2):
        validation = np.concatenate(
            [
                corpus.batch(3, rank, validation_source=source)[:, :-1].reshape(-1)
                for rank in range(4)
            ]
        )
        assert sorted(validation.tolist()) == list(range(source * 1000, source * 1000 + 32))
        training = corpus.batch(source, 0)[:, :-1].reshape(-1)
        assert set(validation).isdisjoint(training)
