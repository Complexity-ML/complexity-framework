"""Legacy conversion helpers must reject the contextual Source67 architecture."""

from copy import deepcopy

import pytest

from complexity.utils.expert_expansion import expanded_expert_config
from complexity.utils.multi_hash_conversion import multi_hash_config


@pytest.mark.parametrize("convert", [multi_hash_config, expanded_expert_config])
@pytest.mark.parametrize(
    "markers",
    [
        {"architecture_profile": "source67", "routing_strategy": "token_id_balanced_hash"},
        {"routing_strategy": "token_id_composed_modulo"},
        {"architecture_profile": "legacy", "routing_strategy": "token_id_composed_modulo"},
        {"architecture_profile": "source67", "routing_strategy": "token_id_composed_modulo"},
    ],
)
def test_contextual_source67_conversion_is_rejected_before_config_or_weights(convert, markers):
    raw = {
        "hidden_size": 32,
        "num_experts": 4,
        "top_k": 2,
        "intermediate_size": 32,
        "mlp_type": "tr_hash_engine",
        **markers,
    }
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="Source67 contextual routing"):
        convert(raw)
    assert raw == before


@pytest.mark.parametrize("convert", [multi_hash_config, expanded_expert_config])
def test_explicit_legacy_config_still_converts(convert):
    raw = {
        "architecture_profile": "legacy",
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_experts": 4,
        "top_k": 2,
        "intermediate_size": 32,
        "mlp_type": "tr_hash_engine",
        "routing_strategy": "token_id_balanced_hash",
    }
    before = deepcopy(raw)
    converted = convert(raw)
    assert converted.routing_strategy != "token_id_composed_modulo"
    assert raw == before
