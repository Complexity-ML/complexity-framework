"""Removed routing transformations fail explicitly before touching model weights."""

import pytest
from complexity.utils import multi_hash_conversion, expert_expansion, token_routed_conversion


@pytest.mark.parametrize(
    "function",
    [
        multi_hash_conversion.multi_hash_config,
        multi_hash_conversion.convert_checkpoint_to_multi_hash,
        multi_hash_conversion.convert_checkpoint_dir_to_multi_hash,
        expert_expansion.expanded_expert_config,
        expert_expansion.convert_checkpoint_to_expanded_experts,
        expert_expansion.convert_checkpoint_dir_to_expanded_experts,
        token_routed_conversion.convert_token_routed_state_dict,
        token_routed_conversion.convert_token_routed_config,
        token_routed_conversion.convert_token_routed_checkpoint,
        token_routed_conversion.convert_token_routed_checkpoint_dir,
    ],
)
def test_retired_routing_transformations_are_rejected(function):
    with pytest.raises(ValueError, match="conversion was removed"):
        function({})
