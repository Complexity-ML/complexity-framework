"""Explicit rejection of removed token-only routing transformations."""


def expanded_expert_config(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")


def convert_checkpoint_to_expanded_experts(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")


def convert_checkpoint_dir_to_expanded_experts(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")
