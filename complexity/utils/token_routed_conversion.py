"""Explicit rejection of removed token-only routing transformations."""


def convert_token_routed_state_dict(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")


def convert_token_routed_config(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")


def convert_token_routed_checkpoint(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")


def convert_token_routed_checkpoint_dir(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")
