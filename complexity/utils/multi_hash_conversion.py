"""Explicit rejection of removed token-only routing transformations."""


def multi_hash_config(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")


def convert_checkpoint_to_multi_hash(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")


def convert_checkpoint_dir_to_multi_hash(*args, **kwargs):
    raise ValueError("This routing conversion was removed; TR-HASH uses only token_id_composed_modulo with four experts and two routes")
