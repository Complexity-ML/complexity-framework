#!/usr/bin/env python3
"""Validate or build the production TR-HASH 125B pretraining corpus."""

from scripts.build_agentic_pretraining_50b import main

if __name__ == "__main__":
    main(
        default_config="configs/agentic_pretraining/tr_hash_pretraining_125b.json",
        default_work_dir="artifacts/tr_hash_pretraining_125b_build",
        default_hf_repo="AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K",
        default_repo_prefix="",
        default_dataset_card="docs/datasets/tr-hash-pretraining-125b-agentic-32k.md",
    )
