"""
Regression test for the FSDP2 save_pretrained rank-0-only corruption
fixed in f2b95d6 (Phase 2).

Before the fix, full_tensor() exceptions were swallowed and the code
fell through to to_local(), writing rank 0's shard only — producing
a safetensors file with ~1/world_size of the real weights.

This test spawns a 2-process gloo group on CPU (no GPU / no torchrun
needed), wraps a small MoE model with fully_shard(), saves, and
asserts that the saved safetensors contains every parameter bit-exact
relative to the unsharded reference.

Run with:
    pytest tests/test_save_pretrained_fsdp.py -v -s
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _small_moe_config():
    from complexity.config import ModelConfig
    return ModelConfig(
        hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=256, vocab_size=1024,
        max_position_embeddings=128, attention_type="gqa",
        mlp_type="token_routed", num_experts=4, shared_expert=True,
        norm_type="rmsnorm", use_qk_norm=True, use_mu_guidance=True,
    )


def _worker(rank: int, world_size: int, save_path: str, ref_path: str, port: int):
    """Init gloo group, shard, save_pretrained, exit."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    from complexity.models import ComplexityModel

    torch.manual_seed(0)
    model = ComplexityModel(_small_moe_config())
    if rank == 0:
        torch.save(model.state_dict(), ref_path)

    mesh = init_device_mesh("cpu", mesh_shape=(world_size,))
    fully_shard(model, mesh=mesh)

    dist.barrier()
    model.save_pretrained(save_path)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="mp.spawn + gloo flaky on Windows")
def test_fsdp2_save_pretrained_full_coverage():
    """Sharded params must be fully gathered in the saved safetensors."""
    from safetensors.torch import load_file

    world_size = 2
    # Use a random-ish port so parallel pytest workers don't collide.
    port = 29500 + (os.getpid() % 1000)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = str(Path(tmp) / "save")
        ref_path = str(Path(tmp) / "ref.pt")

        mp.spawn(
            _worker,
            args=(world_size, save_path, ref_path, port),
            nprocs=world_size,
            join=True,
        )

        ref = torch.load(ref_path, map_location="cpu", weights_only=False)
        saved = load_file(str(Path(save_path) / "model.safetensors"))

    n_ref = sum(v.numel() for v in ref.values())
    n_saved = sum(v.numel() for v in saved.values())

    missing = set(ref) - set(saved)
    extra = set(saved) - set(ref)
    mismatched = [
        k for k in ref
        if k in saved and not torch.equal(ref[k].cpu(), saved[k].cpu())
    ]

    # Param count must match (regression guard for rank-0-only corruption:
    # that bug yielded n_saved ≈ n_ref / world_size).
    assert n_saved == n_ref, (
        f"Param count mismatch: saved {n_saved:,} vs ref {n_ref:,} "
        f"(ratio {n_saved / n_ref:.4f}). Rank-0-only corruption regression?"
    )
    assert not missing, f"Missing keys in saved file: {sorted(missing)[:5]}"
    assert not extra, f"Extra keys in saved file: {sorted(extra)[:5]}"
    assert not mismatched, (
        f"{len(mismatched)} params differ bit-wise from reference; "
        f"first: {mismatched[:3]}"
    )
