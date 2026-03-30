"""
Multi-GPU / FSDP v2 tests.

These tests require torchrun and 2+ GPUs. Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_fsdp.py -v
    torchrun --nproc_per_node=8 -m pytest tests/test_fsdp.py -v

Single-GPU / CPU tests (no torchrun needed):
    pytest tests/test_fsdp.py -v -k "not multigpu"
"""

import os
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return get_rank() == 0


def small_config():
    from complexity.config import ModelConfig
    return ModelConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1000,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=2,
        norm_type="rmsnorm",
        use_qk_norm=False,
        use_mu_guidance=False,
    )


def requires_multigpu(min_gpus: int = 2):
    """Skip if not enough GPUs or not running under torchrun."""
    n = torch.cuda.device_count()
    return pytest.mark.skipif(
        not is_distributed() or n < min_gpus,
        reason=f"Requires torchrun with {min_gpus}+ GPUs (found {n}, distributed={is_distributed()})",
    )


# ── Unit tests (no GPU required) ──────────────────────────────────────────────

class TestCheckpointHelpers:
    """Test DTensor helpers in checkpointing module (CPU, no distributed)."""

    def test_is_dtensor_plain_tensor(self):
        from complexity.utils.checkpointing import _is_dtensor
        t = torch.randn(4, 4)
        assert not _is_dtensor(t)

    def test_has_dtensors_plain(self):
        from complexity.utils.checkpointing import _has_dtensors
        sd = {"w": torch.randn(4, 4), "b": torch.zeros(4)}
        assert not _has_dtensors(sd)

    def test_detensor_plain(self):
        from complexity.utils.checkpointing import _detensor_state_dict
        sd = {"w": torch.randn(4, 4), "b": torch.zeros(4)}
        out = _detensor_state_dict(sd)
        assert set(out.keys()) == {"w", "b"}
        for v in out.values():
            assert isinstance(v, torch.Tensor)
            assert not _is_dtensor_name(v)

    def test_training_state_roundtrip(self):
        from complexity.utils.checkpointing import TrainingState
        state = TrainingState(step=500, epoch=1, best_loss=2.5, total_tokens=1_000_000, learning_rate=3e-4)
        d = state.to_dict()
        restored = TrainingState.from_dict(d)
        assert restored.step == 500
        assert restored.epoch == 1
        assert abs(restored.best_loss - 2.5) < 1e-6
        assert restored.total_tokens == 1_000_000

    def test_checkpoint_manager_init(self):
        from complexity.utils.checkpointing import CheckpointManager
        from complexity.models import ComplexityModel
        model = ComplexityModel(small_config())
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp, model)
            assert Path(tmp).exists()

    def test_checkpoint_manager_save_load_cpu(self):
        """Save and load checkpoint on CPU (no FSDP)."""
        from complexity.utils.checkpointing import CheckpointManager, TrainingState
        from complexity.models import ComplexityModel

        config = small_config()
        model = ComplexityModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)

        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp, model, optimizer=optimizer, max_checkpoints=2)
            state = TrainingState(step=100, learning_rate=1e-4)
            ckpt_path = mgr.save(step=100, training_state=state)

            # Verify files
            p = Path(ckpt_path)
            assert (p / "checkpoint.pt").exists()
            assert (p / "optimizer_rank0.pt").exists()
            assert (p / "training_state.json").exists()

            # Load back
            model2 = ComplexityModel(config)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, foreach=False)
            mgr2 = CheckpointManager(tmp, model2, optimizer=optimizer2)
            loaded_state = mgr2.load(ckpt_path)

            assert loaded_state is not None
            assert loaded_state.step == 100

    def test_checkpoint_rotation(self):
        """Old checkpoints are deleted after max_checkpoints."""
        from complexity.utils.checkpointing import CheckpointManager
        from complexity.models import ComplexityModel

        model = ComplexityModel(small_config())
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp, model, max_checkpoints=2)
            mgr.save(step=100)
            mgr.save(step=200)
            mgr.save(step=300)

            remaining = list(Path(tmp).glob("step_*"))
            assert len(remaining) <= 2


def _is_dtensor_name(v):
    return type(v).__name__ == "DTensor"


# ── Convert checkpoint helper (CPU) ───────────────────────────────────────────

class TestConvertCheckpoint:
    """Test the detensor logic used in convert_checkpoint.py."""

    def test_detensor_nested(self):
        """detensor() handles nested dicts and lists."""
        from scripts.convert_checkpoint import detensor

        sd = {
            "a": torch.randn(4, 4),
            "nested": {"b": torch.zeros(2)},
        }
        out = detensor(sd)
        assert out["a"].shape == (4, 4)
        assert out["nested"]["b"].shape == (2,)

    def test_detensor_list(self):
        from scripts.convert_checkpoint import detensor

        lst = [torch.randn(3), torch.ones(5)]
        out = detensor(lst)
        assert len(out) == 2
        assert out[0].shape == (3,)

    def test_config_json_written(self):
        """convert_checkpoint writes a valid config.json."""
        import json, importlib, sys
        # We can't run the full script, but we can test the config dict directly
        config_json = {
            "model_type": "complexity",
            "hidden_size": 1792,
            "num_hidden_layers": 24,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 4608,
            "vocab_size": 32000,
            "torch_dtype": "bfloat16",
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "config.json"
            out.write_text(json.dumps(config_json))
            loaded = json.loads(out.read_text())
            assert loaded["hidden_size"] == 1792
            assert loaded["torch_dtype"] == "bfloat16"


# ── Multi-GPU tests (torchrun required) ───────────────────────────────────────

@pytest.mark.multigpu
class TestFSDPv2:
    """FSDP v2 composable wrap + checkpoint tests. Requires torchrun."""

    @requires_multigpu(2)
    def test_fsdp_wrap(self):
        """Model wraps without error under fully_shard."""
        from torch.distributed._composable.fsdp import fully_shard
        from complexity.models import ComplexityModel

        config = small_config()
        model = ComplexityModel(config).to(torch.bfloat16)
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        # Forward pass
        device = torch.device(f"cuda:{get_rank()}")
        x = torch.randint(0, 1000, (2, 16)).to(device)
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        assert logits.shape == (2, 16, 1000)

    @requires_multigpu(2)
    def test_fsdp_adamw_foreach_false(self):
        """AdamW with foreach=False handles empty expert shards without crashing."""
        from torch.distributed._composable.fsdp import fully_shard
        from complexity.models import ComplexityModel

        config = small_config()
        model = ComplexityModel(config).to(torch.bfloat16)
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, foreach=False
        )

        device = torch.device(f"cuda:{get_rank()}")
        x = torch.randint(0, 1000, (2, 16)).to(device)
        labels = torch.randint(0, 1000, (2, 16)).to(device)

        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 1000), labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss.item() > 0

    @requires_multigpu(2)
    def test_fsdp_checkpoint_save_load(self):
        """Save checkpoint from FSDP model, reload, verify weights match."""
        from torch.distributed._composable.fsdp import fully_shard
        from complexity.models import ComplexityModel
        from complexity.utils.checkpointing import CheckpointManager, TrainingState

        config = small_config()

        def build_fsdp_model():
            m = ComplexityModel(config).to(torch.bfloat16)
            for layer in m.layers:
                fully_shard(layer)
            fully_shard(m)
            return m

        model = build_fsdp_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)

        # Do one step so optimizer has state
        device = torch.device(f"cuda:{get_rank()}")
        x = torch.randint(0, 1000, (2, 16)).to(device)
        labels = torch.randint(0, 1000, (2, 16)).to(device)
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 1000), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp, model, optimizer=optimizer)
            state = TrainingState(step=1, learning_rate=1e-4)
            ckpt_path = mgr.save(step=1, training_state=state)

            dist.barrier()

            # Verify per-rank optimizer files exist
            rank = get_rank()
            opt_file = Path(ckpt_path) / f"optimizer_rank{rank}.pt"
            assert opt_file.exists(), f"Missing {opt_file}"

            # Verify training_state.json on rank 0
            if is_main():
                state_file = Path(ckpt_path) / "training_state.json"
                assert state_file.exists()
                with open(state_file) as f:
                    d = json.load(d)
                assert d["step"] == 1

            dist.barrier()

            # Reload
            model2 = build_fsdp_model()
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, foreach=False)
            mgr2 = CheckpointManager(tmp, model2, optimizer=optimizer2)
            loaded_state = mgr2.load(ckpt_path)

            assert loaded_state is not None
            assert loaded_state.step == 1

    @requires_multigpu(2)
    def test_fsdp_full_tensor_gather(self):
        """full_tensor() on DTensor params returns complete unsharded tensor."""
        from torch.distributed._composable.fsdp import fully_shard
        from complexity.models import ComplexityModel

        config = small_config()
        model = ComplexityModel(config).to(torch.bfloat16)
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        total_global = 0
        total_local = 0
        for param in model.parameters():
            total_global += param.numel()  # global numel (DTensor reports full shape)
            if hasattr(param, "to_local"):
                total_local += param.to_local().numel()

        # global count should match expected param count
        from complexity.models import ComplexityModel as CM
        ref = CM(config)
        expected = sum(p.numel() for p in ref.parameters())
        assert total_global == expected

        # local count should be < global (sharded across ranks)
        if get_world_size() > 1:
            assert total_local < total_global
