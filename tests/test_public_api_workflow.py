"""CPU integration checks for the supported convenience API workflow."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from complexity import ComplexityModel, ModelConfig
from complexity.api import Dataset, Model, Trainer, TrainerConfig


def small_config():
    return ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=32,
        vocab_size=32,
        num_experts=4,
        top_k=2,
    )


def test_package_metadata_does_not_require_torch_or_optional_surfaces():
    code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Metadata import requested torch")
sys.meta_path.insert(0, BlockTorch())
import complexity
import complexity.api
assert complexity.__version__ == "0.3.0"
assert "Model" in dir(complexity.api)
assert "complexity.training" not in sys.modules
assert "complexity.multimodal" not in sys.modules
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_model_import_does_not_load_multimodal_or_convenience_training():
    code = """
import sys
from complexity.api import Model, ModelConfig
assert Model is not None and ModelConfig is not None
assert "complexity.multimodal" not in sys.modules
assert "complexity.api.trainer" not in sys.modules
assert "complexity.tokenizer" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_model_config_overrides_are_validated_without_mutating_input():
    config = small_config()
    model = Model.create(config, num_hidden_layers=2)
    assert model.config.num_hidden_layers == 2
    assert config.num_hidden_layers == 1
    assert model.config is not config
    with pytest.raises(ValueError, match="hidden_szie"):
        Model.create(config, hidden_szie=24)
    with pytest.raises(ValueError):
        Model.create(config, hidden_size=15)
    with pytest.raises(ValueError, match="hidden_szie"):
        Model.create(hidden_szie=24)


@pytest.mark.parametrize("safe_serialization", [False, True])
def test_wrapper_and_native_model_exports_are_interoperable(tmp_path, safe_serialization):
    model = Model.create(small_config()).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids)["logits"]
    model.save(tmp_path, safe_serialization=safe_serialization)
    native = ComplexityModel.from_pretrained(tmp_path).eval()
    loaded = Model.load(tmp_path).eval()
    assert torch.equal(native(ids)["logits"], expected)
    assert torch.equal(loaded(ids)["logits"], expected)
    assert loaded.config.to_dict() == model.config.to_dict()
    assert list(native.state_dict()) == list(model.module.state_dict())


def test_pytorch_model_pt_load_preserves_training_metadata(tmp_path):
    model = Model.create(small_config()).eval()
    config = model.config.to_dict()
    config["historical_run_metadata"] = "must not become an architecture field"
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(model.module.state_dict(), tmp_path / "model.pt")
    ids = torch.tensor([[1, 2, 3, 4]])
    for path in [tmp_path, tmp_path / "model.pt"]:
        assert torch.equal(Model.load(path).eval()(ids)["logits"], model(ids)["logits"])
    assert torch.equal(
        ComplexityModel.from_pretrained(tmp_path).eval()(ids)["logits"], model(ids)["logits"]
    )


def test_checkpoint_loading_never_succeeds_without_weights(tmp_path):
    small_config().save(str(tmp_path / "config.json"))
    with pytest.raises(FileNotFoundError, match="No model weights"):
        Model.load(tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        Model.load(tmp_path / "missing.pt")


def test_checkpoint_loader_rebuilds_deterministic_buffers_but_rejects_missing_weights(tmp_path):
    model = Model.create(small_config())
    small_config().save(str(tmp_path / "config.json"))
    state = model.module.state_dict()
    omitted = {name: value for name, value in state.items() if not name.endswith(".inv_freq")}
    torch.save(omitted, tmp_path / "model.pt")
    loaded = Model.load(tmp_path)
    assert set(loaded.module.state_dict()) == set(state)
    learned_key = next(name for name, _ in model.module.named_parameters())
    omitted.pop(learned_key)
    torch.save(omitted, tmp_path / "model.pt")
    with pytest.raises(RuntimeError, match="missing="):
        Model.load(tmp_path)


def test_forward_embedded_inputs_follow_native_contract():
    model = Model.create(small_config()).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    hidden = model.module.embed_tokens(ids)
    output = model(inputs_embeds=hidden, routing_ids=ids)
    assert torch.equal(output["logits"], model(ids)["logits"])


def test_trainer_config_preserves_batch_size_and_full_config_extensions():
    config = TrainerConfig(batch_size=3, extra={"use_fsdp": False, "learning_rate": 0.001})
    internal = config.to_training_config()
    assert internal.batch_size == 3
    assert internal.learning_rate == 0.001
    assert internal.use_fsdp is False
    with pytest.raises(ValueError, match="learning_arte"):
        TrainerConfig(extra={"learning_arte": 0.001}).to_training_config()


def test_real_cpu_training_updates_wrapper_model_and_restores_run_state(tmp_path):
    torch.manual_seed(7)
    model = Model.create(small_config())
    initial = model.module.embed_tokens.weight.detach().clone()
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    dataset = Dataset([{"input_ids": ids, "labels": ids.clone()} for _ in range(4)])
    config = TrainerConfig(
        max_steps=1,
        batch_size=2,
        precision="fp32",
        warmup_steps=0,
        lr=0.001,
        save_steps=20,
        eval_steps=0,
        output_dir=str(tmp_path / "run"),
        extra={"use_fsdp": False, "log_dir": str(tmp_path / "logs")},
    )
    trainer = Trainer(model, config)
    summary = trainer.train(dataset)
    assert summary["total_steps"] == 1
    assert trainer.backend.train_dataloader.batch_size == 2
    assert trainer.backend.config.batch_size == 2
    assert not torch.equal(initial, model.module.embed_tokens.weight)
    checkpoint = Path(trainer.save_checkpoint())
    assert json.loads((checkpoint / "training_state.json").read_text())["step"] == 1
    assert (checkpoint / "optimizer_rank0.pt").is_file()
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    assert "scheduler" in payload
    expected_optimizer = deepcopy(trainer.backend.optimizer.state_dict())

    export = tmp_path / "export"
    trainer.save(str(export), safe_serialization=False)
    loaded = Model.load(export)
    assert torch.equal(loaded.module.embed_tokens.weight, model.module.embed_tokens.weight)

    restored_steps = []

    class ResumeCallback:
        def on_resume(self, step):
            restored_steps.append(step)
            restored_state = resumed.backend.optimizer.state_dict()["state"]
            for parameter, state in expected_optimizer["state"].items():
                assert torch.equal(restored_state[parameter]["exp_avg"], state["exp_avg"])
                assert torch.equal(restored_state[parameter]["exp_avg_sq"], state["exp_avg_sq"])
                assert torch.equal(restored_state[parameter]["step"], state["step"])

        def __call__(self, trainer, step, loss):
            assert trainer.optimizer.state  # restored and advanced optimizer moments

    resumed = Trainer(loaded, replace(config, max_steps=2, resume_from=str(checkpoint)))
    resumed.train(dataset, callbacks=[ResumeCallback()])
    assert restored_steps == [1]
    assert resumed.backend.global_step == 2
    assert config.resume_from is None


def test_trainer_rejects_unknown_overrides_before_initialization(tmp_path):
    trainer = Trainer(Model.create(small_config()))
    with pytest.raises(ValueError, match="max_step"):
        trainer.train([], max_step=1)
    assert trainer.backend is None
    with pytest.raises(RuntimeError, match="Call train"):
        trainer.save_checkpoint()


@pytest.mark.parametrize("entrypoint", ["text", "tokens", "stream", "engine", "chat"])
def test_disabled_generation_fails_consistently_before_side_effects(entrypoint):
    from complexity.api import Generate, GenerationConfig

    class UnexpectedTokenizer:
        def encode(self, *args, **kwargs):
            raise AssertionError("Disabled generation tried to tokenize")

        def encode_chat(self, *args, **kwargs):
            raise AssertionError("Disabled generation tried to tokenize")

    config = GenerationConfig(temperature=0.4)
    options = {"config": config, "temperature": 0.7, "tokenizer": UnexpectedTokenizer()}
    with pytest.raises(RuntimeError, match="create_external_backend.*vLLM.*SGLang"):
        if entrypoint == "engine":
            Generate.engine(None)
        else:
            getattr(Generate, entrypoint)(None, "Hello", **options)
    assert config.temperature == 0.4


def test_external_generation_helpers_share_the_canonical_implementation():
    from complexity.api import ExternalGenerationConfig, create_external_backend
    from complexity.inference.external import (
        ExternalGenerationConfig as CanonicalConfig,
    )
    from complexity.inference.external import (
        create_external_backend as canonical_backend,
    )

    assert ExternalGenerationConfig is CanonicalConfig
    assert create_external_backend is canonical_backend


def test_training_configuration_import_does_not_initialize_model_runtime():
    code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Training config import requested torch")
sys.meta_path.insert(0, BlockTorch())
from complexity.training import TrainingConfig, SupervisorManager
assert TrainingConfig(max_steps=1).max_steps == 1
assert SupervisorManager is not None
assert "complexity.training.runner" not in sys.modules
assert "complexity.training.trainer" not in sys.modules
assert "complexity.tokenizer" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tokenizer_metadata_import_does_not_require_torch():
    code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Tokenizer metadata import requested torch")
sys.meta_path.insert(0, BlockTorch())
from complexity.tokenizer import TokenizerConfig, get_special_tokens
from complexity.tokenizer.training import AGENTIC_SPECIAL_TOKENS
from complexity.export.transformers import build_transformers_config
assert TokenizerConfig(vocab_size=384).vocab_size == 384
assert AGENTIC_SPECIAL_TOKENS == get_special_tokens("tr_hash_agentic_reasoning")
assert build_transformers_config({"top_k": 2, "routing_strategy": "token_id_composed_modulo"})["num_experts_per_tok"] == 2
"""
    subprocess.run([sys.executable, "-c", code], check=True)
