"""The native package remains usable without integration extras."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from complexity import _optional

ROOT = Path(__file__).resolve().parents[1]


def test_missing_extra_has_an_actionable_message(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(_optional, "import_module", missing)
    with pytest.raises(
        _optional.MissingOptionalDependencyError,
        match=r'Loading a tokenizer requires tokenizers.*complexity-framework\[tokenizers\]',
    ):
        _optional.require_optional("tokenizers", "tokenizers", feature="Loading a tokenizer")


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("a transitive dependency is missing", name="broken_transitive"),
        ImportError("an installed extension cannot load"),
        RuntimeError("an installed integration cannot initialize"),
    ],
)
def test_guard_preserves_errors_inside_an_installed_integration(monkeypatch, error):
    def broken(name):
        raise error

    monkeypatch.setattr(_optional, "import_module", broken)
    with pytest.raises(type(error)) as raised:
        _optional.require_optional("tokenizers", "tokenizers")
    assert raised.value is error


def test_declared_core_and_integration_profiles_are_separate():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    optional_names = {
        "torch", "torchvision", "transformers", "huggingface-hub", "tokenizers",
        "tiktoken", "sentencepiece", "datasets", "wandb", "tensorboard", "pyarrow", "pandas",
    }
    core = {requirement.split(">", 1)[0].split(";", 1)[0] for requirement in project["dependencies"]}
    assert core.isdisjoint(optional_names)
    extras = project["optional-dependencies"]
    expected = {
        "hf": {"transformers", "huggingface-hub", "tokenizers"},
        "tokenizers": {"tokenizers", "tiktoken", "sentencepiece"},
        "data": {"datasets", "huggingface-hub", "pyarrow", "pandas"},
        "tracking": {"wandb", "tensorboard"},
    }
    for extra, dependencies in expected.items():
        assert {value.split(">", 1)[0] for value in extras[extra]} == dependencies
    assert not any(
        requirement.startswith("torch>") or requirement.startswith("torch=")
        for requirements in extras.values() for requirement in requirements
    ), "Torch must be installed separately with the chosen backend index"


def test_native_compute_evaluation_and_export_without_integration_extras(tmp_path):
    code = r'''
import importlib.abc
import math
import socket
import sys
from pathlib import Path
OPTIONAL = {'transformers', 'tokenizers', 'tiktoken', 'sentencepiece', 'datasets',
            'huggingface_hub', 'wandb', 'tensorboard', 'pyarrow', 'pandas'}
class MissingExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in OPTIONAL:
            raise ModuleNotFoundError('Optional runtime absent: '+fullname, name=root)
sys.meta_path.insert(0, MissingExtras())
def reject(*args, **kwargs):
    raise AssertionError('Native acceptance must not use the network')
socket.socket.connect = reject
socket.create_connection = reject
import torch
from complexity import ComplexityModel, ModelConfig
from complexity.evaluation import EvalConfig, Evaluator
from complexity.export.transformers import export_bundle
from complexity.training import Trainer, TrainingConfig
from complexity.training.callbacks import WandBCallback, TensorBoardCallback
from complexity.training.tensorboard import TensorBoardMetricWriter
import complexity.export.transformers_adapter as adapter
assert set(adapter.__all__) == {'TRHashConfig', 'TRHashForCausalLM'}
root = Path(sys.argv[1])
torch.set_num_threads(1)
torch.manual_seed(7)
model = ComplexityModel(ModelConfig(
    hidden_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=16, shared_intermediate_size=48, vocab_size=64,
    max_position_embeddings=16, mlp_type='tr_hash_engine', num_experts=4, top_k=2,
    routing_strategy='token_id_multi_hash', route_hash_count=2,
    use_custom_kernels=False, use_cggr=False,
)).cpu()
ids = torch.tensor([[1, 7, 13, 2]])
logits = model(ids)['logits']
loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1))
assert math.isfinite(loss.item())
loss.backward()
assert any(parameter.grad is not None for parameter in model.parameters())
assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
           if parameter.grad is not None)
assert Trainer is not None and TrainingConfig(max_steps=1, use_fsdp=False).max_steps == 1
perplexity = Evaluator(model, config=EvalConfig(verbose=False)).perplexity([{'input_ids': ids}])
assert math.isfinite(perplexity) and perplexity > 0
native = root/'native'
model.save_pretrained(native)
restored = ComplexityModel.from_pretrained(native).eval()
model.eval()
with torch.no_grad():
    torch.testing.assert_close(restored(ids)['logits'], model(ids)['logits'], rtol=0, atol=0)
export_bundle(config_path=native/'config.json', weights_path=native/'model.safetensors',
              output=root/'export')
from safetensors.torch import load_file
exported = load_file(root/'export'/'model.safetensors')
original = load_file(native/'model.safetensors')
assert set(exported) == set(original)
for name, weight in original.items():
    torch.testing.assert_close(exported[name], weight, rtol=0, atol=0)
assert WandBCallback('offline-test').wandb is None
assert TensorBoardCallback(str(root/'tracking')).writer is None
assert not TensorBoardMetricWriter(root/'tracking').enabled
assert not (root/'tracking').exists()
assert OPTIONAL.isdisjoint(sys.modules)
print('native-without-extras-ok')
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native-without-extras-ok" in result.stdout


def test_explicit_integrations_fail_before_network_or_output_mutation(tmp_path):
    code = r'''
import importlib.abc
import socket
import sys
from pathlib import Path
class MissingExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in {'transformers', 'tokenizers', 'tiktoken', 'sentencepiece', 'datasets'}:
            raise ModuleNotFoundError('Absent: '+fullname, name=root)
sys.meta_path.insert(0, MissingExtras())
def reject(*args, **kwargs):
    raise AssertionError('Missing integration must fail before network access')
socket.socket.connect = reject
socket.create_connection = reject
from complexity.tokenizer import Tokenizer
from complexity.tokenizer.training import train_tokenizer_from_iterator
from complexity.data import get_tokenizer
from complexity.evaluation import zero_shot
from typer.testing import CliRunner
from complexity.cli.app import app
root = Path(sys.argv[1])
tokenizer = root/'tokenizer.json'
tokenizer.write_text('{}')
calls = [
    (lambda: Tokenizer.load(str(tokenizer)), 'tokenizers'),
    (lambda: Tokenizer.load('tiktoken:cl100k_base'), 'tokenizers'),
    (lambda: Tokenizer.load('llama3-8b'), 'tokenizers'),
    (lambda: get_tokenizer('sentencepiece', model_path='unused.model'), 'tokenizers'),
    (lambda: get_tokenizer('hf', model_path='unused/model'), 'hf'),
    (lambda: train_tokenizer_from_iterator(['hello'], root/'new-tokenizer'), 'tokenizers'),
    (lambda: zero_shot.load_arc_easy(None), 'data'),
    (lambda: __import__('complexity.export.transformers_adapter', fromlist=['TRHashConfig']).TRHashConfig, 'hf'),
]
for call, extra in calls:
    try:
        call()
    except ImportError as error:
        assert 'complexity-framework['+extra+']' in str(error), str(error)
    else:
        raise AssertionError('An absent integration unexpectedly worked')
result = CliRunner().invoke(app, ['tokenize', 'encode', 'hello'])
assert result.exit_code == 1 and 'complexity-framework' in result.output, result.output
assert not (root/'new-tokenizer').exists()
print('missing-extras-actionable-ok')
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing-extras-actionable-ok" in result.stdout
