"""CLI discovery and local tokenization preserve their import/data contracts."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from complexity.cli.app import app

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("This acceptance check must not use the network")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def test_discovery_help_versions_and_legacy_tokens_do_not_import_runtimes():
    code = """
import importlib.abc
import socket
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'datasets', 'numpy', 'triton', 'flash_attn',
            'tiktoken', 'tokenizers', 'wandb', 'tensorboard', 'mlx', 'diffusers',
            'safetensors', 'cv2', 'PIL', 'onnx', 'onnxruntime', 'huggingface_hub',
        }:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, Blocker())
def reject(*args, **kwargs):
    raise AssertionError('Unexpected network call')
socket.socket.connect = reject
socket.create_connection = reject
from typer.main import get_command
from typer.testing import CliRunner
from complexity.cli.app import app
from complexity.cli.core import ComplexityCLI
from complexity.data import ComplexityTemplate, ComplexityTokens, Message
from complexity.tokenizer import Tokenizer
runner = CliRunner()
paths = [[]]
def walk(command, prefix):
    for name, child in getattr(command, 'commands', {}).items():
        path = prefix + [name]
        paths.append(path)
        walk(child, path)
walk(get_command(app), [])
for path in paths:
    result = runner.invoke(app, path + ['--help'])
    assert result.exit_code == 0, (path, result.output, result.exception)
for args in (['--version'], ['info', 'version'], ['info', 'tokens'],
             ['tokenize', 'special-tokens'], ['vision', 'list']):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (args, result.output, result.exception)
tokens = ComplexityTokens()
assert tokens.num_reserved == 2048
assert tokens.get_token_map()['<|call|>'] == 67
assert tokens.get_token_map()['<|begin|>'] == 1
cli = ComplexityCLI()
assert '<|user|>' in cli.format_tokens('bonjour')
assert cli.format_tokens('bonjour', format='chatml') == (
    '<|turn|>user\\nbonjour<|/turn|>\\n<|turn|>assistant\\n'
)
print('all-help-light', len(paths))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("all-help-light ")


def test_data_public_exports_keep_their_implementation_identity():
    import complexity.data as data

    for name, implementation in data._EXPORTS.items():
        module = importlib.import_module(implementation, data.__name__)
        assert getattr(data, name) is getattr(module, name)
    assert set(data.__all__) == set(data._EXPORTS)


def _local_tokenizer(path, *, postprocessor=False):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    path.mkdir()
    tokenizer = Tokenizer(
        WordLevel({"</s>": 0, "<pad>": 1, "<unk>": 2, "hello": 3, "world": 4}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["</s>", "<pad>", "<unk>"])
    if postprocessor:
        tokenizer.post_processor = TemplateProcessing(
            single="$A </s>", special_tokens=[("</s>", 0)]
        )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "format": "complexity",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
        )
    )
    return tokenizer


@pytest.mark.parametrize("postprocessor", [False, True])
def test_native_directory_encode_decode_count_and_preprocess_keep_zero_eos(tmp_path, postprocessor):
    tokenizer = tmp_path / "tokenizer"
    raw = _local_tokenizer(tokenizer, postprocessor=postprocessor)
    source = tmp_path / "input.txt"
    source.write_text("hello world")
    expected = raw.encode("hello world").ids
    encoded = RUNNER.invoke(app, ["tokenize", "encode", "hello world", "--tokenizer", str(tokenizer)])
    assert encoded.exit_code == 0, encoded.output
    assert f"Tokens: {expected}" in encoded.output
    decoded = RUNNER.invoke(app, ["tokenize", "decode", "3,4", "--tokenizer", str(tokenizer)])
    assert decoded.exit_code == 0, decoded.output
    assert "hello world" in decoded.output
    counted = RUNNER.invoke(app, ["tokenize", "count", str(source), "--tokenizer", str(tokenizer)])
    assert counted.exit_code == 0, counted.output
    assert f"Total tokens: {len(expected)}" in counted.output
    output = tmp_path / "data.bin"
    prepared = RUNNER.invoke(
        app,
        ["tokenize", "preprocess", str(source), str(output), "--format", "txt",
         "--tokenizer", str(tokenizer), "--seq-length", "1"],
    )
    assert prepared.exit_code == 0, prepared.output
    assert np.fromfile(output, dtype=np.uint32).tolist() == [3, 4, 0]


def test_hf_alias_honors_the_local_model_argument(tmp_path):
    tokenizer = tmp_path / "tokenizer"
    _local_tokenizer(tokenizer)
    result = RUNNER.invoke(
        app, ["tokenize", "encode", "hello world", "--tokenizer", "hf", "--model", str(tokenizer)]
    )
    assert result.exit_code == 0, result.output
    assert "Tokens: [3, 4]" in result.output


def test_tiktoken_model_argument_and_encoding_override_are_honored(monkeypatch):
    import tiktoken

    from complexity.data import get_tokenizer

    # Real byte-level encoding, kept entirely in memory for this dispatch test.
    local = tiktoken.Encoding(
        name="local-test", pat_str=r"[\s\S]", mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )
    calls = []
    monkeypatch.setattr(tiktoken, "list_encoding_names", lambda: ["local-test"])
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda name: calls.append(name) or local)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: calls.append(name) or local)
    assert get_tokenizer("tiktoken", model_name="example-model").encode("hi") == [104, 105]
    assert calls == ["example-model", "local-test"]
    calls.clear()
    assert get_tokenizer("tiktoken", model_name="ignored", encoding_name="local-test").eos_token_id == 256
    assert calls == ["local-test"]


def test_tokenizer_decode_accepts_lists_numpy_and_torch_tensors(tmp_path):
    import torch

    from complexity.tokenizer import Tokenizer

    path = tmp_path / "tokenizer"
    _local_tokenizer(path)
    tokenizer = Tokenizer.load(str(path))
    for ids in ([3, 4], np.array([3, 4]), torch.tensor([3, 4])):
        assert tokenizer.decode(ids) == "hello world"


def test_extending_legacy_markers_uses_existing_token_spellings(tmp_path):
    from tokenizers import Tokenizer

    from complexity.data import ComplexityTokens

    path = tmp_path / "tokenizer"
    _local_tokenizer(path)
    output = tmp_path / "extended.json"
    result = RUNNER.invoke(
        app, ["tokenize", "extend", str(path), str(output), "--complexity-tokens"]
    )
    assert result.exit_code == 0, result.output
    vocabulary = Tokenizer.from_file(str(output)).get_vocab()
    tokens = ComplexityTokens()
    assert tokens.tool_call in vocabulary
    assert tokens.tool_args in vocabulary
    assert tokens.state_start in vocabulary
    assert "<|think_start|>" not in vocabulary


def test_native_directory_inspection_and_tokenizer_benchmark(tmp_path):
    path = tmp_path / "tokenizer"
    _local_tokenizer(path)
    inspected = RUNNER.invoke(app, ["tokenize", "info", str(path)])
    assert inspected.exit_code == 0, inspected.output
    assert "Vocab size: 5" in inspected.output
    measured = RUNNER.invoke(app, ["tokenize", "benchmark", str(path), "--iterations", "1"])
    assert measured.exit_code == 0, measured.output
    assert "Benchmark" in measured.output


def test_sentencepiece_training_and_model_dispatch_are_local(tmp_path):
    pytest.importorskip("sentencepiece")
    source = tmp_path / "corpus.txt"
    source.write_text(("one two three four five six seven eight nine ten hello world alphabet xyz\n") * 10)
    prefix = tmp_path / "sentencepiece"
    trained = RUNNER.invoke(
        app,
        ["tokenize", "train-sentencepiece", str(source), str(prefix), "--vocab-size", "64",
         "--coverage", "1", "--threads", "1"],
    )
    assert trained.exit_code == 0, trained.output
    model = str(prefix.with_suffix(".model"))
    encoded = RUNNER.invoke(
        app, ["tokenize", "encode", "hello world", "--tokenizer", "sentencepiece", "--model", model]
    )
    assert encoded.exit_code == 0, encoded.output
    from complexity.data import get_tokenizer

    tokenizer = get_tokenizer("sentencepiece", model_name=model)
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
    assert tokenizer.eos_token_id == 2


def test_legacy_native_generation_fails_before_history_or_model_mutation():
    from complexity.cli.core import ComplexityCLI

    cli = ComplexityCLI()
    cli.set_system("original")
    cli.history.add_user("prior")
    before = cli.history.to_list()
    with pytest.raises(RuntimeError, match="create_external_backend"):
        cli.chat("next", system="replacement")
    assert cli.history.to_list() == before
    with pytest.raises(RuntimeError, match="complexity inference generate"):
        cli.generate("hello")
    assert cli.state.model is None


def test_core_model_loader_keeps_false_result_when_the_backend_is_absent():
    code = """
import importlib.abc
import sys
class WithoutTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('PyTorch backend not installed')
sys.meta_path.insert(0, WithoutTorch())
from complexity.cli.core import ComplexityCLI
cli = ComplexityCLI()
assert cli.load_model() is False
assert cli.state.model is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
