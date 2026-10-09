"""Offline contracts for generative scoring, traces, provenance and serving."""

import hashlib
import json
import subprocess
import sys
from types import ModuleType

import pytest

from complexity.evaluation import arc, arc_samples, gsm8k
from complexity.inference.chat_template import default_chat_template
from complexity.inference.external import ExternalGenerationConfig


def test_prompt_helpers_and_cli_help_load_without_ml_runtimes():
    program = """
import importlib.abc
import runpy
import sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'datasets', 'pyarrow'}:
            raise RuntimeError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockML())
from complexity.evaluation import arc, gsm8k
from complexity.inference import create_external_backend
from complexity.inference.chat_template import default_chat_template
from complexity.export.routed import build_mlx_config
gsm8k.build_gsm8k_prompt('Target?', default_chat_template(), num_fewshot=0)
assert gsm8k.extract_answer('Final answer: 4') == 4
assert build_mlx_config({'mlp_type': 'tr_hash_engine'})['mlp_type'] == 'token_routed'
sys.argv = ['arc', '--help']
runpy.run_module('complexity.evaluation.arc', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "evaluate" in result.stdout
    assert "merge" in result.stdout


def test_serving_adapter_preserves_config_and_renders_explicit_template():
    class Client:
        def complete(self, prompt, config):
            self.prompt, self.config = prompt, config
            return "Final answer: B"

    client = Client()
    config = ExternalGenerationConfig(temperature=0.0, extra={"repetition_penalty": 1.05})
    template = default_chat_template()
    generator = arc.ServingGenerator(client, config, chat_template=template)

    assert generator.generate("Question?", seed=17, allowed=("A", "B")) == "Final answer: B"
    assert "Question?" in client.prompt
    assert client.prompt.endswith(template["assistant_prefix"])
    assert client.config.extra == {"seed": 17, "repetition_penalty": 1.05}
    assert config.extra == {"repetition_penalty": 1.05}


def test_arc_evaluation_writes_all_completions_and_integrity(tmp_path):
    examples = [arc.ARCExample("arc_easy", 4, "one", "What is frozen water?", ("steam", "ice"), 1)]
    calls = []

    def generate(prompt, *, seed, allowed):
        calls.append((prompt, seed, allowed))
        return r"The correct answer is \boxed{B}."

    output = tmp_path / "report.json"
    report = arc.evaluate(
        examples, generate, seed=7, output=output, provenance={"model": "fixture"}
    )

    traces = output.with_suffix(".jsonl")
    row = json.loads(traces.read_text())
    assert calls[0][1:] == (7, ("A", "B"))
    assert row["completion"] == r"The correct answer is \boxed{B}."
    assert report["combined"]["native_accuracy"] == 1.0
    assert report["provenance"]["model"] == "fixture"
    assert report["traces_sha256"] == hashlib.sha256(traces.read_bytes()).hexdigest()
    assert json.loads(output.read_text()) == report


def test_arc_cli_evaluation_uses_local_samples_and_supplied_server_adapter(
    monkeypatch, tmp_path, capsys
):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(
            {
                "doc_id": 0,
                "doc": {
                    "id": "one",
                    "question": "What is correct?",
                    "choices": {"label": ["1", "2"], "text": ["one", "two"]},
                    "answerKey": "2",
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        arc, "serving_generator", lambda **kwargs: lambda prompt, **options: "Final answer: B"
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arc",
            "evaluate",
            str(samples),
            "--task",
            "arc_easy",
            "--base-url",
            "http://fixture.invalid",
            "--model",
            "fixture",
            "--output",
            str(output),
        ],
    )

    arc.main()

    report = json.loads(capsys.readouterr().out)
    assert report["complete"] is True
    assert report["combined"]["strict_accuracy"] == 1.0
    assert (
        report["provenance"]["samples_sha256"] == hashlib.sha256(samples.read_bytes()).hexdigest()
    )


def test_gsm8k_final_envelope_numeric_grading_and_retained_traces(tmp_path):
    examples = [{"question": "Target?", "answer": "Work\n#### 4"}]

    def generator(prompt, **kwargs):
        return "The answer is 12.<|think_end|><|final_start|>4<|final_end|>"

    output = tmp_path / "gsm8k.json"

    report = gsm8k.evaluate(
        examples, generator, default_chat_template(), num_fewshot=0, output=output
    )

    assert report["metrics"]["accuracy"] == 1.0
    assert report["num_fewshot"] == 0
    assert json.loads(output.with_suffix(".jsonl").read_text())["predicted"] == "4"
    assert gsm8k.extract_answer("The answer is 12. Final answer: 15") == 15


def test_online_gsm8k_loader_actually_pins_declared_revision(monkeypatch):
    module = ModuleType("datasets")
    calls = []
    module.load_dataset = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or [{"question": "q", "answer": "#### 1"}]
    )
    monkeypatch.setitem(sys.modules, "datasets", module)

    assert len(gsm8k.load_examples(None, None, split="test")) == 1
    assert calls == [
        (("openai/gsm8k", "main"), {"split": "test", "revision": gsm8k.DATASET_REVISION})
    ]


def test_arc_materialization_refuses_overwriting_prior_provenance(tmp_path, monkeypatch):
    marker = tmp_path / "manifest.json"
    marker.write_text("original")
    monkeypatch.setattr(
        arc_samples, "load_dataset", lambda *args, **kwargs: pytest.fail("must fail before loading")
    )

    with pytest.raises(FileExistsError):
        arc_samples.materialize(tmp_path)
    assert marker.read_text() == "original"


def test_merge_rejects_different_decoding_identity(tmp_path):
    shards = []
    for index in range(2):
        report = tmp_path / f"shard_{index}.json"
        report.write_text(
            json.dumps(
                {
                    "backend": "tr_hash_torch",
                    "decoding": {"temperature": index},
                    "elapsed_seconds": 1,
                }
            )
        )
        report.with_suffix(".jsonl").write_text(
            json.dumps(
                {
                    "task": "arc_easy",
                    "doc_id": index,
                    "strict_prediction": "A",
                    "flexible_prediction": "A",
                    "native_prediction": "A",
                    "strict_correct": True,
                    "flexible_correct": True,
                    "native_correct": True,
                }
            )
            + "\n"
        )
        shards.append(report)

    with pytest.raises(ValueError, match="provenance"):
        arc.merge(shards, tmp_path / "merged.json", 2)
    assert not (tmp_path / "merged.json").exists()
