from __future__ import annotations

import json
from pathlib import Path

import pytest

from complexity.training.stack_sft import (
    DatasetMix,
    SourceConfig,
    StackSFTConfig,
    StackSFTDatasetBuilder,
    StackSFTRunner,
)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_stack_sft_config_loads_recipe():
    config = StackSFTConfig.from_yaml("complexity/training/stack_sft/recipes/50m_agentic.yaml")
    assert config.name == "sft-50m-plus5b-agentic-stack"
    assert len(config.stages) == 6
    assert config.stages[2].name == "finalization"
    assert any(source.name == "coding_tool_traces" for source in config.stages[4].mix.sources)
    assert all(source.path for stage in config.stages for source in stage.mix.sources)


def test_stack_sft_dataset_builder_reads_file_backed_sources(tmp_path: Path):
    general = write_jsonl(
        tmp_path / "general.jsonl",
        [
            {"prompt": "User:\nHello\n\nAssistant:\n", "completion": "Hello!"},
            {"instruction": "Explain gravity", "output": "Gravity pulls masses together."},
        ],
    )
    code = write_jsonl(
        tmp_path / "code.jsonl",
        [
            {
                "instruction": "Write Python that adds two numbers.",
                "input": "Use a function.",
                "output": "def add(a, b):\n    return a + b",
            }
        ],
    )
    mix = DatasetMix(
        records=12,
        seed=1,
        sources=[
            SourceConfig(name="general", path=str(general), format="jsonl", weight=0.5),
            SourceConfig(name="code", path=str(code), format="jsonl", weight=0.5),
        ],
    )

    out = StackSFTDatasetBuilder().build(mix, tmp_path / "mixed.jsonl")

    lines = out.read_text(encoding="utf-8").splitlines()
    joined = "\n".join(lines)
    assert len(lines) == 12
    assert "Gravity pulls masses together." in joined
    assert "def add(a, b):" in joined


def test_stack_sft_dataset_builder_reads_messages_source(tmp_path: Path):
    messages = write_jsonl(
        tmp_path / "messages.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        ],
    )
    mix = DatasetMix(
        records=3,
        seed=2,
        sources=[SourceConfig(name="messages", path=str(messages), format="jsonl", weight=1.0)],
    )

    out = StackSFTDatasetBuilder().build(mix, tmp_path / "messages_out.jsonl")

    joined = out.read_text(encoding="utf-8")
    assert "User:\\nWhat is 2+2?" in joined
    assert '"completion": "4"' in joined


def test_stack_sft_rejects_hardcoded_source_mapping():
    with pytest.raises(ValueError, match="hardcoded synthetic"):
        DatasetMix.from_dict({"records": 8, "sources": {"final": 1.0}})


def test_stack_sft_rejects_source_without_path(tmp_path: Path):
    mix = DatasetMix(
        records=1,
        seed=1,
        sources=[SourceConfig(name="bad", kind="chat", weight=1.0)],
    )

    with pytest.raises(ValueError, match="must define a JSONL/parquet path"):
        StackSFTDatasetBuilder().build(mix, tmp_path / "bad.jsonl")


def test_stack_sft_runner_dry_command(tmp_path: Path):
    source = write_jsonl(
        tmp_path / "source.jsonl",
        [{"prompt": "User:\nHi\n\nAssistant:\n", "completion": "Hi!"}],
    )
    config = StackSFTConfig.from_dict(
        {
            "name": "toy-stack",
            "base_checkpoint": "checkpoints/base/step_000001",
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "checkpoints"),
            "stages": [
                {
                    "name": "finalization",
                    "steps": 3,
                    "lr": 1e-5,
                    "mix": {
                        "records": 8,
                        "sources": [
                            {
                                "name": "file",
                                "path": str(source),
                                "format": "jsonl",
                                "weight": 1.0,
                            }
                        ],
                    },
                }
            ],
        }
    )
    runner = StackSFTRunner(config, dry_run=True)
    stage = config.stages[0]
    cmd = runner._stage_command(stage, config.base_checkpoint, tmp_path / "data.jsonl")
    assert "scripts.sft_100m_o200k_tr_local" in cmd
    assert "--checkpoint" in cmd
    assert str(tmp_path / "checkpoints" / "toy-stack-finalization") in cmd
