from pathlib import Path

from complexity.training.stack_sft import DatasetMix, StackSFTConfig, StackSFTDatasetBuilder, StackSFTRunner


def test_stack_sft_config_loads_recipe():
    config = StackSFTConfig.from_yaml("complexity/training/stack_sft/recipes/50m_agentic.yaml")
    assert config.name == "sft-50m-plus5b-agentic-stack"
    assert len(config.stages) == 4
    assert config.stages[1].name == "finalization"
    assert config.stages[1].mix.sources["final"] == 0.70


def test_stack_sft_dataset_builder(tmp_path: Path):
    mix = DatasetMix(
        records=20,
        seed=1,
        sources={
            "chat": 0.4,
            "calculator": 0.3,
            "final": 0.2,
            "reflect": 0.1,
        },
    )
    out = StackSFTDatasetBuilder().build(mix, tmp_path / "stage.jsonl")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    joined = "\n".join(lines)
    assert "Assistant:" in joined
    assert "tool_call" in joined
    assert "Tool result from calculator" in joined


def test_stack_sft_runner_dry_command(tmp_path: Path):
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
                    "mix": {"records": 8, "sources": {"final": 1.0}},
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
