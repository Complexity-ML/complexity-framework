"""Runner for staged SFT curricula."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import StackSFTConfig, StageConfig
from .datasets import StackSFTDatasetBuilder


class StackSFTRunner:
    """Build mixed SFT datasets and chain SFT stages."""

    def __init__(self, config: StackSFTConfig, *, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.dataset_builder = StackSFTDatasetBuilder()

    @classmethod
    def from_yaml(cls, path: str | Path, *, dry_run: bool = False) -> "StackSFTRunner":
        return cls(StackSFTConfig.from_file(path), dry_run=dry_run)

    def run(self, *, start_stage: str | None = None, stop_after: str | None = None) -> str:
        names = [stage.name for stage in self.config.stages]
        if len(set(names)) != len(names):
            raise ValueError("Stack SFT stage names must be unique")
        for selected in (start_stage, stop_after):
            if selected is not None and selected not in names:
                raise ValueError(f"Unknown stack stage: {selected}")
        checkpoint = self.config.base_checkpoint
        active = start_stage is None
        for stage in self.config.stages:
            if stage.name == start_stage:
                active = True
            if not active:
                checkpoint = self._stage_checkpoint(stage)
                continue
            checkpoint = self.run_stage(
                stage,
                checkpoint,
                source_stage=self.config.source_stage
                if stage is self.config.stages[0] or stage.checkpoint
                else "supervised-finetuning",
            )
            if stage.name == stop_after:
                break
        return checkpoint

    def run_stage(
        self, stage: StageConfig, input_checkpoint: str, *, source_stage=None, resume=None
    ) -> str:
        from .stage import run_lora_stage, validate_stack_execution

        config = replace(self.config, source_stage=source_stage or self.config.source_stage)
        validate_stack_execution(config, stage)
        checkpoint = stage.checkpoint or input_checkpoint
        dataset_path = self._stage_dataset_path(stage)
        if self.dry_run:
            print(
                json.dumps(
                    {
                        "workflow": "staged_lora_sft",
                        "stage": stage.name,
                        "checkpoint": checkpoint,
                        "dataset": str(dataset_path),
                        "output": str(self._stage_save_dir(stage)),
                        "source_stage": config.source_stage,
                        "training_started": False,
                        "device": config.device,
                        "precision": config.precision,
                    },
                    sort_keys=True,
                )
            )
            return self._stage_checkpoint(stage)
        if not (Path(checkpoint) / "model.safetensors").is_file():
            raise FileNotFoundError(f"Stack SFT local checkpoint not found: {checkpoint}")
        output = self._stage_save_dir(stage)
        if resume is None and output.exists() and any(output.iterdir()):
            raise FileExistsError(
                "Stack stage output already exists; explicitly resume or use a fresh directory"
            )
        if resume is None:
            self.dataset_builder.build(stage.mix, dataset_path)
        elif not dataset_path.is_file():
            raise FileNotFoundError("Stack SFT resume requires the original mixed JSONL dataset")
        return run_lora_stage(
            config, stage, checkpoint, dataset_path, self._stage_save_dir(stage), resume=resume
        )

    def _stage_dataset_path(self, stage: StageConfig) -> Path:
        if stage.mix.out:
            return Path(stage.mix.out)
        return Path(self.config.data_dir) / self.config.name / f"{stage.name}.jsonl"

    def _stage_save_dir(self, stage: StageConfig) -> Path:
        return Path(self.config.output_dir) / f"{self.config.name}-{stage.name}"

    def _stage_checkpoint(self, stage: StageConfig) -> str:
        return str(self._stage_save_dir(stage) / f"step_{stage.steps:06d}")
