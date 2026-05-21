"""Runner for staged SFT curricula."""

from __future__ import annotations

import subprocess
import sys
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
        return cls(StackSFTConfig.from_yaml(path), dry_run=dry_run)

    def run(self, *, start_stage: str | None = None, stop_after: str | None = None) -> str:
        checkpoint = self.config.base_checkpoint
        active = start_stage is None
        for stage in self.config.stages:
            if stage.name == start_stage:
                active = True
            if not active:
                checkpoint = self._stage_checkpoint(stage)
                continue
            checkpoint = self.run_stage(stage, checkpoint)
            if stage.name == stop_after:
                break
        return checkpoint

    def run_stage(self, stage: StageConfig, input_checkpoint: str) -> str:
        checkpoint = stage.checkpoint or input_checkpoint
        dataset_path = self._stage_dataset_path(stage)
        self.dataset_builder.build(stage.mix, dataset_path)
        cmd = self._stage_command(stage, checkpoint, dataset_path)
        if self.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True)
        return self._stage_checkpoint(stage)

    def _stage_dataset_path(self, stage: StageConfig) -> Path:
        if stage.mix.out:
            return Path(stage.mix.out)
        return Path(self.config.data_dir) / self.config.name / f"{stage.name}.jsonl"

    def _stage_save_dir(self, stage: StageConfig) -> Path:
        return Path(self.config.output_dir) / f"{self.config.name}-{stage.name}"

    def _stage_checkpoint(self, stage: StageConfig) -> str:
        return str(self._stage_save_dir(stage) / f"step_{stage.steps:06d}")

    def _stage_command(self, stage: StageConfig, checkpoint: str, dataset_path: Path) -> list[str]:
        batch_size = stage.batch_size or self.config.batch_size
        seq_len = stage.seq_len or self.config.seq_len
        save_steps = stage.save_steps or self.config.save_steps
        save_total_limit = stage.save_total_limit or self.config.save_total_limit
        grad_ckpt = self.config.grad_ckpt if stage.grad_ckpt is None else stage.grad_ckpt
        loss_chunk_tokens = stage.loss_chunk_tokens or self.config.loss_chunk_tokens
        run_name = f"{self.config.name}-{stage.name}"

        cmd = [
            sys.executable,
            "-m",
            "scripts.sft_100m_o200k_tr_local",
            "--checkpoint",
            checkpoint,
            "--tokenizer",
            self.config.tokenizer,
            "--jsonl",
            str(dataset_path),
            "--steps",
            str(stage.steps),
            "--batch-size",
            str(batch_size),
            "--seq-len",
            str(seq_len),
            "--lr",
            str(stage.lr),
            "--use-custom-kernels",
            self.config.use_custom_kernels,
            "--loss-chunk-tokens",
            str(loss_chunk_tokens),
            "--save-steps",
            str(save_steps),
            "--save-total-limit",
            str(save_total_limit),
            "--save-dir",
            str(self._stage_save_dir(stage)),
            "--run-name",
            run_name,
        ]
        if self.config.bf16:
            cmd.append("--bf16")
        if grad_ckpt:
            cmd.append("--grad-ckpt")
        cmd.extend(stage.extra_args)
        return cmd
