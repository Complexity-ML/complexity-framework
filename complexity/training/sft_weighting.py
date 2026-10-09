"""Optional curriculum loss weights over unchanged, validated SFT v2 packs.

Consumers must pass loss_weight explicitly to the weighted SFT loss API.
This adapter neither changes the native 200M runner nor claims that runner
uses a curriculum, drops examples or rewrites token/label files.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch.utils.data import Dataset

from .curriculum_planning import load_example_index
from .sft_curriculum import audit_stage_loss_targets, load_curriculum, loss_weight_key


class WeightedSFTTokenShardDataset(Dataset):
    """Compose task coefficients with a complete no-truncation shard reader."""

    packing_contract = "supervised-finetuning"

    def __init__(self, dataset, task_weights: Mapping[str, float], *, stage=None):
        if not isinstance(task_weights, Mapping) or not task_weights:
            raise ValueError("SFT task weights must be a nonempty mapping")
        if any(
            not isinstance(key, str)
            or not key
            or type(value) not in {int, float}
            or not math.isfinite(value)
            or value <= 0
            for key, value in task_weights.items()
        ):
            raise ValueError("SFT task weights must be finite and positive")
        self.dataset = dataset
        self.examples = load_example_index(dataset.root)
        if len(self.examples) != len(dataset.examples):
            raise ValueError("Loss metadata does not cover the validated shard")
        for original, example in zip(dataset.examples, self.examples):
            if any(value != example.get(key) for key, value in original.items()):
                raise ValueError("Loss metadata changes the validated example index")
        self._keys = [
            loss_weight_key(stage, row) if stage is not None else row.get("task")
            for row in self.examples
        ]
        if set(self._keys) != set(task_weights):
            raise ValueError("SFT task weights must exactly cover the selected loss keys")
        self.task_loss_weights = {key: float(value) for key, value in task_weights.items()}
        self.loss_target_audit = None

    @classmethod
    def from_curriculum(cls, dataset, config, stage_name):
        """Weight the entire shard; reject a stage that selects/drops rows."""
        curriculum = load_curriculum(config)
        examples = load_example_index(dataset.root)
        audit = audit_stage_loss_targets(examples, curriculum, stage_name)
        if not audit["passed"]:
            raise ValueError("Weighted SFT requires a passing full-shard loss-target audit")
        result = cls(dataset, audit["task_loss_weights"], stage=curriculum.stage(stage_name))
        result.loss_target_audit = audit
        return result

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        weights = torch.zeros_like(sample["labels"], dtype=torch.float32)
        offset = 0
        for position, index in enumerate(self.dataset.packing_plan.packs[item]):
            if position:
                offset += self.dataset.packing_plan.separator_tokens
            length = self.examples[index]["num_tokens"]
            weights[offset : offset + length] = self.task_loss_weights[self._keys[index]]
            offset += length
        if offset > len(weights):
            raise ValueError("SFT loss weights exceed the validated packing window")
        return {**sample, "loss_weight": weights}
