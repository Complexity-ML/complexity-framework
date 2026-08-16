"""
Training module for framework-complexity.

Provides a complete training solution:
- Distributed training with FSDP
- Mixed precision (FP16, BF16)
- Gradient accumulation
- Checkpointing
- Learning rate scheduling
- Logging and metrics

Usage:
    from complexity.training import Trainer, TrainingConfig

    config = TrainingConfig(
        max_steps=100000,
        batch_size=32,
        learning_rate=1e-4,
        precision="bf16",
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
    )

    trainer.train()
"""

from .callbacks import EarlyStoppingCallback, TensorBoardCallback, TqdmCallback, WandBCallback
from .config import TrainingConfig
from .corpus_mixture import (
    PretokenizedCorpusMixtureDataset,
    TextCorpusSource,
    WeightedStreamingTextDataset,
    allocate_weighted_counts,
    validate_corpus_mixture,
)
from .finetuning import (
    FULL_PARAMETER_FINETUNING_PIPELINES,
    TEXT_SUPERVISED_FINETUNING,
    VISION_SUPERVISED_FINETUNING,
    validate_full_parameter_finetuning,
)
from .metrics import MetricsTracker
from .moe_telemetry import detect_num_experts, global_expert_shares, global_tr_diagnostics
from .packing import (
    FRAMEWORK_PACKING_CONTRACTS,
    TOKEN_PACK_PIPELINES,
    PackedEpochSchedule,
    PackingKind,
    PipelinePackingContract,
    TokenPackSchedule,
    resolve_packed_epoch_schedule,
    resolve_token_pack_schedule,
    validate_framework_packing_contracts,
    validate_token_pack_pipeline,
)
from .runner import FineWebStreamingDataset, TrainRunner, resolve_warmup_steps
from .scheduler import get_lr_scheduler, resolve_scheduler_name
from .sequence_packing import (
    EpochSchedule,
    SequencePackingPlan,
    pack_example_lengths,
    resolve_epoch_schedule,
)
from .trainer import Trainer

__all__ = [
    "Trainer",
    "TrainingConfig",
    "MetricsTracker",
    "FULL_PARAMETER_FINETUNING_PIPELINES",
    "TEXT_SUPERVISED_FINETUNING",
    "VISION_SUPERVISED_FINETUNING",
    "validate_full_parameter_finetuning",
    "get_lr_scheduler",
    "resolve_scheduler_name",
    "EarlyStoppingCallback",
    "WandBCallback",
    "TensorBoardCallback",
    "TqdmCallback",
    "global_expert_shares",
    "global_tr_diagnostics",
    "detect_num_experts",
    "TrainRunner",
    "FineWebStreamingDataset",
    "resolve_warmup_steps",
    "TextCorpusSource",
    "PretokenizedCorpusMixtureDataset",
    "WeightedStreamingTextDataset",
    "allocate_weighted_counts",
    "validate_corpus_mixture",
    "PackedEpochSchedule",
    "PackingKind",
    "PipelinePackingContract",
    "FRAMEWORK_PACKING_CONTRACTS",
    "TOKEN_PACK_PIPELINES",
    "resolve_packed_epoch_schedule",
    "TokenPackSchedule",
    "resolve_token_pack_schedule",
    "validate_framework_packing_contracts",
    "validate_token_pack_pipeline",
    "EpochSchedule",
    "SequencePackingPlan",
    "pack_example_lengths",
    "resolve_epoch_schedule",
]
