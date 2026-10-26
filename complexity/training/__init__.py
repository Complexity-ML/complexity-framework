"""Training interfaces loaded only when requested.

    from complexity.training import Trainer, TrainingConfig
    config = TrainingConfig(max_steps=100, precision="fp32")
    # trainer = Trainer(model, config, train_dataloader)
    # trainer.train()

Importing a plan, configuration or job-management helper does not initialize the
model runtime, optional loggers or corpus readers. Existing public imports and
implementation paths are preserved.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "Trainer": (".trainer", "Trainer"),
    "TrainingConfig": (".config", "TrainingConfig"),
    "MetricsTracker": (".metrics", "MetricsTracker"),
    "FULL_PARAMETER_FINETUNING_PIPELINES": (".finetuning", "FULL_PARAMETER_FINETUNING_PIPELINES"),
    "PRETRAINING_STAGE": (".finetuning", "PRETRAINING_STAGE"),
    "REFINEMENT_STAGE": (".finetuning", "REFINEMENT_STAGE"),
    "SUPERVISED_FINETUNING_STAGE": (".finetuning", "SUPERVISED_FINETUNING_STAGE"),
    "TEXT_MODEL_FAMILY": (".finetuning", "TEXT_MODEL_FAMILY"),
    "VISION_MODEL_FAMILY": (".finetuning", "VISION_MODEL_FAMILY"),
    "IMAGE_EDITING_MODEL_FAMILY": (".finetuning", "IMAGE_EDITING_MODEL_FAMILY"),
    "IMAGE_GENERATION_MODEL_FAMILY": (".finetuning", "IMAGE_GENERATION_MODEL_FAMILY"),
    "IMAGE_TEXT_TO_TEXT_MODEL_FAMILY": (".finetuning", "IMAGE_TEXT_TO_TEXT_MODEL_FAMILY"),
    "INTEGRATED_REFINEMENT_MODEL_FAMILIES": (".finetuning", "INTEGRATED_REFINEMENT_MODEL_FAMILIES"),
    "IMAGE_EDITING_SUPERVISED_FINETUNING": (".finetuning", "IMAGE_EDITING_SUPERVISED_FINETUNING"),
    "TEXT_CONTINUED_PRETRAINING": (".finetuning", "TEXT_CONTINUED_PRETRAINING"),
    "TEXT_REFINEMENT": (".finetuning", "TEXT_REFINEMENT"),
    "TEXT_SUPERVISED_FINETUNING": (".finetuning", "TEXT_SUPERVISED_FINETUNING"),
    "VISION_SUPERVISED_FINETUNING": (".finetuning", "VISION_SUPERVISED_FINETUNING"),
    "refinement_corpus_fingerprint": (".finetuning", "refinement_corpus_fingerprint"),
    "validate_full_parameter_finetuning": (".finetuning", "validate_full_parameter_finetuning"),
    "validate_refinement_plan": (".finetuning", "validate_refinement_plan"),
    "validate_training_stage_transition": (".finetuning", "validate_training_stage_transition"),
    "get_lr_scheduler": (".scheduler", "get_lr_scheduler"),
    "resolve_scheduler_name": (".scheduler", "resolve_scheduler_name"),
    "EarlyStoppingCallback": (".callbacks", "EarlyStoppingCallback"),
    "WandBCallback": (".callbacks", "WandBCallback"),
    "TensorBoardCallback": (".callbacks", "TensorBoardCallback"),
    "TqdmCallback": (".callbacks", "TqdmCallback"),
    "global_expert_shares": (".moe_telemetry", "global_expert_shares"),
    "global_tr_diagnostics": (".moe_telemetry", "global_tr_diagnostics"),
    "detect_num_experts": (".moe_telemetry", "detect_num_experts"),
    "TrainRunner": (".runner", "TrainRunner"),
    "FineWebStreamingDataset": (".runner", "FineWebStreamingDataset"),
    "resolve_warmup_steps": (".runner", "resolve_warmup_steps"),
    "TextCorpusSource": (".corpus_mixture", "TextCorpusSource"),
    "PretokenizedCorpusMixtureDataset": (".corpus_mixture", "PretokenizedCorpusMixtureDataset"),
    "WeightedStreamingTextDataset": (".corpus_mixture", "WeightedStreamingTextDataset"),
    "allocate_weighted_counts": (".corpus_mixture", "allocate_weighted_counts"),
    "validate_corpus_mixture": (".corpus_mixture", "validate_corpus_mixture"),
    "TokenShardWriter": (".corpus_preparation", "TokenShardWriter"),
    "prepare_corpus_mixture": (".corpus_preparation", "prepare_corpus_mixture"),
    "build_replay_plan": (".replay", "build_replay_plan"),
    "build_corrective_replay_plan": (".replay", "build_corrective_replay_plan"),
    "SFTTokenShardDataset": (".sft", "SFTTokenShardDataset"),
    "build_sft_plan": (".sft", "build_sft_plan"),
    "execute_sft_config": (".sft", "execute_sft_config"),
    "prepare_sft_dataset": (".sft_preparation", "prepare_sft_dataset"),
    "compile_sft_dataset": (".sft_preparation", "compile_sft_dataset"),
    "encode_agentic_trajectory": (".sft_preparation", "encode_agentic_trajectory"),
    "load_sft_recipe": (".sft_preparation", "load_sft_recipe"),
    "PackedEpochSchedule": (".packing", "PackedEpochSchedule"),
    "PackingKind": (".packing", "PackingKind"),
    "PipelinePackingContract": (".packing", "PipelinePackingContract"),
    "FRAMEWORK_PACKING_CONTRACTS": (".packing", "FRAMEWORK_PACKING_CONTRACTS"),
    "TOKEN_PACK_PIPELINES": (".packing", "TOKEN_PACK_PIPELINES"),
    "resolve_packed_epoch_schedule": (".packing", "resolve_packed_epoch_schedule"),
    "TokenPackSchedule": (".packing", "TokenPackSchedule"),
    "resolve_token_pack_schedule": (".packing", "resolve_token_pack_schedule"),
    "validate_framework_packing_contracts": (".packing", "validate_framework_packing_contracts"),
    "validate_token_pack_pipeline": (".packing", "validate_token_pack_pipeline"),
    "EpochSchedule": (".sequence_packing", "EpochSchedule"),
    "SequencePackingPlan": (".sequence_packing", "SequencePackingPlan"),
    "pack_example_lengths": (".sequence_packing", "pack_example_lengths"),
    "resolve_epoch_schedule": (".sequence_packing", "resolve_epoch_schedule"),
    "SupervisorManager": (".supervisor", "SupervisorManager"),
    "SupervisorProgram": (".supervisor", "SupervisorProgram"),
}

__all__ = [
    "Trainer",
    "TrainingConfig",
    "MetricsTracker",
    "FULL_PARAMETER_FINETUNING_PIPELINES",
    "PRETRAINING_STAGE",
    "REFINEMENT_STAGE",
    "SUPERVISED_FINETUNING_STAGE",
    "TEXT_MODEL_FAMILY",
    "VISION_MODEL_FAMILY",
    "IMAGE_EDITING_MODEL_FAMILY",
    "IMAGE_GENERATION_MODEL_FAMILY",
    "IMAGE_TEXT_TO_TEXT_MODEL_FAMILY",
    "INTEGRATED_REFINEMENT_MODEL_FAMILIES",
    "IMAGE_EDITING_SUPERVISED_FINETUNING",
    "TEXT_CONTINUED_PRETRAINING",
    "TEXT_REFINEMENT",
    "TEXT_SUPERVISED_FINETUNING",
    "VISION_SUPERVISED_FINETUNING",
    "refinement_corpus_fingerprint",
    "validate_full_parameter_finetuning",
    "validate_refinement_plan",
    "validate_training_stage_transition",
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
    "TokenShardWriter",
    "prepare_corpus_mixture",
    "build_replay_plan",
    "build_corrective_replay_plan",
    "SFTTokenShardDataset",
    "build_sft_plan",
    "execute_sft_config",
    "prepare_sft_dataset",
    "compile_sft_dataset",
    "encode_agentic_trajectory",
    "load_sft_recipe",
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
    "SupervisorManager",
    "SupervisorProgram",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(_import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
