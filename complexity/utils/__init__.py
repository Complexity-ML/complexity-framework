"""
Utilities module for framework-complexity.

Provides:
- Checkpointing: Save/load model and training state
- HuggingFace conversion: Load from / save to HF format
- Security: Safe checkpoint loading, audit logging

Usage:
    # Checkpointing
    from complexity.utils import CheckpointManager, TrainingState

    manager = CheckpointManager(
        checkpoint_dir="checkpoints",
        model=model,
        optimizer=optimizer,
    )
    manager.save(step=1000)
    manager.load_latest()

    # HuggingFace conversion
    from complexity.utils import load_from_huggingface, save_to_huggingface

    model = load_from_huggingface("meta-llama/Llama-2-7b-hf", model)
    save_to_huggingface(model, "my_model_hf")

    # Security
    from complexity.utils import safe_torch_load, SecureTrainingContext

    with SecureTrainingContext() as ctx:
        state = ctx.load_checkpoint("model.pt")
"""

from .checkpointing import (
    CheckpointManager,
    TrainingState,
    enable_activation_checkpointing,
    checkpoint_sequential,
)

from .hf_conversion import (
    load_from_huggingface,
    save_to_huggingface,
    push_to_hub,
    from_pretrained,
    convert_hf_state_dict,
    convert_to_hf_state_dict,
)

from .security import (
    safe_torch_load,
    load_safetensors,
    save_safetensors,
    compute_model_hash,
    verify_model_hash,
    ModelManifest,
    create_model_manifest,
    validate_distributed_env,
    validate_input_ids,
    sanitize_config,
    AuditLogger,
    SecureTrainingContext,
    UnsafeCheckpointError,
)

__all__ = [
    # Checkpointing
    "CheckpointManager",
    "TrainingState",
    "enable_activation_checkpointing",
    "checkpoint_sequential",
    # HuggingFace
    "load_from_huggingface",
    "save_to_huggingface",
    "push_to_hub",
    "from_pretrained",
    "convert_hf_state_dict",
    "convert_to_hf_state_dict",
    # Security
    "safe_torch_load",
    "load_safetensors",
    "save_safetensors",
    "compute_model_hash",
    "verify_model_hash",
    "ModelManifest",
    "create_model_manifest",
    "validate_distributed_env",
    "validate_input_ids",
    "sanitize_config",
    "AuditLogger",
    "SecureTrainingContext",
    "UnsafeCheckpointError",
]
