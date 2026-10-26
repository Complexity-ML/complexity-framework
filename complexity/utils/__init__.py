"""
Utilities module for framework-complexity.

Provides:
- Checkpointing: Save/load model and training state
- HuggingFace bundles: Preserve the complete canonical TR-HASH architecture
- Security: Safe checkpoint loading, audit logging
- Safety: Representation Engineering for safe inference

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

    model = load_from_huggingface("my_tr_hash_bundle", model)
    save_to_huggingface(model, "my_model_hf")

    # Security
    from complexity.utils import safe_torch_load, SecureTrainingContext

    with SecureTrainingContext() as ctx:
        state = ctx.load_checkpoint("model.pt")

    # Safety (Representation Engineering)
    from complexity.utils import SafetyClamp, install_safety

    harm_dir = torch.load("harm_direction.pt")
    install_safety(model, harm_dir, threshold=2.0, layers=[-2, -1])
"""

from .checkpointing import (
    CheckpointManager,
    TrainingState,
    checkpoint_sequential,
    enable_activation_checkpointing,
)
from .device import (
    BackendInfo,
    backend_metadata,
    configure_torch_acceleration,
    custom_kernels_enabled,
    get_backend,
    get_backend_info,
    is_nvidia_cuda_available,
    is_rocm_runtime_present,
    log_backend,
    rocm_unavailable_message,
    sdpa_kernel_backends,
    sdpa_kernel_context,
    supports_custom_triton,
)
from .hf_conversion import (
    convert_hf_state_dict,
    convert_to_hf_state_dict,
    create_hf_config,
    from_pretrained,
    load_from_huggingface,
    push_to_hub,
    save_to_huggingface,
)
from .mps import (
    MPSMemoryStats,
    autocast,
    autocast_dtype,
    empty_cache,
    enable_cpu_fallback,
    is_cuda_available,
    is_mps_available,
    is_rocm_available,
    mps_memory_stats,
    seed_all,
    select_device,
    set_memory_watermark,
    setup_mps,
    synchronize,
)
from .safety import (
    ContrastiveSafetyLoss,
    MultiDirectionSafetyClamp,
    SafetyCallback,
    SafetyClamp,
    SafetyConfig,
    get_safety_stats,
    install_safety,
    load_harm_direction,
    remove_safety,
    save_harm_direction,
)
from .security import (
    AuditLogger,
    ModelManifest,
    SecureTrainingContext,
    UnsafeCheckpointError,
    compute_model_hash,
    create_model_manifest,
    load_safetensors,
    safe_torch_load,
    sanitize_config,
    save_safetensors,
    validate_distributed_env,
    validate_input_ids,
    verify_model_hash,
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
    "create_hf_config",
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
    # Safety
    "SafetyConfig",
    "SafetyClamp",
    "MultiDirectionSafetyClamp",
    "ContrastiveSafetyLoss",
    "install_safety",
    "remove_safety",
    "get_safety_stats",
    "SafetyCallback",
    "load_harm_direction",
    "save_harm_direction",
    # MPS / device
    "is_mps_available",
    "is_rocm_available",
    "is_cuda_available",
    "is_nvidia_cuda_available",
    "is_rocm_runtime_present",
    "rocm_unavailable_message",
    "get_backend",
    "get_backend_info",
    "log_backend",
    "custom_kernels_enabled",
    "configure_torch_acceleration",
    "supports_custom_triton",
    "sdpa_kernel_backends",
    "sdpa_kernel_context",
    "BackendInfo",
    "backend_metadata",
    "select_device",
    "set_memory_watermark",
    "enable_cpu_fallback",
    "mps_memory_stats",
    "empty_cache",
    "synchronize",
    "seed_all",
    "autocast",
    "autocast_dtype",
    "setup_mps",
    "MPSMemoryStats",
]
