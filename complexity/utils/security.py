"""
Security utilities for framework-complexity.

Provides security features for training:
- Safe checkpoint loading (pickle security)
- Input validation
- Secure distributed communication
- Audit logging
- Model integrity verification

IMPORTANT: Training LLMs involves risks:
- Malicious checkpoints can execute arbitrary code
- Distributed training requires secure communication
- Data leakage between processes must be prevented
"""

import torch
import torch.nn as nn
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# Safe Checkpoint Loading
# =============================================================================

class UnsafeCheckpointError(Exception):
    """Raised when a checkpoint fails security validation."""
    pass


def safe_torch_load(
    path: Union[str, Path],
    map_location: str = "cpu",
    weights_only: bool = True,
    allowed_classes: Optional[List[type]] = None,
) -> Dict[str, Any]:
    """
    Safely load a PyTorch checkpoint.

    By default, uses weights_only=True to prevent arbitrary code execution
    from malicious pickle files.

    Args:
        path: Path to checkpoint
        map_location: Device to load to
        weights_only: If True, only load tensor data (RECOMMENDED)
        allowed_classes: Additional classes to allow if weights_only=False

    Returns:
        Loaded state dict

    Raises:
        UnsafeCheckpointError: If checkpoint fails validation
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Check file extension
    if path.suffix not in ['.pt', '.pth', '.bin', '.safetensors']:
        raise UnsafeCheckpointError(
            f"Suspicious file extension: {path.suffix}. "
            "Expected .pt, .pth, .bin, or .safetensors"
        )

    # Check file size (warn if very large)
    file_size_gb = path.stat().st_size / (1024 ** 3)
    if file_size_gb > 100:
        logger.warning(f"Large checkpoint: {file_size_gb:.1f}GB. Loading may take a while.")

    # Prefer safetensors format (no pickle, no code execution risk)
    if path.suffix == '.safetensors':
        return load_safetensors(path)

    # Load with weights_only for security
    try:
        if weights_only:
            # PyTorch 2.0+ supports weights_only
            return torch.load(path, map_location=map_location, weights_only=True)
        else:
            # Less secure - only use if you trust the source
            warnings.warn(
                "Loading checkpoint with weights_only=False. "
                "This is a security risk if the checkpoint is from an untrusted source."
            )
            return torch.load(path, map_location=map_location)
    except Exception as e:
        raise UnsafeCheckpointError(f"Failed to load checkpoint: {e}")


def load_safetensors(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load a safetensors checkpoint (most secure format).

    Safetensors is a safe serialization format that:
    - Contains only tensor data
    - Cannot execute arbitrary code
    - Is faster to load
    """
    try:
        from safetensors.torch import load_file
        return load_file(str(path))
    except ImportError:
        raise ImportError(
            "safetensors is required for .safetensors files. "
            "Install with: pip install safetensors"
        )


def save_safetensors(state_dict: Dict[str, torch.Tensor], path: Union[str, Path]):
    """Save a state dict in safetensors format (recommended)."""
    try:
        from safetensors.torch import save_file
        # Filter to only tensors
        tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        save_file(tensor_dict, str(path))
    except ImportError:
        raise ImportError(
            "safetensors is required. Install with: pip install safetensors"
        )


# =============================================================================
# Model Integrity Verification
# =============================================================================

def compute_model_hash(model: nn.Module, algorithm: str = "sha256") -> str:
    """
    Compute a hash of model weights for integrity verification.

    Args:
        model: The model to hash
        algorithm: Hash algorithm (sha256, sha512, md5)

    Returns:
        Hex digest of the hash
    """
    hasher = hashlib.new(algorithm)

    for name, param in sorted(model.named_parameters()):
        # Hash parameter name
        hasher.update(name.encode())
        # Hash parameter data
        hasher.update(param.data.cpu().numpy().tobytes())

    return hasher.hexdigest()


def verify_model_hash(model: nn.Module, expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify model weights against an expected hash.

    Args:
        model: The model to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if hash matches
    """
    actual_hash = compute_model_hash(model, algorithm)
    return hmac.compare_digest(actual_hash, expected_hash)


@dataclass
class ModelManifest:
    """Manifest containing model metadata and security info."""
    model_name: str
    version: str
    hash: str
    hash_algorithm: str
    created_at: str
    created_by: str
    parameters: int
    architecture: Dict[str, Any]
    training_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "hash": self.hash,
            "hash_algorithm": self.hash_algorithm,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "parameters": self.parameters,
            "architecture": self.architecture,
            "training_info": self.training_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        return cls(**data)

    def save(self, path: Union[str, Path]):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelManifest":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def create_model_manifest(
    model: nn.Module,
    model_name: str,
    version: str,
    config: Any,
    created_by: str = "INL",
) -> ModelManifest:
    """Create a manifest for a model."""
    return ModelManifest(
        model_name=model_name,
        version=version,
        hash=compute_model_hash(model),
        hash_algorithm="sha256",
        created_at=datetime.now().isoformat(),
        created_by=created_by,
        parameters=sum(p.numel() for p in model.parameters()),
        architecture=config.to_dict() if hasattr(config, 'to_dict') else {},
    )


# =============================================================================
# Secure Distributed Training
# =============================================================================

def validate_distributed_env() -> Dict[str, Any]:
    """
    Validate distributed training environment for security.

    Returns:
        Dictionary of environment info and warnings
    """
    import torch.distributed as dist

    info = {
        "distributed_initialized": dist.is_initialized(),
        "warnings": [],
        "recommendations": [],
    }

    if dist.is_initialized():
        info["backend"] = dist.get_backend()
        info["world_size"] = dist.get_world_size()
        info["rank"] = dist.get_rank()

        # Check for secure backend
        if info["backend"] == "gloo":
            info["warnings"].append(
                "Using 'gloo' backend. For GPU training, 'nccl' is recommended."
            )

        # Check environment variables
        master_addr = os.environ.get("MASTER_ADDR", "")
        if master_addr and not master_addr.startswith("127.") and master_addr != "localhost":
            info["warnings"].append(
                f"MASTER_ADDR is set to external address: {master_addr}. "
                "Ensure network is secured."
            )

    # Recommendations
    info["recommendations"] = [
        "Use NCCL backend for GPU training",
        "Use private network for distributed training",
        "Enable TLS for cross-node communication if available",
        "Limit MASTER_ADDR to internal IPs",
    ]

    return info


# =============================================================================
# Input Validation
# =============================================================================

def validate_input_ids(
    input_ids: torch.Tensor,
    vocab_size: int,
    max_seq_len: int,
) -> None:
    """
    Validate input token IDs.

    Args:
        input_ids: Input tensor
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length

    Raises:
        ValueError: If validation fails
    """
    if input_ids.dim() != 2:
        raise ValueError(f"Expected 2D input, got {input_ids.dim()}D")

    if input_ids.size(1) > max_seq_len:
        raise ValueError(
            f"Sequence length {input_ids.size(1)} exceeds maximum {max_seq_len}"
        )

    # Check for out-of-range token IDs
    if input_ids.min() < 0:
        raise ValueError(f"Negative token ID found: {input_ids.min()}")

    if input_ids.max() >= vocab_size:
        raise ValueError(
            f"Token ID {input_ids.max()} exceeds vocabulary size {vocab_size}"
        )


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a configuration dictionary.

    Removes potentially dangerous keys and validates values.
    """
    # Keys that should never be in a config file
    dangerous_keys = [
        "eval", "exec", "import", "__", "lambda",
        "os.system", "subprocess", "shell",
    ]

    sanitized = {}
    for key, value in config.items():
        # Check key name
        key_lower = key.lower()
        if any(danger in key_lower for danger in dangerous_keys):
            logger.warning(f"Removed suspicious config key: {key}")
            continue

        # Check string values
        if isinstance(value, str):
            value_lower = value.lower()
            if any(danger in value_lower for danger in dangerous_keys):
                logger.warning(f"Removed suspicious config value for key: {key}")
                continue

        sanitized[key] = value

    return sanitized


# =============================================================================
# Audit Logging
# =============================================================================

class AuditLogger:
    """
    Audit logger for training events.

    Logs security-relevant events for later review.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path) if log_path else Path("audit.log")
        self.events = []

    def log_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description,
            "metadata": metadata or {},
        }
        self.events.append(event)

        # Write to file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(event) + "\n")

        logger.info(f"AUDIT: [{event_type}] {description}")

    def log_checkpoint_load(self, path: str, success: bool, hash: Optional[str] = None):
        """Log checkpoint loading event."""
        self.log_event(
            "CHECKPOINT_LOAD",
            f"Loaded checkpoint from {path}",
            {"path": path, "success": success, "hash": hash},
        )

    def log_training_start(self, config: Dict[str, Any]):
        """Log training start event."""
        self.log_event(
            "TRAINING_START",
            "Training started",
            {"config": sanitize_config(config)},
        )

    def log_training_end(self, metrics: Dict[str, Any]):
        """Log training end event."""
        self.log_event(
            "TRAINING_END",
            "Training completed",
            {"metrics": metrics},
        )

    def log_distributed_init(self, info: Dict[str, Any]):
        """Log distributed initialization."""
        self.log_event(
            "DISTRIBUTED_INIT",
            "Distributed training initialized",
            info,
        )


# =============================================================================
# Secure Training Wrapper
# =============================================================================

class SecureTrainingContext:
    """
    Context manager for secure training.

    Usage:
        with SecureTrainingContext(audit_log="training_audit.log") as ctx:
            ctx.validate_checkpoint(checkpoint_path)
            model = ctx.load_model(checkpoint_path)
            # ... training code ...
    """

    def __init__(
        self,
        audit_log: Optional[str] = None,
        validate_inputs: bool = True,
        safe_loading: bool = True,
    ):
        self.audit = AuditLogger(audit_log)
        self.validate_inputs = validate_inputs
        self.safe_loading = safe_loading

    def __enter__(self):
        self.audit.log_event("CONTEXT_START", "Secure training context started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.audit.log_event(
                "CONTEXT_ERROR",
                f"Error: {exc_type.__name__}: {exc_val}",
            )
        self.audit.log_event("CONTEXT_END", "Secure training context ended")
        return False

    def validate_checkpoint(self, path: str) -> bool:
        """Validate a checkpoint before loading."""
        try:
            path = Path(path)

            # Check file exists and has valid extension
            if not path.exists():
                raise UnsafeCheckpointError(f"Checkpoint not found: {path}")

            if path.suffix not in ['.pt', '.pth', '.bin', '.safetensors']:
                raise UnsafeCheckpointError(f"Invalid extension: {path.suffix}")

            self.audit.log_event(
                "CHECKPOINT_VALIDATED",
                f"Checkpoint validated: {path}",
            )
            return True

        except Exception as e:
            self.audit.log_event(
                "CHECKPOINT_VALIDATION_FAILED",
                f"Validation failed: {e}",
            )
            raise

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Safely load a checkpoint."""
        self.validate_checkpoint(path)
        state_dict = safe_torch_load(path, weights_only=self.safe_loading)
        self.audit.log_checkpoint_load(path, True)
        return state_dict
