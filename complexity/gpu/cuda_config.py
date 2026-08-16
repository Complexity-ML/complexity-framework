"""
CUDA backend configuration for optimal training performance.

Complexity-ML — 2026
"""

import torch
import logging

from complexity.utils.device import configure_torch_acceleration, is_rocm_available

logger = logging.getLogger(__name__)


def configure_cuda_backends(
    tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
) -> None:
    """
    Configure CUDA backends for maximum throughput.

    Args:
        tf32: Enable TF32 for matmul and cuDNN (1.5x speedup on Ampere+)
        cudnn_benchmark: Auto-tune cuDNN kernels (faster after warmup)
        matmul_precision: "highest", "high" (TF32), or "medium" (bf16 accumulate)

    Note:
        TF32 reduces matmul precision from fp32 to ~10 bits mantissa.
        This is fine for training (bf16 is only 7 bits) but may affect
        fp32 inference. Disable for exact fp32 reproducibility.
    """
    if not torch.cuda.is_available():
        return

    if is_rocm_available():
        torch.set_float32_matmul_precision(matmul_precision)
        configure_torch_acceleration(log=True)
        return

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Disable cuDNN's SDPA backend on systems where it errors out — on
    # B200 + cuDNN 9.x the cudnn-frontend planner sometimes raises
    # "No valid execution plans built" for specific head_dim / seq_len
    # combinations. Flash and mem-efficient backends serve the same call
    # with no measurable speed loss.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    torch.set_float32_matmul_precision(matmul_precision)

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"[gpu] CUDA configured: tf32={tf32}, cudnn_benchmark={cudnn_benchmark}, "
                f"matmul_precision={matmul_precision}, device={gpu_name}")
