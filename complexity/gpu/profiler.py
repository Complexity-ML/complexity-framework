"""
GPU profiling utilities for training performance analysis.

Complexity-ML — 2026
"""

import torch
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StepTimer:
    """
    Measure per-step training throughput.

    Usage:
        timer = StepTimer(seq_len=2048, batch_size=64, world_size=8)
        for step in training_loop:
            timer.start()
            # ... training step ...
            timer.stop()
            if step % 100 == 0:
                timer.log()
    """

    def __init__(self, seq_len: int = 2048, batch_size: int = 64,
                 world_size: int = 1, accum: int = 1):
        self.tokens_per_step = batch_size * world_size * accum * seq_len
        self.times = []
        self._start = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.times.append(elapsed)

    @property
    def avg_step_time(self) -> float:
        return sum(self.times[-100:]) / len(self.times[-100:]) if self.times else 0

    @property
    def tokens_per_second(self) -> float:
        avg = self.avg_step_time
        return self.tokens_per_step / avg if avg > 0 else 0

    @property
    def mfu(self, flops_per_token: float = None, gpu_peak_flops: float = None) -> float:
        """Model FLOPs Utilization (MFU). Requires knowing model FLOPs and GPU peak."""
        if flops_per_token is None or gpu_peak_flops is None:
            return 0.0
        achieved = self.tokens_per_second * flops_per_token
        return achieved / gpu_peak_flops

    def log(self, prefix: str = ""):
        avg = self.avg_step_time
        tps = self.tokens_per_second
        logger.info(
            f"[gpu] {prefix}Step: {avg:.3f}s, "
            f"Throughput: {tps/1e6:.2f}M tokens/s, "
            f"{tps/1e3:.0f}K tok/s"
        )

    def reset(self):
        self.times.clear()


def profile_model_flops(model: torch.nn.Module, batch_size: int = 1,
                        seq_len: int = 2048) -> dict:
    """
    Estimate model FLOPs per forward pass.

    Uses the standard 6 * num_params * tokens approximation.
    """
    num_params = sum(p.numel() for p in model.parameters())
    tokens = batch_size * seq_len

    # 6 = 2 (forward matmul) + 2 (backward matmul) + 2 (backward weight update)
    flops_per_step = 6 * num_params * tokens

    return {
        "num_params": num_params,
        "tokens_per_step": tokens,
        "flops_forward": 2 * num_params * tokens,
        "flops_total": flops_per_step,
        "tflops_total": flops_per_step / 1e12,
    }
