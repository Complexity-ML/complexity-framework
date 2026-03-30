"""Training metrics tracking."""

import time
import json
import math
from pathlib import Path
from typing import Dict, Any, List


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[Dict[str, Any]] = []
        self.current_metrics: Dict[str, float] = {}
        self.smoothed_metrics: Dict[str, float] = {}
        self.smoothing = 0.99

        self.start_time = time.time()
        self.step_times: List[float] = []

    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics."""
        self.current_metrics = metrics

        for key, value in metrics.items():
            if key in self.smoothed_metrics:
                self.smoothed_metrics[key] = (
                    self.smoothing * self.smoothed_metrics[key] +
                    (1 - self.smoothing) * value
                )
            else:
                self.smoothed_metrics[key] = value

        record = {
            "step": step,
            "timestamp": time.time() - self.start_time,
            **metrics,
        }
        self.metrics_history.append(record)

    def log_step_time(self, step_time: float):
        """Log time for a step."""
        self.step_times.append(step_time)
        if len(self.step_times) > 100:
            self.step_times.pop(0)

    def get_throughput(self, batch_size: int, seq_len: int) -> float:
        """Get tokens per second."""
        if not self.step_times:
            return 0.0
        avg_time = sum(self.step_times) / len(self.step_times)
        if avg_time == 0:
            return 0.0
        return (batch_size * seq_len) / avg_time

    def save(self, filename: str = "metrics.json"):
        """Save metrics to file."""
        with open(self.log_dir / filename, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "total_steps": len(self.metrics_history),
            "total_time": time.time() - self.start_time,
            "final_loss": self.smoothed_metrics.get("loss", None),
            "average_throughput": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
        }
