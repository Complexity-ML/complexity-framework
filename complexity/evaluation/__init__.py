"""
Evaluation module for framework-complexity.

Provides comprehensive evaluation capabilities:
- Perplexity computation
- Standard benchmarks (MMLU, HellaSwag, etc.)
- Generation quality metrics
- Speed benchmarking

Usage:
    from complexity.evaluation import Evaluator, BenchmarkRunner

    evaluator = Evaluator(model, tokenizer)

    # Perplexity
    ppl = evaluator.perplexity(test_data)

    # Benchmarks
    runner = BenchmarkRunner(model, tokenizer)
    results = runner.run_benchmark("mmlu")
"""

from .evaluator import (
    Evaluator,
    EvalConfig,
    compute_perplexity,
    compute_accuracy,
    compute_f1,
)

from .benchmarks import (
    BenchmarkRunner,
    Benchmark,
    MMLUBenchmark,
    HellaSwagBenchmark,
    WinogradeBenchmark,
    ARCBenchmark,
    TruthfulQABenchmark,
)

from .metrics import (
    RougeScorer,
    BleuScorer,
    BertScoreEvaluator,
    DiversityMetrics,
    CoherenceMetrics,
)

__all__ = [
    # Evaluator
    "Evaluator",
    "EvalConfig",
    "compute_perplexity",
    "compute_accuracy",
    "compute_f1",
    # Benchmarks
    "BenchmarkRunner",
    "Benchmark",
    "MMLUBenchmark",
    "HellaSwagBenchmark",
    "WinogradeBenchmark",
    "ARCBenchmark",
    "TruthfulQABenchmark",
    # Metrics
    "RougeScorer",
    "BleuScorer",
    "BertScoreEvaluator",
    "DiversityMetrics",
    "CoherenceMetrics",
]
