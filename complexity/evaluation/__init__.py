"""Evaluation helpers, loaded only when their implementation is requested."""

from importlib import import_module

_EXPORTS = {
    "Evaluator": (".evaluator", "Evaluator"),
    "EvalConfig": (".evaluator", "EvalConfig"),
    "compute_perplexity": (".evaluator", "compute_perplexity"),
    "compute_accuracy": (".evaluator", "compute_accuracy"),
    "compute_f1": (".evaluator", "compute_f1"),
    "BenchmarkRunner": (".benchmarks", "BenchmarkRunner"),
    "Benchmark": (".benchmarks", "Benchmark"),
    "MMLUBenchmark": (".benchmarks", "MMLUBenchmark"),
    "HellaSwagBenchmark": (".benchmarks", "HellaSwagBenchmark"),
    "WinogradeBenchmark": (".benchmarks", "WinogradeBenchmark"),
    "ARCBenchmark": (".benchmarks", "ARCBenchmark"),
    "TruthfulQABenchmark": (".benchmarks", "TruthfulQABenchmark"),
    "RougeScorer": (".metrics", "RougeScorer"),
    "BleuScorer": (".metrics", "BleuScorer"),
    "BertScoreEvaluator": (".metrics", "BertScoreEvaluator"),
    "DiversityMetrics": (".metrics", "DiversityMetrics"),
    "CoherenceMetrics": (".metrics", "CoherenceMetrics"),
    "Example": (".zero_shot", "Example"),
    "EncodedChoice": (".zero_shot", "EncodedChoice"),
    "encode_choices": (".zero_shot", "encode_choices"),
    "score_examples": (".zero_shot", "score_examples"),
    "summarize": (".zero_shot", "summarize"),
    "load_arc_easy": (".zero_shot", "load_arc_easy"),
    "load_arc_challenge": (".zero_shot", "load_arc_challenge"),
    "load_arc": (".zero_shot", "load_arc"),
    "load_torch_bundle": (".zero_shot", "load_torch_bundle"),
    "evaluate_choices": (".zero_shot", "evaluate_choices"),
    "compare_choices": (".zero_shot", "compare_choices"),
    "load_hellaswag": (".zero_shot", "load_hellaswag"),
    "load_mmlu": (".zero_shot", "load_mmlu"),
    "load_piqa": (".zero_shot", "load_piqa"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
