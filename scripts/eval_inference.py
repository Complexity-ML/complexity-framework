"""
Inference benchmarks for ablation study checkpoints.

Measures latency (ms/token), throughput (tokens/s), and peak VRAM
for all 4 runs. Run 4 is benchmarked both in float and INT8.

Outputs a table + CSV for the paper.

Usage:
    python scripts/eval_inference.py
    python scripts/eval_inference.py --checkpoint-dir ./checkpoints/ablation-150m
    python scripts/eval_inference.py --run 2
    python scripts/eval_inference.py --prompt "The meaning of life is"

INL - 2025
"""

import torch
import argparse
import os
import json
import csv
import time
import math

from complexity.config import ModelConfig
from complexity.models import ComplexityModel


# ── Run metadata ──────────────────────────────────────────────────────────

RUN_INFO = {
    1: ("run1-dense",  "Dense SwiGLU (baseline)", False),
    2: ("run2-full",   "Token-Routed + Mu + INL", False),
    3: ("run3-no-mu",  "Token-Routed + INL (no Mu)", False),
    4: ("run4-inl",    "INL integer-first (float)", False),
    "4-int8": ("run4-inl", "INL integer-first (INT8)", True),
}


# ── Benchmark functions ──────────────────────────────────────────────────

def measure_vram():
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def measure_peak_vram():
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


@torch.no_grad()
def benchmark_prefill(model, input_ids, device, num_runs=20, warmup=5):
    """
    Benchmark prefill (prompt processing) latency.

    Returns:
        avg_ms: Average latency in milliseconds
        std_ms: Standard deviation
    """
    model.eval()

    # Warmup
    for _ in range(warmup):
        _ = model(input_ids)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(input_ids)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
    return avg_ms, std_ms


@torch.no_grad()
def benchmark_generation(model, input_ids, device, max_new_tokens=128,
                         num_runs=10, warmup=3):
    """
    Benchmark autoregressive generation.

    Returns:
        avg_ms_per_token: Average ms per generated token
        throughput: Tokens per second
        total_ms: Total generation time
    """
    model.eval()

    # Warmup
    for _ in range(warmup):
        _ = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
        )

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        generated_tokens = output.shape[1] - input_ids.shape[1]
        times.append(((end - start) * 1000, generated_tokens))

    avg_total_ms = sum(t[0] for t in times) / len(times)
    avg_tokens = sum(t[1] for t in times) / len(times)
    avg_ms_per_token = avg_total_ms / max(avg_tokens, 1)
    throughput = avg_tokens / (avg_total_ms / 1000)

    return avg_ms_per_token, throughput, avg_total_ms


@torch.no_grad()
def benchmark_batch_throughput(model, device, seq_len=512,
                                batch_sizes=[1, 4, 8, 16, 32],
                                num_runs=10, warmup=3, vocab_size=32000):
    """
    Benchmark throughput at different batch sizes.

    Returns:
        List of (batch_size, tokens_per_second)
    """
    model.eval()
    results = []

    for bs in batch_sizes:
        try:
            input_ids = torch.randint(0, vocab_size, (bs, seq_len), device=device)

            # Warmup
            for _ in range(warmup):
                _ = model(input_ids)

            torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(input_ids)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

            avg_time = sum(times) / len(times)
            tokens_per_sec = (bs * seq_len) / avg_time
            results.append((bs, tokens_per_sec))
            print(f"    batch={bs}: {tokens_per_sec:,.0f} tok/s ({avg_time*1000:.1f}ms)")

        except torch.cuda.OutOfMemoryError:
            print(f"    batch={bs}: OOM")
            torch.cuda.empty_cache()
            break

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference benchmarks for ablation runs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ablation-150m")
    parser.add_argument("--run", type=str, default="all",
                        help="Run ID: 1, 2, 3, 4, '4-int8', or 'all'")
    parser.add_argument("--prompt-len", type=int, default=256,
                        help="Prompt length in tokens for benchmarks")
    parser.add_argument("--gen-tokens", type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument("--num-runs", type=int, default=20,
                        help="Number of benchmark runs for averaging")
    parser.add_argument("--output", type=str, default="./inference_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Which runs to evaluate
    if args.run == "all":
        run_ids = [1, 2, 3, 4, "4-int8"]
    else:
        run_ids = [int(args.run) if args.run.isdigit() else args.run]

    # Prompt input
    input_ids = torch.randint(0, 32000, (1, args.prompt_len), device=device)

    results = []

    for run_id in run_ids:
        dirname, description, quantize = RUN_INFO[run_id]

        if quantize:
            model_path = os.path.join(args.checkpoint_dir, dirname, "final-int8")
            if not os.path.exists(model_path):
                # Try quantizing from float checkpoint
                model_path_float = os.path.join(args.checkpoint_dir, dirname, "final")
                if not os.path.exists(model_path_float):
                    print(f"\nSkipping {description}: checkpoint not found")
                    continue
                print(f"\nQuantizing Run 4 from float checkpoint...")
                model = ComplexityModel.from_pretrained(model_path_float, device=str(device))
                model.quantize_all()
            else:
                model = ComplexityModel.from_pretrained(model_path, device=str(device))
        else:
            model_path = os.path.join(args.checkpoint_dir, dirname, "final")
            if not os.path.exists(model_path):
                print(f"\nSkipping {description}: {model_path} not found")
                continue
            model = ComplexityModel.from_pretrained(model_path, device=str(device))

        num_params = model.num_parameters(trainable_only=False)

        print(f"\n{'='*70}")
        print(f"  {description}")
        print(f"  Params: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"{'='*70}")

        # Reset VRAM tracking
        torch.cuda.reset_peak_memory_stats()
        model_vram = measure_vram()
        print(f"\n  Model VRAM: {model_vram:.0f} MB")

        # 1. Prefill benchmark
        print(f"\n  Prefill (prompt={args.prompt_len} tokens):")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefill_ms, prefill_std = benchmark_prefill(
                model, input_ids, device, num_runs=args.num_runs,
            )
        print(f"    Avg: {prefill_ms:.2f} ms (±{prefill_std:.2f})")
        print(f"    Throughput: {args.prompt_len / (prefill_ms / 1000):,.0f} tok/s")

        # 2. Generation benchmark
        print(f"\n  Generation ({args.gen_tokens} tokens):")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ms_per_tok, gen_throughput, total_gen_ms = benchmark_generation(
                model, input_ids, device,
                max_new_tokens=args.gen_tokens,
                num_runs=min(args.num_runs, 10),
            )
        print(f"    Avg: {ms_per_tok:.2f} ms/token")
        print(f"    Throughput: {gen_throughput:.0f} tok/s")
        print(f"    Total: {total_gen_ms:.0f} ms for {args.gen_tokens} tokens")

        # 3. Batch throughput
        print(f"\n  Batch throughput (seq_len=512):")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_results = benchmark_batch_throughput(
                model, device, num_runs=min(args.num_runs, 10),
            )

        peak_vram = measure_peak_vram()
        print(f"\n  Peak VRAM: {peak_vram:.0f} MB")

        # Best batch throughput
        best_batch_tps = max(batch_results, key=lambda x: x[1]) if batch_results else (1, 0)

        results.append({
            "run": str(run_id),
            "description": description,
            "params_M": round(num_params / 1e6, 1),
            "model_vram_MB": round(model_vram, 0),
            "peak_vram_MB": round(peak_vram, 0),
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tok_s": round(args.prompt_len / (prefill_ms / 1000), 0),
            "gen_ms_per_tok": round(ms_per_tok, 2),
            "gen_tok_s": round(gen_throughput, 0),
            "best_batch_tok_s": round(best_batch_tps[1], 0),
            "best_batch_size": best_batch_tps[0],
        })

        # Free GPU
        del model
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────

    if results:
        print(f"\n{'='*90}")
        print(f"  INFERENCE BENCHMARK SUMMARY")
        print(f"{'='*90}")
        header = (f"{'Run':<10} {'Description':<28} {'VRAM':<8} {'Prefill':<10} "
                  f"{'Gen ms/tok':<12} {'Gen tok/s':<10} {'Best Batch':<12}")
        print(header)
        print("-" * 90)
        for r in results:
            print(f"{r['run']:<10} {r['description']:<28} {r['model_vram_MB']:<8.0f} "
                  f"{r['prefill_ms']:<10.2f} {r['gen_ms_per_tok']:<12.2f} "
                  f"{r['gen_tok_s']:<10.0f} {r['best_batch_tok_s']:<12,.0f}")

        # Save CSV
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")

        # Save JSON
        json_path = args.output.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")

        # Key comparisons for paper
        if len(results) >= 2:
            baseline = next((r for r in results if r['run'] == '1'), None)
            int8 = next((r for r in results if r['run'] == '4-int8'), None)
            full = next((r for r in results if r['run'] == '2'), None)

            print(f"\n{'='*70}")
            print(f"  KEY COMPARISONS (for paper)")
            print(f"{'='*70}")

            if baseline and full:
                speedup = baseline['gen_ms_per_tok'] / full['gen_ms_per_tok']
                print(f"  Full arch vs Dense: {speedup:.2f}x generation speed")

            if baseline and int8:
                speedup = baseline['gen_ms_per_tok'] / int8['gen_ms_per_tok']
                vram_ratio = baseline['model_vram_MB'] / max(int8['model_vram_MB'], 1)
                print(f"  INT8 vs Dense: {speedup:.2f}x generation speed, "
                      f"{vram_ratio:.2f}x VRAM reduction")

            if full and int8:
                speedup = full['gen_ms_per_tok'] / int8['gen_ms_per_tok']
                print(f"  INT8 vs Full (float): {speedup:.2f}x generation speed")


if __name__ == "__main__":
    main()
