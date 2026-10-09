"""
Server commands for the Complexity framework.
"""

from pathlib import Path
from typing import Optional

import typer

from ..utils import console, error, print_panel

serve = typer.Typer(name="serve", help="Model serving commands")


@serve.command("start")
def start_server(
    model_path: Path = typer.Argument(..., help="Model checkpoint path"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
    quantize: Optional[str] = typer.Option(
        None, "--quantize", "-q", help="Quantization: int8, int4"
    ),
    max_batch_size: int = typer.Option(32, "--max-batch", help="Maximum batch size"),
    max_seq_length: int = typer.Option(4096, "--max-seq", help="Maximum sequence length"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for authentication"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    ssl_cert: Optional[Path] = typer.Option(None, "--ssl-cert", help="SSL certificate"),
    ssl_key: Optional[Path] = typer.Option(None, "--ssl-key", help="SSL key"),
):
    """
    Explain how to migrate historical native serving to an external runtime.
    """
    console.print(
        error(
            "Native in-process text serving is not supported. Export the checkpoint "
            "for a compatible vLLM/SGLang runtime, start that runtime separately, then "
            "use complexity inference generate/chat/batch with --backend and --base-url. "
            "Detector HTTP serving is available separately under complexity vision serve detector."
        )
    )
    raise typer.Exit(2)


@serve.command("benchmark-server")
def benchmark_server(
    url: str = typer.Argument("http://localhost:8000", help="Existing server URL"),
    prompt: str = typer.Option("Hello, how are you?", "--prompt", "-p"),
    num_requests: int = typer.Option(100, "--requests", "-n", min=1),
    concurrency: int = typer.Option(10, "--concurrency", "-c", min=1),
    max_tokens: int = typer.Option(128, "--max-tokens", min=1),
    model: Optional[str] = typer.Option(
        None, "--model", help="Required model name/id served by the endpoint"
    ),
    backend: str = typer.Option("vllm", "--backend", help="Serving backend: vllm or sglang"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Bearer token for the serving endpoint"
    ),
    timeout: float = typer.Option(
        60.0, "--timeout", min=0.001, help="Per-request timeout in seconds"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable measurements"),
):
    """Measure endpoint request latency and successful request throughput.

    Sends exactly --requests completions, without hidden warmup requests. It
    reports no estimated tokens/sec because actual token usage is unavailable.
    """
    import json
    import math
    import statistics
    import time
    from concurrent.futures import ThreadPoolExecutor

    if not model:
        message = "Specify --model with the model name/id exposed by the serving endpoint."
        if json_output:
            typer.echo(json.dumps({"error": message}), err=True)
        else:
            console.print(error(message))
        raise typer.Exit(1)
    try:
        from complexity.inference import ExternalGenerationConfig, create_external_backend

        client = create_external_backend(
            backend, base_url=url, model=model, api_key=api_key, timeout=timeout
        )
        config = ExternalGenerationConfig(max_tokens=max_tokens)

        def request(_):
            start = time.perf_counter()
            try:
                client.complete(prompt, config)
            except (RuntimeError, ValueError, OSError):
                return None
            return time.perf_counter() - start

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(concurrency, num_requests)) as pool:
            results = list(pool.map(request, range(num_requests)))
        elapsed = time.perf_counter() - start
        latencies = sorted(value for value in results if value is not None)
        successes = len(latencies)
        report = {
            "model": model,
            "backend": backend,
            "requests": num_requests,
            "successful_requests": successes,
            "failed_requests": num_requests - successes,
            "concurrency": min(concurrency, num_requests),
            "elapsed_seconds": elapsed,
            "successful_requests_per_second": successes / elapsed,
            "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
            "p50_latency_seconds": statistics.median(latencies) if latencies else None,
            "p99_latency_seconds": latencies[math.ceil(len(latencies) * 0.99) - 1]
            if latencies
            else None,
            "token_throughput_measured": False,
        }
        if json_output:
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
        else:
            print_panel(
                f"Successful requests: {successes}/{num_requests}\n"
                f"Concurrency: {report['concurrency']}\n"
                f"Elapsed: {elapsed:.3f}s\n"
                f"Successful requests/sec: {report['successful_requests_per_second']:.3f}\n"
                f"Mean latency: {report['mean_latency_seconds']}s\n"
                f"P50 latency: {report['p50_latency_seconds']}s\n"
                f"P99 latency: {report['p99_latency_seconds']}s\n"
                "Token throughput: not measured",
                title="Endpoint measurements",
                style="cyan",
            )
        if successes != num_requests:
            raise typer.Exit(1)
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}), err=True)
        else:
            console.print(error(f"Benchmark failed: {exc}"))
        raise typer.Exit(1) from exc


@serve.command("openai-proxy")
def openai_proxy(
    model_path: Path = typer.Argument(..., help="Model checkpoint"),
    port: int = typer.Option(8000, "--port", "-p"),
    model_name: str = typer.Option("complexity", "--model-name", help="Name to expose"),
):
    """
    Explain migration from the historical native proxy to vLLM/SGLang.
    """
    start_server(
        model_path=model_path,
        host="0.0.0.0",
        port=port,
        device="cuda",
        quantize=None,
        max_batch_size=32,
        max_seq_length=4096,
        api_key=None,
        workers=1,
        ssl_cert=None,
        ssl_key=None,
    )
