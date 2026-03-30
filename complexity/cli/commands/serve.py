"""
Server commands for the Complexity framework.
"""

import typer
from pathlib import Path
from typing import Optional

from ..utils import console, spinner, print_panel, success, error, warning, info

serve = typer.Typer(name="serve", help="Model serving commands")


@serve.command("start")
def start_server(
    model_path: Path = typer.Argument(..., help="Model checkpoint path"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
    quantize: Optional[str] = typer.Option(None, "--quantize", "-q", help="Quantization: int8, int4"),
    max_batch_size: int = typer.Option(32, "--max-batch", help="Maximum batch size"),
    max_seq_length: int = typer.Option(4096, "--max-seq", help="Maximum sequence length"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for authentication"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    ssl_cert: Optional[Path] = typer.Option(None, "--ssl-cert", help="SSL certificate"),
    ssl_key: Optional[Path] = typer.Option(None, "--ssl-key", help="SSL key"),
):
    """
    Start an inference server.

    Examples:
        complexity serve start model.pt
        complexity serve start model.pt --port 8080 --quantize int8
        complexity serve start model.pt --api-key secret123
    """
    if not model_path.exists():
        console.print(error(f"Model not found: {model_path}"))
        raise typer.Exit(1)

    print_panel(
        f"Model: {model_path}\n"
        f"Endpoint: http://{host}:{port}\n"
        f"Device: {device}\n"
        f"Max batch: {max_batch_size}\n"
        f"Max seq length: {max_seq_length}",
        title="Server Configuration",
        style="cyan"
    )

    try:
        with spinner("Loading model..."):
            from complexity.inference import InferenceEngine
            engine = InferenceEngine.from_checkpoint(model_path, device=device)

        console.print(success("Model loaded"))

        # Create FastAPI app
        from fastapi import FastAPI, HTTPException, Depends, Header
        from fastapi.security import HTTPBearer
        from pydantic import BaseModel
        from typing import List, Optional as Opt
        import uvicorn

        app = FastAPI(title="Complexity Inference Server", version="1.0.0")

        class GenerateRequest(BaseModel):
            prompt: str
            max_tokens: int = 256
            temperature: float = 0.7
            top_p: float = 0.9
            top_k: int = 50
            stop: Opt[List[str]] = None

        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatRequest(BaseModel):
            messages: List[ChatMessage]
            max_tokens: int = 512
            temperature: float = 0.7
            stream: bool = False

        class GenerateResponse(BaseModel):
            text: str
            tokens_generated: int

        # Auth dependency
        async def verify_api_key(authorization: str = Header(None)):
            if api_key:
                if not authorization or authorization != f"Bearer {api_key}":
                    raise HTTPException(status_code=401, detail="Invalid API key")
            return True

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/v1/models")
        async def list_models(auth: bool = Depends(verify_api_key)):
            return {
                "data": [
                    {
                        "id": str(model_path.name),
                        "object": "model",
                    }
                ]
            }

        @app.post("/v1/completions")
        async def generate(request: GenerateRequest, auth: bool = Depends(verify_api_key)):
            from complexity.inference import GenerationConfig

            config = GenerationConfig(
                max_new_tokens=min(request.max_tokens, max_seq_length),
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop_strings=request.stop,
            )

            output = engine.generate(request.prompt, config)

            return {
                "id": "gen-1",
                "object": "text_completion",
                "choices": [
                    {
                        "text": output,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ]
            }

        @app.post("/v1/chat/completions")
        async def chat(request: ChatRequest, auth: bool = Depends(verify_api_key)):
            from complexity.inference import GenerationConfig
            from complexity.data import ComplexityTemplate

            template = ComplexityTemplate(format="complexity")
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            prompt = template.format_messages(messages)
            prompt += template.format_header("assistant")

            config = GenerationConfig(
                max_new_tokens=min(request.max_tokens, max_seq_length),
                temperature=request.temperature,
                stop_strings=["<|end|>", "<|user|>"],
            )

            output = engine.generate(prompt, config)

            return {
                "id": "chat-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": output,
                        },
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ]
            }

        console.print(info(f"Starting server on http://{host}:{port}"))
        console.print(info("API endpoints:"))
        console.print("  GET  /health")
        console.print("  GET  /v1/models")
        console.print("  POST /v1/completions")
        console.print("  POST /v1/chat/completions")

        ssl_config = {}
        if ssl_cert and ssl_key:
            ssl_config = {"ssl_certfile": str(ssl_cert), "ssl_keyfile": str(ssl_key)}

        uvicorn.run(app, host=host, port=port, workers=workers, **ssl_config)

    except ImportError as e:
        console.print(error(f"Missing dependency: {e}"))
        console.print(info("Install with: pip install fastapi uvicorn"))
        raise typer.Exit(1)


@serve.command("benchmark-server")
def benchmark_server(
    url: str = typer.Argument("http://localhost:8000", help="Server URL"),
    prompt: str = typer.Option("Hello, how are you?", "--prompt", "-p"),
    num_requests: int = typer.Option(100, "--requests", "-n"),
    concurrency: int = typer.Option(10, "--concurrency", "-c"),
    max_tokens: int = typer.Option(128, "--max-tokens"),
):
    """
    Benchmark an inference server.

    Example:
        complexity serve benchmark-server http://localhost:8000 -n 100 -c 10
    """
    import time
    import asyncio
    import aiohttp

    async def make_request(session, url, payload):
        start = time.perf_counter()
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            result = await resp.json()
            elapsed = time.perf_counter() - start
            return elapsed, result

    async def run_benchmark():
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }

        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warmup
            console.print(info("Warming up..."))
            for _ in range(min(5, num_requests)):
                await make_request(session, url, payload)

            # Benchmark
            console.print(info(f"Running {num_requests} requests with {concurrency} concurrency..."))

            start = time.perf_counter()
            tasks = [make_request(session, url, payload) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.perf_counter() - start

            # Analyze results
            latencies = [r[0] for r in results if not isinstance(r, Exception)]
            errors = sum(1 for r in results if isinstance(r, Exception))

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p50 = sorted(latencies)[len(latencies) // 2]
                p99 = sorted(latencies)[int(len(latencies) * 0.99)]
                throughput = num_requests / total_time

                print_panel(
                    f"Total requests: {num_requests}\n"
                    f"Concurrency: {concurrency}\n"
                    f"Total time: {total_time:.2f}s\n"
                    f"Throughput: {throughput:.2f} req/s\n"
                    f"Avg latency: {avg_latency*1000:.2f}ms\n"
                    f"P50 latency: {p50*1000:.2f}ms\n"
                    f"P99 latency: {p99*1000:.2f}ms\n"
                    f"Errors: {errors}",
                    title="Benchmark Results",
                    style="green"
                )
            else:
                console.print(error("All requests failed"))

    try:
        asyncio.run(run_benchmark())
    except Exception as e:
        console.print(error(f"Benchmark failed: {e}"))
        raise typer.Exit(1)


@serve.command("openai-proxy")
def openai_proxy(
    model_path: Path = typer.Argument(..., help="Model checkpoint"),
    port: int = typer.Option(8000, "--port", "-p"),
    model_name: str = typer.Option("complexity", "--model-name", help="Name to expose"),
):
    """
    Start an OpenAI-compatible proxy server.

    This allows using the model with OpenAI SDK:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

    Example:
        complexity serve openai-proxy model.pt --port 8000
    """
    console.print(info("Starting OpenAI-compatible server..."))
    console.print(info("Use with OpenAI SDK:"))
    console.print(f'  client = OpenAI(base_url="http://localhost:{port}/v1", api_key="none")')

    # This reuses the start command with OpenAI-compatible endpoints
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
