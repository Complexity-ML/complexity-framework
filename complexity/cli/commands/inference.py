"""
Inference commands for the Complexity framework.
"""

from pathlib import Path
from typing import Optional

import typer

from ..utils import console, error, info, spinner, success

inference = typer.Typer(name="inference", help="Inference and generation commands")


@inference.command("generate")
def generate(
    model: str = typer.Argument(..., help="Model name/id served by vLLM or SGLang"),
    prompt: str = typer.Option(None, "--prompt", "-p", help="Text prompt"),
    prompt_file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing prompt"),
    backend: str = typer.Option("vllm", "--backend", help="Serving backend: vllm or sglang"),
    base_url: str = typer.Option(
        "http://localhost:8000", "--base-url", help="OpenAI-compatible server URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Bearer token for the serving endpoint"
    ),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_p: float = typer.Option(0.9, "--top-p", help="Top-p sampling"),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", help="Top-k sampling if supported by backend"
    ),
    timeout: float = typer.Option(60.0, "--timeout", help="Request timeout in seconds"),
):
    """
    Generate text through vLLM or SGLang.

    Complexity does not run native PyTorch generation in-process; start a
    vLLM/SGLang OpenAI-compatible server first, then point this command to it.

    Examples:
        complexity inference generate my-model --backend vllm --base-url http://localhost:8000 --prompt "Hello"
        complexity inference generate my-model --backend sglang --base-url http://localhost:30000 -f prompt.txt
    """
    if prompt is None and prompt_file is None:
        console.print(error("Provide --prompt or --file"))
        raise typer.Exit(1)

    if prompt_file:
        if not prompt_file.exists():
            console.print(error(f"Prompt file not found: {prompt_file}"))
            raise typer.Exit(1)
        prompt = prompt_file.read_text()

    try:
        if backend not in ("vllm", "sglang"):
            console.print(error("--backend must be 'vllm' or 'sglang'"))
            raise typer.Exit(1)

        from complexity.inference import ExternalGenerationConfig, create_external_backend

        client = create_external_backend(
            backend,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        gen_config = ExternalGenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        console.print(
            f"\n[bold cyan]Prompt:[/bold cyan] {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
        )
        console.print(f"[bold green]Generation via {backend}:[/bold green]")

        with spinner(f"Calling {backend} at {base_url}..."):
            output = client.complete(prompt, gen_config)
        console.print(output)

    except ImportError as e:
        console.print(error(f"Missing dependency: {e}"))
        raise typer.Exit(1)
    except Exception as e:
        console.print(error(f"Generation failed: {e}"))
        raise typer.Exit(1)


@inference.command("chat")
def chat_interactive(
    model_path: str = typer.Argument(..., help="Model name/id served by vLLM or SGLang"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    max_tokens: int = typer.Option(
        512, "--max-tokens", "-m", min=1, help="Max tokens per response"
    ),
    temperature: float = typer.Option(0.7, "--temperature", "-t", min=0.0, help="Temperature"),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Legacy local option; configure on the serving host"
    ),
    quantize: Optional[str] = typer.Option(
        None, "--quantize", "-q", help="Legacy local option; configure on the serving host"
    ),
    backend: str = typer.Option("vllm", "--backend", help="Serving backend: vllm or sglang"),
    base_url: str = typer.Option(
        "http://localhost:8000", "--base-url", help="OpenAI-compatible server URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Bearer token for the serving endpoint"
    ),
    timeout: float = typer.Option(60.0, "--timeout", min=0.001, help="Request timeout in seconds"),
):
    """Chat through an existing vLLM/SGLang endpoint with conversation history.

    Device selection and quantization belong to the external server. Historical
    local flags are still recognized and explain how to migrate their usage.
    """
    if device is not None or quantize is not None:
        console.print(
            error(
                "--device and --quantize are server-side settings. Configure them when "
                "starting vLLM/SGLang and omit these flags from this client."
            )
        )
        raise typer.Exit(2)
    try:
        from complexity.inference import ExternalGenerationConfig, create_external_backend

        client = create_external_backend(
            backend,
            base_url=base_url,
            model=str(model_path),
            api_key=api_key,
            timeout=timeout,
        )
        gen_config = ExternalGenerationConfig(max_tokens=max_tokens, temperature=temperature)
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        console.print(info(f"Chat via {backend} at {base_url}"))
        console.print(info("Type 'quit' or 'exit' to end, 'clear' to reset history"))
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ")
            except (KeyboardInterrupt, EOFError):
                console.print(info("Goodbye!"))
                break
            command = user_input.strip().lower()
            if command in {"quit", "exit"}:
                console.print(info("Goodbye!"))
                break
            if command == "clear":
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                console.print(info("History cleared"))
                continue
            if not user_input.strip():
                continue
            request = messages + [{"role": "user", "content": user_input}]
            response = client.chat(request, gen_config)
            console.print("[bold green]Assistant:[/bold green]", response)
            messages = request + [{"role": "assistant", "content": response}]
    except (ImportError, ValueError, RuntimeError) as exc:
        console.print(error(f"Chat failed: {exc}"))
        raise typer.Exit(1) from exc


@inference.command("batch")
def batch_inference(
    model_path: str = typer.Argument(..., help="Model name/id served by vLLM or SGLang"),
    input_file: Path = typer.Argument(..., help="Input JSONL file with prompt or text fields"),
    output_file: Path = typer.Argument(..., help="Output JSONL file"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", min=1),
    batch_size: int = typer.Option(
        8, "--batch-size", "-b", min=1, help="Client chunk size; requests are sequential"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Legacy local option; configure on the serving host"
    ),
    quantize: Optional[str] = typer.Option(
        None, "--quantize", "-q", help="Legacy local option; configure on the serving host"
    ),
    backend: str = typer.Option("vllm", "--backend", help="Serving backend: vllm or sglang"),
    base_url: str = typer.Option(
        "http://localhost:8000", "--base-url", help="OpenAI-compatible server URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Bearer token for the serving endpoint"
    ),
    timeout: float = typer.Option(60.0, "--timeout", min=0.001, help="Request timeout in seconds"),
    force: bool = typer.Option(False, "--force", help="Replace an existing output file"),
):
    """Generate JSONL results through an existing external serving endpoint.

    Input is validated before requests. The output is published only after all
    completions succeed; a failed request leaves an existing output unchanged.
    """
    import json
    import os
    import tempfile

    if device is not None or quantize is not None:
        console.print(
            error("--device and --quantize are server-side settings; omit these client flags.")
        )
        raise typer.Exit(2)
    try:
        if output_file.resolve() == input_file.resolve():
            raise ValueError("Input and output paths must differ")
        if output_file.exists() and not force:
            raise ValueError(f"Output already exists: {output_file}. Use --force to replace it.")
        if not output_file.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {output_file.parent}")
        prompts = []
        with input_file.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON at input line {number}: {exc}") from exc
                prompt = item.get("prompt", item.get("text")) if isinstance(item, dict) else None
                if not isinstance(prompt, str):
                    raise ValueError(f"Input line {number} needs a string prompt or text field")
                prompts.append(prompt)
        from complexity.inference import ExternalGenerationConfig, create_external_backend

        client = create_external_backend(
            backend,
            base_url=base_url,
            model=str(model_path),
            api_key=api_key,
            timeout=timeout,
        )
        config = ExternalGenerationConfig(max_tokens=max_tokens)
        with tempfile.TemporaryDirectory(
            prefix=".complexity-batch-", dir=output_file.parent
        ) as temporary:
            staged = Path(temporary) / "results.jsonl"
            with staged.open("w", encoding="utf-8") as stream:
                for offset in range(0, len(prompts), batch_size):
                    for prompt in prompts[offset : offset + batch_size]:
                        output = client.complete(prompt, config)
                        stream.write(
                            json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False)
                            + "\n"
                        )
            if force:
                os.replace(staged, output_file)
            else:
                # An exclusive publication also prevents clobbering a file
                # created by another process while requests were in flight.
                os.link(staged, output_file)
        console.print(success(f"Saved {len(prompts)} results to {output_file}"))
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        console.print(error(f"Batch inference failed: {exc}"))
        raise typer.Exit(1) from exc


@inference.command("benchmark")
def benchmark(
    model_path: Path = typer.Argument(..., help="Legacy model checkpoint"),
    prompt_length: int = typer.Option(128, "--prompt-len", help="Legacy input length"),
    output_length: int = typer.Option(128, "--output-len", help="Legacy output length"),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes", help="Legacy native batch sizes"),
    warmup: int = typer.Option(3, "--warmup", help="Legacy warmup iterations"),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Legacy benchmark iterations"),
    device: str = typer.Option("cuda", "--device", "-d", help="Legacy local device"),
    quantize: Optional[str] = typer.Option(
        None, "--quantize", "-q", help="Legacy local quantization"
    ),
):
    """Explain migration from removed native benchmarking to endpoint metrics."""
    console.print(
        error(
            "Native checkpoint benchmarking is not supported. Benchmark an existing "
            "endpoint with complexity serve benchmark-server URL --model SERVED_MODEL. "
            "That command measures successful requests and latency; token throughput "
            "requires the serving backend's token-usage metrics."
        )
    )
    raise typer.Exit(2)
