"""
Inference commands for the Complexity framework.
"""

import typer
from pathlib import Path
from typing import Optional, List

from ..utils import console, spinner, print_panel, print_code, success, error, warning, info

inference = typer.Typer(name="inference", help="Inference and generation commands")


@inference.command("generate")
def generate(
    model_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    prompt: str = typer.Option(None, "--prompt", "-p", help="Text prompt"),
    prompt_file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing prompt"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_p: float = typer.Option(0.9, "--top-p", help="Top-p sampling"),
    top_k: int = typer.Option(50, "--top-k", help="Top-k sampling"),
    repetition_penalty: float = typer.Option(1.1, "--rep-penalty", help="Repetition penalty"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device: cuda, cpu, mps"),
    quantize: Optional[str] = typer.Option(None, "--quantize", "-q", help="Quantization: int8, int4, awq"),
):
    """
    Generate text from a prompt.

    Examples:
        complexity inference generate model.pt --prompt "Hello, world"
        complexity inference generate model.pt -f prompt.txt --max-tokens 512
        complexity inference generate model.pt -p "Write code" --quantize int8
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
        with spinner("Loading model..."):
            from complexity.inference import InferenceEngine, GenerationConfig
            from complexity.quantization import quantize_model

            engine = InferenceEngine.from_checkpoint(
                model_path,
                device=device,
            )

            if quantize:
                engine.model = quantize_model(engine.model, method=quantize)

        console.print(success(f"Model loaded on {device}"))

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        console.print(f"\n[bold cyan]Prompt:[/bold cyan] {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")
        console.print("[bold green]Generation:[/bold green]")

        if stream:
            for token in engine.generate_stream(prompt, gen_config):
                console.print(token, end="")
            console.print()
        else:
            with spinner("Generating..."):
                output = engine.generate(prompt, gen_config)
            console.print(output)

    except ImportError as e:
        console.print(error(f"Missing dependency: {e}"))
        raise typer.Exit(1)
    except Exception as e:
        console.print(error(f"Generation failed: {e}"))
        raise typer.Exit(1)


@inference.command("chat")
def chat_interactive(
    model_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-m", help="Max tokens per response"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
    quantize: Optional[str] = typer.Option(None, "--quantize", "-q", help="Quantization"),
):
    """
    Interactive chat with a model.

    Example:
        complexity inference chat model.pt --system "You are a helpful assistant"
    """
    try:
        with spinner("Loading model..."):
            from complexity.inference import InferenceEngine, GenerationConfig
            from complexity.data import ComplexityTokenizer, ComplexityTemplate

            engine = InferenceEngine.from_checkpoint(model_path, device=device)
            template = ComplexityTemplate(format="complexity")

        console.print(success("Model loaded"))
        console.print(info("Type 'quit' or 'exit' to end, 'clear' to reset history"))
        console.print()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop_strings=["<|end|>", "<|user|>"],
        )

        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ")
            except (KeyboardInterrupt, EOFError):
                console.print("\n" + info("Goodbye!"))
                break

            if user_input.lower() in ('quit', 'exit'):
                console.print(info("Goodbye!"))
                break

            if user_input.lower() == 'clear':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                console.print(info("History cleared"))
                continue

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            # Format with template
            prompt = template.format_messages(messages)
            prompt += template.format_header("assistant")

            console.print("[bold green]Assistant:[/bold green] ", end="")

            response_text = ""
            for token in engine.generate_stream(prompt, gen_config):
                console.print(token, end="")
                response_text += token

            console.print("\n")
            messages.append({"role": "assistant", "content": response_text})

    except ImportError as e:
        console.print(error(f"Missing dependency: {e}"))
        raise typer.Exit(1)
    except Exception as e:
        console.print(error(f"Chat failed: {e}"))
        raise typer.Exit(1)


@inference.command("batch")
def batch_inference(
    model_path: Path = typer.Argument(..., help="Model checkpoint"),
    input_file: Path = typer.Argument(..., help="Input JSONL file"),
    output_file: Path = typer.Argument(..., help="Output JSONL file"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    device: str = typer.Option("cuda", "--device", "-d"),
    quantize: Optional[str] = typer.Option(None, "--quantize", "-q"),
):
    """
    Batch inference on a JSONL file.

    Input format: {"prompt": "text"} per line
    Output format: {"prompt": "text", "output": "generated"} per line

    Example:
        complexity inference batch model.pt prompts.jsonl outputs.jsonl
    """
    import json

    if not input_file.exists():
        console.print(error(f"Input file not found: {input_file}"))
        raise typer.Exit(1)

    # Load prompts
    prompts = []
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data.get("prompt", data.get("text", "")))

    console.print(info(f"Loaded {len(prompts)} prompts"))

    try:
        with spinner("Loading model..."):
            from complexity.inference import InferenceEngine, GenerationConfig
            engine = InferenceEngine.from_checkpoint(model_path, device=device)

        console.print(success("Model loaded"))

        gen_config = GenerationConfig(max_new_tokens=max_tokens)

        results = []
        from rich.progress import Progress

        with Progress(console=console) as progress:
            task = progress.add_task("Generating...", total=len(prompts))

            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                outputs = engine.generate_batch(batch, gen_config)

                for prompt, output in zip(batch, outputs):
                    results.append({"prompt": prompt, "output": output})

                progress.advance(task, len(batch))

        # Save results
        with open(output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        console.print(success(f"Saved {len(results)} results to {output_file}"))

    except Exception as e:
        console.print(error(f"Batch inference failed: {e}"))
        raise typer.Exit(1)


@inference.command("benchmark")
def benchmark(
    model_path: Path = typer.Argument(..., help="Model checkpoint"),
    prompt_length: int = typer.Option(128, "--prompt-len", help="Input prompt length"),
    output_length: int = typer.Option(128, "--output-len", help="Output length"),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes", help="Batch sizes to test"),
    warmup: int = typer.Option(3, "--warmup", help="Warmup iterations"),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Benchmark iterations"),
    device: str = typer.Option("cuda", "--device", "-d"),
    quantize: Optional[str] = typer.Option(None, "--quantize", "-q"),
):
    """
    Benchmark inference performance.

    Example:
        complexity inference benchmark model.pt --batch-sizes 1,4,8,16
    """
    import time

    batch_list = [int(b) for b in batch_sizes.split(",")]

    try:
        with spinner("Loading model..."):
            from complexity.inference import InferenceEngine, GenerationConfig
            engine = InferenceEngine.from_checkpoint(model_path, device=device)

        console.print(success("Model loaded"))

        # Create dummy prompt
        dummy_prompt = "Hello " * (prompt_length // 2)
        gen_config = GenerationConfig(max_new_tokens=output_length)

        results = []

        for bs in batch_list:
            prompts = [dummy_prompt] * bs

            # Warmup
            console.print(info(f"Warming up batch size {bs}..."))
            for _ in range(warmup):
                engine.generate_batch(prompts, gen_config)

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                engine.generate_batch(prompts, gen_config)
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            tokens_per_sec = (bs * output_length) / avg_time

            results.append({
                "batch_size": bs,
                "avg_time": f"{avg_time:.3f}s",
                "tokens_per_sec": f"{tokens_per_sec:.1f}",
            })

        # Print results table
        from ..utils import print_table
        print_table(
            "Inference Benchmark Results",
            [
                {"name": "Batch Size", "style": "cyan"},
                {"name": "Avg Time", "style": "yellow"},
                {"name": "Tokens/sec", "style": "green"},
            ],
            [[r["batch_size"], r["avg_time"], r["tokens_per_sec"]] for r in results]
        )

    except Exception as e:
        console.print(error(f"Benchmark failed: {e}"))
        raise typer.Exit(1)
