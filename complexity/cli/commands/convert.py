"""
Model conversion commands for the Complexity framework.
"""

from pathlib import Path
from typing import Optional

import typer

from complexity._optional import require_optional

from ..utils import console, error, info, print_panel, spinner, success, warning

convert = typer.Typer(name="convert", help="Model conversion commands")


@convert.command("to-gguf")
def convert_to_gguf(
    input_path: Path = typer.Argument(..., help="Input model path"),
    output_path: Path = typer.Argument(..., help="Output GGUF file"),
    quantization: str = typer.Option("q4_k_m", "--quant", "-q", help="Quantization: f16, q8_0, q4_k_m, q4_0"),
):
    """
    Convert model to GGUF format for llama.cpp.

    Example:
        complexity convert to-gguf model.pt model.gguf --quant q4_k_m
    """
    if not input_path.exists():
        console.print(error(f"Input not found: {input_path}"))
        raise typer.Exit(1)

    console.print(info(f"Converting to GGUF with {quantization} quantization..."))
    console.print(warning("GGUF conversion requires llama.cpp convert script"))

    # This would integrate with llama.cpp's convert script
    print_panel(
        f"Input: {input_path}\n"
        f"Output: {output_path}\n"
        f"Quantization: {quantization}\n\n"
        "To convert manually:\n"
        "  python llama.cpp/convert.py model_dir/ --outtype f16\n"
        "  ./quantize model.gguf model-q4.gguf q4_k_m",
        title="GGUF Conversion",
        style="yellow"
    )


@convert.command("to-safetensors")
def convert_to_safetensors(
    input_path: Path = typer.Argument(..., help="Input .pt or .bin file"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
):
    """
    Convert PyTorch checkpoint to safetensors format.

    Example:
        complexity convert to-safetensors model.pt
    """
    if not input_path.exists():
        console.print(error(f"Input not found: {input_path}"))
        raise typer.Exit(1)

    if output_path is None:
        output_path = input_path.with_suffix(".safetensors")

    try:
        import torch
        from safetensors.torch import save_file

        with spinner("Loading checkpoint..."):
            state_dict = torch.load(input_path, map_location="cpu")

            # Handle nested state dicts
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        with spinner("Saving safetensors..."):
            save_file(state_dict, output_path)

        console.print(success(f"Saved to: {output_path}"))

        # Show size comparison
        orig_size = input_path.stat().st_size / 1e6
        new_size = output_path.stat().st_size / 1e6
        console.print(info(f"Size: {orig_size:.1f} MB -> {new_size:.1f} MB"))

    except ImportError as e:
        console.print(error(f"Missing dependency: {e}"))
        console.print(info("Install with: pip install safetensors"))
        raise typer.Exit(1)


@convert.command("from-hf")
def convert_from_huggingface(
    model_id: str = typer.Argument(..., help="HuggingFace model ID"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    revision: Optional[str] = typer.Option(None, "--revision", "-r", help="Model revision"),
    token: Optional[str] = typer.Option(None, "--token", help="HuggingFace token"),
):
    """
    Convert HuggingFace model to Complexity format.

    Example:
        complexity convert from-hf meta-llama/Llama-2-7b ./llama2-7b
    """
    try:
        import torch
        require_optional("transformers", "hf")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        output_dir.mkdir(parents=True, exist_ok=True)

        with spinner(f"Downloading {model_id}..."):
            config = AutoConfig.from_pretrained(model_id, revision=revision, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                token=token,
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)

        console.print(success("Model downloaded"))

        # Save in our format
        with spinner("Converting to Complexity format..."):
            # Save config
            import yaml
            complexity_config = {
                "model": {
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_hidden_layers,
                    "num_heads": config.num_attention_heads,
                    "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                    "vocab_size": config.vocab_size,
                    "max_seq_length": getattr(config, "max_position_embeddings", 4096),
                    "rope_theta": getattr(config, "rope_theta", 10000),
                },
                "source": {
                    "hf_model_id": model_id,
                    "revision": revision,
                }
            }

            with open(output_dir / "config.yaml", "w") as f:
                yaml.dump(complexity_config, f)

            # Save weights
            torch.save(model.state_dict(), output_dir / "model.pt")

            # Save tokenizer
            tokenizer.save_pretrained(output_dir / "tokenizer")

        console.print(success(f"Converted to: {output_dir}"))

        print_panel(
            f"Model: {model_id}\n"
            f"Hidden size: {config.hidden_size}\n"
            f"Layers: {config.num_hidden_layers}\n"
            f"Heads: {config.num_attention_heads}\n"
            f"Vocab: {config.vocab_size}",
            title="Conversion Complete",
            style="green"
        )

    except ImportError as e:
        console.print(error(f"Missing dependency: {e}"))
        console.print(info("Install with: pip install \"complexity-framework[hf]\""))
        raise typer.Exit(1)


@convert.command("to-hf")
def convert_to_huggingface(
    input_dir: Path = typer.Argument(..., help="Native config.json + model.safetensors directory"),
    output_dir: Path = typer.Argument(..., help="Standalone Transformers bundle directory"),
    tokenizer_dir: Optional[Path] = typer.Option(None, "--tokenizer", help="Tokenizer assets to copy"),
    force: bool = typer.Option(False, "--force", help="Replace files in an existing output bundle"),
):
    """Export the supported TR-HASH top-2 GQA family through its own adapter.

    This copies native weights and includes model code; it does not translate
    the architecture into a different model family or publish to the Hub.
    """
    from complexity.export.transformers import export_bundle

    try:
        if output_dir.exists() and any(output_dir.iterdir()) and not force:
            raise ValueError("Output directory is not empty; choose a new directory or use --force")
        if tokenizer_dir is None and (input_dir / "tokenizer.json").is_file():
            tokenizer_dir = input_dir
        export_bundle(
            config_path=input_dir / "config.json",
            weights_path=input_dir / "model.safetensors",
            output=output_dir,
            tokenizer_dir=tokenizer_dir,
        )
    except (OSError, ValueError, ImportError) as exc:
        console.print(error(str(exc)))
        raise typer.Exit(1) from exc
    console.print(success(f"Transformers bundle written to: {output_dir}"))


@convert.command("quantize")
def quantize_model(
    input_path: Path = typer.Argument(..., help="Input model"),
    output_path: Path = typer.Argument(..., help="Output path"),
    method: str = typer.Option("int8", "--method", "-m", help="int8, int4, awq, gptq"),
    calibration_data: Optional[Path] = typer.Option(None, "--calibration", "-c", help="Calibration data"),
):
    """
    Quantize a model.

    Examples:
        complexity convert quantize model.pt model-int8.pt --method int8
        complexity convert quantize model.pt model-awq.pt --method awq --calibration data.jsonl
    """
    if not input_path.exists():
        console.print(error(f"Input not found: {input_path}"))
        raise typer.Exit(1)

    if method in ("awq", "gptq") and calibration_data is None:
        console.print(warning(f"{method.upper()} works best with calibration data"))

    try:
        import torch

        from complexity.quantization import quantize_model as do_quantize

        with spinner("Loading model..."):
            state_dict = torch.load(input_path, map_location="cpu")

        with spinner(f"Quantizing with {method}..."):
            quantized = do_quantize(state_dict, method=method)

        with spinner("Saving..."):
            torch.save(quantized, output_path)

        orig_size = input_path.stat().st_size / 1e6
        new_size = output_path.stat().st_size / 1e6
        ratio = new_size / orig_size

        print_panel(
            f"Method: {method}\n"
            f"Original: {orig_size:.1f} MB\n"
            f"Quantized: {new_size:.1f} MB\n"
            f"Compression: {ratio:.2f}x",
            title="Quantization Complete",
            style="green"
        )

    except Exception as e:
        console.print(error(f"Quantization failed: {e}"))
        raise typer.Exit(1)
