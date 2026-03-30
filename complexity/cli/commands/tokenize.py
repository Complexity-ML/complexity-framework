"""
Tokenization commands for the Complexity framework.
"""

import typer
from pathlib import Path
from typing import Optional, List

from ..utils import console, spinner, print_panel, print_table, success, error, warning, info

tokenize = typer.Typer(name="tokenize", help="Tokenization commands")


# =============================================================================
# Format-specific special tokens
# =============================================================================

def _get_format_special_tokens(format_name: str) -> List[str]:
    """Get special tokens for a specific format."""

    # Llama 3 special tokens
    LLAMA3_TOKENS = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",
        # Llama 3.1+ tool calling
        "<|python_tag|>",
        "<|finetune_right_pad_id|>",
        "<|reserved_special_token_5|>",
    ]

    # Mistral special tokens
    MISTRAL_TOKENS = [
        "<s>",
        "</s>",
        "<unk>",
        "[INST]",
        "[/INST]",
        "[TOOL_CALLS]",
        "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]",
        "[TOOL_RESULTS]",
        "[/TOOL_RESULTS]",
    ]

    # ChatML (GPT-4 style) special tokens
    CHATML_TOKENS = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|pad|>",
    ]

    # Gemma special tokens
    GEMMA_TOKENS = [
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<start_of_turn>",
        "<end_of_turn>",
    ]

    # Qwen special tokens
    QWEN_TOKENS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]

    # INL Complexity format (full set)
    COMPLEXITY_TOKENS = []
    try:
        from complexity.data import ComplexityTokens
        ct = ComplexityTokens()
        COMPLEXITY_TOKENS = [
            # Core
            ct.unk_token, ct.bos_token, ct.eos_token, ct.pad_token,
            ct.sep_token, ct.mask_token, ct.cls_token, ct.newline_token,
            # Chat
            ct.turn_start, ct.turn_end, ct.user_start, ct.user_end,
            ct.assistant_start, ct.assistant_end, ct.system_start, ct.system_end,
            # Tools
            ct.tools_start, ct.tools_end, ct.tool_call, ct.tool_call_end,
            ct.tool_args, ct.tool_args_end, ct.tool_result, ct.tool_result_end,
            # Reasoning
            ct.reason_start, ct.reason_end, ct.step_start, ct.step_end,
            ct.conclude, ct.conclude_end, ct.reflect, ct.reflect_end,
            ct.verify, ct.verify_end, ct.plan, ct.plan_end,
            # Code
            ct.code_start, ct.code_end, ct.exec_start, ct.exec_end,
            ct.output_start, ct.output_end, ct.lang_python, ct.lang_bash,
            # Vision
            ct.vision_start, ct.vision_end, ct.image, ct.image_end,
            ct.patch, ct.bbox, ct.segment,
            # Audio
            ct.audio_start, ct.audio_end, ct.speech, ct.transcribe,
            # Robotics
            ct.state_start, ct.state_end, ct.action_start, ct.action_end,
            ct.goal_start, ct.goal_end, ct.trajectory_start, ct.trajectory_end,
            ct.proprio, ct.gripper, ct.joint,
            # Memory
            ct.memory_start, ct.memory_end, ct.retrieve, ct.retrieve_end,
            ct.store, ct.store_end, ct.cite, ct.cite_end,
        ]
    except ImportError:
        pass

    FORMAT_MAP = {
        "complexity": COMPLEXITY_TOKENS,
        "llama3": LLAMA3_TOKENS,
        "llama2": ["<s>", "</s>", "<unk>"],
        "mistral": MISTRAL_TOKENS,
        "chatml": CHATML_TOKENS,
        "gemma": GEMMA_TOKENS,
        "qwen2": QWEN_TOKENS,
        "gpt2": ["<|endoftext|>"],
    }

    return FORMAT_MAP.get(format_name, COMPLEXITY_TOKENS)


# =============================================================================
# Training Tokenizers
# =============================================================================

@tokenize.command("train")
def train_tokenizer(
    data_path: Path = typer.Argument(..., help="Training data (file or directory)"),
    output_path: Path = typer.Argument(..., help="Output tokenizer path"),
    vocab_size: int = typer.Option(32000, "--vocab-size", "-v", help="Vocabulary size"),
    method: str = typer.Option("bpe", "--method", "-m", help="Method: bpe, unigram, wordpiece"),
    base_format: str = typer.Option("complexity", "--format", "-f", help="Token format: complexity, llama3, mistral, chatml, gemma"),
    min_frequency: int = typer.Option(2, "--min-freq", help="Minimum token frequency"),
    special_tokens: Optional[str] = typer.Option(None, "--special-tokens", "-s", help="Comma-separated special tokens"),
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File pattern for directories"),
    sample_size: Optional[int] = typer.Option(None, "--sample", help="Sample N files (for large datasets)"),
    lowercase: bool = typer.Option(False, "--lowercase", "-l", help="Lowercase all text"),
    normalize_unicode: bool = typer.Option(True, "--normalize/--no-normalize", help="Unicode normalization"),
    num_workers: int = typer.Option(4, "--workers", "-w", help="Number of workers"),
):
    """
    Train a new tokenizer from scratch.

    Supports multiple token formats:
    - complexity: INL native format (recommended for new models)
    - llama3: Meta Llama 3 format
    - mistral: Mistral/Mixtral format
    - chatml: OpenAI ChatML format (GPT-4 style)
    - gemma: Google Gemma format

    Examples:
        # Train with Complexity format (default)
        complexity tokenize train ./corpus/ my_tokenizer --vocab-size 50000

        # Train Llama 3 compatible tokenizer
        complexity tokenize train ./corpus/ llama_tok --format llama3 --vocab-size 128000

        # Train Mistral compatible tokenizer
        complexity tokenize train ./corpus/ mistral_tok --format mistral --vocab-size 32000
    """
    import json

    if not data_path.exists():
        console.print(error(f"Data path not found: {data_path}"))
        raise typer.Exit(1)

    # Collect files
    if data_path.is_file():
        files = [data_path]
    else:
        files = list(data_path.rglob(pattern))
        if sample_size and len(files) > sample_size:
            import random
            files = random.sample(files, sample_size)

    if not files:
        console.print(error(f"No files found matching pattern: {pattern}"))
        raise typer.Exit(1)

    console.print(info(f"Found {len(files)} files for training"))

    # Parse special tokens
    base_special_tokens = []
    if special_tokens:
        base_special_tokens = [t.strip() for t in special_tokens.split(",")]

    # Get special tokens based on format
    format_special_tokens = _get_format_special_tokens(base_format)
    base_special_tokens = format_special_tokens + base_special_tokens
    console.print(info(f"Using {base_format} format with {len(format_special_tokens)} special tokens"))

    try:
        # Try tokenizers library first (HuggingFace)
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

        console.print(info(f"Training {method.upper()} tokenizer with vocab size {vocab_size}..."))

        # Create normalizer
        normalizer_list = []
        if normalize_unicode:
            normalizer_list.append(normalizers.NFD())
            normalizer_list.append(normalizers.StripAccents())
        if lowercase:
            normalizer_list.append(normalizers.Lowercase())

        # Create model and trainer based on method
        if method == "bpe":
            model = models.BPE(unk_token="<|unk|>")
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=base_special_tokens,
                show_progress=True,
            )
        elif method == "unigram":
            model = models.Unigram()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=base_special_tokens,
                show_progress=True,
            )
        elif method == "wordpiece":
            model = models.WordPiece(unk_token="<|unk|>")
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=base_special_tokens,
                show_progress=True,
            )
        else:
            console.print(error(f"Unknown method: {method}. Use: bpe, unigram, wordpiece"))
            raise typer.Exit(1)

        # Create tokenizer
        tokenizer = Tokenizer(model)

        if normalizer_list:
            tokenizer.normalizer = normalizers.Sequence(normalizer_list)

        # Pre-tokenizer (split on whitespace and punctuation)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation(),
        ])

        # Decoder
        if method == "bpe":
            tokenizer.decoder = decoders.BPEDecoder()
        elif method == "wordpiece":
            tokenizer.decoder = decoders.WordPiece()

        # Train
        with spinner(f"Training on {len(files)} files..."):
            tokenizer.train([str(f) for f in files], trainer)

        # Save
        output_path = Path(output_path)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".json")

        tokenizer.save(str(output_path))

        # Also save config
        config_path = output_path.with_suffix(".config.json")
        config = {
            "method": method,
            "vocab_size": tokenizer.get_vocab_size(),
            "format": base_format,
            "special_tokens": base_special_tokens,
            "lowercase": lowercase,
            "normalize_unicode": normalize_unicode,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print_panel(
            f"Method: {method.upper()}\n"
            f"Format: {base_format}\n"
            f"Vocab size: {tokenizer.get_vocab_size():,}\n"
            f"Special tokens: {len(base_special_tokens)}\n"
            f"Output: {output_path}\n"
            f"Config: {config_path}",
            title="Tokenizer Training Complete",
            style="green"
        )

    except ImportError:
        console.print(error("tokenizers library not installed"))
        console.print(info("Install with: pip install tokenizers"))
        raise typer.Exit(1)


@tokenize.command("train-sentencepiece")
def train_sentencepiece(
    data_path: Path = typer.Argument(..., help="Training data file"),
    output_prefix: str = typer.Argument(..., help="Output prefix (creates .model and .vocab)"),
    vocab_size: int = typer.Option(32000, "--vocab-size", "-v"),
    model_type: str = typer.Option("bpe", "--type", "-t", help="Model type: bpe, unigram, char, word"),
    character_coverage: float = typer.Option(0.9995, "--coverage", "-c", help="Character coverage"),
    num_threads: int = typer.Option(4, "--threads"),
    max_sentence_length: int = typer.Option(4192, "--max-len"),
    add_complexity_tokens: bool = typer.Option(True, "--complexity-tokens/--no-complexity-tokens"),
):
    """
    Train a SentencePiece tokenizer.

    Example:
        complexity tokenize train-sentencepiece corpus.txt my_sp --vocab-size 50000
    """
    if not data_path.exists():
        console.print(error(f"Data not found: {data_path}"))
        raise typer.Exit(1)

    try:
        import sentencepiece as spm

        # Build user-defined symbols
        user_symbols = []
        if add_complexity_tokens:
            from complexity.data import ComplexityTokens
            ct = ComplexityTokens()
            user_symbols = [
                ct.bos_token, ct.eos_token, ct.pad_token,
                ct.user_start, ct.assistant_start, ct.system_start,
                ct.call_start, ct.call_end, ct.args_start, ct.result_start,
                ct.reason_start, ct.reason_end, ct.step_start, ct.conclude,
                ct.think_start, ct.think_end,
                ct.code_start, ct.code_end,
                ct.image_start, ct.image_end,
                ct.state_start, ct.action_start, ct.trajectory_start,
            ]

        console.print(info(f"Training SentencePiece {model_type} with vocab size {vocab_size}..."))

        with spinner("Training..."):
            spm.SentencePieceTrainer.train(
                input=str(data_path),
                model_prefix=output_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                num_threads=num_threads,
                max_sentence_length=max_sentence_length,
                user_defined_symbols=user_symbols if user_symbols else None,
                pad_id=3,
                unk_id=0,
                bos_id=1,
                eos_id=2,
            )

        print_panel(
            f"Model type: {model_type}\n"
            f"Vocab size: {vocab_size:,}\n"
            f"Output: {output_prefix}.model, {output_prefix}.vocab",
            title="SentencePiece Training Complete",
            style="green"
        )

    except ImportError:
        console.print(error("sentencepiece not installed"))
        console.print(info("Install with: pip install sentencepiece"))
        raise typer.Exit(1)


@tokenize.command("extend")
def extend_tokenizer(
    tokenizer_path: Path = typer.Argument(..., help="Existing tokenizer"),
    output_path: Path = typer.Argument(..., help="Output path"),
    new_tokens: Optional[str] = typer.Option(None, "--tokens", "-t", help="Comma-separated new tokens"),
    tokens_file: Optional[Path] = typer.Option(None, "--file", "-f", help="File with tokens (one per line)"),
    add_complexity_tokens: bool = typer.Option(False, "--complexity-tokens", help="Add Complexity special tokens"),
):
    """
    Extend an existing tokenizer with new tokens.

    Examples:
        complexity tokenize extend tokenizer.json new_tok.json --tokens "<custom1>,<custom2>"
        complexity tokenize extend tokenizer.json new_tok.json --file new_tokens.txt
        complexity tokenize extend gpt2 extended_tok --complexity-tokens
    """
    tokens_to_add = []

    if new_tokens:
        tokens_to_add.extend([t.strip() for t in new_tokens.split(",")])

    if tokens_file and tokens_file.exists():
        tokens_to_add.extend([
            line.strip() for line in tokens_file.read_text().splitlines()
            if line.strip()
        ])

    if add_complexity_tokens:
        from complexity.data import ComplexityTokens
        ct = ComplexityTokens()
        complexity_specials = [
            ct.bos_token, ct.eos_token, ct.pad_token,
            ct.user_start, ct.assistant_start, ct.system_start,
            ct.call_start, ct.call_end, ct.args_start, ct.result_start,
            ct.reason_start, ct.step_start, ct.conclude,
            ct.think_start, ct.think_end,
            ct.code_start, ct.code_end,
            ct.state_start, ct.action_start,
        ]
        tokens_to_add.extend(complexity_specials)

    if not tokens_to_add:
        console.print(error("No tokens to add. Use --tokens, --file, or --complexity-tokens"))
        raise typer.Exit(1)

    try:
        # Try loading with tokenizers library
        from tokenizers import Tokenizer

        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            # Try as a pretrained name
            from tokenizers import Tokenizer
            console.print(error(f"Tokenizer not found: {tokenizer_path}"))
            raise typer.Exit(1)

        # Get current vocab
        current_vocab = set(tokenizer.get_vocab().keys())

        # Filter out existing tokens
        new_tokens_filtered = [t for t in tokens_to_add if t not in current_vocab]

        if not new_tokens_filtered:
            console.print(warning("All tokens already exist in vocabulary"))
            return

        # Add new tokens
        num_added = tokenizer.add_tokens(new_tokens_filtered)

        # Save
        output_path = Path(output_path)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".json")

        tokenizer.save(str(output_path))

        print_panel(
            f"Added {num_added} new tokens\n"
            f"New vocab size: {tokenizer.get_vocab_size():,}\n"
            f"Output: {output_path}",
            title="Tokenizer Extended",
            style="green"
        )

    except ImportError:
        console.print(error("tokenizers library not installed"))
        raise typer.Exit(1)


@tokenize.command("merge")
def merge_tokenizers(
    base_tokenizer: Path = typer.Argument(..., help="Base tokenizer"),
    other_tokenizer: Path = typer.Argument(..., help="Tokenizer to merge from"),
    output_path: Path = typer.Argument(..., help="Output path"),
    num_tokens: int = typer.Option(1000, "--num", "-n", help="Number of tokens to add from other"),
    by_frequency: bool = typer.Option(True, "--by-frequency/--alphabetical", help="Select by frequency"),
):
    """
    Merge tokens from one tokenizer into another.

    Example:
        complexity tokenize merge base.json code_tok.json merged.json --num 5000
    """
    try:
        from tokenizers import Tokenizer

        base = Tokenizer.from_file(str(base_tokenizer))
        other = Tokenizer.from_file(str(other_tokenizer))

        base_vocab = set(base.get_vocab().keys())
        other_vocab = other.get_vocab()

        # Get tokens not in base
        new_tokens = {k: v for k, v in other_vocab.items() if k not in base_vocab}

        if by_frequency:
            # Sort by ID (lower ID = more frequent in BPE)
            sorted_tokens = sorted(new_tokens.keys(), key=lambda x: new_tokens[x])
        else:
            sorted_tokens = sorted(new_tokens.keys())

        tokens_to_add = sorted_tokens[:num_tokens]

        num_added = base.add_tokens(tokens_to_add)

        output_path = Path(output_path)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".json")

        base.save(str(output_path))

        print_panel(
            f"Added {num_added} tokens from {other_tokenizer.name}\n"
            f"New vocab size: {base.get_vocab_size():,}\n"
            f"Output: {output_path}",
            title="Tokenizers Merged",
            style="green"
        )

    except ImportError:
        console.print(error("tokenizers library not installed"))
        raise typer.Exit(1)


# =============================================================================
# Encoding / Decoding
# =============================================================================

@tokenize.command("encode")
def encode_text(
    text: str = typer.Argument(None, help="Text to encode"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File to encode"),
    tokenizer: str = typer.Option("tiktoken", "--tokenizer", "-t", help="Tokenizer: tiktoken, sentencepiece, hf, or path"),
    model: str = typer.Option("gpt2", "--model", "-m", help="Tokenizer model name"),
    show_tokens: bool = typer.Option(False, "--show", "-s", help="Show individual tokens"),
):
    """
    Encode text to tokens.

    Examples:
        complexity tokenize encode "Hello, world!"
        complexity tokenize encode -f document.txt --show
        complexity tokenize encode "Test" --tokenizer ./my_tokenizer.json
    """
    if text is None and file is None:
        console.print(error("Provide text argument or --file"))
        raise typer.Exit(1)

    if file:
        if not file.exists():
            console.print(error(f"File not found: {file}"))
            raise typer.Exit(1)
        text = file.read_text()

    try:
        # Check if tokenizer is a path
        tokenizer_path = Path(tokenizer)
        if tokenizer_path.exists():
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tokenizer_path))
            encoding = tok.encode(text)
            tokens = encoding.ids
            decode_fn = lambda ids: tok.decode(ids)
            decode_single = lambda id: tok.decode([id])
        else:
            from complexity.data import get_tokenizer
            tok = get_tokenizer(tokenizer, model_name=model)
            tokens = tok.encode(text)
            decode_fn = tok.decode
            decode_single = lambda id: tok.decode([id])

        print_panel(
            f"Text length: {len(text)} chars\n"
            f"Token count: {len(tokens)}\n"
            f"Ratio: {len(tokens)/max(len(text),1):.2f} tokens/char",
            title="Encoding Result",
            style="cyan"
        )

        if show_tokens:
            console.print("\n[bold]Token breakdown:[/bold]")
            for i, tok_id in enumerate(tokens[:100]):
                tok_str = decode_single(tok_id)
                console.print(f"  {i:4d}: {tok_id:6d} -> {repr(tok_str)}")

            if len(tokens) > 100:
                console.print(f"  ... ({len(tokens) - 100} more tokens)")

        console.print(f"\n[dim]Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}[/dim]")

    except Exception as e:
        console.print(error(f"Encoding failed: {e}"))
        raise typer.Exit(1)


@tokenize.command("decode")
def decode_tokens(
    tokens: str = typer.Argument(..., help="Comma-separated token IDs"),
    tokenizer: str = typer.Option("tiktoken", "--tokenizer", "-t"),
    model: str = typer.Option("gpt2", "--model", "-m"),
):
    """
    Decode tokens back to text.

    Example:
        complexity tokenize decode "15496,11,995"
    """
    try:
        token_ids = [int(t.strip()) for t in tokens.split(",")]
    except ValueError:
        console.print(error("Invalid token format. Use comma-separated integers."))
        raise typer.Exit(1)

    try:
        tokenizer_path = Path(tokenizer)
        if tokenizer_path.exists():
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tokenizer_path))
            text = tok.decode(token_ids)
        else:
            from complexity.data import get_tokenizer
            tok = get_tokenizer(tokenizer, model_name=model)
            text = tok.decode(token_ids)

        console.print(f"[bold]Decoded text:[/bold]\n{text}")

    except Exception as e:
        console.print(error(f"Decoding failed: {e}"))
        raise typer.Exit(1)


@tokenize.command("count")
def count_tokens(
    path: Path = typer.Argument(..., help="File or directory to count"),
    tokenizer: str = typer.Option("tiktoken", "--tokenizer", "-t"),
    model: str = typer.Option("gpt2", "--model", "-m"),
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File pattern for directories"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r"),
):
    """
    Count tokens in files.

    Examples:
        complexity tokenize count document.txt
        complexity tokenize count ./data/ --pattern "*.jsonl" --recursive
    """
    # Load tokenizer
    tokenizer_path = Path(tokenizer)
    if tokenizer_path.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        encode_fn = lambda t: tok.encode(t).ids
    else:
        from complexity.data import get_tokenizer
        tok = get_tokenizer(tokenizer, model_name=model)
        encode_fn = tok.encode

    files = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
    else:
        console.print(error(f"Path not found: {path}"))
        raise typer.Exit(1)

    if not files:
        console.print(warning(f"No files matching pattern: {pattern}"))
        return

    total_tokens = 0
    total_chars = 0
    file_stats = []

    from rich.progress import Progress

    with Progress(console=console) as progress:
        task = progress.add_task("Counting tokens...", total=len(files))

        for f in files:
            try:
                text = f.read_text()
                tokens = encode_fn(text)
                total_tokens += len(tokens)
                total_chars += len(text)
                file_stats.append((f.name, len(text), len(tokens)))
            except Exception:
                file_stats.append((f.name, "error", "error"))

            progress.advance(task)

    if len(files) > 1:
        print_table(
            f"Token Counts ({len(files)} files)",
            [
                {"name": "File", "style": "cyan"},
                {"name": "Chars", "style": "yellow"},
                {"name": "Tokens", "style": "green"},
            ],
            file_stats[:20]
        )

        if len(files) > 20:
            console.print(f"[dim]... and {len(files) - 20} more files[/dim]")

    print_panel(
        f"Total files: {len(files)}\n"
        f"Total characters: {total_chars:,}\n"
        f"Total tokens: {total_tokens:,}\n"
        f"Average tokens/char: {total_tokens/max(total_chars,1):.3f}",
        title="Summary",
        style="green"
    )


@tokenize.command("preprocess")
def preprocess_data(
    input_path: Path = typer.Argument(..., help="Input file or directory"),
    output_path: Path = typer.Argument(..., help="Output file (.bin)"),
    tokenizer: str = typer.Option("tiktoken", "--tokenizer", "-t"),
    model: str = typer.Option("gpt2", "--model", "-m"),
    seq_length: int = typer.Option(2048, "--seq-length", "-s", help="Sequence length"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Input format: jsonl, txt, parquet"),
    text_key: str = typer.Option("text", "--key", "-k", help="JSON key for text field"),
    num_workers: int = typer.Option(4, "--workers", "-w", help="Number of workers"),
):
    """
    Preprocess data into binary token format.

    Example:
        complexity tokenize preprocess data.jsonl train.bin --seq-length 4096
    """
    import json
    import numpy as np

    # Load tokenizer
    tokenizer_path = Path(tokenizer)
    if tokenizer_path.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        encode_fn = lambda t: tok.encode(t).ids
        eos_id = tok.token_to_id("<|end|>") or tok.token_to_id("</s>") or 2
    else:
        from complexity.data import get_tokenizer
        tok = get_tokenizer(tokenizer, model_name=model)
        encode_fn = tok.encode
        eos_id = getattr(tok, 'eos_token_id', 2)

    # Collect files
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.rglob(f"*.{format}"))

    console.print(info(f"Processing {len(files)} files..."))

    # Load texts
    texts = []
    for f in files:
        if format == "jsonl":
            with open(f) as fp:
                for line in fp:
                    data = json.loads(line)
                    texts.append(data.get(text_key, ""))
        elif format == "txt":
            texts.append(f.read_text())
        elif format == "parquet":
            import pandas as pd
            df = pd.read_parquet(f)
            texts.extend(df[text_key].tolist())

    console.print(info(f"Loaded {len(texts)} documents"))

    # Tokenize
    all_tokens = []

    from rich.progress import Progress

    with Progress(console=console) as progress:
        task = progress.add_task("Tokenizing...", total=len(texts))

        for text in texts:
            tokens = encode_fn(text)
            all_tokens.extend(tokens)
            all_tokens.append(eos_id)
            progress.advance(task)

    # Save as numpy
    all_tokens = np.array(all_tokens, dtype=np.uint32)
    n_seqs = len(all_tokens) // seq_length
    all_tokens = all_tokens[:n_seqs * seq_length]

    all_tokens.tofile(output_path)

    print_panel(
        f"Total tokens: {len(all_tokens):,}\n"
        f"Sequences: {n_seqs:,}\n"
        f"Sequence length: {seq_length}\n"
        f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB",
        title="Preprocessing Complete",
        style="green"
    )


# =============================================================================
# Inspection
# =============================================================================

@tokenize.command("info")
def tokenizer_info(
    tokenizer_path: Path = typer.Argument(..., help="Path to tokenizer"),
    show_vocab: bool = typer.Option(False, "--vocab", "-v", help="Show vocabulary sample"),
    show_special: bool = typer.Option(True, "--special/--no-special", help="Show special tokens"),
):
    """
    Show information about a tokenizer.

    Example:
        complexity tokenize info my_tokenizer.json --vocab
    """
    if not tokenizer_path.exists():
        console.print(error(f"Tokenizer not found: {tokenizer_path}"))
        raise typer.Exit(1)

    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))

        vocab = tok.get_vocab()
        vocab_size = tok.get_vocab_size()

        print_panel(
            f"Path: {tokenizer_path}\n"
            f"Vocab size: {vocab_size:,}",
            title="Tokenizer Info",
            style="cyan"
        )

        if show_special:
            # Find special tokens (tokens with < and >)
            special = [k for k in vocab.keys() if k.startswith("<") and k.endswith(">")]
            special = sorted(special, key=lambda x: vocab[x])

            if special:
                console.print("\n[bold]Special Tokens:[/bold]")
                for t in special[:50]:
                    console.print(f"  {vocab[t]:5d}: {t}")
                if len(special) > 50:
                    console.print(f"  ... ({len(special) - 50} more)")

        if show_vocab:
            console.print("\n[bold]Vocabulary Sample (first 50):[/bold]")
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            for token, idx in sorted_vocab[:50]:
                console.print(f"  {idx:5d}: {repr(token)}")

    except Exception as e:
        console.print(error(f"Failed to load tokenizer: {e}"))
        raise typer.Exit(1)


@tokenize.command("special-tokens")
def show_special_tokens(
    format: str = typer.Option("complexity", "--format", "-f", help="Token format: complexity, chatml, llama"),
):
    """
    Show special tokens for a format.

    Example:
        complexity tokenize special-tokens --format complexity
    """
    from complexity.data import ComplexityTokens

    if format == "complexity":
        tokens = ComplexityTokens()

        categories = {
            "Core (0-31)": ["unk_token", "bos_token", "eos_token", "pad_token", "sep_token", "mask_token"],
            "Chat (32-63)": ["turn_start", "turn_end", "user_start", "assistant_start", "system_start"],
            "Tools (64-127)": ["call_start", "call_end", "args_start", "result_start", "tool_start", "tool_end"],
            "Reasoning (128-191)": ["reason_start", "reason_end", "step_start", "step_end", "conclude", "think_start", "think_end"],
            "Code (192-255)": ["code_start", "code_end", "output_start", "exec_start", "func_start", "class_start"],
            "Vision (256-287)": ["image_start", "image_end", "vision_start", "bbox_start", "caption_start"],
            "Robotics (384-511)": ["state_start", "action_start", "trajectory_start", "proprio_start", "gripper_start"],
        }

        for cat_name, attrs in categories.items():
            console.print(f"\n[bold cyan]{cat_name}[/bold cyan]")
            for attr in attrs:
                if hasattr(tokens, attr):
                    val = getattr(tokens, attr)
                    console.print(f"  {attr:20s} : {val}")
    else:
        console.print(warning(f"Format '{format}' not implemented, showing complexity format"))
        show_special_tokens(format="complexity")


@tokenize.command("benchmark")
def benchmark_tokenizer(
    tokenizer_path: Path = typer.Argument(..., help="Tokenizer to benchmark"),
    data_path: Optional[Path] = typer.Option(None, "--data", "-d", help="Test data file"),
    iterations: int = typer.Option(100, "--iterations", "-n"),
):
    """
    Benchmark tokenizer speed.

    Example:
        complexity tokenize benchmark my_tokenizer.json --data test.txt
    """
    import time

    if not tokenizer_path.exists():
        console.print(error(f"Tokenizer not found: {tokenizer_path}"))
        raise typer.Exit(1)

    # Load test text
    if data_path and data_path.exists():
        test_text = data_path.read_text()
    else:
        test_text = "Hello, world! " * 100 + "This is a test of tokenizer performance. " * 50

    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))

        # Warmup
        for _ in range(10):
            tok.encode(test_text)

        # Benchmark encoding
        start = time.perf_counter()
        for _ in range(iterations):
            encoded = tok.encode(test_text)
        encode_time = time.perf_counter() - start

        # Benchmark decoding
        start = time.perf_counter()
        for _ in range(iterations):
            tok.decode(encoded.ids)
        decode_time = time.perf_counter() - start

        tokens_per_iter = len(encoded.ids)
        chars_per_iter = len(test_text)

        print_panel(
            f"Text length: {chars_per_iter:,} chars\n"
            f"Tokens: {tokens_per_iter:,}\n"
            f"Iterations: {iterations}\n\n"
            f"Encode: {encode_time*1000/iterations:.3f} ms/iter\n"
            f"Decode: {decode_time*1000/iterations:.3f} ms/iter\n"
            f"Encode throughput: {chars_per_iter*iterations/encode_time/1e6:.2f} M chars/s\n"
            f"Decode throughput: {tokens_per_iter*iterations/decode_time/1e6:.2f} M tokens/s",
            title="Tokenizer Benchmark",
            style="green"
        )

    except Exception as e:
        console.print(error(f"Benchmark failed: {e}"))
        raise typer.Exit(1)
