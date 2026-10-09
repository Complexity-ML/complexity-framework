"""
Data module for framework-complexity.

Provides:
- Distributed DataLoaders
- Streaming datasets for large corpora
- Complexity tokenization format
- Chat templates with structured reasoning
- Tool calling helpers
- Data preprocessing pipelines

Complexity Format:
    Native tokenization format with 2048 reserved special tokens:
    - Structured reasoning: <|reason|>, <|step|>, <|conclude|>
    - Tool calling: <|call|>, <|args|>, <|result|>
    - Multi-modal: <|vision|>, <|audio|>, <|video|>

Usage:
    from complexity.data import (
        DistributedDataLoader,
        StreamingDataset,
        ComplexityTokenizer,
        ComplexityTemplate,
        ComplexityTokens,
    )

    # Create streaming dataset
    dataset = StreamingDataset(
        data_path="data/train.jsonl",
        tokenizer=tokenizer,
        seq_length=2048,
    )

    # Complexity tokenizer
    tokenizer = ComplexityTokenizer(
        base_tokenizer=base_tok,
        template=ComplexityTemplate(format="complexity"),
    )

    # Encode chat with reasoning mode
    tokens = tokenizer.encode_chat(
        messages=[{"role": "user", "content": "Solve this problem"}],
        enable_reasoning=True,
        enable_steps=True,
    )

"""

from importlib import import_module as _import_module

_EXPORTS = {
    'DistributedDataLoader': '.dataloader',
    'DataConfig': '.dataloader',
    'create_dataloader': '.dataloader',
    'StreamingDataset': '.streaming',
    'ShardedDataset': '.streaming',
    'InterleavedDataset': '.streaming',
    'TokenShardDataset': '.token_shards',
    'build_token_shard_from_texts': '.token_shards',
    'iter_texts_from_files': '.token_shards',
    'load_token_shard': '.token_shards',
    'token_shard_frequencies': '.token_shards',
    'write_token_shard': '.token_shards',
    'Tokenizer': '.tokenization',
    'BPETokenizer': '.tokenization',
    'SentencePieceTokenizer': '.tokenization',
    'TiktokenWrapper': '.tokenization',
    'HuggingFaceTokenizer': '.tokenization',
    'get_tokenizer': '.tokenization',
    'ComplexityTokens': '.tokenization',
    'ComplexityTemplate': '.tokenization',
    'ComplexityTokenizer': '.tokenization',
    'Message': '.tokenization',
    'TokenFormatMapping': '.tokenization',
    'CompatibleTokenizer': '.tokenization',
    'ToolDefinition': '.tokenization',
    'ToolCall': '.tokenization',
    'ToolCallingMixin': '.tokenization',
    'SpecialTokens': '.tokenization',
    'ChatTemplate': '.tokenization',
    'ChatTokenizer': '.tokenization',
    'TextPreprocessor': '.preprocessing',
    'DataPipeline': '.preprocessing',
    'PackedSequenceCollator': '.preprocessing',
}

__all__ = [
    # DataLoader
    "DistributedDataLoader",
    "DataConfig",
    "create_dataloader",
    # Streaming
    "StreamingDataset",
    "ShardedDataset",
    "InterleavedDataset",
    "TokenShardDataset",
    "build_token_shard_from_texts",
    "iter_texts_from_files",
    "load_token_shard",
    "token_shard_frequencies",
    "write_token_shard",
    # Base tokenizers
    "Tokenizer",
    "BPETokenizer",
    "SentencePieceTokenizer",
    "TiktokenWrapper",
    "HuggingFaceTokenizer",
    "get_tokenizer",
    # Complexity format (primary)
    "ComplexityTokens",
    "ComplexityTemplate",
    "ComplexityTokenizer",
    "Message",
    # Format compatibility (Llama, Mistral, GPT, Gemma)
    "TokenFormatMapping",
    "CompatibleTokenizer",
    # Tool calling
    "ToolDefinition",
    "ToolCall",
    "ToolCallingMixin",
    # Legacy aliases (backward compatibility)
    "SpecialTokens",
    "ChatTemplate",
    "ChatTokenizer",
    # Preprocessing
    "TextPreprocessor",
    "DataPipeline",
    "PackedSequenceCollator",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
