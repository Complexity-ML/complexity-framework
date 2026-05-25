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

from .dataloader import (
    DistributedDataLoader,
    DataConfig,
    create_dataloader,
)

from .streaming import (
    StreamingDataset,
    ShardedDataset,
    InterleavedDataset,
)

from .token_shards import (
    TokenShardDataset,
    build_token_shard_from_texts,
    iter_texts_from_files,
    load_token_shard,
    token_shard_frequencies,
    write_token_shard,
)

from .tokenization import (
    # Base tokenizers
    Tokenizer,
    BPETokenizer,
    SentencePieceTokenizer,
    TiktokenWrapper,
    HuggingFaceTokenizer,
    get_tokenizer,
    # Complexity format (primary)
    ComplexityTokens,
    ComplexityTemplate,
    ComplexityTokenizer,
    Message,
    # Format compatibility (Llama, Mistral, GPT, Gemma)
    TokenFormatMapping,
    CompatibleTokenizer,
    # Tool calling
    ToolDefinition,
    ToolCall,
    ToolCallingMixin,
    # Legacy aliases
    SpecialTokens,  # -> ComplexityTokens
    ChatTemplate,   # -> ComplexityTemplate
    ChatTokenizer,  # -> ComplexityTokenizer
)

from .preprocessing import (
    TextPreprocessor,
    DataPipeline,
    PackedSequenceCollator,
)

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
