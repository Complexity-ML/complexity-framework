# Complexity Framework

Modular Python framework for Transformer and Token-Routed language-model research.

```bash
pip install complexity-framework
```

## What is included

| Area | Description |
| --- | --- |
| Models | GPT/Llama-style decoder models with GQA, RMSNorm, RoPE, dense SwiGLU, and Token-Routed MLPs |
| Training | DDP-friendly local runners, BF16, gradient checkpointing, rotating checkpoints, CSV metrics |
| Tokenization | HuggingFace `tokenizers` folders and local `tiktoken`/`o200k_base` cache folders |
| Losses | Standard causal LM loss and chunked tied-head CE for large vocabularies |
| Utilities | Token-budget planning, checkpoint conversion, FineWeb-Edu streaming helpers |

## Quick Start

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    num_key_value_heads=2,
    vocab_size=200019,
    mlp_type="token_routed",
    num_experts=4,
    shared_expert=True,
    shared_intermediate_size=1536,
    intermediate_size=128,
    top_k=2,
    top_k_primary_weight=0.5,
)

model = ComplexityModel(config)
print(model.num_parameters())
```

## Local o200k Token-Routed Runs

After installing with `pip install -e .`, the package exposes stable console commands:

```bash
cf-plan-run --tokens 30B --gpus 8 --batch-size 256 --seq-len 2048 --tok-s 840000 --save-steps 200
```

```bash
cf-o200k-pretrain \
  --profile 100m \
  --dataset fineweb \
  --tokenizer ./tokenizer-o200k \
  --steps 7153 \
  --batch-size 256 \
  --seq-len 2048 \
  --bf16 \
  --eval-steps 200 \
  --log-steps 10 \
  --save-steps 200 \
  --loss-chunk-tokens 1024 \
  --run-name 30b-100m-o200k-tr
```

The `--profile` flag currently supports:

| Profile | Parameters with o200k | Use |
| --- | ---: | --- |
| `50m` | ~51.9M | quick local and ablation runs |
| `100m` | ~99.7M | main o200k Token-Routed research runs |

## Large-Vocabulary Training

For `o200k_base`, full `[batch, seq, vocab]` logits are too large for practical
training at high batch sizes. Use `--loss-chunk-tokens` to compute the exact same
tied-head cross entropy in token chunks without materializing the full logits tensor.

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
