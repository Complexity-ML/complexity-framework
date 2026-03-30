# Getting Started

## Installation

```bash
pip install complexity-framework

# Development
git clone https://github.com/Complexity-ML/complexity-framework.git
cd complexity-framework
pip install -e .
```

## First Model

```python
from complexity.models import ComplexityModel
from complexity.config import ModelConfig

# 187M Token-Routed model with Mu-Guidance
config = ModelConfig(
    hidden_size=768,
    num_hidden_layers=18,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2048,
    vocab_size=32000,
    mlp_type="token_routed",
    num_experts=4,
    shared_expert=True,
    use_mu_guidance=True,
)
model = ComplexityModel(config)
```

## Training

```bash
# Single GPU
python scripts/train_ablation_150m.py --run 2 --batch-size 128 --target-tokens 500000000

# Multi-GPU
torchrun --nproc_per_node=2 scripts/train_ablation_150m.py -- --run 2 --batch-size 128 --target-tokens 500000000
```

## Inference (vLLM)

```bash
# Deploy on vLLM
python -m vllm.entrypoints.openai.api_server --model /path/to/checkpoint --port 8081

# Query
curl http://localhost:8081/v1/completions -H "Content-Type: application/json" \
  -d '{"model": "/path/to/checkpoint", "prompt": "The meaning of life is", "max_tokens": 100, "temperature": 0.7}'
```

## Evaluation

```bash
# Zero-shot benchmarks (ARC-Easy, HellaSwag)
python scripts/eval_benchmarks.py --checkpoint checkpoints/run2-iso-shared/final --tasks arc_easy,hellaswag
```

## Architecture

![Architecture](../figures/architecture_complexity_deep.png)

## Next Steps

- [Token-Routed MLP](token-routed.md) - Deterministic MoE with sort-and-split dispatch
- [Mu-Guidance](dynamics.md) - Inter-layer communication
- [MoE Comparison](moe.md) - Token-Routed vs Mixtral
- [Training Guide](training.md) - Full training documentation
- [CUDA](cuda.md) - GPU optimizations
