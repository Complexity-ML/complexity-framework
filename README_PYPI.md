# Complexity Framework

PyTorch research framework for TR-GQA, TR-MHA, and deterministic
Token-Routed Mixture-of-Experts (TR-MoE) language models.

## Architecture vocabulary

- **TR-GQA**: grouped-query attention + TR-MoE.
- **TR-MHA**: multi-head attention + TR-MoE.
- **TR-MoE**: dense shared SwiGLU + deterministic token-selected experts.

Token identity selects expert parameters while every selected branch transforms
the current contextual hidden state.

## Installation

Install the PyTorch build for the target CPU, CUDA, ROCm, or MPS environment
first. The package deliberately does not choose a PyTorch wheel.

```bash
pip install torch
pip install complexity-framework
```

Development tools and the optional MCP client:

```bash
pip install -e ".[dev,tools]"
```

## Build TR-GQA

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    num_key_value_heads=2,
    attention_type="gqa",
    vocab_size=200_019,
    mlp_type="token_routed",
    num_experts=4,
    intermediate_size=128,
    shared_expert=True,
    shared_intermediate_size=1536,
    routing_strategy="zipf",
    top_k=2,
    top_k_primary_weight=0.5,
)
model = ComplexityModel(config)
```

For TR-MHA, set `attention_type="mha"` and make
`num_key_value_heads == num_attention_heads`.

## Train

```bash
cf-o200k-pretrain \
  --config configs/run_configs/100m_o200k_tr_rocm_mi350x.yaml
```

The runner records the resolved arguments, model configuration, parameter
count, Git commit, backend metadata, and token accounting in
`runs/<run-name>/run_config.json`.

## Inference

Native `ComplexityModel.generate()` is disabled. Exported checkpoints are
served by a compatible external runtime such as vLLM or SGLang and accessed
through the OpenAI-compatible client.

```python
from complexity.inference import ExternalGenerationConfig, create_external_backend

backend = create_external_backend(
    "vllm",
    base_url="http://localhost:8000",
    model="my-model",
)
text = backend.complete(
    "A computer program is",
    ExternalGenerationConfig(max_tokens=128),
)
```

Full documentation:
<https://github.com/Complexity-ML/complexity-framework/tree/main/docs>

## License

CC BY-NC 4.0.
