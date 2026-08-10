# Getting started

## Requirements

- Python 3.8 or newer;
- a PyTorch build matched to the machine;
- enough memory for the selected vocabulary, model profile, batch, and
  sequence length.

The package does not depend on a generic `torch` wheel because CUDA, ROCm, CPU,
and MPS installations require different builds.

## Install from source

```bash
git clone https://github.com/Complexity-ML/complexity-framework.git
cd complexity-framework

python3 -m venv .venv
source .venv/bin/activate

pip install torch
pip install -e ".[dev,tools]"
```

For Linux GPU backends, consult [GPU and dispatch paths](cuda.md).

## Build a first TR-GQA model

```python
import torch

from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=1,
    attention_type="gqa",
    vocab_size=1_024,
    max_position_embeddings=128,
    mlp_type="tr_hash_engine",
    num_experts=4,
    intermediate_size=64,
    shared_expert=True,
    shared_intermediate_size=256,
    routing_strategy="modulo",
    top_k=2,
    top_k_primary_weight=0.5,
)

model = ComplexityModel(config)
input_ids = torch.randint(0, config.vocab_size, (2, 32))
result = model(input_ids)

print(result["logits"].shape)
print(model.num_parameters())
```

The model passes `input_ids` to every TR-MoE layer as route identifiers.

## Switch to TR-MHA

```python
from dataclasses import replace

tr_mha_config = replace(
    config,
    attention_type="mha",
    num_key_value_heads=config.num_attention_heads,
)
tr_mha_model = ComplexityModel(tr_mha_config)
```

TR-GQA and TR-MHA use the same TR-MoE FFN. Only the attention head layout
changes.

## Run a local smoke training

> `cf-o200k-pretrain` was removed along with the `o200k` training pipeline —
> see [`training.md`](training.md). The commands below no longer run as
> written; a replacement smoke-training entrypoint is not yet in place.
> `--routing-strategy modulo` above also predates the routing-strategy
> guardrail — use `modulo_cyclic` or `token_id_balanced_hash` going forward.

## Save and load

```python
model.save_pretrained("checkpoints/example")
restored = ComplexityModel.from_pretrained("checkpoints/example")
```

`save_pretrained` writes `config.json` plus `model.safetensors` when
`safetensors` is available.

## Generate text

`ComplexityModel.generate()` intentionally raises an error. Serve a compatible
export through vLLM or SGLang and use the external client:

```python
from complexity.inference import ExternalGenerationConfig, create_external_backend

backend = create_external_backend(
    "vllm",
    base_url="http://localhost:8000",
    model="example",
)
print(
    backend.complete(
        "The experiment shows",
        ExternalGenerationConfig(max_tokens=64),
    )
)
```

The external server must itself support the exported architecture.

## Next steps

- [Architecture and naming](architectures.md)
- [TR-MoE internals](token-routed.md)
- [Training](training.md)
- [Run configurations](run_configs.md)
- [API reference](api.md)
