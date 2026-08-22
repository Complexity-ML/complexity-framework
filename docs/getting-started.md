# Getting started

## Requirements

- Python 3.10 or newer;
- a PyTorch build matched to CPU, CUDA, ROCm, or Apple MPS;
- `safetensors` for released checkpoints;
- enough memory for the selected model, batch, sequence length, and optimizer.

PyTorch is intentionally not pinned as a package dependency because its wheel
must match the target backend.

## Install from source

```bash
git clone https://github.com/Complexity-ML/complexity-framework.git
cd complexity-framework

python3 -m venv .venv
source .venv/bin/activate

# Install the appropriate PyTorch build first.
pip install torch
pip install -e ".[dev,tools]"
```

For backend-specific wheels and kernel policy, read
[GPU and dispatch paths](cuda.md).

## Build a small TR-GQA model

This smoke model uses the same multi-hash routing family as the released 200M
checkpoint while remaining small enough for CPU tests.

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
    intermediate_size=64,          # total stored routed width: 4 × 16
    shared_expert=True,
    shared_intermediate_size=256,
    routing_strategy="token_id_multi_hash",
    route_hash_count=2,
    top_k=2,
    top_k_primary_weight=0.5,
    use_custom_kernels=False,
)

model = ComplexityModel(config)
input_ids = torch.randint(0, config.vocab_size, (2, 32))
result = model(input_ids)

print(result["logits"].shape)
print(model.num_parameters())
print(model.layers[0].mlp.capability_summary("cpu"))
```

The model passes the original `input_ids` to every TR-MoE layer. Token IDs
select parameters; the selected experts still transform contextual hidden
states produced by attention.

## Recreate the released 200M shape

Use the exact configuration from `scripts.train_tr_hash_200m_200b.make_config`
rather than copying dimensions by hand:

```python
from complexity import ComplexityModel
from scripts.train_tr_hash_200m_200b import make_config

config = make_config()
model = ComplexityModel(config)

assert model.num_parameters() == 201_194_368
assert model.layers[0].mlp.engine.config.expert_width == 64
assert model.layers[0].mlp.engine.config.stored_routed_width == 256
assert model.layers[0].mlp.engine.config.active_routed_width == 128
```

Constructing this model allocates the full weights. Use the small smoke model
when only validating an installation.

## Load a released checkpoint

Download one complete Hugging Face repository, preserving `config.json`,
`model.safetensors`, tokenizer files, route metadata, and the chat template:

```bash
python -m pip install --upgrade huggingface_hub
```

```bash
hf download AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT \
  --local-dir checkpoints/tr-hash-moe-200m-sft
```

Then load the framework model:

```python
from complexity import ComplexityModel

model = ComplexityModel.from_pretrained("checkpoints/tr-hash-moe-200m-sft")
model.eval()
```

Do not copy only `model.safetensors`: configuration, tokenizer, persisted
routes, and `chat_template.jinja` are part of the release contract.

### Load through Transformers

A Transformers bundle additionally contains `configuration_tr_hash_moe.py`,
`modeling_tr_hash_moe.py`, and the `auto_map` entries that bind them. Load it
as custom model code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)
generated = model.generate(
    **tokenizer("Hello", return_tensors="pt"),
    max_new_tokens=32,
)
```

For a local native checkpoint, create the autonomous bundle with
`scripts/export_tr_hash_transformers.py`. See the adapter
[README](../integrations/transformers/tr_hash_moe/README.md) for the exact
command. The exporter retains the persisted expert-route tables and separates
architectural `num_experts_per_tok` from the sampling parameter `top_k`.

## Save and reload

```python
model.save_pretrained("checkpoints/example")
restored = ComplexityModel.from_pretrained("checkpoints/example")
```

`save_pretrained` writes `config.json` and `model.safetensors` when
`safetensors` is installed. Distributed DTensor/FSDP saves are collective; all
ranks must enter the save call.

## Training entry points

The current production language-model path is:

1. replay-scheduled base pretraining;
2. fresh-optimizer full-parameter refinement;
3. three-epoch full-parameter instruction SFT on the audited 300K v2 dataset;
4. PIQA checkpoint selection and SafeTensors export.

The first three steps are also the default contract for other non-Vision model
families. Refinement means the exact pretraining corpus with fresh optimizer
and scheduler state; task data belongs only to SFT. Vision is explicitly
exempt because its existing clean-image phase is integrated refinement.

See [Training](training.md) for commands and preflight requirements. The old
statement that text training is LoRA-only is no longer true.

## Generate text

`ComplexityModel.generate()` intentionally raises. Production generation for
the released checkpoint belongs to
[TR-Hash-i64](https://github.com/Complexity-ML/TR-Hash-i64), which implements
the custom architecture and exposes OpenAI-compatible endpoints.

```bash
curl http://localhost:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tr-hash-moe-200m",
    "messages": [{"role": "user", "content": "Explain deterministic token routing."}],
    "temperature": 0.4,
    "top_p": 0.85,
    "max_tokens": 384
  }'
```

The framework also contains eager diagnostic generation scripts used for
checkpoint tests. They are not a continuous-batching production server.

## Next steps

- [TR-HASH MoE 200M release](tr-hash-200m-release.md)
- [Architecture and naming](architectures.md)
- [TR-Hash engine](tr-hash-engine.md)
- [Training](training.md)
- [GPU and dispatch paths](cuda.md)
- [API reference](api.md)
