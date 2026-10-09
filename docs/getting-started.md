# Getting started

Begin with a small CPU model to check the environment. A successful forward
pass validates the installation; it does not qualify a distributed checkpoint
restart or model quality.

## Requirements and installation

Use **Python 3.10 or newer** and an isolated environment. PyTorch must match
CPU, CUDA, ROCm or Apple MPS; it is intentionally installed separately from
the framework package.

```bash
git clone https://github.com/Complexity-ML/complexity-framework.git
cd complexity-framework
python3 -m venv .venv
source .venv/bin/activate
make install
```

The portable default selects the local backend. Explicit targets include
`make install-cpu`, `make install-mps`, `make install-cuda` and
`make install-rocm`. If the correct PyTorch build is already present:

```bash
python -m pip install -e .
```

For tests use `python -m pip install -e '.[dev]'`; additional extras are for
specific tasks, such as `[tools]` for MCP and `[detection]` for vision.
See [backend notes](cuda.md) and [vision dependencies](vision/vision-dependency-stack.md).

## Validate without allocating a model

```bash
complexity --version
complexity train validate configs/examples/tr_gqa_cpu.json --json
complexity train inspect scripts/pretrain_1b_fsdp/recipe.json --json
```

The first config is a flat `ModelConfig` example. The second is an existing
FSDP **recipe**, a distinct contract including token budget and asset hashes.
Inspection reports missing local assets honestly; it does not download them
or start training. See [configuration types](run_configs.md).

## Run a small CPU forward pass

```python
import torch
from complexity import ComplexityModel, ModelConfig

config = ModelConfig.load("configs/examples/tr_gqa_cpu.json", strict=True)
model = ComplexityModel(config).eval()
input_ids = torch.randint(0, config.vocab_size, (2, 16))

with torch.no_grad():
    result = model(input_ids)

assert result["logits"].shape == (2, 16, config.vocab_size)
print(model.num_parameters())
```

The example uses untrained weights, synthetic IDs and PyTorch kernels. Each
TR-MoE layer receives the original IDs: identity selects expert parameters;
attention supplies the contextual hidden states they transform.
`strict=True` catches misspelled authoring fields; existing checkpoint loaders
retain their historical metadata tolerance.

## Save and reload a native model

```python
model.save_pretrained("artifacts/local-example")
restored = ComplexityModel.from_pretrained("artifacts/local-example")
restored.eval()
```

A native export contains `config.json` and `model.safetensors` (or
`pytorch_model.bin` when explicitly selected). It is suitable for model
loading and evaluation, and carries the persisted route buffers. It does not
contain optimizer, scheduler or rank RNG state.

A complete local 200M release can be loaded through the same API:

```python
model = ComplexityModel.from_pretrained("checkpoints/tr-hash-moe-200m-sft")
```

Use the exact configuration and tokenizer distributed with the weights,
rather than reconstructing the shape from an archived training script.
[200M release evidence](tr-hash-200m-release.md) documents that separate
lineage. Downloading multi-gigabyte weights is unnecessary for an install check.

## The paused 1B run uses a different checkpoint format

The preserved [1B FSDP recipe](../scripts/pretrain_1b_fsdp/README.md) contains
`recipe.json`, `model.json` and a resume-only four-rank launcher. Its 70B-token
budget, original schedule, tokenizer and distributed state must remain together.
FSDP shards cannot be loaded with `ComplexityModel.from_pretrained`.

The read-only checkpoint inspector accepts a local complete step directory:

```bash
complexity checkpoint inspect /path/to/step_0051000 \
  --recipe scripts/pretrain_1b_fsdp/recipe.json --verify --json
```

Use the actual complete checkpoint name on your machine; `step_0051000` is an
example, not a claim about the latest checkpoint. File/hash checks do not
restore tensors. A later real resume still requires the original FSDP runner
and a bounded save/reload check. See [workflows](workflows.md).

## Choose the next workflow

| Next task | Guide |
|---|---|
| Create a new CPU training experiment with explicit data | [Workflows](workflows.md#new-local-training-experiment) |
| Prepare a later 1B resume | [Workflows](workflows.md#paused-1b-pretraining) |
| Evaluate a native checkpoint without changing it | [Workflows](workflows.md#evaluation) |
| Use a Transformers custom-model bundle or an existing server | [API](api.md#external-inference) |
| Extend components | [Framework architecture](framework-architecture.md), [custom models](custom-models.md) |

`ComplexityModel.generate()` is intentionally disabled. Generation uses the
[Transformers adapter](../integrations/transformers/tr_hash_moe/README.md) or
an external service that implements TR-HASH and its persisted routes.
