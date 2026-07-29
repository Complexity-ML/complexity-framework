# Complexity Framework

Research framework for controlled language-model architecture experiments in
PyTorch. The repository contains matched GQA and MHA baselines, contextual
Write--Read--Value (W/R/V) attention, deterministic token-routed feed-forward
paths, TR-MHA pilots, training utilities, and reproducibility artifacts.

This is an active research codebase rather than a finished pretrained model.
Every numerical claim below is tied to a specific configuration, token budget,
hardware setup, and evaluation protocol.

## Current research paths

| Path | What changes | Current evidence |
| --- | --- | --- |
| **GQA / MHA controls** | Standard causal softmax attention | Matched baselines |
| **Contextual W/R/V** | Contextual read heads with shared write/value heads | Three paired H200 seeds |
| **TR-MHA** | Dense MHA plus small token-selected attention adapters | Short MPS pilots |
| **Shared + routed FFN** | Dense shared SwiGLU plus deterministic token-ID experts | Short matched MPS pilot |
| **Lexical controls** | Fixed, balanced, random, round-robin, and semantic-LSH routes | Ablations and diagnostics |

Historical Mu-Guidance, convolutional, fixed-state, and lexical-write variants
remain available as controls. They are not presented as the canonical
architecture.

## Evidence snapshot

### Controlled W/R/V comparison

Six approximately 98.2M-parameter runs use the same FineWeb-Edu shard,
tokenizer, 100,007,936-token budget, optimizer, peak learning rate, BF16 mode,
evaluation protocol, and seeds 42--44.

| Architecture | Held-out NLL, mean ± sample SD | Mean training throughput |
| --- | ---: | ---: |
| GQA | 4.703035 ± 0.017657 | 125,021 tok/s |
| Contextual W/R/V | **4.684432 ± 0.021137** | 121,448 tok/s |

W/R/V is lower on all three paired seeds, with a mean paired difference of
-0.018604 NLL and a 2.86% training-throughput penalty. The 95% t interval
[-0.0541, 0.0169] includes zero, so this is reported as a consistent
small-scale observation, not statistical significance or universal
superiority.

Sources and raw measurements are under [`paper/tmlr`](paper/tmlr).

### TR-MHA and routed-FFN pilot

The matched 99,487,680-parameter, seed-42 MPS pilot trains for 1,024,000 tokens.
The strongest configuration keeps MHA dense and uses a width-1,296 shared
SwiGLU path plus two deterministic width-40 token-ID experts:

| Architecture | Final eval NLL | Eval PPL |
| --- | ---: | ---: |
| GQA dense | 7.359221 | 1570.61 |
| MHA dense | 7.369812 | 1587.34 |
| MHA + shared FFN + balanced token-ID experts | **7.321415** | **1512.34** |

This is a single-seed short pilot. It validates the implementation and motivates
replication; it does not establish scaling or production-speed gains. See
[`TR_MHA.md`](TR_MHA.md) for the complete ablation table.

## Installation

PyTorch is intentionally not installed by the package because its wheel must
match the selected CPU, CUDA, ROCm, or MPS backend.

```bash
git clone https://github.com/Complexity-ML/complexity-framework.git
cd complexity-framework

python3 -m venv .venv
source .venv/bin/activate

# Install the appropriate PyTorch build first.
pip install torch
pip install -e ".[dev,tools]"
```

For CUDA and ROCm environments, use the backend-specific PyTorch index or the
helpers described in [`docs/cuda.md`](docs/cuda.md).

## Build a model

```python
from complexity.config import ModelConfig
from complexity.models import ComplexityModel

config = ModelConfig(
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    num_key_value_heads=8,
    attention_type="mha",
    vocab_size=200_019,
    mlp_type="token_routed",
    num_experts=4,
    intermediate_size=160,
    shared_expert=True,
    shared_intermediate_size=1296,
    routing_strategy="modulo_balanced_secondary",
    top_k=2,
    top_k_primary_weight=0.5,
)

model = ComplexityModel(config)
```

The primary route is derived deterministically from token identity. The
secondary table is built offline to balance estimated token-frequency load.
Both selected experts transform the current contextual hidden state; token
identity selects parameters, not a context-free output.

## Run a bounded training experiment

```bash
python -m complexity.training.o200k_pretrain \
  --config configs/run_configs/experiments_100m/100m_params_mha_modulo_balanced_shared_1296_mps.yaml
```

Run configurations intentionally record model size, token budget, seed,
optimizer, evaluation cadence, and output location. Local dataset and tokenizer
paths must be adapted to the machine running the experiment.

## Inference boundary

The framework builds, trains, evaluates, and exports models. Native
`model.generate()` is deliberately disabled; production generation is delegated
to an OpenAI-compatible vLLM or SGLang server.

```bash
complexity inference generate my-model \
  --backend vllm \
  --base-url http://localhost:8000 \
  --prompt "A computer program is"
```

The external client is available directly:

```python
from complexity.inference import (
    ExternalGenerationConfig,
    create_external_backend,
)

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

## Reproducibility

```bash
# Architecture, serving-boundary, and MCP tests
python -m pytest -q \
  tests/test_tr_mha.py \
  tests/test_100m_ablation_configs.py \
  tests/test_external_inference.py \
  tests/test_mcp_official.py \
  tests/test_models.py

# Standalone W/R/V artifact
cd paper/tmlr/standalone_artifact
python -m pytest -q tests
```

The repository tracks lightweight configurations, raw metrics, checksums,
generated tables, and the standalone mini-framework. Multi-gigabyte datasets
and checkpoints are excluded from Git.

To rebuild the anonymous supplement:

```bash
python paper/tmlr/scripts/generate_controlled_tables.py
python paper/tmlr/scripts/build_anonymous_supplement.py
```

Generated submission packages are written to `paper/tmlr/submission/` and are
not tracked.

## Experimental integrations

- OpenAI-compatible vLLM/SGLang inference client.
- Official MCP stdio client under `complexity.mcp`.
- Experimental shared online-RL and MPS wrappers.
- MLX conversion, generation, and PyTorch-parity utilities.
- Expert-route, CKA, semantic-LSH, and routed-vocabulary-head diagnostics.

These integrations are research utilities and should be validated for the
target deployment before production use.

## Repository map

```text
complexity/                  model, training, inference, MCP, and RL code
configs/run_configs/         explicit experiment configurations
tests/                       architecture and integration tests
paper/tmlr/                  paper, measurements, and standalone artifact
figures/                     generated diagnostic figures
scripts/                     training, evaluation, conversion, and audit tools
spikes/                      isolated performance prototypes
```

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture reference](docs/architectures.md)
- [Token-routed MLP](docs/token-routed.md)
- [Training](docs/training.md)
- [CUDA and serving](docs/cuda.md)
- [TR-MHA pilot](TR_MHA.md)

Some historical documentation describes earlier controls and should be read
together with the dated configuration and artifact it references.

## License

[CC BY-NC 4.0](LICENSE) — Creative Commons Attribution-NonCommercial 4.0.
