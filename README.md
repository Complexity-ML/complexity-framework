# Complexity Framework

Research framework for **TR-GQA**, **TR-MHA**, and deterministic
**Token-Routed Mixture-of-Experts (TR-MoE)** language models in PyTorch.

The central design keeps contextual sequence processing in attention while
token identity selects a small FFN parameter subspace. A dense shared SwiGLU
path remains active for every token.

```text
token IDs ──► embeddings ──► GQA (TR-GQA) or MHA (TR-MHA) ──► TR-MoE FFN ──► logits
                                                           │
                                                           ├─ shared SwiGLU: every token
                                                           └─ routed experts: fixed token-ID table
```

This is an active research codebase, not a finished foundation model. Numerical
results are reported only with their model size, token budget, seed, hardware,
and evaluation protocol.

## Architecture names

The names below describe complete decoder configurations, not competing
definitions of MoE.

| Name | Attention | Feed-forward path | Configuration |
| --- | --- | --- | --- |
| **TR-GQA** | Grouped-query attention | TR-MoE | `attention_type="gqa"`, `mlp_type="token_routed"` |
| **TR-MHA** | Multi-head attention | TR-MoE | `attention_type="mha"`, `mlp_type="token_routed"` |
| **Dense GQA** | Grouped-query attention | Dense SwiGLU | `attention_type="gqa"`, `mlp_type="swiglu"` |
| **Dense MHA** | Multi-head attention | Dense SwiGLU | `attention_type="mha"`, `mlp_type="swiglu"` |

TR-MoE is therefore shared by both TR-GQA and TR-MHA:

```text
TR-MoE(x, token_id) = shared_swiglu(x)
                    + weighted_sum(selected_token_experts(x))
```

The repository also contains `attention_type="tr_mha"` and `"tr_mha_v2"`.
Those are separate, experimental token-routed residual adapters inside MHA.
They must not be confused with the main **TR-MHA = MHA + TR-MoE** pairing.

## TR-MoE

`TokenRoutedMLP` provides:

- a dense shared SwiGLU branch for common contextual computation;
- four narrow routed experts by default;
- deterministic per-layer token-to-expert tables;
- top-k routes without a learned router or auxiliary balancing loss;
- `zipf`, `modulo`, `modulo_balanced_secondary`, `round_robin`, `random`,
  and experimental `lsh_hidden` routing;
- a universal PyTorch dispatch path and an optional CUDA/Triton CGGR path;
- optional routing and shared/routed RMS telemetry.

Frequency-balanced routing is approximate and depends on the supplied tokenizer
frequency artifact. Without frequencies, `zipf` deliberately falls back to
token-ID modulo routing.

## Research paths and evidence

### TR-GQA

TR-GQA is the default o200k pretraining path. Tracked profiles cover
approximately 50M, 100M, 300M, 1B, and 8B parameters, with local, ROCm, and
cluster-planning configurations under
[`configs/run_configs`](configs/run_configs).

These configurations establish implementation and launch contracts. A planned
large run is not presented as a completed result.

### TR-MHA

A matched 99,487,680-parameter MPS pilot compares GQA, dense MHA, and
MHA + TR-MoE for 1,024,000 training tokens with seed 42:

| Architecture | Final evaluation NLL | Evaluation PPL |
| --- | ---: | ---: |
| Dense GQA | 7.359221 | 1570.61 |
| Dense MHA | 7.369812 | 1587.34 |
| **TR-MHA: MHA + shared TR-MoE** | **7.321415** | **1512.34** |

This is a short, single-seed pilot. It validates the implementation and
motivates replication; it does not establish scaling or statistical
significance. See [`TR_MHA.md`](TR_MHA.md).

### Contextual W/R/V control

W/R/V is a distinct attention experiment, not a replacement name for TR-MoE.
Six approximately 98.2M-parameter H200 runs use matched settings and seeds
42--44:

| Architecture | Held-out NLL, mean ± sample SD | Mean training throughput |
| --- | ---: | ---: |
| GQA | 4.703035 ± 0.017657 | 125,021 tok/s |
| Contextual W/R/V | **4.684432 ± 0.021137** | 121,448 tok/s |

The mean paired difference is -0.018604 NLL. Its 95% t interval
[-0.0541, 0.0169] includes zero, so the result is reported as a consistent
small-scale observation rather than statistical significance. Raw evidence is
under [`paper/tmlr`](paper/tmlr).

## Installation

PyTorch is intentionally not a package dependency because its wheel must match
the target CPU, CUDA, ROCm, or MPS backend.

```bash
git clone https://github.com/Complexity-ML/complexity-framework.git
cd complexity-framework

python3 -m venv .venv
source .venv/bin/activate

# Install the PyTorch build for this machine first.
pip install torch
pip install -e ".[dev,tools]"
```

Backend-specific notes are in [`docs/cuda.md`](docs/cuda.md).

## Build TR-GQA and TR-MHA

```python
from dataclasses import replace

from complexity import ComplexityModel, ModelConfig

tr_gqa = ModelConfig(
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

tr_mha = replace(
    tr_gqa,
    attention_type="mha",
    num_key_value_heads=tr_gqa.num_attention_heads,
)

tr_gqa_model = ComplexityModel(tr_gqa)
tr_mha_model = ComplexityModel(tr_mha)
```

Both models require the original `token_ids` during the forward pass because
TR-MoE routes from token identity while transforming contextual hidden states.

## Train from a tracked configuration

```bash
cf-o200k-pretrain \
  --config configs/run_configs/100m_o200k_tr_rocm_mi350x.yaml
```

For a bounded MPS TR-MHA pilot, adapt the local dataset and tokenizer paths in:

```bash
cf-o200k-pretrain \
  --config configs/run_configs/experiments_100m/100m_params_mha_modulo_balanced_shared_1296_mps.yaml
```

YAML settings, CLI overrides, resume validation, token accounting, and cluster
plans are documented in [`docs/run_configs.md`](docs/run_configs.md).

## Inference boundary

The framework owns model definition, training, evaluation, conversion, and
serving clients. Native `ComplexityModel.generate()` is intentionally disabled.
Text generation is delegated to an OpenAI-compatible vLLM or SGLang runtime.

```bash
complexity inference generate my-model \
  --backend vllm \
  --base-url http://localhost:8000 \
  --prompt "A computer program is"
```

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

Serving compatibility depends on the external runtime supporting the exported
TR-MoE architecture. The client alone does not add model support to upstream
vLLM or SGLang.

## Validation

```bash
python -m pytest -q \
  tests/test_models.py \
  tests/test_o200k_pretrain.py \
  tests/test_100m_ablation_configs.py \
  tests/test_tr_mha.py \
  tests/test_external_inference.py
```

The repository tracks configurations, metrics, tables, and lightweight
reproducibility artifacts. Multi-gigabyte datasets and checkpoints are excluded
from Git.

## Repository map

```text
complexity/                  model, training, inference, MCP, and RL code
complexity_cuda/             optional CUDA/Triton kernels
configs/run_configs/         explicit experiment and cluster configurations
tests/                       architecture and integration tests
paper/tmlr/                  controlled W/R/V evidence package
scripts/                     training, conversion, evaluation, and audit tools
spaces/                      Hugging Face Space wrappers
spikes/                      isolated research prototypes
```

## Documentation

- [Documentation index](docs/index.md)
- [Architecture and naming](docs/architectures.md)
- [TR-MoE internals](docs/token-routed.md)
- [Getting started](docs/getting-started.md)
- [Training](docs/training.md)
- [Run configurations](docs/run_configs.md)
- [GPU and dispatch paths](docs/cuda.md)
- [API reference](docs/api.md)
- [Historical and experimental components](docs/dynamics.md)

## License

[CC BY-NC 4.0](LICENSE). Commercial use is not permitted by this repository's
current license.
