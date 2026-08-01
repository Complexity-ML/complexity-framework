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
- explicit `modulo_cyclic` routing with no corpus counts;
- `zipf`, `round_robin`, `random`, and experimental `lsh_hidden` are retained
  only as research controls;
- a universal PyTorch dispatch path and an optional CUDA/Triton CGGR path;
- optional routing and shared/routed RMS telemetry.

The legacy names `modulo` and `modulo_balanced_secondary` remain accepted only
for historical ablation configurations. New TR-GQA and TR-MHA runs use
`modulo_cyclic`, so the canonical architecture has no dataset-frequency
pre-pass or frequency-dependent routing table.

## Research paths and evidence

### Matched 99.49M MPS summary

The seed-42 selection pilot compares dense and token-routed FFNs under both
GQA and MHA. Every run has 99,487,680 parameters, uses the same
1,024,000-token training budget, and evaluates on the same 5% tail. Routing
statistics use the training partition only.

| Attention | FFN | Final evaluation NLL | Evaluation PPL | NLL vs matched dense |
| --- | --- | ---: | ---: | ---: |
| GQA | Dense SwiGLU | 7.596686 | 1991.58 | — |
| GQA | **TR-MoE (TR-GQA)** | **7.536167** | **1874.63** | **-0.060519** |
| MHA | Dense SwiGLU | 7.586145 | 1970.70 | — |
| MHA | **TR-MoE (TR-MHA)** | **7.536471** | **1875.20** | **-0.049674** |

Both attention families show the same direction at this short budget. The
selected widths were subsequently frozen and paired with their dense controls
under seed 43; the routed direction repeats for GQA and MHA. These remain
architecture pilots on one small corpus and two seeds, not scaling or
statistical claims. Full protocol, throughput context, and machine-readable
results are in [`RESULTS_100M_MPS.md`](RESULTS_100M_MPS.md) and
[`results/matched_gqa_mha_mps_100m.csv`](results/matched_gqa_mha_mps_100m.csv).

### TR-GQA

TR-GQA is the default o200k pretraining path. Tracked profiles cover
approximately 50M, 100M, 300M, 1B, and 8B parameters, with local, ROCm, and
cluster-planning configurations under
[`configs/run_configs`](configs/run_configs).

The completed primary comparison uses one matched seed per architecture,
306.5M parameters, and an 8B-token FineWeb-Edu training budget. At the last
common evaluation checkpoint (step 7,500; 7.864B tokens processed):

| Architecture | Evaluation-stream NLL | Evaluation PPL | Training throughput |
| --- | ---: | ---: | ---: |
| Dense GQA + dense SwiGLU | 2.948246 | 19.07 | ~0.95M tok/s |
| **TR-GQA: GQA + shared top-2 TR-MoE** | **2.932897** | **18.78** | ~0.75M tok/s |

The NLL difference is -0.015349 in favor of TR-GQA at this checkpoint. This is
a token-matched, single-seed observation, not a claim of statistical
significance or general superiority. The fixed evaluation stream comes from
the FineWeb-Edu training split and is therefore diagnostic rather than held
out. The routed implementation is also approximately 21% slower in training.
The matched measurements are published in
[`corrected_300m_scaling.csv`](https://github.com/Complexity-ML/tmlr-paper-pool/blob/main/supplementary_code/results/corrected_300m_scaling.csv).

An additional matched 99,487,680-parameter MPS pilot tests how much of the
1,648-unit FFN budget should be assigned to the routed branch. All runs use the
same local FineWeb-Edu sample, 5% held-out tail, optimizer, 1,024,000-token
budget, and seed 42. Routing frequencies are computed from the training
partition only.

| Routed width | Shared width | Final evaluation NLL | Evaluation PPL |
| ---: | ---: | ---: | ---: |
| Dense GQA | 1,648 | 7.596686 | 1991.58 |
| 64 | 1,584 | 7.598139 | 1994.48 |
| 128 | 1,520 | 7.570862 | 1940.81 |
| 160 | 1,488 | 7.547887 | 1896.73 |
| **256** | **1,392** | **7.536167** | **1874.63** |

Because width 256 was selected on this sweep, it was rerun against Dense GQA
with seed 43:

| Seed 43 architecture | Final evaluation NLL | Evaluation PPL |
| --- | ---: | ---: |
| Dense GQA | 7.530082 | 1863.26 |
| **TR-GQA, routed width 256** | **7.492290** | **1794.16** |

The seed-43 difference is -0.037792 NLL in favor of TR-GQA. This confirms the
direction on a second initialization, but it uses the same small corpus and
evaluation tail. It is not a multi-corpus or scaling result. Protocol details
and machine-readable values are in [`TR_GQA.md`](TR_GQA.md) and
[`results/tr_gqa_mps_100m.csv`](results/tr_gqa_mps_100m.csv).

A stricter routing control keeps the same attention backbone, shared path,
experts, data order, optimizer, and token budget while replacing the fixed
lookup with a learned contextual top-2 router:

| Attention | Seed | Dense | Learned contextual top-2 | Fixed token-ID top-2 |
| --- | ---: | ---: | ---: | ---: |
| GQA | 42 | 7.596686 | 7.602109 | **7.536167** |
| GQA | 43 | 7.530082 | 7.548665 | **7.492290** |
| MHA | 42 | 7.586145 | 7.592847 | **7.536471** |
| MHA | 43 | 7.726971 | 7.666093 | **7.541488** |

The learned control adds only 15,360 routing parameters (0.0154%). Fixed
routing is lower in NLL in all four short local runs, but two seeds do not
establish general superiority. Full details are in
[`RESULTS_100M_MPS.md`](RESULTS_100M_MPS.md).

The other tracked configurations establish implementation and launch
contracts. A planned large run is not presented as a completed result.

### TR-MHA

Corrected matched MPS pairs compare dense MHA with MHA + TR-MoE for
1,024,000 training tokens:

| Seed | Architecture | Final evaluation NLL | Evaluation PPL | NLL delta |
| ---: | --- | ---: | ---: | ---: |
| 42 | Dense MHA | 7.586145 | 1970.70 | — |
| 42 | **TR-MHA: MHA + shared TR-MoE** | **7.536471** | **1875.20** | **-0.049674** |
| 43 | Dense MHA | 7.726971 | 2268.72 | — |
| 43 | **TR-MHA: MHA + shared TR-MoE** | **7.541488** | **1884.63** | **-0.185483** |

The mean paired NLL difference is -0.117579 in favor of TR-MHA. Routing
statistics exclude the evaluation tail. These remain short two-seed pilots;
the variation between paired differences precludes a statistical claim. See
[`RESULTS_100M_MPS.md`](RESULTS_100M_MPS.md). The separate routed-attention
adapter experiments are documented in [`TR_MHA.md`](TR_MHA.md).

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
    routing_strategy="modulo_cyclic",
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

## Instruction fine-tuning from binary shards

The SFT runner accepts pre-tokenized, indexed o200k shards with separate
`input_ids.bin` (`uint32`) and `labels.bin` (`int32`) files. Prompt and padding
labels must be `-100`; only assistant tokens contribute to the causal loss.
The held-out `eval` shard is finite and is never repeated during evaluation.

```bash
python -m scripts.sft_100m_o200k_tr_local \
  --checkpoint /path/to/pretrained/checkpoint.pt \
  --sft-bin /path/to/atlas-instruct-o200k \
  --tokenizer ./tokenizer-o200k \
  --steps 900 \
  --batch-size 32 \
  --seq-len 512 \
  --lr 1e-5 \
  --bf16 \
  --freeze-token-io \
  --eval-at-start \
  --eval-steps 10 \
  --eval-batches 0 \
  --save-best \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.001 \
  --save-steps 0 \
  --save-model-only \
  --run-name sft-atlas-instruct \
  --save-dir checkpoints/sft-atlas-instruct
```

For small o200k instruction corpora, `--freeze-token-io` preserves the large
token embedding and tied output table while adapting the transformer and
TR-Hash blocks. Evaluation at step zero establishes the pretrained baseline;
`--save-best` writes validation-selected checkpoints under `SAVE_DIR/best`, and
patience stops the run after consecutive non-improving evaluations.
`--save-model-only` omits AdamW and scheduler state for compact evaluation and
inference checkpoints. The held-out shard should contain at least 500
independently authored examples before its NLL is treated as a stable capability
estimate.

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
