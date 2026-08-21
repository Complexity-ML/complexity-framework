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

This is the implementation behind the released **TR-HASH MoE 200M** lineage,
not a claim of a general-purpose foundation model. Numerical results are
reported only with their exact checkpoint, token budget, and evaluation
protocol.

## Current reference release: TR-HASH MoE 200M

The current public reference is a 201,194,368-parameter decoder with 16 layers,
hidden size 896, GQA (14 query heads / 2 KV heads), a 32,000-token vocabulary,
one shared width-3,072 SwiGLU path, and four stored width-64 residual experts
(256 total routed width, 128 active with top-2).
Multi-hash rendezvous voting compiles each token ID to a persisted top-2 route;
there is no learned router or load-balancing loss.

| Phase | Progress represented by released weights | Loss / perplexity signal | PIQA acc / acc_norm |
| --- | --- | --- | --- |
| Base pretraining | 165,298 steps; 129,995,636,736 token exposures | Last logged training minibatch: 2.652628 / 14.19 | 65.45% / 65.61% |
| Full-parameter refinement | Step 8,156 / 17,802; 32.07B additional unique-token exposures | Terminal displayed training loss / PPL: 2.3208 / 10.2 | 68.66% / 68.39% |
| Full-parameter SFT | 3 epochs; 238.9M supervised tokens; epoch 2 promoted | Epoch-3 matched eval: 1.220861 / 3.39 | **68.82% / 69.31%** (epoch 2) |

The refinement was intentionally stopped at 45.8% of its planned 70B-token
pass. Consequently, `160B` is a rounded lineage label for approximately
162.07B source-token exposures, not a claim that the planned refinement
completed. The released assistant is a **full-parameter SFT, not LoRA**.
PIQA uses all 1,838 validation examples, zero-shot causal continuation
log-likelihood, no chat template, and maximum sequence length 2,048. Training
loss, held-out SFT loss, and PIQA are different measurements and are not
presented as interchangeable.

- [130B base checkpoint](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-130B)
- [≈162B interrupted refinement checkpoint](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement)
- [Released full-parameter SFT](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT)
- [Live 200M chat](https://www.complexity-ai.fr/ai-lab)
- [200M release paper](https://www.complexity-ai.fr/papers/tr-hash-200m-multi-hash-routing.pdf)

## Architecture names

The names below describe complete decoder configurations, not competing
definitions of MoE.

| Name | Attention | Feed-forward path | Configuration |
| --- | --- | --- | --- |
| **TR-GQA** | Grouped-query attention | TR-MoE | `attention_type="gqa"`, `mlp_type="tr_hash_engine"` |
| **TR-MHA** | Multi-head attention | TR-MoE | `attention_type="mha"`, `mlp_type="tr_hash_engine"` |

The canonical language-model path is scoped to TR-Hash MoE. The old
`swiglu`/`gelu`/`geglu`/`standard` registry entries and learned-router MoE were
removed. A separately named deterministic-initialization dense control
(`dense_deterministic`) and experimental lexical modules remain available;
they are not the default TR-GQA/TR-MHA path.

TR-MoE is therefore shared by both TR-GQA and TR-MHA:

```text
TR-MoE(x, token_id) = shared_swiglu(x)
                    + weighted_sum(selected_token_experts(x))
```

The repository also contains `attention_type="tr_mha"` and `"tr_mha_v2"`.
Those are separate, experimental token-routed residual adapters inside MHA.
They must not be confused with the main **TR-MHA = MHA + TR-MoE** pairing.

## TR-MoE

`TRHashEngineMLP` (`mlp_type="tr_hash_engine"`, aliased `tr_hash_moe`) is the
canonical implementation, backed by `complexity.tr_hash.TRHashEngine`:

- a dense shared SwiGLU branch for common contextual computation;
- narrow routed experts, `num_experts` in `{1, 2, 4, 8, 16}`;
- deterministic per-layer token-to-expert tables, re-derived from
  `routing_strategy` at construction — `modulo_cyclic` (no corpus counts) or
  `token_id_balanced_hash`; the released 200M uses
  `token_id_multi_hash` with two rendezvous-hash channels;
- top-k routes without a learned router or auxiliary balancing loss;
- a universal PyTorch dispatch path and optional CUDA/Triton CGGR / hash-native
  fused paths, selected automatically per shape (`use_cggr`, `use_custom_kernels`);
- runtime-configurable capacity: an allocated model can be shrunk to fewer
  active experts and/or a narrower per-expert width at any point via
  `engine.set_active_capacity(num_experts=..., expert_width=...)` (or
  declaratively via `ModelConfig(active_num_experts=..., active_expert_width=...)`)
  — still fully deterministic ID/hash routing, just over a smaller pool.

`zipf`, `round_robin`, `random`, and `lsh_hidden` routing, and the historical
`TokenRoutedMLP` dispatch implementation (`mlp_type="token_routed"`), were
removed to keep the framework scoped to deterministic token-ID / hash-table
routing only. Legacy values can still be parsed from an old `ModelConfig`, but
constructing `ComplexityModel` rejects them before a layer is built.
Existing `token_routed`-format checkpoints still load — convert them first:

```python
from complexity.utils.token_routed_conversion import convert_token_routed_checkpoint_dir

model = convert_token_routed_checkpoint_dir("/path/to/old/checkpoint")
```

This renames the checkpoint's tensors to `TRHashEngineMLP`'s layout and
transplants the exact trained routing table (not a re-derived one), so the
converted model is numerically equivalent to the original.

## Research paths and evidence

Earlier measurements in this section compared TR-GQA/TR-MHA against Dense GQA/MHA
and a learned contextual router, produced by the `o200k` pretraining pipeline.
Both the Dense architecture and that pipeline were removed this cycle to
refocus the framework on TR-Hash MoE only; they will return later as explicit
baselines, at which point these comparisons will be rerun against the current
implementation rather than restated from the removed one. The original
protocol, tables, and machine-readable results are preserved for the
historical record in [`RESULTS_100M_MPS.md`](RESULTS_100M_MPS.md),
[`TR_GQA.md`](TR_GQA.md), [`TR_MHA.md`](TR_MHA.md), and
[`results/`](results/).

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

For TR-HASH Vision, install Torch and Torchvision from the same backend index
plus the detector runtime stack in one command:

```bash
make install-vision-cuda  # NVIDIA
make install-vision-rocm  # AMD
make install-vision-cpu
```

The dependency contract is documented in
[`docs/vision-dependency-stack.md`](docs/vision-dependency-stack.md).

## Build TR-GQA and TR-MHA

```python
from dataclasses import replace

from complexity import ComplexityModel, ModelConfig

tr_gqa = ModelConfig(
    hidden_size=896,
    num_hidden_layers=16,
    num_attention_heads=14,
    num_key_value_heads=2,
    attention_type="gqa",
    vocab_size=32_000,
    mlp_type="tr_hash_engine",
    num_experts=4,
    intermediate_size=256,  # total routed width: 4 experts × 64
    shared_expert=True,
    shared_intermediate_size=3072,
    routing_strategy="token_id_multi_hash",
    route_hash_count=2,
    top_k=2,
    top_k_primary_weight=0.5,
    shared_output_scale=1.0,
    routed_output_scale=2.0,
    use_qk_norm=True,
    max_position_embeddings=2048,
    tie_word_embeddings=True,
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

## Reproduce the 200M training lineage

The released text pipeline has three distinct stages. Refinement is continued
language-model training with a fresh optimizer; it is not instruction SFT.
The final assistant stage is full-parameter instruction SFT, not LoRA.

| Stage | Production entry point |
|---|---|
| 130B replay pretraining | `scripts/vast_pretrain_tr_hash_200m_70b_replay.sh` |
| 32.07B unique-token refinement | `scripts/vast_finetune_tr_hash_200m_70b_unique_phase2.sh` |
| Audited 300K full SFT v2 | `scripts/vast_sft_200m_clean_v2_full_3e.sh` |
| Three-checkpoint PIQA + behavior evaluation | `scripts/vast_eval_200m_clean_sft_v2_all.sh` |
| Promotion-gated F32 SafeTensors release | `scripts/export_sft_v2_release.py` |

These `vast_*` scripts are tracked production profiles for `/workspace`
machines. They validate the expected checkpoint, dataset manifest, tokenizer,
distributed world size, Liger availability, kernel policy, and save/evaluation
boundaries before optimization. Read [Training](docs/training.md) and adapt the
path variables rather than copying a command blindly.

The former `cf-o200k-pretrain` driver and its Dense comparison path were
removed. Their YAMLs and runbook remain explicitly historical evidence, not
current launchers. `cf-plan-run` and `cf-plan-cluster` still perform arithmetic
and validation without launching a job.

## Audit pretrained routed-expert geometry

The 3-D visualization script measures a released routed checkpoint directly.
It supports both historical converted bundles and current `tr_hash_engine`
exports, uses natural PIQA validation text without a chat template, captures
the contextual MLP input at selected layers, and plots each top-2 routed
residual contribution separately. It never substitutes random token IDs or
includes the shared MLP output in the projected vector.

```bash
pip install -e ".[viz]"
python scripts/viz_pretrained_expert_tsne_3d.py \
  --checkpoint /path/to/TR-HASH-MoE-200M-160B-SFT \
  --probe /path/to/physicaliqa-train-dev/dev.jsonl \
  --layers 0,3,7,11,15 \
  --model-label "TR-HASH MoE 200M full SFT" \
  --source-token-exposure 162065132681 \
  --sft-tokens 238900000 \
  --output artifacts/evaluations/tr_hash_moe_200m_full_sft_tsne/expert_tsne_3d.html
```

The older 492.1M/20B visualization remains reproducible by pointing the same
script at its converted checkpoint and selecting layers `0,5,11,17,23`.

The command also writes a deterministic compressed point table and a metadata
manifest containing the checkpoint, tokenizer/config, probe, HTML, and point
hashes. The t-SNE remains an exploratory visualization, not evidence of expert
specialization, downstream quality, or superiority over another architecture.

## Full-parameter instruction SFT

The current SFT runner is `scripts.sft_tr`. It supports explicit
`--full-parameter` training as well as experimental LoRA mode. The promoted
200M recipe uses:

- the step-8,156 refinement checkpoint as its source;
- the audited 300,000-example SFT-v2 train split and 3,000-example eval split;
- final-assistant-only labels, no truncation, and packed 2,048-token sequences;
- three epochs, BF16 training, AdamW, a continuous cosine schedule, and Liger plus the
  tested custom CUDA/Triton path;
- complete held-out evaluation and a resumable checkpoint at every epoch;
- PIQA evaluation of all three epoch checkpoints before root promotion.

The exact command and preflight checks are tracked in
[`scripts/vast_sft_200m_clean_v2_full_3e.sh`](scripts/vast_sft_200m_clean_v2_full_3e.sh).
See [Training](docs/training.md) for the argument contract and
[TR-HASH MoE 200M release](docs/tr-hash-200m-release.md) for measured results.

## Inference boundary

The framework owns model definition, training, evaluation, conversion, and
diagnostic eager generation. Native `ComplexityModel.generate()` remains
disabled so a custom architecture is not accidentally presented as supported
by a generic serving stack. The released 200M model is served by
[`TR-Hash-i64`](https://github.com/Complexity-ML/TR-Hash-i64), which implements
the persisted TR-Hash routes and exposes an OpenAI-compatible API.

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

The external client labels are currently `vllm` and `sglang`, but the HTTP
contract is OpenAI-compatible. Selecting a client label does not add TR-MoE
support to an upstream runtime. Use TR-Hash-i64 for the released model, or
explicitly implement the architecture and its persisted route tables in the
chosen server.

### Hugging Face Transformers adapter

The released checkpoint can also be packaged as an autonomous Transformers
custom model. The adapter preserves all native safetensors names and exposes
`AutoConfig`, `AutoModelForCausalLM`, causal loss, KV-cache generation, and the
persisted deterministic route tables:

```bash
python scripts/export_tr_hash_transformers.py \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokenizer-dir /path/to/tokenizer \
  --output /path/to/hub-bundle
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

The standalone source and its compatibility notes live in
[`integrations/transformers/tr_hash_moe`](integrations/transformers/tr_hash_moe).
This PyTorch reference adapter does not imply native vLLM, SGLang, or AutoRound
support; each optimized runtime still needs an explicit TR-Hash implementation.

## Validation

```bash
python -m pytest -q \
  tests/test_models.py \
  tests/test_tr_hash_engine.py \
  tests/test_tr_hash_200m_pretraining.py \
  tests/test_sft_bin.py \
  tests/test_sft_v2_production_contract.py \
  tests/test_sft_v2_regression_gate.py \
  tests/test_tr_hash_dynamic_moe.py \
  tests/test_token_routed_to_tr_hash_conversion.py \
  tests/test_tr_mha.py \
  tests/test_external_inference.py \
  tests/test_tr_hash_transformers_adapter.py
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
deploy/                      supervisor and deployment configurations
spikes/                      isolated research prototypes
```

## Documentation

- [Documentation index](docs/index.md)
- [TR-HASH MoE 200M release](docs/tr-hash-200m-release.md)
- [Architecture and naming](docs/architectures.md)
- [TR-MoE internals](docs/tr-hash-engine.md)
- [Getting started](docs/getting-started.md)
- [Training](docs/training.md)
- [Efficient training](docs/efficient.md)
- [Run configurations](docs/run_configs.md)
- [GPU and dispatch paths](docs/cuda.md)
- [API reference](docs/api.md)
- [Historical TokenRoutedMLP migration](docs/token-routed.md)

## License

[CC BY-NC 4.0](LICENSE). Commercial use is not permitted by this repository's
current license.
