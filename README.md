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
| **TR-GQA** | Grouped-query attention | TR-MoE | `attention_type="gqa"`, `mlp_type="tr_hash_engine"` |
| **TR-MHA** | Multi-head attention | TR-MoE | `attention_type="mha"`, `mlp_type="tr_hash_engine"` |

This framework is now scoped to TR-Hash MoE only: dense (SwiGLU/GeGLU) and
learned-router baselines were removed and will return later as explicit
comparisons against TR-Hash — see "Research paths and evidence" below.

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
  `token_id_balanced_hash`;
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
routing only; constructing a config with any of them raises a clear error.
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
    mlp_type="tr_hash_engine",
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

The `cf-o200k-pretrain` CLI and the `complexity/training/o200k/` pipeline it
drove were removed with the rest of this cycle's TR-Hash refocus, along with
the Dense architecture they were built to compare against. A replacement
training entrypoint is not yet in place. The tracked YAML configurations
under [`configs/run_configs`](configs/run_configs) and the settings they
describe (token accounting, resume validation, cluster plans) still document
the intended run shapes — see [`docs/run_configs.md`](docs/run_configs.md) —
but currently need a driver to execute against. `cf-plan-run` and
`cf-plan-cluster` remain available for token-budget and cluster-sizing
arithmetic independent of any specific pipeline.

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

For conversational adaptation, `configs/sft_conversation_v16.yaml` provides
two runtime-only stages. `casual-only` selects the 400 source-pair-distinct
training dialogues while 20 separate pairs remain held out.
`conversation-blend` retains the first stage and targets a final 571-row,
approximately 70% casual / 20% empathy / 10% practical mixture. Weighted selection is
deterministic, accounts for rows retained from the previous stage, and never
duplicates or rewrites the canonical dataset.

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
  tests/test_tr_hash_engine.py \
  tests/test_tr_hash_dynamic_moe.py \
  tests/test_token_routed_to_tr_hash_conversion.py \
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
