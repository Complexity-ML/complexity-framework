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
  --checkpoint artifacts/remote_runs/tr_hash_moe_500m_20b/hf_base \
  --probe /path/to/physicaliqa-train-dev/dev.jsonl \
  --layers 0,5,11,17,23 \
  --output artifacts/evaluations/tr_hash_500m_pretrain_tsne/expert_tsne_3d.html
```

For the released 200M full-SFT checkpoint, use layers `0,3,7,11,15` and pass
release-specific labels and token accounting through `--model-label`,
`--artifact-label`, `--source-token-exposure`, and `--sft-tokens`.

The command also writes a deterministic compressed point table and a metadata
manifest containing the checkpoint, tokenizer/config, probe, HTML, and point
hashes. The t-SNE remains an exploratory visualization, not evidence of expert
specialization, downstream quality, or superiority over another architecture.

## LoRA instruction fine-tuning from binary shards

The 500M LoRA-SFT runner accepts pre-tokenized, indexed native-32k shards with separate
`input_ids.bin` (`uint32`) and `labels.bin` (`int32`) files. Prompt and padding
labels must be `-100`; only assistant tokens contribute to the causal loss.
The held-out `eval` shard is finite and is never repeated during evaluation.

```bash
python -m scripts.sft_500m_32k_tr \
  --checkpoint /path/to/pretrained/checkpoint.pt \
  --sft-bin /path/to/complexity-atlas-posttrain/tokenized/32k-v2 \
  --tokenizer ./tokenizer \
  --steps 900 \
  --batch-size 32 \
  --seq-len 512 \
  --lr 5e-5 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --bf16 \
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

The runner requires a positive LoRA rank and does not expose a full-parameter
SFT path. LoRA keeps the pretrained model weights frozen while adapting the
selected attention, shared MLP, and TR-Hash expert projections. Evaluation at
step zero establishes the pretrained baseline;
`--save-best` writes validation-selected checkpoints under `SAVE_DIR/best`, and
patience stops the run after consecutive non-improving evaluations.
`--save-model-only` omits AdamW and scheduler state for compact evaluation and
inference checkpoints. The held-out shard should contain at least 500
independently authored examples before its NLL is treated as a stable capability
estimate.

### Two-dimensional full-shard loss balancing (Card Corpus V2)

`configs/sft_500m_32k_v2_balanced.yaml` keeps every training example for one
complete epoch (`max_examples: all`, `balance_by: none`). It does not create a
smaller curriculum. At startup, the binary-shard loader counts the supervised
assistant labels that are actually visible after the 512-token context window,
then derives a loss multiplier for each task or explicit `task × domain` cell.
Balancing is hierarchical:

1. each loss group receives its configured global share;
2. each semantic loss cell receives its configured share inside that group.

| Invariant | Raw full shard | Optimizer behavior |
|---|---:|---:|
| Rows visited per epoch | 100% | 100% |
| Distilled reasoning | natural row/token count | 20% of weighted token loss |
| Natural conversation | natural row/token count | 20% of weighted token loss |
| Instruction + structured | natural row/token count | 60% of weighted token loss |

The first dimension is the group mixture. Measured on the 224,654-row audited
V2 32K train shard, it changes the effective gradient mixture without changing
row exposure:

| Loss group | Raw supervised-token share | Weighted-loss target |
|---|---:|---:|
| Distilled reasoning | 90.74% | 20% |
| Natural conversation | 1.41% | 20% |
| Instruction + structured | 7.85% | 60% |

The second dimension controls the task mixture inside each group. The current
audited targets and measured coefficients are:

| Group | Task family | Global weighted-loss target | Runtime coefficient |
|---|---|---:|---:|
| Reasoning | `reasoning_verification` | 9.25% | 0.1953x |
| Reasoning | `explanation_learning` | 8.00% | 0.2005x |
| Reasoning | `casual_reasoning` cell | 1.50% | 0.4555x |
| Reasoning | `planning_comparison` | 0.50% | 6.1332x |
| Reasoning | `troubleshooting` | 0.50% | 4.7781x |
| Reasoning | `critique_revision` | 0.25% | 25.6761x |
| Conversation | `casual_social` cell | 14.00% | 23.7657x |
| Conversation | `conversation_empathy` | 6.00% | 7.3117x |
| Instruction | `context_clarification` | 10.00% | 6.3253x |
| Instruction | `extraction_classification` | 8.00% | 9.0486x |
| Instruction | `grounded_qa` | 8.00% | 11.6392x |
| Instruction | `practical_action` | 4.00% | 25.3305x |
| Instruction | `brainstorming_creativity` | 3.50% | 25.6163x |
| Instruction | `safety_uncertainty` | 7.00% | 6.2611x |
| Instruction | `summarization_synthesis` | 7.00% | 5.3167x |
| Instruction | `writing_transformation` | 6.00% | 6.0676x |
| Instruction | `casual_instruction` cell | 6.50% | 6.6077x |

For loss cell `c` in group `g`, the runtime multiplier is:

```text
loss_weight(c) = group_target(g) * cell_target(c | g) / raw_token_share(c)
```

The group targets sum to one, every group's task targets sum to one, and every
selected loss cell must appear exactly once. `max_task_loss_weight: 30.0` is a hard
guard: planning and training fail if a shard change would require a larger
coefficient. The measured targets, coefficients, achieved group shares, and
achieved task shares are printed before any optimizer step.

The YAML expresses both dimensions directly:

```yaml
loss_groups:
  natural_conversation:
    target_share: 0.20
    tasks: [casual_social, conversation_empathy]
    task_target_shares:
      casual_social: 0.70
      conversation_empathy: 0.30
```

See [Two-dimensional full-shard SFT weighting](docs/sft-full-shard-2d-weighting.md)
for the complete framework contract, runtime audit fields, continuation
semantics, failure conditions, and regression suite.

With eight ranks and 24 examples per rank, this shard produces 1,171 optimizer
steps. The end of step 1,171 is the single-epoch save/evaluation boundary; the
base profile does not start a second epoch automatically. The tracked
`configs/sft_500m_32k_v2_balanced_continuation.yaml` profile performs two
additional complete passes from a selected epoch-one checkpoint using the same
two-dimensional targets. Every row is still shuffled and consumed; only its
assistant-token contribution to the gradient changes.

```bash
python -m scripts.run_sft_curriculum \
  --checkpoint artifacts/tr_hash_moe_500m_20b_hf \
  --sft-bin artifacts/complexity_card_corpus_v2_229026/tokenized/32k-v2 \
  --curriculum-config configs/sft_500m_32k_v2_balanced.yaml \
  --through-stage full-shard-weighted \
  --output-root artifacts/tr_hash_500m_32k_v2_weighted_lora \
  --world-size 8 --batch-size 24 --lora-rank 32 --lora-alpha 32 \
  --dry-run
```

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
deploy/                      supervisor and deployment configurations
spikes/                      isolated research prototypes
```

## Documentation

- [Documentation index](docs/index.md)
- [Architecture and naming](docs/architectures.md)
- [TR-MoE internals](docs/tr-hash-engine.md)
- [Getting started](docs/getting-started.md)
- [Training](docs/training.md)
- [Two-dimensional full-shard SFT weighting](docs/sft-full-shard-2d-weighting.md)
- [Run configurations](docs/run_configs.md)
- [GPU and dispatch paths](docs/cuda.md)
- [API reference](docs/api.md)
- [Historical TokenRoutedMLP migration](docs/token-routed.md)

## License

[CC BY-NC 4.0](LICENSE). Commercial use is not permitted by this repository's
current license.
