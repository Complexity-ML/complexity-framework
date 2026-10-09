# Complexity Framework

A PyTorch research framework for **TR-HASH language models**: contextual
attention combined with deterministic token-ID routing to a shared SwiGLU path
and a small pool of experts. The framework owns model construction, training,
checkpoint handling and evaluation; numerical results belong to a specific
checkpoint and protocol.

The existing [TR-HASH-MoE-1B-70B-Agentic-Pretraining](https://huggingface.co/Pacific-i64/TR-HASH-MoE-1B-70B-Agentic-Pretraining)
run has a preserved **70-billion-token pretraining budget**. It is currently
paused and resumed periodically. Its model, original schedule, tokenizer and
FSDP checkpoint contract remain in `scripts/pretrain_1b_fsdp/`; improving the
framework does not create a new training recipe.

## Choose a task

| Task | Start here |
|---|---|
| Install and check a small CPU model | [Getting started](docs/getting-started.md) |
| Understand modules and public interfaces | [Framework architecture](docs/framework-architecture.md), [API](docs/api.md) |
| Author and validate a new configuration | [Run configurations](docs/run_configs.md) |
| Inspect or prepare a later resume of the paused 1B run | [Workflows](docs/workflows.md), [preserved FSDP recipe](scripts/pretrain_1b_fsdp/README.md) |
| Prepare data, train or evaluate a new local experiment | [Workflows](docs/workflows.md) |
| Review the separate 200M release and its measurements | [200M release](docs/tr-hash-200m-release.md) |
| Train/evaluate/export vision models | [Vision package and command tree](docs/vision/index.md) |
| Explore multimodal research | [Documentation index](docs/index.md) |

## Install

Use Python 3.10 or newer and an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

The installer selects the local backend. Explicit alternatives are
`make install-cpu`, `make install-mps`, `make install-cuda`, and
`make install-rocm`. PyTorch is installed separately from framework extras so
its backend matches the machine. If PyTorch is already installed correctly:

```bash
python -m pip install -e .
# Only needed for development checks:
python -m pip install -e '.[dev]'
```

Kernel and vision dependencies are optional; see [backend notes](docs/cuda.md)
and the [vision dependency contract](docs/vision/vision-dependency-stack.md).

## Check the installation

This configuration allocates a small, randomly initialized CPU model. It does
not download weights, resume a checkpoint or run an optimizer.

```bash
complexity train validate configs/examples/tr_gqa_cpu.json
complexity train inspect scripts/pretrain_1b_fsdp/recipe.json
```

```python
import torch
from complexity import ComplexityModel, ModelConfig

config = ModelConfig.load("configs/examples/tr_gqa_cpu.json", strict=True)
model = ComplexityModel(config).eval()
input_ids = torch.randint(0, config.vocab_size, (1, 16))
with torch.no_grad():
    logits = model(input_ids)["logits"]
assert logits.shape == (1, 16, config.vocab_size)
print(model.num_parameters())
```

For a native weight export, load the complete local directory:

```python
model = ComplexityModel.from_pretrained("checkpoints/my-model")
```

A distributed 1B training checkpoint is a different format. Use its original
FSDP restore and resume path; do not pass its shards to `from_pretrained`.

## Architecture in one view

```text
token IDs ─► embeddings ─► contextual attention ─► TR-MoE ─► logits
     │                                               ├─ shared SwiGLU
     └─ persisted per-layer route table ──────────────└─ selected experts
```

**TR-GQA** means GQA + TR-MoE; **TR-MHA** means MHA + TR-MoE. The canonical
MLP registry key is `tr_hash_engine` (`tr_hash_moe` is an alias). Experts
transform contextual hidden states while token identity selects the routes.
There is no learned router or auxiliary load-balancing loss in this path.

`intermediate_size` is the total stored routed width, divided by
`num_experts`; the shared branch uses `shared_intermediate_size`. Original
route buffers are part of checkpoint compatibility. Experimental attention
adapters named `tr_mha` and `tr_mha_v2` are separate research components, not
synonyms for the TR-MHA decoder. See [architecture and naming](docs/architectures.md).

## Training and checkpoint boundaries

Use `complexity.training.Trainer` with explicit dataloaders for a new local
experiment. The generic CLI validates and inspects configurations; a dataset
and a compatible checkpoint loader still belong to the concrete runner.
Read-only inspection does not prove a successful distributed restart.

The preserved 1B recipe targets 534,057 optimizer updates at 131,072 tokens
per update: **69,999,919,104 tokens**, the integer-update realization of the
70B budget. Refinement and SFT are separate future stages. Resuming this run
must retain its original optimizer, data cursor, scheduler and rank RNG state.
See [workflows](docs/workflows.md) before restoring or launching anything.

Native `save_pretrained` exports weights for loading/evaluation; it is not an
optimizer-state resume checkpoint. Production text generation is an external
serving boundary; `ComplexityModel.generate()` is intentionally disabled.
The [Transformers adapter](integrations/transformers/tr_hash_moe/README.md)
and [external inference API](docs/api.md#external-inference) document the
available interfaces.

## Release evidence

The separate released 200M lineage consists of base pretraining, interrupted
full-parameter refinement and completed full-parameter SFT v2. Its [release
page](docs/tr-hash-200m-release.md) retains architecture, checkpoint links,
token exposure, held-out metrics, PIQA protocol and limitations. Those results
do not qualify the paused 1B model or establish the proposed cost/quality
advantage; that remains a research question.

Historical experiment evidence remains in [RESULTS_100M_MPS.md](RESULTS_100M_MPS.md),
[TR_GQA.md](TR_GQA.md) and [TR_MHA.md](TR_MHA.md). Some referenced raw CSV
artifacts are absent from this checkout and are explicitly marked in those notes. Historical
launch profiles and experimental modules are labelled in the [documentation
index](docs/index.md).

## Development checks

```bash
make test-core  # selected current local model/API/config/CLI/vision contracts
make test       # complete repository suite
make lint       # the whole package, including tracked data modules
```

The complete local suite passes after useful deleted-script responsibilities
were integrated into the package. See [refactor status](docs/refactor-status.md)
for current results and validation scopes. GPU kernels, full-scale model quality and a real
four-rank checkpoint restart require separate validation against the selected
hardware and artifacts.

## Repository map

```text
complexity/                  public Python package and reusable components
complexity/vision/           grouped detection, image, video and visual-language code
complexity_cuda/             optional CUDA/Triton kernels
configs/                     model examples, plans and experiment configurations
scripts/pretrain_1b_fsdp/     preserved 1B model, recipe and resume tooling
scripts/                     preserved tooling and compatibility launchers
integrations/                external model/runtime adapters
tests/                       contract and regression checks
docs/                        task guides, release evidence and research notes
spikes/                      isolated research prototypes
```

[Documentation index](docs/index.md) · [API reference](docs/api.md) ·
[Framework architecture](docs/framework-architecture.md)

## License

[CC BY-NC 4.0](LICENSE). Commercial use is not permitted by this repository's
current license.
