# Standalone contextual W/R/V review artifact

This directory is a small, self-contained implementation of the architecture evaluated in the accompanying paper. It does **not** import or require the authors' research framework.

## Included

- contextual Write/Read/Value attention using PyTorch SDPA/FlashAttention dispatch;
- matched grouped-query attention control;
- per-head R/W (Q/K for the control) RMS normalization and RoPE;
- tied lexical-object modulation and deterministic micro-expert residuals;
- a causal LM with tied token embeddings;
- deterministic FineWeb document splitting, training, evaluation, CSV logging, and four paper/ablation configs;
- focused tests for causality, full/incremental equivalence, lexical-off behavior, parameter counts, optimization, and framework independence.

The paper-size constructors reproduce the realized trainable parameter counts exactly:

- GQA: `98,179,844`
- contextual W/R/V, lexical residual disabled: `98,195,204`

## Install and test

Create a clean Python 3.11+ environment and install:

```bash
python -m pip install -e '.[test]'
pytest -q
```

## Run the model

```python
import torch
from mini_wrv import ModelConfig, TinyLanguageModel

model = TinyLanguageModel(ModelConfig.paper(attention_type="wrv"))
output = model(torch.randint(0, 200019, (1, 32)))
print(output["logits"].shape)
```

## Reproduce a training condition

Download the pinned FineWeb-Edu shard and verify it first:

- repository: `HuggingFaceFW/fineweb-edu`
- configuration: `sample-10BT`
- revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- shard: `sample/10BT/000_00000.parquet`
- SHA-256: `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871`

Then run, for example:

```bash
mini-wrv-train \
  --config configs/wrv_seed42.yaml \
  --fineweb-parquet /path/to/000_00000.parquet \
  --output runs/wrv-seed42
```

Documents whose zero-based index is divisible by 20 form the deterministic 5% evaluation stream; all other documents form the training stream. Documents are tokenized with `o200k_base`, separated by EOT, concatenated, and emitted as overlapping next-token windows of 2,048 inputs plus one target token.

## Scope and provenance

The raw CSV metrics and realized JSON configurations in `evidence/` come from the archived H200 runs. This standalone source is a minimal review implementation of the same equations, dimensions, data split, and parameterization; it is not claimed to be a byte-identical copy of every internal training utility used during the original runs. No checkpoint is required or included.
