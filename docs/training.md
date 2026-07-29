# Training

The supported research runner is `cf-o200k-pretrain`, implemented by
`complexity.training.o200k_pretrain`.

It runs on CPU, MPS, CUDA, or ROCm through PyTorch and supports single-process
or DDP execution.

## What the runner records

Rank zero writes:

```text
runs/<run-name>/
├── run_config.json
└── metrics.csv
```

`run_config.json` contains:

- resolved YAML and CLI arguments;
- `ModelConfig`;
- trainable parameter count;
- Git commit;
- world size and backend metadata;
- tokens per step and total tokens.

`metrics.csv` includes training/evaluation loss, perplexity, learning rate,
throughput, and optional expert diagnostics.

## TR-GQA

The o200k profiles default to:

```text
attention_type = gqa
mlp_type       = token_routed
shared_expert  = true
num_experts    = 4
```

Example:

```bash
cf-o200k-pretrain \
  --profile 100m \
  --dataset tokens \
  --tokens-path /data/fineweb_tokens \
  --tokenizer ./tokenizer-o200k \
  --steps 1000 \
  --batch-size 8 \
  --seq-len 2048 \
  --routing-strategy zipf \
  --top-k 2 \
  --top-k-primary-weight 0.5 \
  --bf16 \
  --run-name tr-gqa-100m
```

For `dataset=tokens`, the shard must provide `tokens.bin` and
`tokens.idx.json`.

## TR-MHA

TR-MHA keeps `mlp_type=token_routed` and changes attention:

```bash
cf-o200k-pretrain \
  --profile 100m \
  --dataset text \
  --text-file /data/sample.txt \
  --tokenizer ./tokenizer-o200k \
  --attention-type mha \
  --num-key-value-heads 8 \
  --routing-strategy modulo_balanced_secondary \
  --top-k 2 \
  --top-k-primary-weight 0.5 \
  --steps 250 \
  --run-name tr-mha-100m
```

The head count must match the selected profile. A tracked, parameter-matched
example is:

```text
configs/run_configs/experiments_100m/
└── 100m_params_mha_modulo_balanced_shared_1296_mps.yaml
```

## Data modes

| Mode | Input | Notes |
| --- | --- | --- |
| `random` | none | smoke tests only |
| `text` | `--text-file` | local text, deterministic train/eval split |
| `tokens` | `--tokens-path` | memory-mapped pretokenized data |
| `fineweb` | dataset stream | may require network and dataset dependencies |

For `text` and `tokens`, the runner can derive token-frequency counts used by
`zipf` and `modulo_balanced_secondary`.

## Loss backends

- `chunked`: exact tied-head causal cross entropy in token chunks;
- `liger`: fused linear cross entropy when `liger-kernel` is available;
- `auto`: select Liger when available, otherwise chunked.

`--loss-chunk-tokens` controls memory use for the chunked path. It does not
change the target distribution.

## Optimization

The runner supports AdamW and the experimental `muon_tr` optimizer. Do not
compare optimizer variants without matching all remaining settings.

Relevant controls include:

- BF16 autocast;
- gradient checkpointing;
- shared-expert token chunking;
- optional `torch.compile`;
- gradient clipping;
- label smoothing and z-loss;
- shared/routed gate schedules;
- top-k primary-weight schedules;
- optional expert-diversity penalty.

## Expert telemetry

Enable with:

```bash
--moe-telemetry
```

This records route shares, dead-expert count, shared/routed RMS, gates, and
gradient norms. It adds synchronization and reduction overhead, so throughput
measurements should state whether telemetry was enabled.

## DDP

```bash
torchrun --nproc_per_node=8 \
  -m complexity.training.o200k_pretrain \
  --config configs/run_configs/100m_o200k_tr_30b_b300.yaml
```

Token accounting in the current runner is:

```text
tokens_per_step = batch_size × sequence_length × world_size
```

The local runner does not currently expose gradient accumulation. Cluster-plan
YAMLs may include it for arithmetic validation, but they are not automatically
launchable by the local DDP runner.

## Evaluation language

Always distinguish:

- training NLL;
- evaluation NLL on a training-split stream;
- genuinely held-out NLL;
- downstream evaluation.

A fixed stream from the training split is repeatable but not held out.

## Resume safety

```bash
cf-o200k-pretrain \
  --config path/to/run.yaml \
  --resume checkpoints/run/step-1000
```

The runner compares the previous and current resolved configuration and rejects
training-critical mismatches. `--force-resume` should be used only after
manually reviewing the difference.

## Reproducible comparison checklist

- same tokenizer and vocabulary;
- same data order and split;
- same token budget;
- same parameter count or explicitly reported mismatch;
- same seed set;
- same optimizer and schedule;
- same precision and hardware;
- same telemetry and compile settings;
- same evaluation checkpoints;
- throughput measured over the same interval.
