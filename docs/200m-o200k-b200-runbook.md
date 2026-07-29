# Matched 200M o200k / 4B-token B200 runbook

This protocol trains a parameter-matched Dense GQA and fixed token-ID TR-GQA
pair. Both models contain exactly 200,081,920 trainable parameters. Each run
uses 3,999,793,152 target tokens with the same seed, tokenizer, optimizer,
schedule, frozen training stream, and disjoint held-out stream.

## Storage layout

The launcher defaults to local server NVMe:

```text
/workspace/data/fineweb_edu_o200k_4b/
  train/tokens.bin
  eval/tokens.bin
  dataset_manifest.json

/workspace/artifacts/complexity-200m-o200k/
  checkpoints/
  logs/
  releases/
```

The frozen binary dataset occupies about 16.07 GB. Keep at least 35 GB free for
dataset files, checkpoints, exported model weights, metrics, and logs.

FineWeb-Edu is pinned to revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The preparation command downloads
one Parquet file at a time, tokenizes it, appends EOS between documents, writes
little-endian uint32 token streams, and deletes the raw Parquet file. Training
then performs no dataset network I/O.

Every 200th source document is held out for evaluation. Held-out documents are
never written into the training stream.

## First rental: Dense run

Install the CUDA build of PyTorch first, then install this repository and run:

```bash
cd /workspace/complexity-framework

./scripts/run_200m_o200k_4xb200.sh prepare
./scripts/run_200m_o200k_4xb200.sh verify
./scripts/run_200m_o200k_4xb200.sh smoke-dense
./scripts/run_200m_o200k_4xb200.sh dense
./scripts/run_200m_o200k_4xb200.sh collect-dense
```

Do not start the full run unless `verify` and the 10-step smoke test both pass.

`collect-dense` creates a release directory containing:

- model-only `model.safetensors`;
- exact model and run configurations;
- `metrics.csv` and the complete training log;
- tokenizer files;
- dataset manifests and partition indices;
- SHA-256 digests for every release file and the full optimizer checkpoint.

The full resumable checkpoint remains under `checkpoints/`. Download it only if
an exact optimizer-state resume is needed.

## Before destroying the rental

Download at least:

```text
/workspace/artifacts/complexity-200m-o200k/releases/dense-gqa-200m-o200k-4b-s42/
```

For exact resume, also download:

```text
/workspace/artifacts/complexity-200m-o200k/checkpoints/dense-gqa-200m-o200k-4b-s42/
```

Verify the downloaded files against `artifact_manifest.json` before destroying
the server. The TR run must later reuse the original token shard binaries or
verified byte-identical copies with the same SHA-256 values.

## Second rental: fixed token-ID TR run

Restore the frozen dataset, run `verify`, then:

```bash
cd /workspace/complexity-framework

./scripts/run_200m_o200k_4xb200.sh tr
./scripts/run_200m_o200k_4xb200.sh collect-tr
```

Never rebuild the dataset with a new revision, tokenizer, or partition rule
between the Dense and TR runs.
