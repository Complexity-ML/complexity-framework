# Corpus streaming and replay

Corpus loading, tokenization and replay planning live inside the framework.
Use [current data workflows](workflows.md#prepare-and-inspect-text-data) for
`complexity data prepare`, `inspect` and `replay`. The 200M budgets below are
recorded release evidence; they do not replace the paused 1B/70B recipe.

## Current interfaces

- `complexity.training.corpus_preparation`: quota-controlled token shards and
  manifests from explicit sources/tokenizer; preparation does not upload.
- `complexity.training.corpus_mixture`: local or lazily cached Hub uint16
  streams, manifest validation, per-rank reading and replay-plan consumption.
- `complexity.training.replay`: deterministic zero-copy selection,
  quality-ranked replay and correction of already double-exposed shards.

A local mixture can be inspected and planned without training:

```bash
complexity data inspect /path/to/local-mixture
complexity data replay /path/to/local-mixture /path/to/replay-config.json \
  /path/to/new-replay-plan.json
```

Replay configuration explicitly names `unique_token_budgets`, `replay_passes`
and `row_alignment`. Optional `quality_scores` maps every candidate shard path
to a finite score; incomplete coverage fails. Without scores, selection follows
stable manifest order. Exposure correction uses
`already_double_exposed_shards` and fresh backfill shards, and cannot be mixed
with quality-ranked selection.

For remote loading, use the package API with a pinned dataset revision:

```python
from complexity.training import PretokenizedCorpusMixtureDataset

dataset = PretokenizedCorpusMixtureDataset(
    "hf://datasets/Pacific-i64/data-32k-200b-tokens",
    revision="PINNED_DATASET_REVISION",  # replace with the audited immutable revision
    cache_dir="/workspace/token-cache",
    cache_max_bytes=10 * 1024**3,
    prefetch_shards=1,
)
```

Manifests are fetched during preflight, while binary shards are loaded lazily,
memory-mapped and evicted from a bounded shared cache. Inspection checks
manifests and file sizes; the reader checks supplied shard hashes. A fresh
cache, remote credentials and sustained download performance remain separate
operational checks. A valid plan does not launch or qualify training.

## Recorded 200M source budgets

The immutable 200B pretokenized mixture supplied the recorded 70B unique-token
selection and approximately 130B trained-token exposures:

| Source | Unique tokens | Passes |
|---|---:|---:|
| DCLM | 20B | 1 |
| FineWeb-Edu-Dedup | 25B | 2 |
| Stack-Edu | 8B | 2 |
| FineMath | 5B | 3 |
| InfiWebMath | 5B | 3 |
| Cosmopedia v2 | 7B | 2 |

Use the plan's exact `unique_tokens` and `trained_tokens`: row alignment can
change rounded totals. Unique source material and repeated training exposure
are different quantities. These source maps are available as explicit
reference constants in `complexity.training.replay`; callers must supply their
chosen maps to the planner.

The old `build_tr_hash_70b_replay_plan` script's reusable behavior is now the
packaged planner. The one-off `vast_pretrain_tr_hash_200m_70b_replay.sh` release
launcher remains unavailable; its historical 24-GiB cache setting and completed
130B checkpoint do not constitute a current resume command.

The loader preserves per-rank cursor/resume inputs and checks declared
exposure against the plan. A concrete trainer must still own the optimizer
horizon and compatible checkpoint state. For job supervision use the shared
[Supervisor interface](supervisor-jobs.md), and for measurements consult the
[200M release evidence](tr-hash-200m-release.md).
