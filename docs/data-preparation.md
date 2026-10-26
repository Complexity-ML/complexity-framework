# Data preparation


Corpus preparation and replay planning are package operations. The example
config describes a small local-text mixture; its dry run reads no source data,
loads no tokenizer and creates no output:

```bash
complexity data prepare configs/examples/corpus_mixture.json artifacts/local-mixture --dry-run
```

It reports 64 rows and 65,536 trained tokens. Supply `data/train.jsonl` with a
`text` field and an existing local tokenizer before creating the actual mixture:

```bash
complexity data prepare configs/examples/corpus_mixture.json artifacts/local-mixture \
  --tokenizer /path/to/local-tokenizer
complexity data inspect artifacts/local-mixture
complexity data replay artifacts/local-mixture configs/examples/corpus_replay.json \
  artifacts/local-replay.json
```

Preparation requires `sources`, `target_tokens`, `seq_len`,
`global_batch_sequences` and `rows_per_shard`. Sources identify an explicit
local file or dataset, text field and mixture weight. The uint16 token format
supports at most 65,536 tokenizer IDs. Requested tokens are rounded to complete
optimizer updates; exhausted sources fail rather than silently repeating data.
Existing completed output is not overwritten. Uploads are a separate operation.

Replay requires explicit `unique_token_budgets`, `replay_passes` and
`row_alignment`; optional shard `quality_scores` and
`already_double_exposed_shards` are mutually exclusive. It writes selection
metadata without copying or retokenizing shards. `inspect` checks manifests
and file sizes; the training reader verifies shard hashes. A plan is not a
training launch. See [streaming and replay](tr-hash-200m-streaming-data.md) for
the loader API and the separate recorded 200M source budgets.

Native Agentic SFT preparation lives in
`complexity.training.sft_preparation`. `prepare_sft_dataset` selects already
projected local prompt/completion records from explicit train/validation source
iterables and a quota recipe. It checks complete native tool/final targets,
rejects reasoning/legacy markers in targets, deduplicates prompts/source IDs
across splits and fails on exhausted quotas or overlength examples. It creates
JSONL files and source hashes, without downloading or publishing anything.
`mini_projected_record` and `parse_when2call_response` provide reusable native
tool projections. Source revisions and source adapters remain caller inputs.

Compile existing local `train.jsonl` and `eval.jsonl` into token-shard v2 with
the same native Agentic tokenizer and an explicit loss policy:

```bash
complexity data sft /path/to/jsonl-dataset /path/to/new-shards \
  --tokenizer /path/to/native-agentic-tokenizer \
  --sequence-length 2048 --assistant-supervision final_assistant_only
```

Compilation verifies declared source hashes, fixed marker IDs and complete
examples; it saves next-token labels with the final EOS, hashes, tokenizer and
chat metadata. `final_assistant_only` masks prior assistant actions and is
readable by the native SFT workflow. `all_assistant_turns` preserves the older
trajectory objective; that different objective is explicitly rejected by the
direct-SFT ablation reader. The encoder never invents reasoning or truncates
responses. Existing output is not overwritten, and a failed build leaves no
completed output. These checks are local artifact contracts, not dataset or
model quality measurements.

The [historical balanced recipe](../configs/examples/agentic_sft_balanced_100k.json)
records **100,000 train / 5,000 validation** examples, matching its archived
source. The old 200K/10K assertion was added in cleanup commit `1a90efd` without
a matching allocation recipe; the test now checks the verified historical
counts and provenance. This profile is historical and not release eligible.

## Collect exports and metrics


Collect a local run's model weights and explicitly selected metadata with a
JSON configuration tagged `complexity-artifact-collection-v1`:

```bash
complexity checkpoint collect /path/to/collection.json
complexity jobs metrics /path/to/metrics.jsonl --logdir /path/to/tensorboard
```

Collection requires `run_name`, `checkpoint_root`, `run_dir`, `data_root`,
`tokenizer` and a new `output_dir`; paths are relative to the configuration.
Optional `run_files` and `data_files` select metadata, and `log` selects a text
log. The default metadata names match the old 200M run layout. Native
SafeTensors exports are copied exactly; legacy `checkpoint.pt` is loaded with
`weights_only=True` and its tensor-only model is converted. The resulting
manifest records file sizes and SHA-256 hashes. Optimizer/RNG state is excluded:
this is a model export, not a resumable checkpoint or a Transformers conversion.
Native SFT optimization checkpoints supply their architecture from
`identity.plan.model_config`; the collector also accepts this explicit format.
Existing output, escaping metadata paths and unsafe serialized payloads fail.

Metrics export reads recorded CSV or JSONL scalars with a `step` field. It
requires the optional TensorBoard dependency; it does not start or monitor
training. `complexity.training.tensorboard.mirror_once` also supports a caller's
writer and deduplicates completed steps while retrying incomplete rows.
