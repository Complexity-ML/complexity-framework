# TR-HASH 125B corpus build runbook

This runbook prepares the production corpus build on the 120-core CPU host.
It deliberately does **not** permit final tokenization with an unvalidated or
mutable tokenizer.

## 1. Static preflight (safe before tokenizer completion)

```bash
cd /workspace/complexity-framework
python -m scripts.build_tr_hash_pretraining_125b \
  --curriculum configs/agentic_pretraining/tr_hash_pretraining_125b_curriculum.json \
  --validate-only
```

Expected invariants:

- requested tokens: `125000000000`;
- buckets: `75000000000` foundation and `50000000000` agentic;
- every source allocation is a multiple of the 64-sequence global batch;
- tokenizer status is `validated` in the canonical config after step 2; an
  older checkout must remain `pending-validation` and refuse to build.

## 2. Freeze and pin the final Agentic 32K tokenizer

First publish the validated tokenizer files to
`AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic`, obtain its immutable 40-character
revision, and download that exact snapshot. Never point the corpus build at a
moving `main` checkout.

The local snapshot must contain at least:

- `tokenizer.json` with exactly 32,000 IDs;
- `agentic_tokenizer_manifest.json`;
- all markers from the `tr_hash_agentic_reasoning` format;
- tokenizer configuration and chat template files.

Pin the snapshot into the corpus config:

```bash
python -m scripts.pin_tr_hash_pretraining_tokenizer \
  --config configs/agentic_pretraining/tr_hash_pretraining_125b.json \
  --tokenizer /workspace/tokenizers/tr-hash-agentic-32k-<REVISION> \
  --revision <40-CHARACTER-REVISION>
```

This command validates the vocabulary and marker IDs, then records the
revision, manifest SHA-256 and `tokenizer.json` SHA-256 atomically. Review and
commit that config change before launching production.

## 3. Two-stage architecture and capacity pilot

The production launcher no longer filters, globally merges and packs raw
network streams in one round-robin loop. That design measured only about
68,500 reference tokens/s because the slowest upstream repeatedly blocked the
other sources.

The corrected pipeline has two restart-safe stages:

1. twelve independent source processes fetch, filter, decontaminate, apply
   within-source exact deduplication, tokenize only for quota accounting, then
   upload deterministic gzip candidate shards under `_candidates/`; every
   upload is size/SHA-256 verified before its local file is evicted;
2. one canonical-order merger downloads one candidate shard at a time,
   verifies it, performs global exact deduplication, packs final uint16 token
   shards, uploads/verifies them under `production/`, then evicts both inputs
   and outputs locally.

Before the full build, use a private pilot repository and temporarily lower
the source targets in a copied config. Do not edit the canonical 125B budgets:

```bash
export TR_HASH_125B_CREATE_PRIVATE_REPO=1
export TR_HASH_125B_HF_REPO=AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K-Pilot
export TR_HASH_125B_TOKENIZER=/workspace/tokenizers/tr-hash-agentic-32k-<REVISION>
export TR_HASH_125B_WORK_DIR=/workspace/builds/tr-hash-pretraining-125b-pilot
export TR_HASH_125B_STAGE_ONLY=1
export TR_HASH_125B_SOURCE_WORKERS=12

scripts/run_tr_hash_pretraining_125b_build.sh
```

Inspect every per-source retained/scanned ratio, the aggregate throughput and
the remote candidate hashes. Do not start the canonical 125B build unless its
extrapolated extraction time is acceptable and every source can supply its
quota plus the configured 5% candidate margin.

## 4. Production launch

Store `HF_TOKEN` in a root-readable environment file rather than the command
line or repository. The build uploads each shard, checks the remote size and
SHA-256, commits its SQLite state, and only then evicts the local shard.

The launcher runs candidate extraction first and starts final packing only when
all source manifests are complete. A restart uses the same command and paths.
Only verified shards are durable progress; an uncommitted candidate partial is
discarded and deterministically regenerated. The final merger is independent
of source completion timing and preserves canonical config order.

```bash
export HF_TOKEN='<loaded from protected environment file>'
export TR_HASH_125B_TOKENIZER=/workspace/tokenizers/tr-hash-agentic-32k-<REVISION>
export TR_HASH_125B_WORK_DIR=/workspace/builds/tr-hash-pretraining-125b
export TR_HASH_125B_HF_REPO=AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K
export TR_HASH_125B_SOURCE_WORKERS=12

scripts/run_tr_hash_pretraining_125b_build.sh
```

For systemd, place the variables in a mode-0600 `EnvironmentFile` and use the
launcher as `ExecStart`. The same command and work directory are the resume
operation: committed shards are not regenerated, while missing manifests are
republished.

## 5. Live monitoring

```bash
journalctl -u tr-hash-pretraining-125b.service -f -o cat
```

During extraction, inspect the independent durable states without modifying
them:

```bash
find /workspace/builds/tr-hash-pretraining-125b/candidates -name state.sqlite3 -print0 | \
  xargs -0 -n1 sqlite3 \
  'SELECT source, scanned, retained_tokens, retained_records FROM progress;'
```

During final packing:

```bash
sqlite3 /workspace/builds/tr-hash-pretraining-125b/final/state.sqlite3 \
  'SELECT source, rows_done, scanned, source_tokens FROM progress ORDER BY source;'
```

## 6. Release acceptance

Keep the Hugging Face dataset private until all of the following exist and
agree:

- root `README.md`, `mixture_manifest.json` and `pretrain_plan.json`;
- `_metadata/config.json` and `_metadata/curriculum.json`;
- one complete `corpora/<source>/manifest.json` per frozen source;
- remote size and SHA-256 for every shard;
- `unique_tokens == trained_tokens`, each shard referenced exactly once and
  `source_passes == 1` in the runtime plan;
- non-zero protected-prompt count and a stable protected-index SHA-256;
- audited source and row-level redistribution rights.

Only after this audit should the dataset visibility be changed from private.

Run the remote metadata/LFS audit with:

```bash
python -m scripts.audit_tr_hash_pretraining_125b_release
```

This checks every manifest and curriculum invariant and compares every remote
shard against its expected LFS size and SHA-256 without downloading the shard
payloads again.
