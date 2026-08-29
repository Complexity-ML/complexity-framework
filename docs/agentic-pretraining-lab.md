# TR-HASH agentic pretraining lab

This laboratory creates a new TR-HASH lineage. It does not resize or mutate the
canonical 32K tokenizer or any released model.

## Contract

- Vocabulary size remains exactly 32,000.
- IDs 0-19 are reserved before pretraining for roles, tool calls/results,
  planning, memory, private reasoning, and final answers.
- The remaining 31,980 entries are learned from the audited raw-text mixture.
- The pretraining corpus is 60% general text and 40% agentic-selected text.
- ARC Easy, ARC Challenge, PIQA, GSM8K, HellaSwag, MMLU, TruthfulQA, and
  Winogrande references are excluded by the first-pass contamination filter.
- Every retained document records its source, content SHA-256, source record ID,
  detected agentic signals, and selection score.
- Public redistribution remains blocked until every source license is audited.

## Build pipeline

```bash
python -m scripts.build_agentic_pretraining_corpus \
  --config configs/agentic_pretraining/tr_hash_small_agentic_50m.json \
  --output-dir artifacts/tr_hash_small_agentic_raw

python -m scripts.train_tr_hash_agentic_tokenizer \
  --corpus-dir artifacts/tr_hash_small_agentic_raw \
  --output-dir artifacts/tr_hash_agentic_tokenizer_32k

python -m scripts.tokenize_agentic_pretraining_corpus \
  --corpus-dir artifacts/tr_hash_small_agentic_raw \
  --tokenizer artifacts/tr_hash_agentic_tokenizer_32k \
  --output artifacts/tr_hash_small_agentic_50m_tokens \
  --target-tokens 50000000 \
  --seq-len 1024 \
  --global-batch-sequences 8
```

The tokenization stage rounds the requested budget to complete optimizer
updates. For this configuration it produces 50,003,968 trained tokens and a
single-pass `pretrain_plan.json` named `unique_core`.

## Home-server run

First supervise the CPU-only dataset and tokenizer preparation. This job does
not reserve the eGPU and can coexist with inference:

```bash
sudo /home/boris/TR-Hash-Server/.venv/bin/tr-hash-server job submit \
  /home/boris/complexity-framework/tr-hash-agentic-data-50m.toml \
  --user boris \
  --group boris \
  --enable-on-boot
```

Follow preparation:

```bash
sudo journalctl -u tr-hash-job-tr-hash-agentic-data-50m.service -f -o cat
```

Then submit the GPU training job:

```bash
sudo /home/boris/TR-Hash-Server/.venv/bin/tr-hash-server job submit \
  /home/boris/complexity-framework/tr-hash-small-agentic-50m.toml \
  --user boris \
  --group boris \
  --enable-on-boot
```

Follow the run:

```bash
sudo journalctl -u tr-hash-job-tr-hash-small-agentic-50m.service -f -o cat
```

The persistent TensorBoard service reads all experiment directories under
`artifacts/`; access remains private through an SSH tunnel to port 6006.
