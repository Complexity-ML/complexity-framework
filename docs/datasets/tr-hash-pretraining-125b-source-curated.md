---
pretty_name: TR-HASH Pretraining 125B Agentic 32K
language:
- en
- fr
task_categories:
- text-generation
license: other
---

# TR-HASH Pretraining 125B — Agentic 32K

Private, source-curated pretraining artifact for the TR-HASH Agentic 32K model
line. It contains 125B packed token exposures: 75B foundation and 50B
agentic/procedural content.

The corpus uses the immutable, validated 32,000-ID revision of
`AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic`. It is not compatible with the
older TR-HASH 32K tokenizer.

## Composition

| Bucket | Tokens | Purpose |
|---|---:|---|
| Foundation | 75B | English and French knowledge, educational web text, math and synthetic textbooks |
| Agentic | 50B | Educational code, procedures, debugging, verification, planning, math reasoning and capped tool-use trajectories |

All source repositories, configurations, revisions, token budgets, and license
audit notes are pinned in the published `_metadata/config.json`.

## Materialization disclosure

This is the high-throughput **source-curated direct** build. Documents are read
from curated upstream subsets and tokenized directly. The build intentionally
does **not** claim per-document quality filtering, agentic-signal filtering,
benchmark decontamination, or global exact-document deduplication.

Each one-billion-token `uint16` shard is uploaded, checked against its remote
size and SHA-256, committed to restart state, and only then deleted locally.
The dataset remains private until source licenses, shard hashes, token budgets,
and this materialization disclosure have been audited.

## Curriculum

The runtime plan consumes every packed row once:

1. `foundation-first`: 60B foundation + 15B agentic.
2. `agentic-intensification`: 15B foundation + 35B agentic.

Shard boundaries are aligned to the phase split. `unique_tokens` and
`trained_tokens` describe packed token positions; source documents may repeat
because global deduplication is disabled in this direct build.
