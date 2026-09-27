---
license: other
library_name: complexity
language:
- en
- fr
pipeline_tag: text-generation
---
# TR-HASH MoE 1B — Agentic Pretraining

Pretraining in progress: 1.01B parameters, 8 routed experts (top-2), a shared branch, 16K context and a 32K tokenizer. Target: approximately 70B tokens. This is not an instruction-tuned model.

`latest.json` identifies the latest verified, resumable checkpoint. The repository retains the three latest regular checkpoints plus the latest interruption checkpoint, at most four snapshots. Old checkpoint history is purged automatically.

Checkpoints include distributed model and optimizer state. Consolidated inference weights will be published at the root after training completes.

[Training recipe and resumption](training/README.md).
