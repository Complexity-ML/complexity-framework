# Local training

## Integrated demonstration

Run the complete local workflow with a new or empty output directory:

```bash
complexity train demo --output artifacts/demo-cpu --device cpu
```

On a Mac where PyTorch reports MPS available:

```bash
complexity train demo --output artifacts/demo-mps --device mps
```

The command creates a TR-HASH model with 30,912 parameters, eight synthetic
training sequences and four held-out sequences. Each sequence has 12 tokens;
labels are already next-token targets, with masked context and a supervised
terminal EOS. Attention dropout exercises RNG restoration. There is no
download, tokenizer, HF access or tracking dependency.

The native Trainer performs two updates, publishes a complete transactional
checkpoint, reconstructs the model and AdamW optimizer, and resumes to update
four. The full cosine horizon remains four throughout. An explicit loader
restores the data cursor, so the resumed updates consume the remaining rows.
An independent uninterrupted four-update run verifies the continued weights,
AdamW moments, scheduler, RNG, sample order and logits. The command then
evaluates held-out loss, exports native SafeTensors and checks logits after
reloading that export. These are integration checks on synthetic data;
held-out loss is not a language-quality benchmark or an expected improvement.

Standard output is one JSON report. Diagnostics use standard error. The report
has `format="complexity-local-training-demo-v1"`, `passed`, `device`,
`planned_updates=4`, `pause_update=2`, `resumed_from_update=2`, `final_update=4`
and `data_cursor=4` (four batches, each containing two rows). The `checks`
object contains `model`, `optimizer`, `scheduler`, `rng`, `logits`, `cursor`,
`sample_order`, `parameters_updated` and `export_reload`, each with a `passed`
flag. Tensor comparisons also report `max_abs_diff`. CPU requires exact
equality; MPS uses the reported `comparison_tolerance=1e-6` for floating model,
optimizer and logits comparisons. RNG and scheduler comparisons remain exact.

Outputs include `fixtures.json`, `plan.json`, `report.json`,
`resumed/checkpoints/interrupted_2`, `resumed/checkpoints/final_4`, a separate
`reference` run and `export/config.json` plus `export/model.safetensors`.
The JSON report includes their absolute checkpoint/export/report paths.
An existing nonempty directory is rejected without overwriting its contents.
An unavailable explicitly requested MPS device fails before output creation;
there is no automatic CPU fallback. Run this single-process demonstration
without `torchrun`. Published models, the paused 1B run and its checkpoints
are outside its scope.

## Python API with explicit dataloaders

The reusable training API takes explicit dataloaders. This complete example
performs four diagnostic updates on synthetic tokens; it is not a language
model recipe or the paused 1B run. Temporary output avoids confusing smoke
artifacts with training checkpoints.

```python
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from complexity import ComplexityModel, ModelConfig
from complexity.training import Trainer, TrainingConfig

config = ModelConfig.load("configs/examples/tr_gqa_cpu.json", strict=True)
model = ComplexityModel(config)
generator = torch.Generator().manual_seed(42)
examples = [
    {"input_ids": torch.randint(0, config.vocab_size, (16,), generator=generator)}
    for _ in range(8)
]
loader = DataLoader(examples, batch_size=2, shuffle=False)

with TemporaryDirectory(prefix="complexity-smoke-") as output:
    training = TrainingConfig(
        max_steps=4,
        batch_size=2,
        learning_rate=1e-4,
        warmup_steps=0,
        lr_scheduler="constant",
        precision="fp32",
        use_fsdp=False,
        eval_steps=0,
        save_steps=4,
        num_workers=0,
        checkpoint_dir=str(Path(output) / "checkpoints"),
        log_dir=str(Path(output) / "logs"),
    )
    trainer = Trainer(model=model, config=training, train_dataloader=loader, device="cpu")
    trainer.train()
```

Pass `device="cpu"` or `device="mps"` to select a device explicitly. An
unavailable requested device fails clearly. Omitting `device` retains the
historical selection of available CUDA or CPU. Do not launch this example
under `torchrun` unless you intentionally want a distributed experiment.
For real data, batches provide `input_ids` and optional `attention_mask` and
`labels`; explicit labels are already next-token targets and `-100` masks
excluded positions. Dataloader reconstruction,
tokenizer identity and data cursor tracking belong to the experiment.

`batch_size` is per rank; gradient accumulation determines optimizer-update
boundaries. Configure these from the actual dataloader, rather than treating a
configuration's batch count as proof of its effective token budget. See
[optimizer boundaries](training-boundaries.md).

## Resume or initialize a new stage

For the generic trainer, an exact continuation uses `TrainingConfig.resume_from`
and the original model, dataloader, optimizer and schedule settings. A finite
loader must restore its cursor or use the runner's deterministic skip policy;
arbitrary iterable datasets do not gain replay guarantees from a checkpoint
alone.

Weights-only initialization uses a native export with
`TrainingConfig.init_checkpoint`; it starts new optimization state. Changed
corpus, optimizer grouping, training objective or schedule belongs to a new
experiment or stage, not an exact resume.

See [workflows](workflows.md) for packaged SFT, refinement and the preserved 1B runner.
