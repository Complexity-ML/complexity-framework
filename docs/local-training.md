# Local training with the Python API


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
    trainer = Trainer(model=model, config=training, train_dataloader=loader)
    trainer.train()
```

The trainer selects an available CUDA device or CPU. For a CPU-only check on a
GPU host, run the example with `CUDA_VISIBLE_DEVICES=""`. Do not launch it
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
