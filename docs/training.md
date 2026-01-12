# Training Guide

Train models with the Complexity Framework.

## Quick Start

```python
from complexity.api import Tokenizer, Model, Dataset, Trainer, TrainerConfig

# Load components
tokenizer = Tokenizer.load("llama-7b")
model = Model.load("llama-7b", device="cuda")
dataset = Dataset.load("./train.jsonl", tokenizer=tokenizer)

# Train
trainer = Trainer(model, dataset)
trainer.train()
```

## Training Configuration

```python
from complexity.api import TrainerConfig

config = TrainerConfig(
    # Optimization
    learning_rate=1e-4,
    weight_decay=0.1,
    warmup_steps=1000,
    max_steps=100000,

    # Batch
    batch_size=4,
    gradient_accumulation_steps=8,

    # Mixed precision
    fp16=True,
    bf16=False,

    # Checkpointing
    save_steps=1000,
    save_dir="./checkpoints",

    # Logging
    log_steps=100,
    wandb_project="my-project",
)

trainer = Trainer(model, dataset, config=config)
```

## Small Budget Training

For limited GPU memory:

```python
from complexity.api import Efficient, SmallModels

# Small model (~125M params)
model = SmallModels.tiny_llm(vocab_size=32000)

# Enable memory optimizations
Efficient.enable_checkpointing(model)
model, optimizer, scaler = Efficient.mixed_precision(model, optimizer)

# Estimate memory usage
mem = Efficient.estimate_memory(model, batch_size=4, seq_len=2048)
print(f"Estimated: {mem['total_gb']:.1f} GB")

# Get recommended config for your GPU
config = Efficient.recommend_config(vram_gb=12, training=True)
```

## INL Dynamics Integration

For stable long training:

```python
from complexity.api import INLDynamics

# Add dynamics to your model
dynamics = INLDynamics(
    hidden_size=768,
    beta_max=2.0,      # CRITICAL: keeps beta in [0, 2]
    velocity_max=10.0,  # Prevents runaway
)

# In forward pass
h_next, v_next = dynamics(hidden_states, velocity_states)
```

**Why velocity tracking?**
- Acts as damper/shock absorber
- Stabilizes training on frequent tokens (the, a, is)
- Prevents explosion at long training (400k+ steps)

## Streaming Dataset

For large datasets:

```python
from complexity.api import StreamingDataset, DataPipeline

# Stream from disk
dataset = StreamingDataset(
    path="./data/train.jsonl",
    tokenizer=tokenizer,
    max_seq_len=2048,
    shuffle_buffer=10000,
)

# Or use DataPipeline for complex preprocessing
pipeline = DataPipeline(
    source="./data/",
    tokenizer=tokenizer,
    transforms=[
        lambda x: x["text"],
        # Add more transforms
    ],
)
```

## Distributed Training

```python
import torch.distributed as dist
from complexity.api import Trainer

# Initialize distributed
dist.init_process_group("nccl")

# Trainer handles DDP automatically
trainer = Trainer(
    model,
    dataset,
    distributed=True,
)
trainer.train()
```

## Checkpointing

```python
from complexity.api import Trainer

trainer = Trainer(model, dataset)

# Save checkpoint
trainer.save_checkpoint("./checkpoint-1000")

# Resume training
trainer.load_checkpoint("./checkpoint-1000")
trainer.train()
```

## Gradient Checkpointing

Save memory by recomputing activations:

```python
from complexity.api import Efficient

# Enable gradient checkpointing
Efficient.enable_checkpointing(model)

# Or manually on specific layers
for layer in model.layers[::2]:  # Every other layer
    layer.gradient_checkpointing = True
```

## Mixed Precision

```python
from complexity.api import MixedPrecision

# Setup
scaler = MixedPrecision.setup(model, optimizer)

# Training loop
with MixedPrecision.autocast():
    loss = model(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Monitoring

```python
from complexity.api import Debug

# Parameter count
print(Debug.count_params(model))  # "125M"

# Memory usage
mem = Debug.memory_usage()
print(f"GPU: {mem['allocated_gb']:.1f} GB")

# Model summary
Debug.print_summary(model)

# INL Dynamics stats
if hasattr(model, 'dynamics'):
    stats = model.dynamics.get_dynamics_stats()
    print(f"mu mean: {stats['mu_mean']:.3f}")
```

## Common Issues

### Out of Memory
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision (fp16/bf16)
4. Use smaller sequence length

### Training Instability
1. Check beta is clamped to [0, 2]
2. Enable velocity tracking
3. Reduce learning rate
4. Add gradient clipping

### Slow Training
1. Use Flash Attention
2. Enable mixed precision
3. Increase batch size if memory allows
4. Use O(N) architectures for long sequences

## See Also

- [Efficient Training](efficient.md) - Memory optimization
- [INL Dynamics](dynamics.md) - Stability system
- [CUDA Optimizations](cuda.md) - Flash Attention
