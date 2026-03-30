# Efficient - Small Budget Training

**Optimizations for training on limited hardware.**

## Recommendations by GPU

```python
from complexity.api import Efficient

# Detect VRAM
vram = Efficient.get_vram()
print(f"Available VRAM: {vram:.1f} GB")

# Recommended config
config = Efficient.recommend_config(vram_gb=vram, training=True)
print(config)
```

| VRAM | Model | Batch | Seq Len | Optimizations |
|------|-------|-------|---------|---------------|
| 80 GB (A100) | medium | 32 | 4096 | - |
| 40 GB (A100) | small | 16 | 4096 | flash_attention |
| 24 GB (3090/4090) | tiny | 8 | 2048 | flash, checkpointing |
| 12 GB (3060/4070) | micro | 4 | 1024 | flash, checkpointing, mixed_precision |
| 8 GB (3050) | nano | 2 | 512 | checkpointing, mixed_precision, accumulation |

## Pre-configured Models

```python
# Nano (~10M params) - CPU friendly
model = Efficient.nano_llm(vocab_size=32000)

# Micro (~30M params) - Single GPU
model = Efficient.micro_llm(vocab_size=32000)

# Tiny (~125M params) - Consumer GPU (RTX 3060+)
model = Efficient.tiny_llm(vocab_size=32000)

# Small (~350M params) - RTX 3080+
model = Efficient.small_llm(vocab_size=32000)
```

## Gradient Checkpointing

Reduces memory at the cost of compute (~30% slower).

```python
model = Efficient.enable_checkpointing(model)
```

## Mixed Precision (FP16/BF16)

2x less memory, faster.

```python
model, optimizer, scaler = Efficient.mixed_precision(model, optimizer)

# Training loop
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(inputs)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Quantization

### INT8 Inference

```python
model = Efficient.quantize.dynamic(model)  # INT8 dynamic
```

### INT4 Inference

```python
model = Efficient.quantize.static(model, bits=4)
```

### Estimate Savings

```python
savings = Efficient.estimate_savings(model, bits=4)
print(savings)
# {'original': '7.00 GB', 'quantized': '1.75 GB', 'savings': '75.0%'}
```

## Memory Estimation

```python
mem = Efficient.estimate_memory(
    model,
    batch_size=4,
    seq_len=2048,
)
print(mem)
# {
#     'model': '500 MB',
#     'activations': '2.1 GB',
#     'gradients': '500 MB',
#     'optimizer': '1 GB',
#     'total': '4.1 GB'
# }
```

## Gradient Accumulation

Simulates a larger batch.

```python
accumulation_steps = 8
effective_batch_size = batch_size * accumulation_steps

for step, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Complete Small Budget Setup

```python
from complexity.api import Efficient, Helpers
import torch

# 1. Create model
model = Efficient.tiny_llm(vocab_size=32000).cuda()
print(f"Params: {Helpers.count_params(model)}")

# 2. Optimizations
Efficient.enable_checkpointing(model)

# 3. Optimizer with mixed precision
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model, optimizer, scaler = Efficient.mixed_precision(model, optimizer)

# 4. Estimate memory
mem = Efficient.estimate_memory(model, batch_size=4, seq_len=2048)
print(f"Estimated memory: {mem['total']}")

# 5. Training loop
accumulation_steps = 4

for step, batch in enumerate(dataloader):
    with torch.autocast("cuda"):
        loss = model(batch["input_ids"]) / accumulation_steps

    scaler.scale(loss).backward()

    if (step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Memory-Efficient Optimizers

```python
from complexity.training import get_optimizer

# AdaLomo - O(1) memory (no optimizer states!)
optimizer = get_optimizer(model, "adalomo", lr=1e-2)

# 8-bit Adam - 75% less memory
optimizer = get_optimizer(model, "adam8bit", lr=1e-4)

# LION - No variance state
optimizer = get_optimizer(model, "lion", lr=1e-5)
```
