# API reference

This page covers the stable research-facing surface. The source remains the
authority for experimental fields.

## Top-level imports

```python
from complexity import ComplexityModel, ModelConfig
```

The package also exports:

- component registries and registration decorators;
- GQA, MHA, and MQA classes;
- the canonical TR-Hash MLP and experimental registered components;
- normalization and position-embedding components.

## `ModelConfig`

### Model shape

| Field | Meaning |
| --- | --- |
| `hidden_size` | residual width |
| `num_hidden_layers` | decoder depth |
| `intermediate_size` | routed expert pool width or dense FFN width |
| `vocab_size` | tokenizer vocabulary |
| `max_position_embeddings` | configured context bound |

### Attention

| Field | Meaning |
| --- | --- |
| `attention_type` | registry key such as `gqa`, `mha`, `tr_mha_v2` |
| `num_attention_heads` | query head count |
| `num_key_value_heads` | K/V head count |
| `use_qk_norm` | Q/K RMS normalization |
| `use_sdpa` | PyTorch SDPA path |
| `sliding_window` | optional local-attention window |

### TR-MoE

| Field | Meaning |
| --- | --- |
| `mlp_type` | set `tr_hash_engine` for TR-MoE |
| `num_experts` | routed expert count |
| `routing_strategy` | `modulo_cyclic` or `token_id_balanced_hash` |
| `shared_expert` | enable dense shared SwiGLU |
| `shared_intermediate_size` | shared branch width |
| `shared_expert_chunk_tokens` | chunk shared computation over tokens |
| `top_k` | number of deterministic expert routes |
| `top_k_primary_weight` | blend assigned to primary route |
| `use_shared_routed_gates` | learn shared/routed scalar gates |
| `collect_moe_telemetry` | collect route/RMS diagnostics |
| `use_custom_kernels` | custom-kernel policy |
| `use_cggr` | grouped-GEMM policy |

Configuration validates shape, routing, and range invariants in
`ModelConfig.__post_init__`.

## `ComplexityModel`

### Construct

```python
model = ComplexityModel(config)
model = ComplexityModel.from_config("config.yaml")
model = ComplexityModel.from_pretrained("checkpoint-directory")
```

### Forward

```python
result = model(
    input_ids,
    attention_mask=None,
    past_key_values=None,
    use_cache=False,
    return_hidden_states=False,
    return_logits=True,
)
```

Return mapping:

| Key | Value |
| --- | --- |
| `logits` | `[batch, sequence, vocabulary]`, or `None` |
| `last_hidden_state` | final normalized hidden states |
| `past_key_values` | optional per-layer cache/state list |
| `hidden_states` | optional embedding and layer states |

Set `return_logits=False` for fused or chunked tied-head loss paths.

### Save and load

```python
model.save_pretrained("checkpoint")
restored = ComplexityModel.from_pretrained("checkpoint")
```

For distributed DTensor/FSDP saves, every rank must enter
`save_pretrained` because full-tensor gathering is collective.

### Generation

`model.generate()` intentionally raises `RuntimeError`. Use the external
serving client.

## External inference

```python
from complexity.inference import (
    ExternalGenerationConfig,
    OpenAICompatibleBackend,
    create_external_backend,
)
```

`create_external_backend` accepts `"vllm"` or `"sglang"` and calls
`/v1/completions` or `/v1/chat/completions`.

```python
backend = create_external_backend(
    "vllm",
    base_url="http://localhost:8000",
    model="tr-gqa",
)
answer = backend.chat(
    [{"role": "user", "content": "Summarize the experiment."}],
    ExternalGenerationConfig(max_tokens=128),
)
```

The client is synchronous and non-streaming in the current implementation.

## Component registries

```python
from complexity.core.registry import (
    ATTENTION_REGISTRY,
    MLP_REGISTRY,
    NORMALIZATION_REGISTRY,
    POSITION_REGISTRY,
    register_attention,
    register_mlp,
)
```

Principal attention keys:

```text
gqa, mha, mqa
tr_mha, tr_mha_v2
lexical_gqa, lexical_key_gqa
causal_conv, causal_state_conv, causal_fast_weight_conv
```

Principal MLP keys:

```text
tr_hash_engine, tr_hash_moe
dense_deterministic
lexical_modulated, lexical_channel_modulated, lexical_object_micro_expert
```

`swiglu`/`gelu`/`geglu`/`standard`/`mixtral`/`token_routed` were removed —
constructing a config with any of them raises a clear error pointing at the
replacement (see [`token-routed.md`](token-routed.md) for `token_routed`
checkpoints specifically).

Several aliases exist for checkpoint compatibility. New documentation should
use the principal key.

## Direct TR-MoE module

```python
from complexity.core.mlp import MLPConfig, TRHashEngineMLP

layer = TRHashEngineMLP(
    MLPConfig(
        hidden_size=384,
        intermediate_size=128,
        vocab_size=32_000,
        num_experts=4,
        shared_expert=True,
        shared_intermediate_size=1536,
        top_k=2,
    )
)
output = layer(hidden_states, token_ids=input_ids)
```

Useful diagnostics:

```python
layer.engine.last_backend
layer.capability_summary()
layer.training_telemetry()
```

## Official MCP client

Install:

```bash
pip install -e ".[tools]"
```

Imports:

```python
from complexity.mcp import (
    MCPTool,
    MCPToolResult,
    OfficialMCPStdioClient,
    OfficialMCPStdioConfig,
)
```

The wrapper launches and calls an MCP server through the official Python SDK
stdio transport. It does not reimplement tools.

## CLI entry points

```text
complexity
cf-plan-run
cf-plan-cluster
cf-image-train
cf-image-edit-train
cf-detector-train
cf-vision-pretrain
cf-detector-serve
cf-sensor-fusion-train
cf-sensor-fusion-submit
```

`cf-o200k-pretrain` and `cf-check-pipeline` were removed along with the
`o200k` training pipeline (see [`training.md`](training.md)). `cf-plan-run`
and `cf-plan-cluster` remain for token-budget and cluster-sizing arithmetic.
Some older `complexity` subcommands remain experimental.
