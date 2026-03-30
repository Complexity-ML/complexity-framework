# CUDA / GPU Optimizations

## Attention

| Type | Complexity | Memory | Usage |
|------|-----------|--------|-------|
| Flash Attention (SDPA) | O(N^2) | O(N) | Default, up to 8k tokens |
| Sliding Window | O(NxW) | O(W) | Long sequences |

### GQA with Flash Attention

```python
# 12 query heads, 4 KV heads (Complexity-Deep default)
# Uses PyTorch SDPA with Flash backend
attn = GroupedQueryAttention(config)
```

### KV Cache Fix

When using SDPA with KV cache (autoregressive generation), `is_causal=True` must be disabled when `q_len != kv_len`:

```python
# is_causal=True only valid when q_len == kv_len (no KV cache)
use_causal = (attn_mask is None) and (q.shape[2] == k.shape[2])
attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)
```

## Token-Routed MLP Dispatch

### Sort-and-Split (current)

```python
sort_idx = expert_ids.argsort(stable=True)
sorted_x = flat[sort_idx]
# BMM: [E, N/E, hidden] @ [E, hidden, inter*2]
gu = torch.bmm(sorted_x.view(E, chunk, hidden), gate_up_proj)
```

- Fullgraph compatible with `torch.compile`
- Each expert processes exactly N/E tokens
- No dynamic shapes

### Fused Cross-Entropy

```python
from complexity_cuda.fused_cross_entropy import fused_cross_entropy
# Never materializes full logits tensor
loss = fused_cross_entropy(hidden_states, embed_weight, labels)
```

## vLLM Integration

The model runs on vLLM with:
- **PagedAttention**: KV cache management
- **CUDA Graphs**: captured for decode steps
- **Custom splitting_op**: deterministic routing in eager mode during graph replay
- **204 tok/s** on RTX 5060 Ti (16GB)

## torch.compile Notes

- Expert weights as `nn.Parameter` (not 3D indexed tensors) for XBLOCK compatibility
- `token_to_expert` buffer stored outside compiled module to avoid pickle issues
- `mu_init` in `ComplexityForCausalLM` (not compiled) to avoid serialization errors

## Memory Tips

| Model | GPU Memory | Config |
|-------|-----------|--------|
| 187M (training) | ~31 GB | bf16, batch 128, seq 2048 |
| 187M (inference, vLLM) | ~0.4 GB | bf16, PagedAttention |

## See Also

- [Token-Routed MLP](token-routed.md)
- [Training](training.md)
