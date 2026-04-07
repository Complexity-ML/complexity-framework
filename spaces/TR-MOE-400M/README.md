---
title: TR-MoE-400M
emoji: "⚡"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
models:
  - Pacific-i64/TR-MoE-400M
---

# TR-MoE-400M Inference

Token-Routed MoE (383.5M params, ~105M active per token) served via vllm-cuda-graph-i64.

## API

```bash
curl -X POST https://Pacific-i64-TR-MOE-400M.hf.space/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 100, "temperature": 0.7}'
```

## Model

- **Architecture**: Token-Routed MLP + Mu-Guidance + Shared Lexical Expert
- **Parameters**: 383.5M total, ~105M active per token
- **Experts**: 4 (deterministic Zipf-balanced routing)
- **Training**: 8B tokens on FineWeb-Edu
- **Inference**: vllm-cuda-graph-i64 with FlashAttention + CUDA graphs

## Links

- [Model](https://huggingface.co/Pacific-i64/TR-MoE-400M)
- [Paper (TMLR)](https://openreview.net/forum?id=jZq6EVboC6)
- [Framework](https://github.com/Complexity-ML/complexity-framework)
- [vllm-cuda-graph-i64](https://github.com/Complexity-ML/vllm-cuda_graph)
