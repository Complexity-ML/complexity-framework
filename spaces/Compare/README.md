---
title: Compare TR-MoE vs Dense
emoji: "\u2696"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
---

# TR-MoE vs Dense — Side-by-Side Comparison

Same prompt, two architectures (384M iso-params), real-time comparison.

Calls [TR-MoE-400M](https://huggingface.co/spaces/Pacific-i64/TR-MOE-400M) and [Dense-400M](https://huggingface.co/spaces/Pacific-i64/Dense-400M) in parallel.

## API

```bash
# Compare both
curl -X POST https://Pacific-i64-Compare.hf.space/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 100}'

# TR-MoE only
curl -X POST https://Pacific-i64-Compare.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}'

# Dense only
curl -X POST https://Pacific-i64-Compare.hf.space/dense \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}'
```
