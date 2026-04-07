---
title: Dense-400M
emoji: "\u2B50"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
models:
  - Pacific-i64/Dense-400M
---

# Dense SwiGLU Baseline (384.5M)

Dense transformer baseline for iso-parameter comparison with TR-MoE-400M.

## API

```bash
curl -X POST https://Pacific-i64-Dense-400M.hf.space/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 100, "temperature": 0.7}'
```
