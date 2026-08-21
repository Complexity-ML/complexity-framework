# TR-HASH Hugging Face Transformers adapter

These two Python modules form the autonomous `trust_remote_code` adapter for
the public TR-HASH MoE checkpoints:

- `configuration_tr_hash_moe.py` registers `model_type = "tr_hash_moe"`;
- `modeling_tr_hash_moe.py` implements `TRHashForCausalLM`, legacy and modern
  KV caches, causal generation, the persisted multi-hash route tables, and the
  shared-plus-routed SwiGLU block.

The implementation intentionally keeps the native safetensors names. Exporting
a checkpoint does not transpose, rename, merge, or otherwise rewrite weights.

## Build a Hub bundle

```bash
python scripts/export_tr_hash_transformers.py \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokenizer-dir /path/to/tokenizer \
  --output /path/to/hub-bundle
```

The resulting directory can be loaded directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)
output = model.generate(
    **tokenizer("Hello", return_tensors="pt"),
    max_new_tokens=32,
)
```

`num_experts_per_tok` is used for architectural top-k routing. It is kept
separate from Transformers' `top_k` sampling parameter so loading the model
does not silently change generation defaults.
