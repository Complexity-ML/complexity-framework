"""
TR-MoE-400M — HuggingFace Spaces via vllm-i64 server.
"""

import os
import torch
from pathlib import Path

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
PORT = int(os.environ.get("PORT", 7860))

from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
from vllm_i64.models.complexity_deep.model import ComplexityDeepModel
from vllm_i64.api.server import I64Server
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

print(f"Loading model from {MODEL_DIR}...")
config = ComplexityDeepConfig.from_json(str(Path(MODEL_DIR) / "config.json"))
model = ComplexityDeepModel(config)
state = load_file(str(Path(MODEL_DIR) / "model.safetensors"), device="cpu")
model.load_framework_checkpoint(state)
model.eval()

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    from vllm_i64.engine.i64_engine import I64Engine
    model = model.cuda()
    engine = I64Engine(model=model, num_experts=config.num_experts, vocab_size=config.vocab_size)
else:
    from vllm_i64.cpu.engine import CPUEngine
    engine = CPUEngine(model=model, num_experts=config.num_experts, vocab_size=config.vocab_size)

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)

server = I64Server(
    engine=engine,
    tokenizer=tokenizer,
    model_name="Pacific-i64/TR-MoE-400M",
    host="0.0.0.0",
    port=PORT,
)
server.run()
