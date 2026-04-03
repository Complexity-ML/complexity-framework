FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    safetensors \
    transformers \
    huggingface-hub \
    aiohttp \
    numpy

# v2: RoPE fix + KV cache fix + token_quality_vector guard
RUN pip install --no-cache-dir git+https://github.com/Complexity-ML/vllm-i64.git@main

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Pacific-i64/TR-MoE-400M', local_dir='/app/model')"

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
