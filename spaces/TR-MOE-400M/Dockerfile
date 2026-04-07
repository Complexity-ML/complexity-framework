FROM python:3.12-slim

WORKDIR /app

ENV OMP_NUM_THREADS=4
ENV VLLM_TARGET_DEVICE=cuda
ENV CUDA_VISIBLE_DEVICES=0
ENV VLLM_LOGGING_LEVEL=WARNING

RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    https://github.com/Complexity-ML/vllm-cuda_graph/releases/download/v0.2.0/vllm-0.1.0-cp312-cp312-linux_x86_64.whl

RUN pip install --no-cache-dir huggingface-hub

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Pacific-i64/TR-MoE-400M', local_dir='/app/model')"

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
