# PROTEUS miner image (bittensor neuron). CPU-only ON PURPOSE: the miner does no
# GPU compute itself, it serves requests by calling a vLLM server over HTTP
# (expert.py -> VLLM_HOST). The GPU work lives in a separate vllm/vllm-openai
# container. Keeping them apart avoids torch/vLLM/bittensor dependency conflicts.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU torch (bittensor needs torch, but the miner never touches the GPU).
RUN pip install --no-cache-dir "setuptools<81" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir \
        bittensor==8.5.1 \
        bittensor-cli==8.4.4 \
        "uvicorn<0.34" \
        "pydantic==2.13.4" \
        numpy

WORKDIR /app
COPY . .
ENV PYTHONPATH=/app
# expert.py reads these; overridable at runtime.
ENV VLLM_HOST=http://proteus-vllm:8000
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# args (netuid, endpoint, wallet, hotkey, axon port) passed by the launcher.
ENTRYPOINT ["python", "neurons/miner.py"]
