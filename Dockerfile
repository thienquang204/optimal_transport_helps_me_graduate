# syntax=docker/dockerfile:1
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Matryoshka-MMPOT experiment"
LABEL org.opencontainers.image.description="MRL versus pairwise Matryoshka-MMPOT training and benchmarking"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    HF_DATASETS_CACHE=/data/huggingface/datasets \
    TORCH_HOME=/data/torch

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

COPY matryoshka_real_mmpot_experiment.py matryoshka_mmpot_experiment.py \
     csr_vs_mmpot_imagenet.py download_imagenet.py summarize_results.py \
     container_pipeline.sh run_csr_vs_mmpot_imagenet.sh ./

RUN chmod +x /app/container_pipeline.sh /app/run_csr_vs_mmpot_imagenet.sh

RUN mkdir -p /data/huggingface /data/torch /output

VOLUME ["/data", "/output"]

ENTRYPOINT ["python", "/app/matryoshka_real_mmpot_experiment.py"]
CMD ["--help"]
