# Start from the NVIDIA official image (ubuntu-22.04 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""

# Define installation arguments
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Set apt source
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
    echo "deb ${APT_SOURCE} jammy main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-updates main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-backports main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

# Install systemctl
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

# Install tini
RUN apt-get update && \
    apt-get install -y tini && \
    apt-get clean

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip

# Install torch-2.5.1 + vllm-0.7.3
RUN pip install --no-cache-dir vllm==0.7.3 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 tensordict \
    transformers>=4.49.0 accelerate datasets peft \
    ray codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc qwen-vl-utils

# Install flash_attn-2.7.4.post1
RUN pip uninstall -y transformer-engine flash-attn && \
    wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
