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
ARG VLLM_COMMIT=227578480d71fc94ef46ca77fb69496412158d68

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

# Install vllm-0.7.4-nightly
RUN pip install --no-cache-dir vllm --pre --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}" && \
    git clone -b verl_v1 https://github.com/hiyouga/vllm.git && \
    cp -r vllm/vllm/ /usr/local/lib/python3.10/dist-packages/

# Install torch-2.5.1
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 tensordict torchdata \
    transformers>=4.49.0 accelerate datasets peft hf_transfer \
    ray codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc qwen-vl-utils wandb liger-kernel \
    pytest yapf py-spy pyext

# Install flash_attn-2.7.4.post1
RUN pip uninstall -y transformer-engine flash-attn && \
    wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Fix cv2
RUN pip uninstall -y pynvml nvidia-ml-py && \
    pip install --no-cache-dir nvidia-ml-py>=12.560.30 opencv-python-headless==4.8.0.74 fastapi==0.115.6
