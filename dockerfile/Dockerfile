# Ev-MEFormer runtime image.
#
# Baseline verified on the lab machine:
#   Python 3.8 / PyTorch 1.13.1+cu117 / MMCV-Full 1.6.0 / MMDet 2.24.0 /
#   MMDet3D 1.0.0rc5 / MMSeg 0.29.1 / spconv-cu117 2.3.6 / flash-attn 1.0.5
#
# Do not mix cu111 / cu117 / cu118 wheels of torch, spconv and cumm: the CUDA
# extension ABIs disagree and the failure usually only shows up once training
# has already started.

# Step 1: CUDA 11.8 development base image (nvcc is required to build flash-attn).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Step 2: Environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"
# Architectures flash-attn is compiled for. flash-attn 1.0.5 requires SM 8.0+.
# 8.0 = A100, 8.6 = RTX 30xx / A6000, 8.9 = RTX 40xx / 6000 Ada.
# Trim this list to your own GPUs to shorten the build considerably.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV MAX_JOBS=4

# Step 3: Base system dependencies.
RUN apt-get update && apt-get install -y \
    wget \
    git \
    gedit \
    vim \
    build-essential \
    ninja-build \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Step 4: PyTorch first, so the packages below compile against a fixed ABI.
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Step 5: Remaining Python packages.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Step 6: flash-attn is a hard requirement — projects/mmdet3d_plugin/models/utils/
# attention.py imports it at module import time, so the plugin cannot even be
# loaded without it. Version 1.0.5 is the last release compatible with torch 1.13.
# It is built from source here; this needs nvcc but no GPU, and takes a while.
RUN pip install packaging ninja && \
    pip install flash-attn==1.0.5 --no-build-isolation

# Step 7: The repository is bind-mounted at /Ev-MEFormer by scripts/start.sh.
# Mount the repository ROOT (not projects/): configs declare
# `plugin_dir = 'projects/mmdet3d_plugin/'` relative to the working directory.
WORKDIR /Ev-MEFormer

# Step 8: Default command.
CMD ["/bin/bash"]
