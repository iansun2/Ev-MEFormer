#!/usr/bin/env bash
# Build the Ev-MEFormer image.
#
# flash-attn 1.0.5 is compiled from source inside the image, which needs nvcc
# (present in the CUDA devel base image) but no GPU at build time. Expect a long
# build; TORCH_CUDA_ARCH_LIST in the Dockerfile controls which SM targets are built.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build "${REPO_ROOT}/dockerfile" -t ev-meformer
