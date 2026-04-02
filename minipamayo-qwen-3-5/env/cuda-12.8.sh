#!/usr/bin/env sh

# Repo-local CUDA toolkit selection for canonical flash-attn runs.
# Usage:
#   cd /home/masa/minipamayo/minipamayo-qwen-3-5
#   . ./env/cuda-12.8.sh

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
