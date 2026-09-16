#!/bin/bash

# RunPod training startup script
# This script sets up the environment and starts training

set -euo pipefail

echo "Starting training on RunPod..."

# Set environment variables
export PYTHONPATH=/workspace
export HF_HOME=/workspace/.cache/huggingface

# Create cache directory
mkdir -p /workspace/.cache/huggingface

# Activate virtual environment if using uv
if command -v uv &> /dev/null; then
    echo "Using uv for package management"
    uv sync --frozen --extra gpu --no-dev
else
    echo "Install uv before using this launcher" >&2
    exit 1
fi

# Start training
echo "Starting training with config: ${CONFIG_PATH:-configs.yaml}"
uv run --frozen --no-sync python run_amazon2014.py --config-path "${CONFIG_PATH:-configs.yaml}" \
    --num-gpus "${NUM_GPUS:-1}" "$@"
