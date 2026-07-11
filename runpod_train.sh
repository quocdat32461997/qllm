#!/bin/bash

# RunPod training startup script
# This script sets up the environment and starts training

set -e

echo "Starting training on RunPod..."

# Set environment variables
export PYTHONPATH=/workspace
export HF_HOME=/workspace/.cache/huggingface
export WANDB_API_KEY=${WANDB_API_KEY:-""}

# Create cache directory
mkdir -p /workspace/.cache/huggingface

# Activate virtual environment if using uv
if command -v uv &> /dev/null; then
    echo "Using uv for package management"
    uv sync --frozen
else
    echo "uv not found, installing dependencies manually"
    pip install -r requirements.txt || echo "No requirements.txt found"
fi

# Start training
echo "Starting training with config: ${CONFIG_PATH:-configs.yaml}"
uv run python train.py --config-path "${CONFIG_PATH:-configs.yaml}"
