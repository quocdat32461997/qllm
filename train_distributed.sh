#!/bin/bash

# Distributed training launcher (DeepSpeed ZeRO via HF Trainer).
#
# Memory sharding — NOT plain DDP — is what lets Qwen3-8B fit: ZeRO splits the
# optimizer states (stage 2) and optionally the parameters (stage 3) across the
# GPUs / offloads them to CPU. Set the DeepSpeed stage in the yaml config via
#   trainer.deepspeed: ds_configs/zero2.json   (or zero3.json)
#
# Usage:
#   NUM_GPUS=4 CONFIG_PATH=configs_qwen8b.yaml bash train_distributed.sh

set -e

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
# Reduces fragmentation OOMs from the multi-forward compute_loss.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Avoid tokenizers fork warnings under multi-worker dataloaders.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG_PATH="${CONFIG_PATH:-configs_qwen8b.yaml}"

# Default to every visible GPU on the node.
if [ -z "${NUM_GPUS}" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS="$(nvidia-smi -L | wc -l)"
    else
        NUM_GPUS=1
    fi
fi

echo "Launching distributed training: ${NUM_GPUS} GPU(s), config=${CONFIG_PATH}"

# torchrun works because the HF Trainer initializes DeepSpeed from the json
# referenced by trainer.deepspeed. (`deepspeed --num_gpus=${NUM_GPUS} train.py`
# is an equivalent alternative launcher.)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    train.py --config-path "${CONFIG_PATH}"
