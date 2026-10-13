#!/bin/bash

# RunPod entrypoint — Semantic-ID quantization training (train_quant.py).
#
# Env setup -> optional dependency install -> ensure the Amazon 2014 data exists ->
# launch train_quant.py (single- or multi-GPU). Designed to run on a RunPod pod whose
# image is the torch/CUDA template (e.g. runpod/pytorch:...-torch280-...); the template
# supplies torch/CUDA/Python, this script adds the app libraries and runs training.
#
# Env overrides:
#   CONFIG_PATH      yaml config (default: configs/configs_smol_quant.yaml)
#                    set configs/configs_qwen8b_quant.yaml for the 8B multi-GPU run
#   NUM_GPUS         GPUs to use (default: all visible; 1 -> plain python, >1 -> torchrun)
#   SKIP_INSTALL     1 to skip `uv sync` (deps already present in the image/venv)
#   SKIP_PREPROCESS  1 to skip the download+preprocess step (data already present)
#   WANDB_MODE       e.g. offline to disable W&B uploads
#
# Usage:
#   bash train_quant.sh
#   CONFIG_PATH=configs/configs_qwen8b_quant.yaml NUM_GPUS=2 bash train_quant.sh

set -e

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Reduces fragmentation OOMs from the multi-forward compute_loss.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}"

CONFIG_PATH="${CONFIG_PATH:-configs/configs_smol_quant.yaml}"

# 1. Dependencies. The torch/CUDA template already ships torch; `uv sync` adds the
#    training libraries (transformers, trl, peft, datasets, ... + deepspeed via --extra
#    gpu). Set SKIP_INSTALL=1 to reuse an environment that already has them.
if command -v uv &> /dev/null; then
    RUN=(uv run); [ "${SKIP_INSTALL:-0}" = "1" ] || uv sync --frozen --extra gpu
else
    RUN=()
fi

# 2. Data. preprocess.sh is idempotent (skips existing downloads/artifacts).
if [ "${SKIP_PREPROCESS:-0}" != "1" ]; then
    bash preprocess.sh
fi

# 3. Launch. Default to every visible GPU; ZeRO (trainer.deepspeed in the yaml) is
#    driven by the HF Trainer under torchrun for the multi-GPU 8B configs.
if [ -z "${NUM_GPUS}" ]; then
    if command -v nvidia-smi &> /dev/null; then NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"; else NUM_GPUS=1; fi
fi

echo "train_quant: config=${CONFIG_PATH} gpus=${NUM_GPUS}"
# Always launch via torchrun (even nproc_per_node=1) so the torch-distributed env
# vars (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR) are set. DeepSpeed then uses the
# env:// backend instead of MPI auto-discovery, which would require mpi4py.
"${RUN[@]}" torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    train_quant.py --config-path "${CONFIG_PATH}"
