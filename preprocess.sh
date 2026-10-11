#!/bin/bash

# RunPod data-pipeline startup script.
#
# Downloads the Amazon 2014 per-category files (meta + 5-core reviews) and runs the
# P5-style preprocessing to build the sequential-recommendation artifacts
# (sequential_data.txt, datamaps.json, item2attributes.json, negative_samples.txt)
# that both data_module pipelines consume. Run this once before training.
#
# Env overrides:
#   CATEGORIES     space-separated category names (default: the 3 configured ones)
#   DATA_DIR       where raw meta/reviews gz files live (default: ./amazon2014_meta)
#   OUT_DIR        where per-category artifacts are written (default: ./amazon2014_seq)
#   SEED           preprocessing seed (default: 2022, as in P5)
#   SAMPLE_TYPE    negative sampling: random | pop (default: random)
#   SKIP_DOWNLOAD  set to 1 to reuse already-downloaded gz files
#
# Usage:
#   bash preprocess.sh
#   CATEGORIES="Beauty" bash preprocess.sh
#   SKIP_DOWNLOAD=1 OUT_DIR=/workspace/amazon2014_seq bash preprocess.sh

set -e

echo "Starting Amazon 2014 data pipeline on RunPod..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH:-/workspace}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# Create cache directory
mkdir -p "${HF_HOME}"

CATEGORIES="${CATEGORIES:-Beauty Sports_and_Outdoors Toys_and_Games}"
DATA_DIR="${DATA_DIR:-./amazon2014_meta}"
OUT_DIR="${OUT_DIR:-./amazon2014_seq}"
SEED="${SEED:-2022}"
SAMPLE_TYPE="${SAMPLE_TYPE:-random}"

# Prefer uv (matches the rest of the repo); fall back to plain python.
if command -v uv &> /dev/null; then
    echo "Using uv for package management"
    uv sync --frozen
    PY=(uv run python)
else
    echo "uv not found, using system python"
    PY=(python)
fi

echo "Categories : ${CATEGORIES}"
echo "Data dir   : ${DATA_DIR}"
echo "Output dir : ${OUT_DIR}"
echo "Seed       : ${SEED}  |  negatives: ${SAMPLE_TYPE}"

# 1. Download meta + 5-core reviews for each category (idempotent: skips existing files).
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "SKIP_DOWNLOAD=1 set; reusing existing files in ${DATA_DIR}"
else
    echo "Downloading meta + reviews into ${DATA_DIR}..."
    "${PY[@]}" download_amazon_2014.py --out "${DATA_DIR}" ${CATEGORIES}
fi

# 2. Build the P5 sequential artifacts per category into OUT_DIR/<Category>/.
echo "Running P5 preprocessing..."
"${PY[@]}" preprocess_amazon_2014.py ${CATEGORIES} \
    --data-dir "${DATA_DIR}" \
    --out "${OUT_DIR}" \
    --seed "${SEED}" \
    --sample-type "${SAMPLE_TYPE}"

echo "Done. Artifacts written under ${OUT_DIR}/<Category>/"
echo "Point configs.yaml 'amazon_2014_sequential_dir: ${OUT_DIR}' to use them."
