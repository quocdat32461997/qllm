FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Install dependencies using uv
RUN uv python install 3.12 && uv sync --frozen --extra gpu --no-dev --python 3.12
COPY *.py *.yaml *.sh ./
COPY ds_configs ./ds_configs

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/.cache/huggingface

# Create cache directory
RUN mkdir -p /workspace/.cache/huggingface

# Expose port for MLflow if needed
EXPOSE 5000

# Default command
CMD ["bash", "runpod_train.sh", "--download"]
