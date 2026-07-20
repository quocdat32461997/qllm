FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY train.py ./
COPY trainers.py ./
COPY data_module.py ./
COPY constants.py ./
COPY configs.yaml ./

# Install dependencies using uv
RUN uv sync --frozen

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
CMD ["uv", "run", "python", "train.py", "--config-path", "configs.yaml"]
