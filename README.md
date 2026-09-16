# qllm: semantic IDs and generative recommendation

Train a shared causal language model to encode product metadata as discrete semantic IDs and reconstruct the product, then fine-tune that checkpoint for next-item recommendation.

The Amazon 2014 pipeline supports **Beauty**, **Sports and Outdoors**, and **Toys and Games**, following the category selection in TIGER. It includes chronological interaction splits, stable ASIN lookup with collision handling, and catalog-constrained Recall/NDCG evaluation.

```bash
uv sync --frozen                       # local CPU validation
uv run --frozen python -m pytest -q
uv run --frozen python run_amazon2014.py --dry-run

# On a GPU host:
uv sync --frozen --extra gpu
uv run --frozen python run_amazon2014.py --download --categories Beauty
```

See [the Amazon 2014 guide](docs/AMAZON2014.md) for dataset research, protocol decisions, SmolLM/Qwen runs, LoRA, resume commands, evaluation, and limits. [Validation record](docs/VALIDATION.md).

Primary entry points: `amazon2014.py` (prepare), `train.py` (semantic training), `export_semantic_ids.py` (freeze the item index), `train_recommendation.py` (next-item fine-tuning), and `evaluate_recommendation.py` (ranking). `run_amazon2014.py` orchestrates separate category experiments. The older Amazon 2023 metadata loader remains available with `dataset_version: '2023'`.

For RunPod, build the supplied Dockerfile and mount persistent storage for `/workspace/data`, `/workspace/outputs`, and `/workspace/.cache`. `CONFIG_PATH` chooses the model config, `NUM_GPUS` controls distributed training, and the container's startup script runs the complete pipeline. Tracking is optional and off by default.
