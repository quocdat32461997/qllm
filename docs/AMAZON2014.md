# Amazon 2014 retraining and recommendation

## Dataset choice and comparison plan

Use **Beauty**, **Sports and Outdoors**, and **Toys and Games**, trained and reported separately. Sports and Outdoors is a single category, not two datasets. Beauty is the first debugging experiment; run all three before drawing cross-domain conclusions. This recommendation follows shared benchmarks in the papers below; it is not a bibliometric claim that these are the most frequent categories in all recommender research.

| Primary source | Relevant benchmark evidence | Use in this project |
| --- | --- | --- |
| [TIGER, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf), §4.1 and Appendix C | Amazon 1996–July 2014; Beauty, Sports and Outdoors, Toys and Games | Main semantic-ID recommendation comparison |
| [GRAM, ACL 2025](https://aclanthology.org/2025.acl-long.1596.pdf), §4 and Appendix G | Same three 2014 categories, plus Yelp; 5-core; last/penultimate holdouts; Recall/NDCG at 5 and 10 | Later generative baseline using the same category set |
| [LETTER, original May 2024 version](https://arxiv.org/html/2405.07314v1), §4.1.1 | Beauty, Instruments, Yelp; Amazon citation points to a different release | Useful token-diversity/collaborative-signal comparison, but its Beauty numbers are not automatically comparable |

TIGER Appendix C reference counts (not locally measured results):

| Category | Users | Items | Mean sequence length |
| --- | ---: | ---: | ---: |
| Beauty | 22,363 | 12,101 | 8.87 |
| Sports and Outdoors | 35,598 | 18,357 | 8.32 |
| Toys and Games | 19,412 | 11,924 | 8.63 |

Suggested comparison axes: semantic code utilization and collisions; content reconstruction; next-item Recall@5/10 and NDCG@5/10; training/inference cost; then separate cold-start experiments. For model baselines, prioritize TIGER, SASRec, and GRAM, with a random-ID ablation to test whether learned codes help. Their training implementations are not included here. Record parameter count and training budget: SmolLM-135M and Qwen3-8B differ substantially from TIGER's roughly 13M-parameter recommender. Use multiple seeds for final comparisons. Do not mix published metrics from different dataset releases or preprocessing/evaluation protocols.

The research notes are background, not additional implementation instructions. Their RL/GRPO, multimodal, search, and cold-start proposals are future experiments. This implementation adds supervised next-item recommendation after semantic-ID retraining. It does not claim an exact TIGER model reproduction or a verified hierarchical meaning for the shared code vocabulary.

## Data protocol

`amazon2014.py` downloads the official [McAuley Amazon 2014 5-core reviews and category metadata](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html). It accepts gzip or plain JSON lines and safely parses the Python-literal metadata format with `ast.literal_eval`.

- Preserve all source review events and ratings; review text is never a training feature. Iteratively enforce five interactions per user and item. The official 5-core inputs should already meet this condition.
- Sort each user's events by timestamp, breaking ties by source row order. Last event is test; penultimate event is validation; every earlier nonempty prefix is a training example. Keep the latest 20 history items. Validation history contains training events; test history also includes the validation event.
- Use only title, description, brand, and categories as product text. Related-item graphs and sales-rank fields are excluded. ASIN is retained for joins, not embedded into product descriptions. Missing titles are reported; entirely missing item metadata fails preparation.
- Keep the full filtered interaction catalog for the metadata stage and candidate index. This is a **transductive catalog** experiment: held-out interaction targets' metadata is available, but their held-out interactions never become recommendation training labels. The stage-1 random metadata holdout measures reconstruction, not cold start.
- Preserve repeated user-item events and report their count. No per-user deduplication, positive-rating threshold, global time split, or unseen-item filtering is silently applied.
- Save raw and processed SHA-256 hashes, counts, policy choices, and timestamp-tie policy in `manifest.json`. Training checks prepared artifact hashes. Compare observed counts with the table before reporting benchmark metrics.

Each category directory has `products.jsonl`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`. Synthetic fixtures in the tests are not Amazon benchmark results.

## GPU run

Run commands from the repository root. The default config is SmolLM-135M, Beauty for standalone commands, with all three categories selected by the pipeline launcher. The Qwen config uses Qwen3-8B and ZeRO-2. Both are fresh retraining from their pretrained base, replacing the prior Amazon 2023 run for these experiments. Existing checkpoints are not deleted.

```bash
uv sync --frozen --extra gpu

# Prepare all three categories without loading a model.
uv run --frozen python amazon2014.py --download

# Inspect every command; this performs no training or downloads.
uv run --frozen python run_amazon2014.py --dry-run

# Entire SmolLM pipeline, one category first (uses prepared data).
uv run --frozen python run_amazon2014.py --categories Beauty

# Independent runs on all three categories.
uv run --frozen python run_amazon2014.py

# Qwen3-8B: training stages use torchrun; export/evaluation use one GPU.
uv run --frozen python run_amazon2014.py \
  --config-path configs_qwen8b.yaml --num-gpus 4
```

Add `--download` to the pipeline launcher to download missing raw files automatically. The raw files are several hundred MB compressed; full model downloads require additional space. Export/evaluation load the whole model on a single device, so that GPU must fit it. Distributed CUDA/DeepSpeed execution requires validation on the actual GPU host; local CPU tests do not establish its memory requirements.

The launcher writes an experiment YAML in `outputs/amazon2014/<model>/<category>/`, then invokes the existing semantic trainer, catalog export, recommendation trainer, and test evaluator. It refuses to overwrite completed final checkpoints and refuses a different config at an existing experiment path. Use a new `--output-root` for a new seed or altered settings. `--stages` can restart at a completed boundary:

```bash
uv run --frozen python run_amazon2014.py --categories Beauty \
  --stages export recommendation evaluate
```

For an interrupted individual training stage, use the standalone CLI with `--resume-from-checkpoint PATH` to restore optimizer/scheduler state. For recommendation, still pass the original stage-1 checkpoint and frozen index used by the run:

```bash
uv run --frozen python train.py --config-path outputs/amazon2014/smollm135m/Beauty/experiment.yaml \
  --resume-from-checkpoint outputs/amazon2014/smollm135m/Beauty/semantic/checkpoint-1000

uv run --frozen python train_recommendation.py \
  --config-path outputs/amazon2014/smollm135m/Beauty/experiment.yaml \
  --checkpoint outputs/amazon2014/smollm135m/Beauty/semantic/final \
  --index outputs/amazon2014/smollm135m/Beauty/semantic_index.json \
  --resume-from-checkpoint outputs/amazon2014/smollm135m/Beauty/recommendation/checkpoint-1000
```

Use `NUM_GPUS=4 CONFIG_PATH=... TRAIN_SCRIPT=train_recommendation.py bash train_distributed.sh --checkpoint ... --index ...` to resume with the distributed launcher. ZeRO-2 is the supported starting configuration for the custom multiple-forward semantic loss; ZeRO-3 is not validated.

`trainer.report_to: []` disables external experiment tracking. Set it to `[wandb]` to enable it; project comes from `experiment_name` and optional account from `wandb_entity`. Set `use_lora: true` for stage 1 and `recommendation.use_lora: true` for stage 2 when desired. The embeddings and LM head must remain trainable (`modules_to_save`). When continuing a stage-1 adapter into a new stage-2 adapter, the trainer saves a merged `recommendation/stage1_base/` snapshot so reloading stage 2 retains the learned stage-1 weights. Keep that directory with the adapter and update the adapter base path if moving it between hosts.

## Checkpoints and stable item lookup

Stage 1 uses the existing shared causal LM, autoregressive Gumbel straight-through selection, format loss, and product reconstruction. Fixes include per-example assistant masking, independent initialization of added code tokens, float32 Gumbel probabilities, non-mutating embedding updates, valid gradient-accumulation scaling, and deterministic evaluation. The configured temperature ends at 0.1 to avoid essentially zero gradients. Temperature affects gradient softness; it does not itself remove Gumbel sampling noise during training.

`export_semantic_ids.py` encodes **all catalog items** using the first prompt template, all available features within the character budget, greedy selection, and no dropout/Gumbel noise. It stores integer code values independently of tokenizer token IDs. Same-catalog collisions receive a stable suffix assigned by lexicographically sorted ASIN within each bucket; every item receives one suffix token, including singleton buckets. The six learned code tokens remain unchanged. Complete identifiers have six code tokens plus one disambiguation token between the semantic boundaries.

Export reports raw semantic-code collision rate and largest collision bucket. The suffix guarantees lookup uniqueness, not semantic quality: collapsed codes remain a failed representation even if the recommender learns suffixes. Inspect these statistics before committing to a long stage-2 run. Do not regenerate the index after recommendation training. The final recommendation checkpoint includes a copy and checksum of its frozen index.

Stage 2 predicts the next item's complete identifier from the prior identifiers. Only the assistant target contributes to the supervised loss; user/system text and padding are masked. Token limits fail explicitly instead of silently cutting off targets. Recommended defaults are initial experiment settings (one semantic epoch and three recommendation epochs), not tuned benchmark hyperparameters. Stage 2 selects its checkpoint by validation loss; tune or select by validation ranking metrics for a publication-quality comparison, without consulting test metrics.

## Evaluation and validation

The recommender evaluator uses a catalog prefix trie and beam search over the full candidate catalog, reports Recall/NDCG at 5 and 10, and checks index integrity. This is approximate top-k generation, not exhaustive scoring of every item. It uses no sampled negatives and no seen-item exclusion. If a comparison baseline excludes seen items, align that policy explicitly. All items have equal identifier length, and the collision suffix is included in model scoring.

```bash
uv run --frozen python evaluate_recommendation.py \
  --checkpoint outputs/amazon2014/smollm135m/Beauty/recommendation/final --split validation

uv run --frozen python evaluation.py \
  --config-path outputs/amazon2014/smollm135m/Beauty/experiment.yaml \
  --checkpoint outputs/amazon2014/smollm135m/Beauty/semantic/final --max-samples 500

# Offline CPU tests: raw-data fixtures, both training stages, full/LoRA checkpoint
# reload, deterministic ID export, constrained decoding, and hand-computed metrics.
uv run --frozen python -m pytest -q
```

The tests initialize a tiny random Llama locally and do real forward/backward optimization and checkpoint reloads. They require no model download or GPU. Passing them confirms mechanics, not recommendation quality or successful full retraining. See `VALIDATION.md` for the execution record from this implementation session.
