# Local validation — 16 September 2026

Full SmolLM/Qwen retraining and recommendation fine-tuning were deliberately **not run** on this Mac, following the user's instruction to prepare locally and leave full training for a GPU. No trained benchmark result is claimed.

## Actual Amazon 2014 preparation

Downloaded and gzip-validated the six official review/metadata files (582,494,887 bytes compressed in total). Prepared all three categories locally. Raw files and generated datasets are under ignored `data/amazon2014/`; their sizes make them unsuitable for Git. The portable record of source and processed file hashes is [amazon2014_data_manifest.json](amazon2014_data_manifest.json).

| Category | Users | Items | Review events | Training prefixes | Validation / test examples each | Missing titles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beauty | 22,363 | 12,101 | 198,502 | 131,413 | 22,363 | 7 |
| Sports and Outdoors | 35,598 | 18,357 | 296,337 | 189,543 | 35,598 | 90 |
| Toys and Games | 19,412 | 11,924 | 167,597 | 109,361 | 19,412 | 59 |

Users/items match TIGER Appendix C; review counts match the official 5-core files and GRAM Table 10. No user-item duplicates or missing catalog records were found. Products with missing titles remain in the benchmark and use `Unknown product` with their available features according to the configured feature probability. Source metadata, including missing titles, is retained rather than dropping interactions.

Checked SHA-256 hashes of every prepared artifact, nondecreasing timestamps in every history, history lengths at most 20, and one validation/test example per user. Fixture tests additionally check that held-out future events never enter that user's training prefixes. These per-user splits are not a global temporal cutoff.

Python's system certificate store failed the HTTPS download on this host. The system `curl` client successfully verified HTTPS certificates, downloaded the files, and each gzip stream was validated in full before preparation. No certificate-verification bypass was used. The portable downloader in `amazon2014.py` uses Python HTTPS; a host with a custom corporate CA must configure its trusted certificate store.

## Executed model tests

The offline pytest suite performs real CPU optimization with a tiny randomly initialized Llama and local tokenizer. It verifies:

- Chronological splits, history caps, iterative k-core pruning, safe literal parsing, sample limits, and rejection of changed data hashes.
- A nonzero reconstruction gradient reaching the first code-selector logits through the straight-through bottleneck.
- Unmasked code IDs, assistant/padding loss masking, explicit length errors, deterministic/batch-invariant ID export, gradient accumulation, and gradient checkpointing.
- Two optimization steps in stage 1, complete catalog export, two optimization steps in stage 2, checkpoint save/reload, and constrained generation.
- All four full/LoRA stage transitions; a merged stage-1 adapter is retained in the stage-2 adapter's base snapshot, and full stage-2 training updates backbone parameters.
- Unique item lookup despite semantic-code collisions, valid catalog decoding, and hand-computed Recall/NDCG values.

Result: **10 tests passed**. PEFT emits the expected notice that resized embedding layers are saved in adapter checkpoints. Synthetic ranking metrics are correctness checks, not performance estimates.

Downloaded only the actual SmolLM-135M-Instruct and Qwen3-8B tokenizer/config files (no pretrained model weights). Checked the real chat-template assistant boundaries and loss masks for approximately 128 sampled products per category and model. Maximum sampled source/reconstruction lengths were 324 tokens for SmolLM and 282 for Qwen, below the configured 1,000-token limits.

The environment installed PyTorch 2.7.1, Transformers 5.14.1, Datasets 3.0.0, and PEFT 0.19.1 using the updated lock file. Python compilation, CLI help for all entry points, shell syntax, diff whitespace checks, and SmolLM/Qwen multi-GPU command-plan generation passed.

## GPU work remaining

Run the commands in [AMAZON2014.md](AMAZON2014.md) on a CUDA host, inspect exported code usage/collisions, and obtain real validation/test metrics. GPU memory fit, CUDA/DeepSpeed distributed execution, the Docker image build, full pretrained-weight loading, and convergence have not been tested here. The supplied epoch counts and learning rates are starting settings and require validation-based tuning. Multiple seeds and matched baseline implementations are still required for publishable comparisons.
