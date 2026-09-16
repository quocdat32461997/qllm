> Amazon 2014 next-item recommendation is now implemented in `train_recommendation.py` and `evaluate_recommendation.py`. See [the run guide](docs/AMAZON2014.md). The broader retrieval and clustering experiments below remain proposals.

# Evaluation Plan — Semantic-ID Quantization

Evaluating a learned **semantic-ID tokenizer** (an LLM that maps product text to
`codebook_size` discrete codes, each from a `codebook_range` alphabet, trained
with a Gumbel-softmax + reconstruction objective). This is the TIGER / RQ-VAE
lineage of discrete quantization for generative retrieval, and there is no
single clean supervised metric — so we evaluate across four tiers, from cheap
intrinsic training signals up to the downstream task that is the real goal.

Key principle used throughout: **perplexity, collision, and uniqueness are
nonlinear statistics of the full usage distribution.** They must be accumulated
over the whole split and reduced *once* — never averaged over per-batch values
(the same reason LLM perplexity accumulates total NLL over a corpus rather than
averaging per-batch perplexities).

---

## Tier 1 — Intrinsic codebook health (training-time signal)

**Purpose.** Watch codebook utilization *evolve* during training as the
Gumbel-softmax temperature anneals from random (high, noise-driven usage) to
committed (data-driven usage). This is a **trend** signal, read per batch.

**Cadence.** Every training step, logged from `compute_loss`.

**Scope.** Per batch (fixed batch size ⇒ constant bias ⇒ the trend is still
informative even though the absolute value is capped by
`batch_size × codebook_size`).

| Metric | Definition | Reads as |
| --- | --- | --- |
| `semantic_id_perplexity` | `exp(-Σ p_k log p_k)` over pooled code usage | 1 = collapse, → `codebook_range` = uniform |
| `semantic_id_diversity` | unique codes used / `codebook_range` | fraction of alphabet touched, ∈ [0,1] |
| `semantic_id_perplexity_pos_{j}` | per-position perplexity (each of the `codebook_size` slots) | catches a single position collapsing while the pool looks healthy |

**Interpretation.**
- Expect perplexity to start high (Gumbel noise), dip as the model commits, then
  settle. Where it settles is the signal: toward `codebook_range` = healthy;
  toward 1 = collapse.
- High diversity + low perplexity = wide but heavily skewed usage → early-stage
  collapse. The **disagreement** between the two is itself diagnostic.
- Judge per-batch numbers against their own early-training baseline, not against
  `codebook_range` (the batch-size cap prevents reaching it). Lean on logger
  smoothing.

**Status.** ✅ Implemented — `trainers.py`:
`_compute_codebook_perplexity`, `_compute_diversity_score`,
`_compute_per_position_perplexity`, logged in `compute_loss`.

---

## Tier 2 — Semantic-ID quality (dataset-level)

**Purpose.** Measure the *true* codebook utilization and the distinguishability
of the learned IDs — the primary defense against **codebook collapse** (a
fraction of codes used, semantically unrelated items mapped together).

**Cadence.** On demand over the test split (post-training or from a checkpoint).

**Scope.** Whole split, accumulated then reduced once.

| Metric | Definition | Target |
| --- | --- | --- |
| `eval/codebook_perplexity` | pooled `exp(entropy)` over the dataset usage histogram | high, → `codebook_range` |
| `eval/active_codes` / `eval/dead_codes` | codes used ≥ 1× / never used | few dead codes |
| `eval/active_code_ratio` | active / `codebook_range` | → 1.0 |
| `eval/collision_rate` | fraction of items sharing an identical `codebook_size`-tuple | low |
| `eval/uniqueness` | distinct ID tuples / items | high, → 1.0 |
| `eval/num_unique_ids`, `eval/num_items` | bookkeeping | — |

**Interpretation.**
- Uniqueness is the foundational sanity check against collapse: a collapsed
  codebook has near-zero uniqueness regardless of perplexity.
- Collision nuance (future refinement): not all collisions are harmful — two
  genuinely similar products sharing an ID is benign; unrelated ones colliding
  is harmful. A "qualification-aware" split (valid-pair masking) distinguishes
  them. Raw collision rate is the first-pass distinguishability check.

**Status.** ✅ Implemented — `evaluation.py`: `SemanticIdUsageEvaluator`
(accumulator), driven by `evaluate_test_split`.

---

## Tier 3 — Reconstruction fidelity

**Purpose.** Confirm the codes actually *retain* the item's information: text
reconstructed from the generated semantic IDs should match the source product
text. A proxy for information content, not the end goal (high fidelity with a
collapsed codebook is still bad — always read alongside Tier 2).

**Cadence.** Training-time loss signal + on-demand text metrics over the test
split.

**Scope.** Per item; averaged over the split.

| Metric | Definition | Reads as |
| --- | --- | --- |
| `reconstruction_loss` | training reconstruction objective | lower = better fidelity |
| `eval/recon_exact_match` | normalized exact string match (prefix-stripped) | strict fidelity |
| `eval/recon_token_f1` | token-overlap F1 | soft content match |
| `eval/recon_rouge_l` | LCS-based ROUGE-L F1 | ordered content match |

**Method.** The reconstruction injects the *real* generated code tokens into the
semantic-ID slots of the reconstruction prompt (mirroring the training scatter,
hard tokens instead of soft embeddings), drops the reference answer, and lets
the model complete the product text. Metrics are dependency-free (no
`evaluate` / `rouge_score`) and strip the fixed answer prefix so they score the
payload, not boilerplate.

**Status.** ✅ Implemented — `trainers.py`: `reconstruct_from_semantic_ids`;
`evaluation.py`: `reconstruction_text_metrics`, wired in `evaluate_test_split`.

---

## Tier 4 — Downstream task (gold standard)

**Purpose.** The metric the field trusts: feed the generated semantic IDs into a
generative retrieval / recommendation model and measure task performance. All
intrinsic tiers are proxies for this.

**Cadence.** Separate downstream training + eval run, once the tokenizer is
stable.

**Scope.** Standard retrieval/recommendation protocol.

| Metric | Definition | Setting |
| --- | --- | --- |
| Recall@K | fraction of relevant items retrieved in top-K | K = 5, 10 (rec); 50, 100 (retrieval) |
| NDCG@K | rank-weighted relevance | same K |
| Hit@K / MRR | any hit in top-K / mean reciprocal rank | sequential rec |
| Semantic-structure check | do prefix-sharing IDs (same first code) form coherent item clusters? | tests hierarchy is meaningful, not just unique |

**Plan.**
1. Freeze the trained semantic-ID tokenizer; assign IDs to the full item corpus.
2. Train a generative retriever (sequence-to-semantic-ID) on interaction data.
3. Report Recall@K / NDCG@K vs. baselines (e.g. random IDs, hash IDs) to isolate
   the value of *learned* semantic structure.
4. Optional: cluster-coherence audit on shared ID prefixes.

**Status.** ⬜ Not implemented — requires a downstream dataset + retriever
outside this repo's current scope.

---

## Running the evaluation

```bash
# Tier 2 + Tier 3 over the test split, from a trained checkpoint
.venv/bin/python evaluation.py \
  --config-path configs_qwen8b.yaml \
  --checkpoint outputs/semantic_ids/checkpoint-1000 \
  --max-samples 500
```

- Omit `--checkpoint` to score the untrained base model as a baseline.
- Writes `eval_metrics.json` to `<output_dir>/` (override with `--output`).
- Tier 1 is emitted automatically during `train.py` (per-step wandb logs).

## Summary of status

| Tier | Signal | Cadence | Status |
| --- | --- | --- | --- |
| 1 — codebook health | perplexity, diversity, per-position | per batch (train) | ✅ |
| 2 — ID quality | perplexity, dead codes, collision, uniqueness | per dataset (eval) | ✅ |
| 3 — reconstruction | recon loss, EM / token-F1 / ROUGE-L | train loss + eval | ✅ |
| 4 — next-item recommendation | Recall@5/10, NDCG@5/10 | separate run | Implemented; full GPU experiment pending |

## References

- [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/pdf/2305.05065)
- [Adapting LLMs by Integrating Collaborative Semantics (LC-Rec)](https://arxiv.org/pdf/2311.09049)
- [Taming the Long Tail: Robust Semantic ID Generation](https://arxiv.org/pdf/2510.25622)
- [Semantic IDs for Recommender Systems at Snapchat](https://arxiv.org/html/2604.03949v1)
- [Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning](https://arxiv.org/html/2603.00632v1)
- [HiD-VAE: Interpretable Generative Recommendation via Hierarchical Semantic IDs](https://arxiv.org/pdf/2508.04618)
