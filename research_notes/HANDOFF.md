# Handoff: semantic-ID quantization (qllm), state as of 2026-11-07

Branch `shuffle-merged-dataset` (latest `fbaa7b0`). Repo: github.com/quocdat32461997/qllm (public).

## Goal
One decoder-only LLM is the semantic-ID tokenizer, the decoder and (later) the recommender.
`train_quant.py` / `QuanSFTTrainer` (`trainers.py`): the LLM reads product text, emits 6 code
tokens (99 codes each) via Gumbel-softmax + straight-through estimator, then reconstructs the
product text from those codes. Main problem: **codebook collapse** on larger Qwen models.

## Current training objective (`trainers.py`)
- `loss = 0.1·format + 1.0·reconstruction + 1.0·KL` (`guessing_weight`, `reconstruction_weight`,
  `kl_weight`).
- **KL to uniform** per position: `log 99 − H(p̄)`. p̄ comes from the clean softmax of the code
  logits (no Gumbel noise, no temperature, as in wav2vec 2.0).
- **Sliding window** (`db9ead7`): p̄ averages micro-batch × `gradient_accumulation_steps` products.
  It holds detached usage of the previous M−1 micro-batches plus the current one. The current
  micro-batch's gradient is scaled by window/micro-batch, with the value unchanged.
- **Commitment loss: off** (`commitment_weight` 0.0). The block is gated and fixed, but its
  input-vs-output-embedding form is 0 for tied models.
- **Temperature:** fixed at `temperature_final` when `temperature_initial` is unset.
- **Memory fix** (`d5e715a`): `logits_to_keep=1` in the 6 code-selection forwards.
- Logged: `kl_loss`, `semantic_id_soft_perplexity_pos_{0..5}`, hard `semantic_id_perplexity_pos_*`,
  `semantic_id_uniqueness`, format and reconstruction losses.

## Configs
| Config | Model | Batch × accumulation | τ | Notes |
|---|---|---|---|---|
| `configs/configs_qwen06b_quant.yaml` | Qwen3-0.6B (tied) | 32 × 2 | 1.0 | fp32, no DeepSpeed, ~30 GB, ~4 h on H100 |
| `configs/configs_qwen8b_quant.yaml` | Qwen3-8B (untied) | 4 × 16 | 1.0 | bf16, DeepSpeed ZeRO-2 with CPU-offloaded Adam, ~31 GB at batch 4 |

The transformer body is frozen; only `embed_tokens` and `lm_head` train.

## Runs (W&B entity `quocdat32461997`, project `quan_semantic_ids`)
| Run | What | Result |
|---|---|---|
| `zbcnlzq4` | Qwen3-0.6B, KL 1.0, format 0.1, batch 32×2, τ 1.0 | Done, 1298 steps. recon 4.03 → 1.61; soft perplexity 99/99 all positions; uniqueness 1.0. Saved: `/workspace/qllm-ssd/outputs/qwen06b_quant/` |
| `rev1f6tw` | Qwen3-8B, τ **1e-10**, batch 4×2 (before the window) | **Collapsed**: 1–2 codes per position, uniqueness ~0.5, KL stuck ~4.3. Stopped at step 995 |
| `7l54x24n`, `nd4nazpv` | 0.6B attempts | OOM at step 205 (fixed) |

Why the 8B collapsed: per-micro-batch KL floor log(99/4) ≈ 3.2; τ 1e-10 makes the
straight-through softmax one-hot (no reconstruction gradient into selection); a saturated softmax
kills the KL gradient (‖h‖ ≈ 129); degenerate code-row init.

## Key findings
- **Every Qwen3 model puts the new code tokens on untrained padding rows** (matrix 151,936 rows vs
  tokenizer 151,669, so the resize shrinks and no init runs). 0.6B: 10/99 distinct rows; 1.7B:
  7/99 (cos 1.000); 8B lm_head: 9/99, cos 1.000. SmolLM (no collapse) gets HF mean-init rows.
- **Teacher-forcing bypass:** reconstruction improves even when codes collapse; the decoder uses
  the text.
- The 0.6B's soft perplexity of 99 can't tell "flat per-product distributions" from "confident but
  diverse". Per-product entropy H(pᵢ) is not logged yet.

## Infrastructure (RunPod)
- Network volume `33a42lbdq0` (US-NE-1, 150 GB) at `/workspace`. Pods must be in US-NE-1.
- Worktree on the volume: `/workspace/qllm-ssd` (detached checkout of this branch). Data symlinks
  `amazon2014_meta`, `amazon2014_seq`, venv `.venv` → `/workspace/qllm`. `/workspace/qllm` itself is
  the user's checkout on `main`; don't touch it.
- `run_detached.sh <config> <log>`:
  - loads `WANDB_API_KEY` / `HF_TOKEN` from `/proc/1/environ`;
  - sets `HF_HOME=/root/.cache/huggingface`;
  - **puts `/workspace/qllm/.venv/bin` on PATH** (DeepSpeed needs `ninja` for CPU Adam);
  - runs `torchrun`.
- Chain pattern: `chain*.sh` runs training, then `runpodctl stop pod $RUNPOD_POD_ID` (the API key is
  in `/proc/1/environ`). Progress goes to `outputs/chain.log`.
- Pods:
  - `q3vhgkxi2pq5lt` (H100, $3.99/hr): stopped; its host was full.
  - `rsrbueehjm5xxn` (RTX PRO 6000): stopped; **GPU fails CUDA init every time**, terminate it.
  - The user's `5uubvlaotfrd00` and `rmvfd88xzyupvb`: don't use without asking.
- As of 2026-11-03, US-NE-1 had no free GPUs.

## Launch the 8B (when a GPU frees up in US-NE-1)
1. Start `q3vhgkxi2pq5lt` (or create an H100 / RTX PRO 6000 pod in US-NE-1 with the volume, same
   image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`).
2. Check CUDA: `torch.ones(2, device="cuda")`.
3. `cd /workspace/qllm-ssd && git fetch && git checkout --detach origin/shuffle-merged-dataset`.
4. Start a chain: `run_detached.sh configs/configs_qwen8b_quant.yaml outputs/<log>`, then stop the pod.
5. Watch the first steps (W&B URL, no OOM), then the per-position soft perplexity.

## Open to-dos (priority order)
1. **Initialize the code rows** (mean/cov or k-means of centered h; normal norms) instead of padding rows.
2. **Token dropout** on the reconstruction input (30–70%) to close the teacher-forcing bypass.
3. **Log per-product entropy H(pᵢ)**; consider a mutual-information objective H(p̄) − γ·mean H(pᵢ).
4. Balance alternatives: bias balancing (loss-free, DeepSeek-V3 style), hidden-state queue for the
   KL, or an EMA (see the HTML notes in `/Users/ericngo/Desktop/projects/*.html`, local only).
5. Cosine logits to avoid saturation on the 8B; one codebook (select with E) for the untied 8B.
6. Format/reconstruction logits only at label positions (memory → larger micro-batches).
7. Scheduled sampling for reconstruction (`research_notes/TODO.md`, local only).
8. Evaluation for the paper: corpus uniqueness/collisions, true-vs-shuffled codes, Recall/NDCG vs
   TIGER/LETTER/DIGER, cold-start.

Notion: decision pages under "Can LLM describe a product's attributes via Semantic-IDs?".
