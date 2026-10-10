"""
Evaluate a trained semantic-ID tokenizer on the recommendation test split.

Tier-2 (semantic-ID quality) + Tier-3 (reconstruction fidelity) over the
``build_amazon_datasets`` eval split — the same split ``train_rec.py`` validates
on. Semantic-ID generation is funnelled through the dedicated generator in
``generate_semantic_ids.py`` (``code_token_to_index`` /
``semantic_ids_to_code_indices``) so there is a single source of generation
truth.

Outputs (into ``--output-dir``, default ``<trainer.output_dir>/rec_eval``):
    rec_eval_metrics.json       Tier-2 + Tier-3 metrics
    semantic_id_assignments.jsonl   per-item product_name -> semantic_ids
                                    (the item corpus -> ID map; step 1 of the
                                    EVALUATION.md Tier-4 downstream plan)

Usage:
    python evaluate_rec.py \\
        --config-path configs.yaml \\
        --checkpoint outputs/semantic_ids/checkpoint-1000 \\
        --max-samples 500

Omit ``--checkpoint`` to score the untrained base model as a baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any

import torch

from data_module import QuantDataCollator, build_amazon_datasets
from evaluate_quant import SemanticIdUsageEvaluator, _build_eval_trainer
from generate_semantic_ids import code_token_to_index, semantic_ids_to_code_indices

# Prefix the model is trained to emit before the reconstructed product text;
# stripped from both sides so metrics score the payload, not the boilerplate.
_RECON_PREFIX = "The product name and/or its metadata are: "


# --------------------------------------------------------------------------- #
# Tier 3 — reconstruction fidelity (dependency-free text metrics)
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    text = text.strip()
    if text.startswith(_RECON_PREFIX):
        text = text[len(_RECON_PREFIX):]
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def _token_f1(pred: str, ref: str) -> float:
    pred_tokens, ref_tokens = _tokens(pred), _tokens(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l(pred: str, ref: str) -> float:
    pred_tokens, ref_tokens = _tokens(pred), _tokens(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def reconstruction_text_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    if not predictions:
        return {
            "eval/recon_exact_match": 0.0,
            "eval/recon_token_f1": 0.0,
            "eval/recon_rouge_l": 0.0,
        }
    exact = [float(_normalize(p) == _normalize(r)) for p, r in zip(predictions, references)]
    f1 = [_token_f1(p, r) for p, r in zip(predictions, references)]
    rouge = [_rouge_l(p, r) for p, r in zip(predictions, references)]
    n = len(predictions)
    return {
        "eval/recon_exact_match": sum(exact) / n,
        "eval/recon_token_f1": sum(f1) / n,
        "eval/recon_rouge_l": sum(rouge) / n,
    }


def _reference_text(reconstruction_prompt: list[dict[str, str]]) -> str:
    """The reference product text is the assistant turn of the recon prompt."""
    return reconstruction_prompt[-1]["content"]


def evaluate_rec_test_split(
    trainer: Any,
    eval_dataset: Any,
    batch_size: int,
    codebook_size: int,
    codebook_range: int,
    max_samples: int | None = None,
    log_examples: int = 5,
) -> tuple[dict[str, float], list[dict[str, Any]], list[tuple[str, str]]]:
    """Generate IDs for the test split and compute Tier-2 + Tier-3 metrics.

    Returns ``(metrics, assignments, examples)`` where ``assignments`` is the
    per-item corpus -> semantic-ID map and ``examples`` are
    ``(prediction, reference)`` reconstruction pairs for qualitative review.
    """
    collator = QuantDataCollator()
    usage = SemanticIdUsageEvaluator(
        codebook_token_ids=trainer.codebook_token_ids,
        codebook_range=codebook_range,
        codebook_size=codebook_size,
    )
    id_to_code = code_token_to_index(trainer)

    predictions: list[str] = []
    references: list[str] = []
    assignments: list[dict[str, Any]] = []

    total = len(eval_dataset)
    if max_samples is not None:
        total = min(max_samples, total)

    trainer.model.eval()
    for start in range(0, total, batch_size):
        items = [eval_dataset[i] for i in range(start, min(start + batch_size, total))]
        batch = collator(items)

        # Generate semantic IDs from each item's own guessing prompt (goal-1
        # generation path) and accumulate Tier-2 usage statistics.
        semantic_ids, _ = trainer.generate_semantic_ids(batch["guessing_prompt"])
        usage.update(semantic_ids)

        # Tier-3: reconstruct product text from those IDs.
        recon = trainer.reconstruct_from_semantic_ids(
            batch["reconstruction_prompt"], semantic_ids
        )
        predictions.extend(recon)
        references.extend(_reference_text(p) for p in batch["reconstruction_prompt"])

        # Per-item corpus -> ID assignment (reuses the shared mapping helper).
        code_rows = semantic_ids_to_code_indices(id_to_code, semantic_ids, codebook_size)
        for item, codes in zip(items, code_rows):
            assignments.append(
                {
                    "product_name": item.get("product_name"),
                    "semantic_ids": codes,
                    "semantic_id_tokens": [
                        f"<|CODE_{c}|>" if c >= 0 else None for c in codes
                    ],
                }
            )

    metrics = usage.compute()
    metrics.update(reconstruction_text_metrics(predictions, references))

    examples = list(zip(predictions[:log_examples], references[:log_examples]))
    return metrics, assignments, examples


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Evaluate semantic-ID quality on the recommendation test split."
    )
    parser.add_argument("--config-path", type=str, default="configs.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Trained checkpoint dir (full-FT save or LoRA adapter). "
        "If omitted, the untrained base model is evaluated.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of test items evaluated.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size (default: per_device_eval_batch_size).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write metrics JSON + assignments "
        "(default: <trainer.output_dir>/rec_eval).",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    _, eval_dataset = build_amazon_datasets(config)
    trainer = _build_eval_trainer(config, args.checkpoint)

    max_samples = args.max_samples
    if max_samples is None:
        max_samples = config.get("max_eval_samples")
    batch_size = args.batch_size or config["trainer"].get(
        "per_device_eval_batch_size", 2
    )

    with torch.no_grad():
        metrics, assignments, examples = evaluate_rec_test_split(
            trainer=trainer,
            eval_dataset=eval_dataset,
            batch_size=batch_size,
            codebook_size=config["codebook_size"],
            codebook_range=config["codebook_range"],
            max_samples=max_samples,
        )

    print("Recommendation semantic-ID test-split evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    for i, (pred, ref) in enumerate(examples):
        print(f"  [example {i}] pred={pred!r} | ref={ref!r}")

    output_dir = args.output_dir or os.path.join(
        config["trainer"]["output_dir"], "rec_eval"
    )
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "rec_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    assignments_path = os.path.join(output_dir, "semantic_id_assignments.jsonl")
    with open(assignments_path, "w", encoding="utf-8") as handle:
        for record in assignments:
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote {len(assignments)} semantic-ID assignments to {assignments_path}")


if __name__ == "__main__":
    main()
