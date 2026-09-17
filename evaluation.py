"""
Intrinsic evaluation for learned semantic-ID quantization.

Two tiers of metrics, computed over the *test split* (dataset-level, not
per-batch — perplexity/collision are nonlinear statistics of the full usage
distribution, so they must be accumulated then reduced once):

Tier 2 — semantic-ID quality / codebook health
    * codebook perplexity        exp(entropy) of the pooled code-usage dist
    * active / dead codes         coverage of the codebook_range alphabet
    * collision rate              fraction of items sharing an identical ID tuple
    * uniqueness                  distinct ID tuples / items (inverse of collision)

Tier 3 — reconstruction fidelity
    * exact match / token-F1 / ROUGE-L between the product text reconstructed
      from the generated semantic IDs and the reference product text.

All metrics are dependency-free (no `evaluate`, `sacrebleu`, `rouge_score`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any

import torch

from data_module import QuantDataCollator

# Prefix the model is trained to emit before the reconstructed product text;
# stripped from both sides so metrics score the payload, not the boilerplate.
_RECON_PREFIX = "The product name and/or its metadata are: "


# --------------------------------------------------------------------------- #
# Tier 2 — codebook usage / semantic-ID quality (dataset-level accumulator)
# --------------------------------------------------------------------------- #
class SemanticIdUsageEvaluator:
    """
    Accumulates code-usage statistics across batches, then reduces once.

    Feed each batch's generated semantic-ID tensor via ``update``; call
    ``compute`` after the whole split to get the final metrics.
    """

    def __init__(
        self,
        codebook_token_ids: torch.Tensor,
        codebook_range: int,
        codebook_size: int,
    ) -> None:
        self.codebook_range = codebook_range
        self.codebook_size = codebook_size
        # Map codebook vocab id -> code index in [0, codebook_range).
        self._id_to_code = {
            int(token_id): idx
            for idx, token_id in enumerate(codebook_token_ids.tolist())
        }
        self._counts = torch.zeros(codebook_range, dtype=torch.long)
        self._id_tuples: Counter = Counter()
        self._num_items = 0

    def update(self, semantic_ids: torch.Tensor) -> None:
        # Only the first codebook_size columns are generated code positions.
        codes = semantic_ids[:, : self.codebook_size].detach().cpu().tolist()
        for row in codes:
            mapped = tuple(self._id_to_code.get(int(t), -1) for t in row)
            self._id_tuples[mapped] += 1
            self._num_items += 1
            for code in mapped:
                if code >= 0:
                    self._counts[code] += 1

    def compute(self) -> dict[str, float]:
        total = int(self._counts.sum().item())
        if total == 0 or self._num_items == 0:
            return {
                "eval/codebook_perplexity": 0.0,
                "eval/active_codes": 0,
                "eval/dead_codes": self.codebook_range,
                "eval/active_code_ratio": 0.0,
                "eval/collision_rate": 0.0,
                "eval/uniqueness": 0.0,
                "eval/num_unique_ids": 0,
                "eval/num_items": self._num_items,
            }

        probs = self._counts.float() / total
        nonzero = probs[probs > 0]
        perplexity = torch.exp(-(nonzero * nonzero.log()).sum()).item()

        active = int((self._counts > 0).sum().item())
        unique_tuples = len(self._id_tuples)
        # Items whose ID tuple is shared by at least one other item.
        collided_items = sum(c for c in self._id_tuples.values() if c > 1)

        return {
            "eval/codebook_perplexity": perplexity,
            "eval/active_codes": active,
            "eval/dead_codes": self.codebook_range - active,
            "eval/active_code_ratio": active / self.codebook_range,
            "eval/collision_rate": collided_items / self._num_items,
            "eval/uniqueness": unique_tuples / self._num_items,
            "eval/num_unique_ids": unique_tuples,
            "eval/num_items": self._num_items,
        }


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


# --------------------------------------------------------------------------- #
# Driver — run Tier 2 + Tier 3 over the test split
# --------------------------------------------------------------------------- #
def evaluate_test_split(
    trainer: Any,
    eval_dataset: Any,
    batch_size: int,
    max_samples: int | None = None,
    log_examples: int = 5,
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """
    Generate semantic IDs + reconstructions for the test split and compute the
    Tier-2 (codebook quality) and Tier-3 (reconstruction fidelity) metrics.

    Returns ``(metrics, examples)`` where ``examples`` is a list of
    ``(prediction, reference)`` pairs for qualitative inspection.
    """
    collator = QuantDataCollator()
    usage = SemanticIdUsageEvaluator(
        codebook_token_ids=trainer.codebook_token_ids,
        codebook_range=trainer.args.codebook_range,
        codebook_size=trainer.args.codebook_size,
    )

    predictions: list[str] = []
    references: list[str] = []

    total = len(eval_dataset)
    if max_samples is not None:
        total = min(max_samples, total)

    trainer.model.eval()
    for start in range(0, total, batch_size):
        items = [eval_dataset[i] for i in range(start, min(start + batch_size, total))]
        batch = collator(items)

        # Tier 2: generate semantic IDs and accumulate usage.
        semantic_ids, _ = trainer.generate_semantic_ids(batch["guessing_prompt"])
        usage.update(semantic_ids)

        # Tier 3: reconstruct product text from those IDs, collect pred/ref.
        recon = trainer.reconstruct_from_semantic_ids(
            batch["reconstruction_prompt"], semantic_ids
        )
        predictions.extend(recon)
        references.extend(
            _reference_text(p) for p in batch["reconstruction_prompt"]
        )

    metrics = usage.compute()
    metrics.update(reconstruction_text_metrics(predictions, references))

    examples = list(zip(predictions[:log_examples], references[:log_examples]))
    return metrics, examples


# --------------------------------------------------------------------------- #
# CLI driver — load a trained checkpoint and score the test split
# --------------------------------------------------------------------------- #
def _build_eval_trainer(config: dict[str, Any], checkpoint: str | None):
    """
    Reconstruct the tokenizer + model + trainer for evaluation.

    * Full fine-tune: weights (and tokenizer) are loaded from ``checkpoint``,
      which is a HF Trainer save with the already-resized embedding table.
    * LoRA: the base model is loaded, the semantic-ID tokens are re-added and
      the embeddings resized to match training, then the adapter in
      ``checkpoint`` is attached.
    """
    # Imported here so the metric utilities above stay import-light.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from constants import (
        BOS_SEMANTIC_SESSION,
        BOS_SEMANTIC_TOKEN,
        EOS_SEMANTIC_TOKEN,
    )
    from trainers import QuanSFTTrainer, QuantConfig

    use_lora = bool(config.get("use_lora"))
    trainer_cfg = config["trainer"]
    torch_dtype = torch.bfloat16 if trainer_cfg.get("bf16", False) else torch.float32

    # For full-FT the checkpoint dir carries the resized tokenizer + weights.
    tokenizer_src = config["model_name"]
    if checkpoint and not use_lora:
        tokenizer_src = checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint if (checkpoint and not use_lora) else config["model_name"],
        torch_dtype=torch_dtype,
        attn_implementation=trainer_cfg.get("attn_implementation", "sdpa"),
    )

    # Ensure the semantic-ID tokens exist and the embedding table matches
    # training (idempotent if the checkpoint already contains them).
    codebook_tokens = [f"<|CODE_{i}|>" for i in range(config["codebook_range"])]
    tokenizer.add_special_tokens(
        {
            "extra_special_tokens": [
                BOS_SEMANTIC_TOKEN,
                EOS_SEMANTIC_TOKEN,
                BOS_SEMANTIC_SESSION,
            ]
            + codebook_tokens
        }
    )
    model.resize_token_embeddings(len(tokenizer))

    if use_lora and checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint)

    trainer_args = QuantConfig(
        output_dir=trainer_cfg["output_dir"],
        per_device_eval_batch_size=trainer_cfg.get("per_device_eval_batch_size", 2),
        report_to=[],
        bf16=trainer_cfg.get("bf16", False),
        seed=config.get("seed", 42),
        codebook_size=config["codebook_size"],
        codebook_range=config["codebook_range"],
        max_source_length=config["max_source_length"],
        max_target_length=config["max_target_length"],
        generation_max_new_tokens=config.get("generation_max_new_tokens", 24),
        generation_temperature=config.get("generation_temperature", 1.0),
        generation_top_p=config.get("generation_top_p", 0.9),
        temperature_initial=trainer_cfg.get("temperature_initial"),
        temperature_final=trainer_cfg.get("temperature_final"),
    )

    trainer = QuanSFTTrainer(
        model=model,
        args=trainer_args,
        tokenizer=tokenizer,
        data_collator=QuantDataCollator(),
    )
    return trainer


def main() -> None:
    import yaml

    from data_module import build_amazon_datasets

    parser = argparse.ArgumentParser(
        description="Evaluate learned semantic-ID quantization on the test split."
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
        "--output",
        type=str,
        default=None,
        help="Where to write the metrics JSON (default: <output_dir>/eval_metrics.json).",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    _, eval_dataset = build_amazon_datasets(config)
    trainer = _build_eval_trainer(config, args.checkpoint)

    max_samples = args.max_samples
    if max_samples is None:
        max_samples = config.get("max_eval_samples")

    metrics, examples = evaluate_test_split(
        trainer=trainer,
        eval_dataset=eval_dataset,
        batch_size=config["trainer"].get("per_device_eval_batch_size", 2),
        max_samples=max_samples,
    )

    print("Semantic-ID test-split evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    for i, (pred, ref) in enumerate(examples):
        print(f"  [example {i}] pred={pred!r} | ref={ref!r}")

    output_path = args.output or os.path.join(
        config["trainer"]["output_dir"], "eval_metrics.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
