"""
Intrinsic quantization evaluation for learned semantic-IDs — Tier-2 only.

Tier 2 — semantic-ID quality / codebook health, computed over the *test split*
(dataset-level, not per-batch — perplexity/collision are nonlinear statistics of
the full usage distribution, so they must be accumulated then reduced once):
    * codebook perplexity        exp(entropy) of the pooled code-usage dist
    * active / dead codes         coverage of the codebook_range alphabet
    * collision rate              fraction of items sharing an identical ID tuple
    * uniqueness                  distinct ID tuples / items (inverse of collision)

Tier-3 reconstruction fidelity lives in ``evaluate_rec.py`` (the recommendation
evaluation), which reuses ``SemanticIdUsageEvaluator`` / ``_build_eval_trainer``
from this module.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

import torch

from data_module import QuantDataCollator


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
# Driver — run Tier 2 over the test split
# --------------------------------------------------------------------------- #
def evaluate_test_split(
    trainer: Any,
    eval_dataset: Any,
    batch_size: int,
    max_samples: int | None = None,
) -> dict[str, float]:
    """
    Generate semantic IDs for the test split and compute the Tier-2 codebook
    quality metrics. Returns the metrics dict.
    """
    collator = QuantDataCollator()
    usage = SemanticIdUsageEvaluator(
        codebook_token_ids=trainer.codebook_token_ids,
        codebook_range=trainer.args.codebook_range,
        codebook_size=trainer.args.codebook_size,
    )

    total = len(eval_dataset)
    if max_samples is not None:
        total = min(max_samples, total)

    trainer.model.eval()
    for start in range(0, total, batch_size):
        items = [eval_dataset[i] for i in range(start, min(start + batch_size, total))]
        batch = collator(items)

        # Generate semantic IDs and accumulate codebook usage.
        semantic_ids, _ = trainer.generate_semantic_ids(batch["guessing_prompt"])
        usage.update(semantic_ids)

    return usage.compute()


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

    metrics = evaluate_test_split(
        trainer=trainer,
        eval_dataset=eval_dataset,
        batch_size=config["trainer"].get("per_device_eval_batch_size", 2),
        max_samples=max_samples,
    )

    print("Semantic-ID quantization (Tier-2) test-split evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    output_path = args.output or os.path.join(
        config["trainer"]["output_dir"], "eval_metrics.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
