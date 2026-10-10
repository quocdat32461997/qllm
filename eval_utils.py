"""
Shared evaluation utilities for the semantic-ID scripts.

Houses the pieces reused across ``evaluate_quant.py`` (Tier-2 quantization),
``evaluate_rec.py`` (Tier-2 + Tier-3 recommendation) and
``generate_semantic_ids.py`` (dedicated generation), so no eval script imports
another's internals:

    * ``SemanticIdUsageEvaluator`` — dataset-level Tier-2 usage accumulator.
    * ``_build_eval_trainer``      — rebuild tokenizer + model (+LoRA) + trainer
                                      for inference from a config + checkpoint.
"""

from __future__ import annotations

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
# Trainer loading
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
