"""
Dedicated semantic-ID generation — input-agnostic.

Loads a trained semantic-ID tokenizer checkpoint and generates the
``codebook_size``-tuple of semantic IDs for *any* text input, decoupled from the
Amazon dataset pipeline. This is the single source of truth for semantic-ID
generation: ``evaluate_rec.py`` imports ``generate_for_prompts`` from here rather
than re-implementing the batch-generate loop.

Input (raw text only, so generation stays independent of any dataset schema):
  * ``--text "..."`` (repeatable) — raw product-text strings
  * ``--input-file FILE`` — one product text per line

Output: JSONL (``--output``), one record per item:
  ``{"product_text", "semantic_ids", "semantic_id_tokens", "semantic_id_text"}``

Usage:
    python generate_semantic_ids.py \\
        --config-path configs.yaml \\
        --checkpoint outputs/semantic_ids/checkpoint-1000 \\
        --text "Wireless gaming mouse with RGB" \\
        --input-file products.txt \\
        --output outputs/semantic_ids/generated_ids.jsonl

Omit ``--checkpoint`` to generate from the untrained base model (baseline).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from constants import EOS_SEMANTIC_TOKEN
from data_module import (
    ASSISTANT_PROMPT,
    PROMPT_TEMPLATES,
    SYSTEM_PROMPT,
)


# --------------------------------------------------------------------------- #
# Prompt construction — identical to AmazonSemanticIdDataset.__getitem__'s
# ``guessing_prompt`` so a trained model sees exactly the format it learned on.
# --------------------------------------------------------------------------- #
def build_guessing_prompt(
    product_text: str,
    codebook_size: int,
    codebook_range: int,
    template_idx: int = 0,
) -> list[dict[str, str]]:
    """Build the system/user/assistant chat turns for one product text.

    The assistant turn carries ``codebook_size`` ``EOS_SEMANTIC_TOKEN``
    placeholders that the generation loop overwrites with the real codes — this
    mirrors the training-time prompt exactly.
    """
    template = PROMPT_TEMPLATES[template_idx % len(PROMPT_TEMPLATES)]
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                codebook_size=codebook_size,
                codebook_range=codebook_range,
            ),
        },
        {
            "role": "user",
            "content": template.format(product_text=product_text),
        },
        {
            "role": "assistant",
            "content": ASSISTANT_PROMPT.format(
                SEMANTIC_IDS="".join([EOS_SEMANTIC_TOKEN] * codebook_size)
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Code-index mapping — vocab token id -> code index in [0, codebook_range).
# Mirrors SemanticIdUsageEvaluator._id_to_code.
# --------------------------------------------------------------------------- #
def code_token_to_index(trainer: Any) -> dict[int, int]:
    return {
        int(token_id): idx
        for idx, token_id in enumerate(trainer.codebook_token_ids.tolist())
    }


def semantic_ids_to_code_indices(
    id_to_code: dict[int, int],
    semantic_ids: torch.Tensor,
    codebook_size: int,
) -> list[list[int]]:
    """Map a generated ``(batch, seq)`` token-id tensor to code-index rows.

    Keeps only the first ``codebook_size`` columns (the generated code
    positions) and maps each vocab token id to its code index in
    ``[0, codebook_range)``; unknown tokens map to ``-1``. Shared by the text
    generation path and ``evaluate_rec.py`` so the mapping lives in one place.
    """
    codes = semantic_ids[:, :codebook_size].detach().cpu().tolist()
    return [[id_to_code.get(int(t), -1) for t in row] for row in codes]


# --------------------------------------------------------------------------- #
# Core generation loop — the single source of truth reused by evaluate_rec.py.
# --------------------------------------------------------------------------- #
def generate_for_prompts(
    trainer: Any,
    guessing_prompts: list[list[dict[str, str]]],
    batch_size: int,
    codebook_size: int,
) -> list[dict[str, Any]]:
    """Generate semantic IDs for pre-built guessing prompts.

    Returns one record per prompt with the code indices (``semantic_ids``), the
    human-readable ``<|CODE_i|>`` tokens, and the decoded ``semantic_id_text``.
    """
    id_to_code = code_token_to_index(trainer)
    records: list[dict[str, Any]] = []

    trainer.model.eval()
    for start in range(0, len(guessing_prompts), batch_size):
        chunk = guessing_prompts[start : start + batch_size]
        semantic_ids, semantic_id_texts = trainer.generate_semantic_ids(chunk)
        rows = semantic_ids_to_code_indices(id_to_code, semantic_ids, codebook_size)
        for code_indices, sid_text in zip(rows, semantic_id_texts):
            records.append(
                {
                    "semantic_ids": code_indices,
                    "semantic_id_tokens": [
                        f"<|CODE_{c}|>" if c >= 0 else None for c in code_indices
                    ],
                    "semantic_id_text": sid_text,
                }
            )
    return records


def generate_for_texts(
    trainer: Any,
    texts: list[str],
    batch_size: int,
    codebook_size: int,
    codebook_range: int,
    template_idx: int = 0,
) -> list[dict[str, Any]]:
    """Generate semantic IDs for arbitrary raw product-text strings."""
    prompts = [
        build_guessing_prompt(text, codebook_size, codebook_range, template_idx)
        for text in texts
    ]
    records = generate_for_prompts(trainer, prompts, batch_size, codebook_size)
    for text, record in zip(texts, records):
        record["product_text"] = text
    # Put product_text first for readability.
    return [
        {"product_text": r.pop("product_text"), **r} for r in records
    ]


# --------------------------------------------------------------------------- #
# Trainer loading (reuses evaluation._build_eval_trainer)
# --------------------------------------------------------------------------- #
def load_generator(
    config_path: str,
    checkpoint: str | None,
) -> tuple[Any, dict[str, Any]]:
    """Load config + rebuild the inference trainer. Returns ``(trainer, config)``."""
    import yaml

    from eval_utils import _build_eval_trainer

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    trainer = _build_eval_trainer(config, checkpoint)
    return trainer, config


# --------------------------------------------------------------------------- #
# Input collection
# --------------------------------------------------------------------------- #
def _collect_inputs(texts: list[str] | None, input_file: str | None) -> list[str]:
    collected: list[str] = []
    if texts:
        collected.extend(t for t in texts if t.strip())
    if input_file:
        with open(input_file, "r", encoding="utf-8") as handle:
            collected.extend(line.strip() for line in handle if line.strip())
    return collected


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate semantic IDs for arbitrary product text."
    )
    parser.add_argument("--config-path", type=str, default="configs.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Trained checkpoint dir (full-FT save or LoRA adapter). "
        "If omitted, the untrained base model is used.",
    )
    parser.add_argument(
        "--text",
        type=str,
        action="append",
        default=None,
        help="A raw product-text string (repeatable).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="A text file with one product text per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write the JSONL output (default: "
        "<output_dir>/generated_semantic_ids.jsonl).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size (default: per_device_eval_batch_size).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of inputs processed.",
    )
    parser.add_argument(
        "--template-idx",
        type=int,
        default=0,
        help="Which PROMPT_TEMPLATES entry to use when wrapping the text.",
    )
    args = parser.parse_args()

    inputs = _collect_inputs(args.text, args.input_file)
    if not inputs:
        parser.error("Provide at least one input via --text and/or --input-file.")
    if args.max_samples is not None:
        inputs = inputs[: args.max_samples]

    trainer, config = load_generator(args.config_path, args.checkpoint)
    batch_size = args.batch_size or config["trainer"].get(
        "per_device_eval_batch_size", 2
    )

    with torch.no_grad():
        records = generate_for_texts(
            trainer=trainer,
            texts=inputs,
            batch_size=batch_size,
            codebook_size=config["codebook_size"],
            codebook_range=config["codebook_range"],
            template_idx=args.template_idx,
        )

    output_path = args.output or os.path.join(
        config["trainer"]["output_dir"], "generated_semantic_ids.jsonl"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    for record in records[:10]:
        print(f"  {record['semantic_ids']}  <-  {record['product_text']!r}")
    print(f"Wrote {len(records)} semantic-ID records to {output_path}")


if __name__ == "__main__":
    main()
