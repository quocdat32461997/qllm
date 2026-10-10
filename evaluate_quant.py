"""
Intrinsic quantization evaluation for learned semantic-IDs — Tier-2.

Tier 2 — semantic-ID quality / codebook health, computed over the *test split*
(dataset-level, not per-batch — perplexity/collision are nonlinear statistics of
the full usage distribution, so they must be accumulated then reduced once):
    * codebook perplexity        exp(entropy) of the pooled code-usage dist
    * active / dead codes         coverage of the codebook_range alphabet
    * collision rate              fraction of items sharing an identical ID tuple
    * uniqueness                  distinct ID tuples / items (inverse of collision)

Two modes, one CLI:
  * default           Tier-2 summary over the whole (merged) test split from
                      ``build_amazon_datasets`` → ``eval_metrics.json``.
  * ``--by-category`` loads each category separately (keeping provenance) and
                      additionally reports semantic-ID *collapse* (how many
                      products share an identical code tuple) and the
                      per-codebook-position code-value distribution, with PNG
                      plots → ``quant_eval_metrics.json`` + ``*.png``.

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

from data_module import AmazonSemanticIdDataset, QuantDataCollator

DEFAULT_CATEGORIES = ["All_Beauty", "Sports_and_Outdoors", "Toys_and_Games"]


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
# Per-category accumulator: usage metrics + per-position + full-tuple counts
# --------------------------------------------------------------------------- #
class CategoryStats:
    """Wraps a ``SemanticIdUsageEvaluator`` and adds per-position histograms."""

    def __init__(
        self,
        codebook_token_ids: torch.Tensor,
        codebook_range: int,
        codebook_size: int,
    ) -> None:
        self.codebook_range = codebook_range
        self.codebook_size = codebook_size
        self.usage = SemanticIdUsageEvaluator(
            codebook_token_ids, codebook_range, codebook_size
        )
        self._id_to_code = dict(self.usage._id_to_code)
        # One code-value Counter per codebook position.
        self.position_counts: list[Counter] = [
            Counter() for _ in range(codebook_size)
        ]
        # Full code-tuple counts (basis for the collapse histogram).
        self.tuple_counts: Counter = Counter()

    def update(self, semantic_ids: torch.Tensor) -> None:
        self.usage.update(semantic_ids)
        codes = semantic_ids[:, : self.codebook_size].detach().cpu().tolist()
        for row in codes:
            mapped = tuple(self._id_to_code.get(int(t), -1) for t in row)
            self.tuple_counts[mapped] += 1
            for pos, code in enumerate(mapped):
                self.position_counts[pos][code] += 1

    def compute(self, top_shared: int = 20) -> dict[str, Any]:
        metrics = self.usage.compute()
        # Strip the "eval/" prefix for readability in this report.
        metrics = {k.replace("eval/", ""): v for k, v in metrics.items()}

        # Collapse: histogram of how many tuples are shared by N products.
        multiplicity_hist: Counter = Counter()
        for count in self.tuple_counts.values():
            multiplicity_hist[count] += 1
        most_shared = [
            {"tuple": list(tpl), "count": cnt}
            for tpl, cnt in self.tuple_counts.most_common(top_shared)
        ]

        metrics["collapse"] = {
            # {products_sharing_a_tuple: number_of_such_tuples}
            "multiplicity_histogram": {
                str(k): multiplicity_hist[k] for k in sorted(multiplicity_hist)
            },
            "max_products_sharing_one_id": (
                max(self.tuple_counts.values()) if self.tuple_counts else 0
            ),
            "most_shared_ids": most_shared,
        }
        metrics["position_histograms"] = {
            str(pos): {
                str(code): cnt for code, cnt in sorted(counter.items())
            }
            for pos, counter in enumerate(self.position_counts)
        }
        return metrics


# --------------------------------------------------------------------------- #
# Whole-split driver — Tier 2 over the merged test split
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
# Per-category dataset loading (keeps category provenance)
# --------------------------------------------------------------------------- #
def build_eval_datasets_by_category(
    config: dict[str, Any],
    categories: list[str],
) -> dict[str, AmazonSemanticIdDataset]:
    """
    Load each category separately and return its eval split, mirroring the
    split logic in ``data_module.build_amazon_datasets`` (same seed / ratio) so
    the eval items match those the model was validated on during training.
    """
    from datasets import load_dataset

    dataset_kwargs = {
        "codebook_size": config["codebook_size"],
        "codebook_range": config["codebook_range"],
        "feature_probability": config.get("feature_probability", 0.5),
        "seed": config.get("seed", 42),
        "max_num_chars": config.get("max_num_chars"),
    }
    split_ratio = config.get("eval_split_ratio", 0.02)
    seed = config.get("seed", 42)
    max_total_samples = config.get("max_total_samples")

    datasets: dict[str, AmazonSemanticIdDataset] = {}
    for category in categories:
        raw = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        )
        if max_total_samples:
            raw = raw.select(range(min(max_total_samples, len(raw))))
        split = raw.train_test_split(test_size=split_ratio, seed=seed)
        datasets[category] = AmazonSemanticIdDataset(
            split["test"], **dataset_kwargs
        )
    return datasets


def run_category(
    trainer: Any,
    dataset: AmazonSemanticIdDataset,
    batch_size: int,
    codebook_token_ids: torch.Tensor,
    codebook_range: int,
    codebook_size: int,
    global_stats: CategoryStats,
    max_samples: int | None,
) -> CategoryStats:
    collator = QuantDataCollator()
    stats = CategoryStats(codebook_token_ids, codebook_range, codebook_size)

    total = len(dataset)
    if max_samples is not None:
        total = min(max_samples, total)

    trainer.model.eval()
    for start in range(0, total, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, total))]
        batch = collator(items)
        semantic_ids, _ = trainer.generate_semantic_ids(batch["guessing_prompt"])
        stats.update(semantic_ids)
        global_stats.update(semantic_ids)
    return stats


# --------------------------------------------------------------------------- #
# Plotting (matplotlib is a plotting-only dependency)
# --------------------------------------------------------------------------- #
def write_plots(
    per_category: dict[str, dict[str, Any]],
    global_metrics: dict[str, Any],
    codebook_size: int,
    codebook_range: int,
    output_dir: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - env dependent
        print(f"[warn] matplotlib unavailable ({exc}); skipping PNG plots.")
        return

    categories = list(per_category.keys())
    codes = list(range(codebook_range))

    # One grouped bar chart per codebook position (series = category).
    width = 0.8 / max(len(categories), 1)
    for pos in range(codebook_size):
        fig, ax = plt.subplots(figsize=(16, 4))
        for offset, category in enumerate(categories):
            hist = per_category[category]["position_histograms"][str(pos)]
            heights = [hist.get(str(c), 0) for c in codes]
            xs = [c + offset * width for c in codes]
            ax.bar(xs, heights, width=width, label=category)
        ax.set_title(f"Codebook position {pos} — code-value distribution")
        ax.set_xlabel("code value")
        ax.set_ylabel("num products")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, f"codebook_pos_{pos}_dist.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)

    # Collapse: tuple-multiplicity histogram (global).
    hist = global_metrics["collapse"]["multiplicity_histogram"]
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = sorted(int(k) for k in hist)
    ax.bar([str(x) for x in xs], [hist[str(x)] for x in xs])
    ax.set_title("Semantic-ID collapse — products sharing an identical ID tuple")
    ax.set_xlabel("num products sharing the same ID tuple")
    ax.set_ylabel("num distinct ID tuples")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "collision_histogram.png"), dpi=120)
    plt.close(fig)


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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _run_whole_split(args, config, trainer, batch_size: int) -> None:
    from data_module import build_amazon_datasets

    _, eval_dataset = build_amazon_datasets(config)

    max_samples = args.max_samples
    if max_samples is None:
        max_samples = config.get("max_eval_samples")

    metrics = evaluate_test_split(
        trainer=trainer,
        eval_dataset=eval_dataset,
        batch_size=batch_size,
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


def _run_by_category(args, config, trainer, batch_size: int) -> None:
    datasets = build_eval_datasets_by_category(config, args.categories)

    codebook_size = config["codebook_size"]
    codebook_range = config["codebook_range"]
    output_dir = args.output_dir or os.path.join(
        config["trainer"]["output_dir"], "quant_eval"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Global accumulator spans every category (overall collapse picture).
    global_stats = CategoryStats(
        trainer.codebook_token_ids, codebook_range, codebook_size
    )

    per_category: dict[str, dict[str, Any]] = {}
    for category, dataset in datasets.items():
        print(f"\n=== {category}: {len(dataset)} eval products ===")
        stats = run_category(
            trainer=trainer,
            dataset=dataset,
            batch_size=batch_size,
            codebook_token_ids=trainer.codebook_token_ids,
            codebook_range=codebook_range,
            codebook_size=codebook_size,
            global_stats=global_stats,
            max_samples=args.max_samples_per_category,
        )
        per_category[category] = stats.compute()

    global_metrics = global_stats.compute()

    report = {
        "checkpoint": args.checkpoint,
        "categories": args.categories,
        "codebook_size": codebook_size,
        "codebook_range": codebook_range,
        "global": global_metrics,
        "per_category": per_category,
    }

    # Console summary.
    print("\n===== Global semantic-ID collapse =====")
    for key in (
        "num_items",
        "num_unique_ids",
        "collision_rate",
        "uniqueness",
        "codebook_perplexity",
        "active_codes",
        "dead_codes",
    ):
        print(f"  {key}: {global_metrics[key]}")
    print(
        "  max products sharing one id: "
        f"{global_metrics['collapse']['max_products_sharing_one_id']}"
    )
    for category, metrics in per_category.items():
        print(f"\n----- {category} -----")
        print(f"  num_items: {metrics['num_items']}")
        print(f"  num_unique_ids: {metrics['num_unique_ids']}")
        print(f"  collision_rate: {metrics['collision_rate']}")
        print(f"  uniqueness: {metrics['uniqueness']}")
        print(f"  codebook_perplexity: {metrics['codebook_perplexity']}")

    metrics_path = os.path.join(output_dir, "quant_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote metrics to {metrics_path}")

    write_plots(per_category, global_metrics, codebook_size, codebook_range, output_dir)
    print(f"Wrote plots to {output_dir}")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description=(
            "Intrinsic semantic-ID quantization eval (Tier-2). Default: summary "
            "over the whole merged test split. With --by-category: per-category "
            "collapse + codebook distribution + PNG plots."
        )
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
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size (default: per_device_eval_batch_size).",
    )
    parser.add_argument(
        "--by-category",
        action="store_true",
        help="Run the per-category collapse + codebook-distribution analysis "
        "(with PNG plots) instead of the whole-split Tier-2 summary.",
    )
    # Whole-split mode.
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="[whole-split] Cap the number of test items evaluated.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="[whole-split] Metrics JSON path (default: <output_dir>/eval_metrics.json).",
    )
    # Per-category mode.
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="[--by-category] Amazon-Reviews-2023 raw_meta_<X> category config names.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="[--by-category] Output dir for metrics JSON + PNGs "
        "(default: <output_dir>/quant_eval).",
    )
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=None,
        help="[--by-category] Cap products per category (default: entire eval split).",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    trainer = _build_eval_trainer(config, args.checkpoint)
    batch_size = args.batch_size or config["trainer"].get(
        "per_device_eval_batch_size", 2
    )

    if args.by_category:
        _run_by_category(args, config, trainer, batch_size)
    else:
        _run_whole_split(args, config, trainer, batch_size)


if __name__ == "__main__":
    main()
