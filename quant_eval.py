"""
Quantization eval for learned semantic-IDs — collapse + per-category codebook
distribution.

Two things this script answers that ``evaluation.py`` does not:

1. **Semantic-ID collapse** — how many products share the *exact same* codebook
   series (identical full code tuple). Reported globally and per category via
   collision_rate / uniqueness plus a tuple-multiplicity histogram and the
   most-shared tuples.

2. **Per-codebook product distribution, per category** — for ``All_Beauty``,
   ``Sports_and_Outdoors`` and ``Toys_and_Games`` (overridable), both:
     * per-position code-value histograms (does a codebook collapse / do
       categories separate?), and
     * full-tuple distribution (how many unique IDs each category occupies).

``evaluation.py`` merges categories and drops the label, so this script loads
each category separately (mirroring the same split logic/seed) to keep
provenance, then reuses the model-loading and usage-metric machinery from
``evaluation.py``.

Usage:
    python quant_eval.py --config-path configs.yaml \\
        --checkpoint workspace/semantic_ids/checkpoint-1000

Outputs (into --output-dir, default <trainer.output_dir>/quant_eval):
    quant_eval_metrics.json     global + per-category metrics + collapse tables
    codebook_pos_{i}_dist.png   grouped bar chart per codebook position
    collision_histogram.png     tuple-multiplicity (collapse) histogram
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

import torch
from datasets import load_dataset

from data_module import AmazonSemanticIdDataset, QuantDataCollator
from evaluation import SemanticIdUsageEvaluator, _build_eval_trainer

DEFAULT_CATEGORIES = ["All_Beauty", "Sports_and_Outdoors", "Toys_and_Games"]


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
        # Strip the "eval/" prefix for readability in this script's report.
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
# Generation over a category's full eval split
# --------------------------------------------------------------------------- #
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
# CLI driver
# --------------------------------------------------------------------------- #
def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Semantic-ID collapse + per-category codebook distribution."
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
        "--categories",
        type=str,
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Amazon-Reviews-2023 raw_meta_<X> category config names.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write metrics JSON + PNGs "
        "(default: <trainer.output_dir>/quant_eval).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size (default: per_device_eval_batch_size).",
    )
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=None,
        help="Cap products per category (default: entire eval split).",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    trainer = _build_eval_trainer(config, args.checkpoint)
    datasets = build_eval_datasets_by_category(config, args.categories)

    codebook_size = config["codebook_size"]
    codebook_range = config["codebook_range"]
    batch_size = args.batch_size or config["trainer"].get(
        "per_device_eval_batch_size", 2
    )
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


if __name__ == "__main__":
    main()
