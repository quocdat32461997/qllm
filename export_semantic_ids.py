"""Freeze deterministic catalog IDs between semantic training and recommendation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import yaml

from amazon2014 import sha256_file
from data_module import AmazonSemanticIdDataset, load_amazon_catalog
from evaluation import _build_eval_trainer


def build_index(asins, codes, codebook_size, codebook_range):
    if len(asins) != len(codes) or len(set(asins)) != len(asins) or not asins:
        raise ValueError("Catalog IDs must be nonempty, unique, and aligned with codes")
    groups = defaultdict(list)
    for asin, code in zip(asins, codes):
        if len(code) != codebook_size or any(not 0 <= i < codebook_range for i in code):
            raise ValueError(f"Invalid semantic code for {asin}: {code}")
        groups[tuple(code)].append(asin)
    items = {}
    for code, bucket in sorted(groups.items()):
        for suffix, asin in enumerate(sorted(bucket)):
            items[asin] = {"codes": list(code), "suffix": suffix}
    return {
        "schema_version": 1, "codebook_size": codebook_size, "codebook_range": codebook_range,
        "suffix_count": max(map(len, groups.values())), "items": items,
        "statistics": {"items": len(items), "unique_semantic_ids": len(groups),
            "collision_item_fraction": sum(len(b) for b in groups.values() if len(b) > 1) / len(items),
            "largest_collision_bucket": max(map(len, groups.values())),
            "active_codes_per_position": [len(Counter(c[i] for c in codes)) for i in range(codebook_size)]},
    }


def export_index(config, checkpoint, output, batch_size=8):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    catalog = load_amazon_catalog(config)
    dataset = AmazonSemanticIdDataset(catalog, config["codebook_size"], config["codebook_range"],
        config.get("max_num_chars", 600), feature_probability=1.0, canonical_prompt=True)
    trainer = _build_eval_trainer(config, checkpoint)
    trainer.model.eval()
    token_to_code = {int(token): i for i, token in enumerate(trainer.codebook_token_ids)}
    asins, codes = [], []
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        ids, _ = trainer.generate_semantic_ids([row["guessing_prompt"] for row in rows])
        asins.extend(row["asin"] for row in rows)
        codes.extend([[token_to_code[t] for t in row] for row in ids.cpu().tolist()])
    index = build_index(asins, codes, config["codebook_size"], config["codebook_range"])
    index["provenance"] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "catalog_sha256": sha256_file(Path(config["data_dir"]) / "products.jsonl"),
        "data_manifest_sha256": sha256_file(Path(config["data_dir"]) / "manifest.json"),
        "prompt": "first template; feature_probability=1; greedy; no Gumbel noise",
        "max_num_chars": config.get("max_num_chars", 600),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index["statistics"], indent=2))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="configs.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    export_index(yaml.safe_load(Path(args.config_path).read_text()), args.checkpoint, args.output, args.batch_size)


if __name__ == "__main__":
    main()
