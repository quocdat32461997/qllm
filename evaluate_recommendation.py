"""Full-catalog next-item evaluation with valid-item constrained beam search."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from amazon2014 import sha256_file
from model_utils import load_checkpoint
from recommendation import load_index, RecommendationDataset, evaluate_recommendations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-path", help="Defaults to the checkpoint's experiment.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    saved = yaml.safe_load((checkpoint / "experiment.yaml").read_text())
    config = yaml.safe_load(Path(args.config_path).read_text()) if args.config_path else saved
    index_path = checkpoint / "semantic_index.json"
    if saved["semantic_index_sha256"] != sha256_file(index_path):
        raise ValueError("The checkpoint's frozen item index has changed")
    index = load_index(index_path, config["data_dir"])
    model, tokenizer = load_checkpoint(config, checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rec = config["recommendation"]
    dataset = RecommendationDataset(Path(config["data_dir"]) / f"{args.split}.jsonl", index,
        rec.get("max_history", 20), args.max_samples)
    metrics = evaluate_recommendations(model, tokenizer, dataset, index,
        rec.get("beam_size", 20), tuple(rec.get("ks", [5, 10])),
        rec.get("eval_batch_size", 8), rec.get("max_length", 1024))
    metrics.update(split=args.split, semantic_index_sha256=sha256_file(index_path),
                   data_manifest_sha256=sha256_file(Path(config["data_dir"]) / "manifest.json"))
    output = Path(args.output) if args.output else checkpoint / f"{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
