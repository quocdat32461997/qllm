"""Reproducible Amazon 2014 preparation; no model dependencies required."""

import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

CATEGORIES = ("Beauty", "Sports_and_Outdoors", "Toys_and_Games")
DATA_URL = "https://snap.stanford.edu/data/amazon/productGraph"
METADATA_FIELDS = ("asin", "title", "description", "brand", "categories", "price")


def canonical_category(value):
    aliases = {"beauty": "Beauty", "sports": "Sports_and_Outdoors",
               "sports_and_outdoors": "Sports_and_Outdoors", "toys": "Toys_and_Games",
               "toys_and_games": "Toys_and_Games"}
    key = value.lower().replace(" & ", "_and_").replace(" ", "_")
    if key not in aliases:
        raise ValueError(f"Unknown Amazon 2014 category {value!r}; choose {CATEGORIES}")
    return aliases[key]


def read_records(path):
    """2014 metadata is often Python literals despite its .json suffix."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = ast.literal_eval(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_no}: {exc}") from exc
            yield row


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    print(f"Downloading {url} -> {path}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as dest:
        shutil.copyfileobj(response, dest)
    # An HTML error page must never become a cached dataset.
    with gzip.open(partial, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    partial.replace(path)


def filter_k_core(events, min_user=5, min_item=5):
    while True:
        users = Counter(e[0] for e in events)
        items = Counter(e[1] for e in events)
        kept = [e for e in events if users[e[0]] >= min_user and items[e[1]] >= min_item]
        if len(kept) == len(events):
            return kept
        events = kept


def sequence_splits(events, max_history=20):
    if max_history < 1:
        raise ValueError("max_history must be positive")
    by_user = defaultdict(list)
    for user, asin, timestamp, order in events:
        by_user[user].append((timestamp, order, asin))
    splits = {name: [] for name in ("train", "validation", "test")}
    for user, values in sorted(by_user.items()):
        values.sort()  # Original row order resolves equal timestamps deterministically.
        if len(values) < 3:
            raise ValueError(f"User {user} needs at least three events for leave-two-out")
        for i in range(1, len(values)):
            split = "test" if i == len(values) - 1 else "validation" if i == len(values) - 2 else "train"
            history = values[max(0, i - max_history):i]
            splits[split].append({
                "user_id": user,
                "history": [v[2] for v in history],
                "target": values[i][2],
                "history_timestamps": [v[0] for v in history],
                "target_timestamp": values[i][0],
            })
    return splits


def prepare_category(category, reviews_path, metadata_path, output_dir,
                     min_user=5, min_item=5, max_history=20):
    category = canonical_category(category)
    if min_user < 3 or min_item < 1:
        raise ValueError("min_user must be >= 3 and min_item >= 1")
    events = []
    for order, row in enumerate(read_records(reviews_path)):
        try:
            user, asin = row["reviewerID"], row["asin"]
            timestamp = int(row["unixReviewTime"])
            if not isinstance(user, str) or not user or not isinstance(asin, str) or not asin:
                raise ValueError("empty/non-string reviewerID or asin")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid review {order + 1} in {reviews_path}: {exc}") from exc
        events.append((user, asin, timestamp, order))
    raw_count = len(events)
    events = filter_k_core(events, min_user, min_item)
    if not events:
        raise ValueError("No interactions remain after k-core filtering")
    catalog = {e[1] for e in events}
    products = {}
    for row in read_records(metadata_path):
        asin = row.get("asin")
        if asin in catalog:
            if asin in products:
                raise ValueError(f"Duplicate catalog metadata for {asin}")
            products[asin] = {key: row[key] for key in METADATA_FIELDS if key in row}
    missing = sorted(catalog - products.keys())
    if missing:
        raise ValueError(f"Metadata missing for {len(missing)} items, e.g. {missing[:5]}; "
                         "refusing to silently change the benchmark catalog")
    splits = sequence_splits(events, max_history)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "products.jsonl", (products[asin] for asin in sorted(products)))
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    manifest = {
        "schema_version": 1, "dataset": "amazon_2014", "category": category,
        "users": len({e[0] for e in events}), "items": len(catalog),
        "raw_interactions": raw_count, "interactions": len(events),
        "duplicate_user_item_events": len(events) - len({(e[0], e[1]) for e in events}),
        "missing_titles": sum(not row.get("title") for row in products.values()),
        "min_user": min_user, "min_item": min_item, "max_history": max_history,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "protocol": "per-user timestamp order; final event test, penultimate validation; all earlier prefixes train",
        "timestamp_ties": "original review row order", "rating_filter": None,
        "duplicate_policy": "preserve source events",
        "catalog_scope": "all items in filtered interactions (transductive metadata only)",
        "sources": {
            "reviews": {"path": str(reviews_path), "sha256": sha256_file(reviews_path)},
            "metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        },
        "artifacts": {name: sha256_file(output_dir / name) for name in
                      ("products.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl")},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("dataset") != "amazon_2014":
        raise ValueError("Expected an Amazon 2014 manifest")
    for name, expected in manifest["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Prepared data changed: {directory / name}; rerun preparation")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/amazon2014/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/amazon2014/processed"))
    parser.add_argument("--download", action="store_true", help="Download missing official 5-core and metadata files")
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=20)
    args = parser.parse_args()
    for category in dict.fromkeys(map(canonical_category, args.categories)):
        reviews = args.raw_dir / f"reviews_{category}_5.json.gz"
        metadata = args.raw_dir / f"meta_{category}.json.gz"
        if args.download:
            download_file(f"{DATA_URL}/categoryFiles/{reviews.name}", reviews)
            download_file(f"{DATA_URL}/categoryFiles/{metadata.name}", metadata)
        manifest = prepare_category(category, reviews, metadata, args.output_dir / category,
                                    args.min_user, args.min_item, args.max_history)
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
