#!/usr/bin/env python
"""Build P5-style sequential-recommendation data from Amazon 2014 reviews.

This faithfully reproduces the processing in the P5 paper's preprocessing notebook
(https://github.com/jeykigung/P5, ``preprocess/data_preprocess_amazon.ipynb``) for the
Amazon 2014 dataset, producing the core sequential artifacts:

    <out>/<category>/sequential_data.txt   "<uid> <iid> <iid> ..." per user, time-ordered
    <out>/<category>/datamaps.json         user2id / item2id / id2user / id2item (+ attrs)
    <out>/<category>/item2attributes.json  item-id -> [attribute-id, ...]
    <out>/<category>/negative_samples.txt  "<uid> <neg> <neg> ..." (99 negatives per user)

Inputs (download with ``download_amazon_2014.py``):
    <data-dir>/reviews_<Category>_5.json.gz   (5-core reviews -> interaction sequences)
    <data-dir>/meta_<Category>.json.gz        (product metadata -> item attributes)

P5 parameters (kept identical): rating_score=0.0, user_core=5, item_core=5,
attribute_core=0, test_num=99, seed=2022. The P5 notebook processes one category into
its own directory; we do the same per category, so each category keeps a disjoint,
1-indexed id space (matching the notebook's one-dir-per-dataset layout).

Usage:
    python preprocess_amazon_2014.py Beauty Sports_and_Outdoors Toys_and_Games \
        --data-dir ./amazon2014_meta --out ./amazon2014_seq
"""

import argparse
import ast
import gzip
import json
import os
import random
from collections import defaultdict
from typing import Any, Iterator

import numpy as np

# --- P5 constants -----------------------------------------------------------------
SEED = 2022
RATING_SCORE = 0.0  # reviews with overall <= this are dropped (0.0 keeps all ratings)
USER_CORE = 5
ITEM_CORE = 5
ATTRIBUTE_CORE = 0
TEST_NUM = 99


def _parse_gzip_dicts(path: str) -> Iterator[dict[str, Any]]:
    """Yield dict records from a loose-JSON gzip file (one Python-dict literal / line).

    The 2014 files use single-quoted Python-dict literals, so each line is parsed with
    ``ast.literal_eval`` (the safe equivalent of P5's ``eval``); malformed lines are
    skipped. Mirrors ``data_module._iter_amazon_2014_records``' parsing strategy.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if isinstance(record, dict):
                yield record


def read_amazon_reviews(path: str, rating_score: float) -> list[tuple[str, str, int]]:
    """P5 ``Amazon``: collect (reviewerID, asin, unixReviewTime) above a rating floor."""
    datas: list[tuple[str, str, int]] = []
    for inter in _parse_gzip_dicts(path):
        try:
            if float(inter["overall"]) <= rating_score:
                continue
            datas.append((inter["reviewerID"], inter["asin"], int(inter["unixReviewTime"])))
        except (KeyError, TypeError, ValueError):
            continue
    return datas


def get_interaction(datas: list[tuple[str, str, int]]) -> dict[str, list[str]]:
    """P5 ``get_interaction``: group by user and order each user's items by timestamp."""
    user_seq: dict[str, list[tuple[str, int]]] = {}
    for user, item, time in datas:
        user_seq.setdefault(user, []).append((item, time))

    user_items: dict[str, list[str]] = {}
    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        user_items[user] = [item for item, _ in item_time]
    return user_items


def check_Kcore(
    user_items: dict[str, list[str]], user_core: int, item_core: int
) -> tuple[dict[str, int], dict[str, int], bool]:
    """P5 ``check_Kcore``: per-user / per-item interaction counts and a k-core flag."""
    user_count: dict[str, int] = defaultdict(int)
    item_count: dict[str, int] = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for num in user_count.values():
        if num < user_core:
            return user_count, item_count, False
    for num in item_count.values():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True


def filter_Kcore(
    user_items: dict[str, list[str]], user_core: int, item_core: int
) -> dict[str, list[str]]:
    """P5 ``filter_Kcore``: iteratively drop under-core users/items until stable.

    Each pass removes users below ``user_core`` and, for the rest, items below
    ``item_core``; it repeats until the remaining graph is a k-core. Items are removed
    while iterating a copy of each user's list (vs P5's in-place ``.remove``) so a
    single pass drops every under-core item; the k-core fixpoint is identical either
    way. Empty user lists are pruned so they cannot linger below ``user_core``.
    """
    user_count, item_count, is_kcore = check_Kcore(user_items, user_core, item_core)
    while not is_kcore:
        for user in list(user_items.keys()):
            if user_count[user] < user_core:
                user_items.pop(user)
            else:
                for item in list(user_items[user]):
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
                if not user_items[user]:
                    user_items.pop(user)
        user_count, item_count, is_kcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(
    user_items: dict[str, list[str]],
) -> tuple[dict[str, list[str]], int, int, dict[str, dict[str, str]]]:
    """P5 ``id_map``: shuffle users, then assign 1-indexed string ids to users/items."""
    user2id: dict[str, str] = {}
    item2id: dict[str, str] = {}
    id2user: dict[str, str] = {}
    id2item: dict[str, str] = {}
    user_id = 1
    item_id = 1
    final_data: dict[str, list[str]] = {}

    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        final_data[user2id[user]] = iids

    data_maps = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    }
    return final_data, user_id - 1, item_id - 1, data_maps


def load_amazon_meta(path: str, data_maps: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """P5 ``Amazon_meta``: load meta records for items present in ``item2id`` only."""
    item_asins = set(data_maps["item2id"].keys())
    metas: dict[str, dict[str, Any]] = {}
    for info in _parse_gzip_dicts(path):
        asin = info.get("asin")
        if asin in item_asins:
            metas[asin] = info
    return metas


def get_attribute_amazon(
    meta_infos: dict[str, dict[str, Any]],
    data_maps: dict[str, Any],
    attribute_core: int,
) -> tuple[int, float, dict[str, Any], dict[str, list[int]]]:
    """P5 ``get_attribute_Amazon``: build item attributes from categories + brand.

    Each category path contributes its labels except the first (root) one, plus the
    product ``brand``; attributes appearing at least ``attribute_core`` times are kept
    (0 keeps all). Returns attribute count, mean attributes/item, the updated datamaps
    (with ``attribute2id`` / ``id2attribute`` / ``attributeid2num``), and the
    item-id -> [attribute-id] map.
    """
    attributes: dict[str, int] = defaultdict(int)
    for info in meta_infos.values():
        for cates in info.get("categories", []):
            for cate in cates[1:]:  # drop the root category (uninformative)
                attributes[cate] += 1
        brand = info.get("brand")
        if brand:
            attributes[brand] += 1

    new_meta: dict[str, list[str]] = {}
    for iid, info in meta_infos.items():
        kept: list[str] = []
        brand = info.get("brand")
        if brand and attributes[brand] >= attribute_core:
            kept.append(brand)
        for cates in info.get("categories", []):
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    kept.append(cate)
        new_meta[iid] = kept

    attribute2id: dict[str, int] = {}
    id2attribute: dict[str, str] = {}
    attributeid2num: dict[int, int] = defaultdict(int)
    attribute_id = 1
    items2attributes: dict[str, list[int]] = {}
    attribute_lens: list[int] = []

    for iid, attrs in new_meta.items():
        item_id = data_maps["item2id"][iid]
        items2attributes[item_id] = []
        for attribute in attrs:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[str(attribute_id)] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))

    data_maps["attribute2id"] = attribute2id
    data_maps["id2attribute"] = id2attribute
    data_maps["attributeid2num"] = {str(k): v for k, v in attributeid2num.items()}
    avg_attribute = float(np.mean(attribute_lens)) if attribute_lens else 0.0
    return len(attribute2id), avg_attribute, data_maps, items2attributes


def sample_test_data(
    user_items: dict[str, list[str]], test_num: int, sample_type: str = "random"
) -> dict[str, list[str]]:
    """P5 ``sample_test_data``: draw ``test_num`` negatives per user (not in their seq)."""
    item_count: dict[int, int] = defaultdict(int)
    int_seqs: dict[str, list[int]] = {}
    for user, items in user_items.items():
        int_items = [int(i) for i in items]
        int_seqs[user] = int_items
        for item in int_items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    counts = list(item_count.values())
    total = float(np.sum(counts))
    probability = [c / total for c in counts]

    user_neg_items: dict[str, list[str]] = {}
    for user, seq in int_seqs.items():
        seen = set(seq)
        test_samples: list[str] = []
        picked: set[str] = set()
        while len(test_samples) < test_num:
            if sample_type == "random":
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else:  # "pop": popularity-weighted
                sample_ids = np.random.choice(
                    all_item, test_num, replace=False, p=probability
                )
            for item in sample_ids:
                sid = str(item)
                if item not in seen and sid not in picked:
                    test_samples.append(sid)
                    picked.add(sid)
        user_neg_items[user] = test_samples[:test_num]
    return user_neg_items


def process(category: str, data_dir: str, out_root: str, sample_type: str = "random") -> None:
    """Run the full P5 pipeline for one category and write the four artifact files."""
    reviews_path = os.path.join(data_dir, f"reviews_{category}_5.json.gz")
    meta_path = os.path.join(data_dir, f"meta_{category}.json.gz")
    for path, kind, flag in (
        (reviews_path, "5-core reviews", "--reviews"),
        (meta_path, "metadata", "--meta"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Amazon 2014 {kind} file not found for '{category}': {path}. "
                f"Download it with `python download_amazon_2014.py {flag} "
                f"--out {data_dir} {category}`."
            )

    out_dir = os.path.join(out_root, category)
    os.makedirs(out_dir, exist_ok=True)

    datas = read_amazon_reviews(reviews_path, RATING_SCORE)
    user_items = get_interaction(datas)
    print(f"[{category}] raw interactions: {len(datas)} (ratings > {RATING_SCORE})")

    user_items = filter_Kcore(user_items, USER_CORE, ITEM_CORE)
    user_items, user_num, item_num, data_maps = id_map(user_items)

    user_count, item_count, _ = check_Kcore(user_items, USER_CORE, ITEM_CORE)
    user_lens = list(user_count.values())
    interact_num = int(np.sum(user_lens))
    sparsity = (1 - interact_num / (user_num * item_num)) * 100 if user_num and item_num else 0.0
    print(
        f"[{category}] {USER_CORE}-core: users={user_num} items={item_num} "
        f"interactions={interact_num} avg_user_len={np.mean(user_lens):.2f} "
        f"sparsity={sparsity:.2f}%"
    )

    meta_infos = load_amazon_meta(meta_path, data_maps)
    attribute_num, avg_attribute, data_maps, item2attributes = get_attribute_amazon(
        meta_infos, data_maps, ATTRIBUTE_CORE
    )
    print(
        f"[{category}] attributes={attribute_num} avg_attr/item={avg_attribute:.2f} "
        f"(meta items matched: {len(meta_infos)}/{item_num})"
    )

    negatives = sample_test_data(user_items, TEST_NUM, sample_type)

    seq_path = os.path.join(out_dir, "sequential_data.txt")
    with open(seq_path, "w", encoding="utf-8") as out:
        for user, items in user_items.items():
            out.write(user + " " + " ".join(items) + "\n")

    with open(os.path.join(out_dir, "item2attributes.json"), "w", encoding="utf-8") as out:
        json.dump(item2attributes, out)

    with open(os.path.join(out_dir, "datamaps.json"), "w", encoding="utf-8") as out:
        json.dump(data_maps, out)

    with open(os.path.join(out_dir, "negative_samples.txt"), "w", encoding="utf-8") as out:
        for user, samples in negatives.items():
            out.write(user + " " + " ".join(samples) + "\n")

    print(f"[{category}] wrote artifacts to {out_dir}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "categories",
        nargs="+",
        help="Category names, e.g. Beauty Sports_and_Outdoors Toys_and_Games",
    )
    parser.add_argument(
        "--data-dir",
        default="./amazon2014_meta",
        help="Directory with reviews_<Cat>_5.json.gz and meta_<Cat>.json.gz",
    )
    parser.add_argument(
        "--out",
        default="./amazon2014_seq",
        help="Output root; artifacts are written to <out>/<Category>/",
    )
    parser.add_argument(
        "--sample-type",
        choices=("random", "pop"),
        default="random",
        help="Negative-sampling strategy (default: random, as in P5)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (P5 uses 2022)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for category in args.categories:
        process(category, args.data_dir, args.out, args.sample_type)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
