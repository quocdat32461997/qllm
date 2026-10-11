import ast
import gzip
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Iterator

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset

from constants import BOS_SEMANTIC_TOKEN, EOS_SEMANTIC_TOKEN

SYSTEM_PROMPT = (
    f"You are a marketing analyst who specializes in product categorization and feature extraction."
    f"Semantic-IDs are {{codebook_size}} integers that each value ranges from <|CODE_0|> to <|CODE_{{codebook_range}}|>, separated by commands, "
    f"and placed between {BOS_SEMANTIC_TOKEN} and {EOS_SEMANTIC_TOKEN}."
)

PROMPT_TEMPLATES = (
    f"Please analyze the following product and its features: {{product_text}}. Then, generate semantic-IDs that meaningfully represent the product.",
    f"Analyze this catalog item and convert it into semantic-IDs: {{product_text}}.",
    f"Read the product information and map it to semantic-IDs for recommendation cold-start: {{product_text}}.",
    f"Remember the following product and summarize it by generating semantic-IDs: {{product_text}}.",
    f"Given the following product, produce semantic-IDs that capture its meaning: {{product_text}}.",
)
ASSISTANT_PROMPT = (
    f"The semantic-IDs are: {BOS_SEMANTIC_TOKEN}{{SEMANTIC_IDS}}{EOS_SEMANTIC_TOKEN}"
)

RECONSTRUCTION_PROMPT_TEMPLATE = f"The semantic-IDs are: {BOS_SEMANTIC_TOKEN}{{SEMANTIC_IDS}}{EOS_SEMANTIC_TOKEN}. Recover the product name and/or its metadata."
RECONSTRUCTION_ASSISTANT_PROMPT = (
    f"The product name and/or its metadata are: {{product_text}}"
)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return " ".join(_stringify(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify(item) for item in value)
    return str(value).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_product_name(example: dict[str, Any]) -> str:
    for key in ("title", "product_title", "name"):
        value = _clean_text(_stringify(example.get(key)))
        if value:
            return value
    return "Unknown product"


def extract_product_features(example: dict[str, Any]) -> list[str]:
    feature_chunks = []
    for key in ("features", "description", "details", "categories"):
        value = _clean_text(_stringify(example.get(key)))
        if value:
            feature_chunks.append(value)
    deduped = []
    seen = set()
    for chunk in feature_chunks:
        if chunk not in seen:
            deduped.append(chunk)
            seen.add(chunk)
    # return " ".join(deduped)
    return deduped


def build_product_text(
    product_name: str,
    product_features: str,
    include_features: bool,
    max_num_chars: int,
) -> str:
    if include_features and product_features:
        max_num_chars -= len(f"{product_name}. Features: ")

        feature_idx = 0
        while (
            feature_idx < len(product_features)
            and max_num_chars - len(product_features[feature_idx]) > 0
        ):
            max_num_chars -= len(product_features[feature_idx]) + 1  # +1 for space
            feature_idx += 1
        return f"""{product_name}. Features: {" ".join(product_features[:feature_idx])}"""  # noqa
    return product_name


def format_semantic_ids(ids: list[int]) -> str:
    encoded = ",".join(str(value) for value in ids)
    return f"{BOS_SEMANTIC_TOKEN}{encoded}{EOS_SEMANTIC_TOKEN}"


def parse_semantic_ids(
    text: str,
    codebook_size: int,
    codebook_range: int,
    fallback_ids: list[int] | None = None,
) -> list[int]:
    """Parse and normalize semantic ID values from generated text.

    Args:
        text: Generated text containing semantic ID numbers
        codebook_size: Number of semantic IDs required
        codebook_range: Maximum valid value for each semantic ID
        fallback_ids: Optional fallback values if parsing fails

    Returns:
        List of exactly `codebook_size` semantic ID values, each in range
        [1, codebook_range]
    """
    values = [int(match) for match in re.findall(r"\d+", text)]
    if fallback_ids:
        values.extend(fallback_ids)
    normalized = []
    for value in values:
        clipped = max(1, min(codebook_range, value))
        normalized.append(clipped)
        if len(normalized) == codebook_size:
            break
    while len(normalized) < codebook_size:
        normalized.append(1)
    return normalized


class AmazonSemanticIdDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        codebook_size: int,
        codebook_range: int,
        max_num_chars: int,
        feature_probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.codebook_size = codebook_size
        self.codebook_range = codebook_range
        self.feature_probability = feature_probability
        self.seed = seed
        self.max_num_chars = max_num_chars

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        example = self.records[index]
        rng = random.Random(self.seed + index)

        product_name = extract_product_name(example)
        product_features = extract_product_features(example)
        include_features = bool(product_features) and (
            rng.random() < self.feature_probability
        )
        prompt_template = PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)]
        product_text = build_product_text(
            product_name=product_name,
            product_features=product_features,
            include_features=include_features,
            max_num_chars=self.max_num_chars,
        )
        return {
            "guessing_prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        codebook_size=self.codebook_size,
                        codebook_range=self.codebook_range,
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_template.format(product_text=product_text),
                },
                {
                    "role": "assistant",
                    "content": ASSISTANT_PROMPT.format(
                        SEMANTIC_IDS="".join([EOS_SEMANTIC_TOKEN] * self.codebook_size)
                    ),
                },
            ],
            "reconstruction_prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        codebook_size=self.codebook_size,
                        codebook_range=self.codebook_range,
                    ),
                },
                {
                    "role": "user",
                    "content": RECONSTRUCTION_PROMPT_TEMPLATE.format(
                        SEMANTIC_IDS="".join([EOS_SEMANTIC_TOKEN] * self.codebook_size)
                    ),
                },
                {
                    "role": "assistant",
                    "content": RECONSTRUCTION_ASSISTANT_PROMPT.format(
                        product_text=product_text
                    ),
                },
            ],
            "product_name": product_name,
            "product_features": product_features,
        }


class QuantDataCollator:
    def __call__(self, features: list[dict[str, str]]) -> dict[str, list[str]]:
        batch = defaultdict(list)
        for feature in features:
            for key, value in feature.items():
                batch[key].append(value)
        return dict(batch)


def _resolve_categories(config: dict[str, Any]) -> list[str]:
    categories = config.get("categories")
    if not categories:
        categories = [config["category"]]
    return categories


def _finalize_datasets(
    merged_dataset: HFDataset, config: dict[str, Any]
) -> tuple[Dataset, Dataset]:
    """Shared tail for building train/eval datasets from a merged HF dataset.

    Applies optional total-sample truncation, performs the train/test split, then
    wraps each split in an ``AmazonSemanticIdDataset``. Reused by both the 2023 and
    2014 loaders so the sampling/splitting behaviour stays identical across versions.
    """
    max_total_samples = config.get("max_total_samples")
    if max_total_samples:
        merged_dataset = merged_dataset.select(
            range(min(max_total_samples, len(merged_dataset)))
        )

    split_ratio = config.get("eval_split_ratio", 0.02)
    split_dataset = merged_dataset.train_test_split(
        test_size=split_ratio,
        seed=config.get("seed", 42),
    )

    dataset_kwargs = {
        "codebook_size": config["codebook_size"],
        "codebook_range": config["codebook_range"],
        "feature_probability": config.get("feature_probability", 0.5),
        "seed": config.get("seed", 42),
        "max_num_chars": config.get("max_num_chars"),
    }
    train_records = split_dataset["train"]
    eval_records = split_dataset["test"]

    train_limit = config.get("max_train_samples")
    if train_limit:
        train_records = train_records.select(
            range(min(train_limit, len(train_records)))
        )

    eval_limit = config.get("max_eval_samples")
    if eval_limit:
        eval_records = eval_records.select(range(min(eval_limit, len(eval_records))))

    return (
        AmazonSemanticIdDataset(train_records, **dataset_kwargs),
        AmazonSemanticIdDataset(eval_records, **dataset_kwargs),
    )


def _flatten_amazon_2014_categories(categories: Any) -> list[str]:
    """Flatten the 2014 nested ``categories`` (list of category paths) to strings.

    The 2014 format stores categories as a list of paths, each path being a list of
    increasingly specific labels, e.g. ``[["Sports & Outdoors", "Dance"], [...]]``.
    Each path is collapsed into one string so the result is a flat ``list[str]`` with a
    stable schema for Arrow.
    """
    paths: list[str] = []
    entries = categories if isinstance(categories, (list, tuple)) else [categories]
    for entry in entries:
        text = _clean_text(_stringify(entry))
        if text:
            paths.append(text)
    return paths


def normalize_amazon_2014_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map a raw Amazon 2014 product dict onto a fixed, extractor-friendly schema.

    Raw 2014 records have ragged keys (many optional fields) and shapes that differ
    from 2023 (single-string ``description``, nested ``categories``, ``brand`` instead
    of ``store``). Normalizing to a constant set of keys/types gives ``Dataset`` a
    stable schema and lets the shared ``extract_product_name`` /
    ``extract_product_features`` helpers work unchanged (``brand`` is surfaced via the
    ``details`` field they already read).
    """
    brand = _clean_text(_stringify(record.get("brand")))
    return {
        "asin": _stringify(record.get("asin")),
        "title": _clean_text(_stringify(record.get("title"))),
        "description": _clean_text(_stringify(record.get("description"))),
        "categories": _flatten_amazon_2014_categories(record.get("categories")),
        "details": f"Brand: {brand}" if brand else "",
    }


def _iter_amazon_2014_records(
    path: str, allowed_asins: list[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Yield normalized product metadata dicts from a single Amazon 2014 gzip file.

    The 2014 metadata files store one product per line as a Python-dict literal
    (loose JSON with single quotes), so each line is parsed with ``ast.literal_eval``
    rather than ``json.loads``. Malformed lines are skipped, and each parsed record is
    passed through ``normalize_amazon_2014_record`` for a consistent schema.

    When ``allowed_asins`` is given, only products whose ``asin`` is in that set are
    emitted. This aligns the Semantic-ID item universe with the P5 sequential
    pipeline's 5-core ``datamaps`` (see ``build_amazon_2014_dataset``).
    """
    allowed = set(allowed_asins) if allowed_asins is not None else None
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(record, dict):
                continue
            if allowed is not None and record.get("asin") not in allowed:
                continue
            yield normalize_amazon_2014_record(record)


def _load_datamaps_item_asins(sequential_dir: str, category: str) -> list[str]:
    """Return the raw item asins kept by the P5 pipeline for one category.

    Reads ``<sequential_dir>/<category>/datamaps.json`` (written by
    ``preprocess_amazon_2014.py``) and returns the ``item2id`` keys, i.e. the asins of
    the 5-core-filtered items that appear in user sessions.
    """
    path = os.path.join(sequential_dir, category, "datamaps.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"P5 datamaps not found for category '{category}': {path}. Aligned mode "
            f"needs the sequential artifacts; run "
            f"`python preprocess_amazon_2014.py {category} --out '{sequential_dir}'` "
            f"first (or unset 'amazon_2014_sequential_dir' for the full meta dump)."
        )
    with open(path, "r", encoding="utf-8") as handle:
        datamaps = json.load(handle)
    return list(datamaps["item2id"].keys())


def _load_amazon_2014_category(
    data_dir: str,
    category: str,
    filename_template: str,
    allowed_asins: list[str] | None = None,
) -> HFDataset:
    """Load one Amazon 2014 category into a HuggingFace ``Dataset``.

    Materializing into an HF ``Dataset`` keeps the downstream pipeline
    (``concatenate_datasets`` / ``train_test_split`` / ``select``) identical to the
    2023 loader. ``allowed_asins`` optionally narrows the products to the P5-filtered
    item set.
    """
    path = os.path.join(data_dir, filename_template.format(category=category))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Amazon 2014 metadata file not found for category '{category}': {path}. "
            f"Download the per-category 'meta_<Category>.json.gz' file from "
            f"https://jmcauley.ucsd.edu/data/amazon/index_2014.html into "
            f"'{data_dir}' (or set 'amazon_2014_filename_template')."
        )
    return HFDataset.from_generator(
        _iter_amazon_2014_records,
        gen_kwargs={"path": path, "allowed_asins": allowed_asins},
    )


def build_amazon_2014_dataset(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    """Build train/eval datasets from local Amazon 2014 product metadata.

    Unlike the 2023 dataset, the 2014 metadata is not a HuggingFace Hub config; it is
    read from local per-category gzip files named ``meta_<Category>.json.gz`` (override
    via ``amazon_2014_filename_template``) inside ``config['amazon_2014_dir']``.

    When ``amazon_2014_sequential_dir`` is set (and ``amazon_2014_restrict_to_datamaps``
    is not disabled), the products are restricted to the P5-filtered item set from that
    directory's per-category ``datamaps.json`` so the Semantic-ID vocabulary matches the
    items that appear in user sessions. Without it, every metadata item is used
    (original behaviour).
    """
    categories = _resolve_categories(config)
    data_dir = config["amazon_2014_dir"]
    filename_template = config.get(
        "amazon_2014_filename_template", "meta_{category}.json.gz"
    )

    sequential_dir = config.get("amazon_2014_sequential_dir")
    restrict = config.get("amazon_2014_restrict_to_datamaps", bool(sequential_dir))
    if restrict and not sequential_dir:
        raise ValueError(
            "amazon_2014_restrict_to_datamaps is set but amazon_2014_sequential_dir "
            "is not; point it at the preprocess_amazon_2014.py output directory."
        )

    datasets = [
        _load_amazon_2014_category(
            data_dir,
            category,
            filename_template,
            allowed_asins=(
                _load_datamaps_item_asins(sequential_dir, category)
                if restrict
                else None
            ),
        )
        for category in categories
    ]

    merged_dataset = (
        datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    )

    return _finalize_datasets(merged_dataset, config)


class AmazonSequentialDataset(Dataset):
    """Per-user sequential-recommendation records from the P5 pipeline.

    Each item is one user's chronologically-ordered interaction sequence (and, when
    available, its 99 evaluation negatives), as produced by
    ``preprocess_amazon_2014.py``. Item/user ids are the P5 1-indexed string ids, scoped
    to ``category`` (ids are only comparable within a category). Prompt construction is
    intentionally left out — training on sessions is a separate concern; this just
    exposes the sequences in the same ``torch.utils.data.Dataset`` contract as the
    Semantic-ID datasets.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _read_sequential_file(path: str) -> dict[str, list[str]]:
    """Parse a P5 ``"<id> <tok> <tok> ..."``-per-line file into ``id -> [tok, ...]``."""
    mapping: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1:]
    return mapping


def _load_amazon_2014_sequential_records(
    sequential_dir: str, category: str
) -> list[dict[str, Any]]:
    """Load one category's P5 artifacts into per-user sequence records."""
    cat_dir = os.path.join(sequential_dir, category)
    seq_path = os.path.join(cat_dir, "sequential_data.txt")
    if not os.path.isfile(seq_path):
        raise FileNotFoundError(
            f"Sequential data not found for category '{category}': {seq_path}. Run "
            f"`python preprocess_amazon_2014.py {category} --out '{sequential_dir}'` first."
        )

    user_items = _read_sequential_file(seq_path)

    neg_path = os.path.join(cat_dir, "negative_samples.txt")
    user_negatives = (
        _read_sequential_file(neg_path) if os.path.isfile(neg_path) else {}
    )

    records: list[dict[str, Any]] = []
    for user_id, item_ids in user_items.items():
        records.append(
            {
                "category": category,
                "user_id": user_id,
                "item_ids": item_ids,
                "neg_items": user_negatives.get(user_id, []),
            }
        )
    return records


def _finalize_sequential_datasets(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[Dataset, Dataset]:
    """Shuffle/split per-user records, mirroring ``_finalize_datasets`` knobs.

    Honours ``max_total_samples``, ``eval_split_ratio``, ``seed``, ``max_train_samples``
    and ``max_eval_samples`` so the sequential loader behaves like the Semantic-ID one.
    """
    records = list(records)
    rng = random.Random(config.get("seed", 42))
    rng.shuffle(records)

    max_total_samples = config.get("max_total_samples")
    if max_total_samples:
        records = records[:max_total_samples]

    split_ratio = config.get("eval_split_ratio", 0.02)
    n_eval = int(len(records) * split_ratio)
    if records and n_eval == 0:
        n_eval = 1
    eval_records = records[:n_eval]
    train_records = records[n_eval:]

    train_limit = config.get("max_train_samples")
    if train_limit:
        train_records = train_records[:train_limit]
    eval_limit = config.get("max_eval_samples")
    if eval_limit:
        eval_records = eval_records[:eval_limit]

    return (
        AmazonSequentialDataset(train_records),
        AmazonSequentialDataset(eval_records),
    )


def build_amazon_2014_sequential_dataset(
    config: dict[str, Any],
) -> tuple[Dataset, Dataset]:
    """Build train/eval sequential-recommendation datasets from P5 artifacts.

    Reads the per-category artifacts under ``config['amazon_2014_sequential_dir']``
    (created by ``preprocess_amazon_2014.py``) for each configured category and returns
    ``AmazonSequentialDataset`` train/eval splits of per-user interaction sequences.
    """
    sequential_dir = config.get("amazon_2014_sequential_dir")
    if not sequential_dir:
        raise ValueError(
            "dataset_mode 'sequential' requires 'amazon_2014_sequential_dir' to point "
            "at the preprocess_amazon_2014.py output directory."
        )

    categories = _resolve_categories(config)
    records: list[dict[str, Any]] = []
    for category in categories:
        records.extend(
            _load_amazon_2014_sequential_records(sequential_dir, category)
        )

    return _finalize_sequential_datasets(records, config)


def build_amazon_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    dataset_mode = str(config.get("dataset_mode", "metadata"))
    if dataset_mode == "sequential":
        return build_amazon_2014_sequential_dataset(config)

    dataset_version = str(config.get("dataset_version", 2023))
    if dataset_version == "2014":
        return build_amazon_2014_dataset(config)

    categories = _resolve_categories(config)

    datasets = []
    for category in categories:
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        )
        datasets.append(dataset)

    merged_dataset = (
        datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    )

    return _finalize_datasets(merged_dataset, config)
