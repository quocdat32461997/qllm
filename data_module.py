import random
import re
from collections import defaultdict
from typing import Any

from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset


PROMPT_TEMPLATES = (
    "Please analyze the following product and its features: {product_text}. Then, generate semantic-IDs that meaningfully represent the product. The semantic-IDs are:",
    "Analyze this catalog item and convert it into semantic-IDs: {product_text}. The semantic-IDs are:",
    "Read the product information and map it to semantic-IDs for recommendation cold-start: {product_text}. The semantic-IDs are:",
    "Remember the following product and summarize it as semantic-IDs: {product_text}. The semantic-IDs are:",
    "Given the following product, produce semantic-IDs that capture its meaning: {product_text}. The semantic-IDs are:",
)

RECONSTRUCTION_PROMPT_TEMPLATE = (
    "########## The semantic-IDs are: {semantic_ids}. Recover the product name."
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


def extract_product_features(example: dict[str, Any]) -> str:
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
    return " ".join(deduped)


def build_product_text(
    product_name: str,
    product_features: str,
    include_features: bool,
) -> str:
    if include_features and product_features:
        return f"{product_name}. Features: {product_features}"
    return product_name


def format_semantic_ids(ids: list[int]) -> str:
    encoded = ",".join(str(value) for value in ids)
    return f"<semantic-id>{encoded}</semantic-id>"


def parse_semantic_ids(
    text: str,
    codebook_size: int,
    codebook_range: int,
    fallback_ids: list[int] | None = None,
) -> list[int]:
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
        feature_probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.codebook_size = codebook_size
        self.codebook_range = codebook_range
        self.feature_probability = feature_probability
        self.seed = seed

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
        )

        return {
            "guessing_prompt": prompt_template.format(product_text=product_text),
            "reconstruction_prompt_template": RECONSTRUCTION_PROMPT_TEMPLATE,
            "reconstruction_target": product_name,
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


def build_amazon_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    categories = config.get("categories")
    if not categories:
        categories = [config["category"]]

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
    }
    train_records = [dict(row) for row in split_dataset["train"]]
    eval_records = [dict(row) for row in split_dataset["test"]]

    train_limit = config.get("max_train_samples")
    if train_limit:
        train_records = train_records[:train_limit]

    eval_limit = config.get("max_eval_samples")
    if eval_limit:
        eval_records = eval_records[:eval_limit]

    return (
        AmazonSemanticIdDataset(train_records, **dataset_kwargs),
        AmazonSemanticIdDataset(eval_records, **dataset_kwargs),
    )
