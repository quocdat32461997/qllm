"""Next-item examples, collision-safe item IDs, constrained retrieval, and metrics."""
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from amazon2014 import read_records, sha256_file, validate_prepared
from constants import BOS_SEMANTIC_TOKEN, EOS_SEMANTIC_TOKEN
from model_utils import chat_ids, position_ids, tokenize_supervised


def load_index(path, data_dir=None):
    index = json.loads(Path(path).read_text())
    size, width = index["codebook_size"], index["codebook_range"]
    if index.get("schema_version") != 1 or not index["items"] or index["suffix_count"] < 1:
        raise ValueError("Invalid semantic index")
    seen = set()
    for item in index["items"].values():
        codes, suffix = item["codes"], item["suffix"]
        if len(codes) != size or any(type(c) is not int or not 0 <= c < width for c in codes):
            raise ValueError("Invalid code in index")
        if type(suffix) is not int or not 0 <= suffix < index["suffix_count"]:
            raise ValueError("Invalid collision suffix")
        key = tuple(codes) + (suffix,)
        if key in seen:
            raise ValueError("Index contains duplicate complete item identifiers")
        seen.add(key)
    if data_dir is not None:
        validate_prepared(data_dir)
        if index["provenance"]["catalog_sha256"] != sha256_file(Path(data_dir) / "products.jsonl"):
            raise ValueError("Semantic index was exported for a different catalog")
        catalog = {r["asin"] for r in read_records(Path(data_dir) / "products.jsonl")}
        if catalog != set(index["items"]):
            raise ValueError("Index must cover every catalog item exactly once")
    return index


def item_tokens(item):
    return [BOS_SEMANTIC_TOKEN, *[f"<|CODE_{c}|>" for c in item["codes"]],
            f"<|ITEM_SUFFIX_{item['suffix']}|>", EOS_SEMANTIC_TOKEN]


def add_item_tokens(model, tokenizer, index):
    tokenizer.add_special_tokens({"extra_special_tokens":
        [f"<|ITEM_SUFFIX_{i}|>" for i in range(index["suffix_count"])]},
        replace_extra_special_tokens=False)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def recommendation_prompt(history, index):
    encoded = "\n".join("".join(item_tokens(index["items"][asin])) for asin in history)
    return [{"role": "system", "content": "Predict the next item from the chronological interaction history. "
            "Reply with exactly one item identifier."},
            {"role": "user", "content": "History (oldest first):\n" + encoded}]


class RecommendationDataset(Dataset):
    def __init__(self, path, index, max_history=20, max_samples=None):
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.rows = list(read_records(path))
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"Empty recommendation split: {path}")
        self.index, self.max_history = index, max_history
        for row in self.rows:
            if not row["history"] or any(a not in index["items"] for a in [*row["history"], row["target"]]):
                raise ValueError("Recommendation rows require nonempty histories and indexed items")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        messages = recommendation_prompt(row["history"][-self.max_history:], self.index)
        messages.append({"role": "assistant", "content": "".join(item_tokens(self.index["items"][row["target"]]))})
        return {"messages": messages}


class RecommendationCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, rows):
        batch = tokenize_supervised(self.tokenizer, [r["messages"] for r in rows], self.max_length)
        batch["position_ids"] = position_ids(batch["attention_mask"])
        return batch


class CatalogTrie:
    def __init__(self, tokenizer, index):
        self.root, self.reverse = {}, {}
        self.eos = tokenizer.convert_tokens_to_ids(EOS_SEMANTIC_TOKEN)
        for asin, item in sorted(index["items"].items()):
            tokens = item_tokens(item)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            if any(tokenizer.convert_ids_to_tokens(i) != t for i, t in zip(ids, tokens)):
                raise ValueError("Checkpoint tokenizer does not contain the exact index vocabulary")
            self.reverse[tuple(ids)] = asin
            node = self.root
            for value in ids:
                node = node.setdefault(value, {})
        self.length = index["codebook_size"] + 3

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            if token not in node:
                return [self.eos]  # Finished/padded beam only.
            node = node[token]
        return sorted(node) if node else [self.eos]

    def decode(self, sequence):
        return self.reverse.get(tuple(sequence[:self.length]))


def ranking_metrics(predictions, targets, ks=(5, 10)):
    if len(predictions) != len(targets) or not targets or any(k < 1 for k in ks):
        raise ValueError("Need aligned nonempty predictions/targets and positive cutoffs")
    totals = {f"{metric}@{k}": 0.0 for k in ks for metric in ("recall", "ndcg")}
    for candidates, target in zip(predictions, targets):
        candidates = list(dict.fromkeys(candidates))
        rank = candidates.index(target) + 1 if target in candidates else math.inf
        for k in ks:
            if rank <= k:
                totals[f"recall@{k}"] += 1
                totals[f"ndcg@{k}"] += 1 / math.log2(rank + 1)
    return {key: value / len(targets) for key, value in totals.items()}


@torch.no_grad()
def evaluate_recommendations(model, tokenizer, dataset, index, beam_size=20, ks=(5, 10),
                             batch_size=8, max_length=1024):
    if beam_size < max(ks) or batch_size < 1:
        raise ValueError("beam_size must cover the largest metric cutoff; batch_size must be positive")
    trie = CatalogTrie(tokenizer, index)
    beam_size = min(beam_size, len(index["items"]))
    model.eval()
    predictions, targets = [], []
    invalid = 0
    for start in range(0, len(dataset), batch_size):
        rows = dataset.rows[start:start + batch_size]
        prompts = [chat_ids(tokenizer, recommendation_prompt(r["history"][-dataset.max_history:], index), True)
                   for r in rows]
        if max(map(len, prompts)) + trie.length > max_length:
            raise ValueError("Recommendation prompt + item identifier exceeds max_length")
        batch = tokenizer.pad([{"input_ids": p, "attention_mask": [1] * len(p)} for p in prompts],
                              padding=True, return_tensors="pt").to(model.device)
        prefix_length = batch["input_ids"].shape[1]
        def allowed(batch_id, sequence):
            return trie.allowed(sequence[prefix_length:].tolist())
        outputs = model.generate(**batch, do_sample=False, num_beams=beam_size,
            num_return_sequences=beam_size, max_new_tokens=trie.length,
            eos_token_id=trie.eos, pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=allowed, length_penalty=0.0, use_cache=True)
        for i, row in enumerate(rows):
            generated = outputs[i * beam_size:(i + 1) * beam_size, prefix_length:].tolist()
            items = [trie.decode(sequence) for sequence in generated]
            invalid += sum(item is None for item in items)
            predictions.append(list(dict.fromkeys(item for item in items if item is not None)))
            targets.append(row["target"])
    metrics = ranking_metrics(predictions, targets, ks)
    metrics.update(examples=len(targets), invalid_sequences=invalid, beam_size=beam_size,
                   candidate_count=len(index["items"]), candidate_policy="full catalog; no seen-item exclusion",
                   decoding="catalog-constrained beam search (approximate top-k)")
    return metrics
