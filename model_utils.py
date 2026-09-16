"""Shared checkpoint loading and chat loss masking for both training stages."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import BOS_SEMANTIC_TOKEN, EOS_SEMANTIC_TOKEN, BOS_SEMANTIC_SESSION


def add_semantic_tokens(tokenizer, model, codebook_range):
    tokenizer.add_special_tokens({"extra_special_tokens": [
        BOS_SEMANTIC_TOKEN, EOS_SEMANTIC_TOKEN, BOS_SEMANTIC_SESSION,
        *[f"<|CODE_{i}|>" for i in range(codebook_range)],
    ]}, replace_extra_special_tokens=False)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        # Mean-based BF16 vocabulary expansion can give all new code tokens
        # identical embeddings, making the selector's reconstruction gradient zero.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_checkpoint(config, checkpoint=None, trainable=False):
    source = str(checkpoint or config["model_name"])
    experiment = Path(source, "experiment.yaml")
    if checkpoint and experiment.exists():
        import yaml
        saved = yaml.safe_load(experiment.read_text())
        for key in ("codebook_size", "codebook_range"):
            if saved[key] != config[key]:
                raise ValueError(f"Checkpoint and config disagree on {key}")
    adapter = Path(source, "adapter_config.json").exists()
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must have a padding or EOS token")
    tokenizer.padding_side = "left"
    base_source = source
    if adapter:
        from peft import PeftConfig
        base_source = PeftConfig.from_pretrained(source).base_model_name_or_path
    dtype = torch.bfloat16 if config.get("trainer", {}).get("bf16", False) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_source, dtype=dtype,
        attn_implementation=config.get("trainer", {}).get("attn_implementation", "sdpa"),
    )
    add_semantic_tokens(tokenizer, model, config["codebook_range"])
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, source, is_trainable=trainable)
    model.config.use_cache = False
    return model, tokenizer


def chat_ids(tokenizer, messages, generation=False):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False,
        add_generation_prompt=generation, enable_thinking=False,
    )


def tokenize_supervised(tokenizer, prompts, max_length):
    """Mask the exact assistant prefix per row; fail instead of truncating labels."""
    rows, starts = [], []
    for messages in prompts:
        ids = chat_ids(tokenizer, messages)
        prefix = chat_ids(tokenizer, messages[:-1], generation=True)
        if ids[:len(prefix)] != prefix:
            raise ValueError("Chat template generation prefix does not match the assistant turn")
        if len(ids) > max_length:
            raise ValueError(f"Chat has {len(ids)} tokens, exceeding {max_length}; "
                             "increase the sequence limit or reduce product/history length")
        if len(ids) <= len(prefix):
            raise ValueError("No assistant target tokens")
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        starts.append(len(prefix))
    batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    for i, (row, start) in enumerate(zip(rows, starts)):
        offset = batch["input_ids"].shape[1] - len(row["input_ids"]) if tokenizer.padding_side == "left" else 0
        labels[i, :offset + start] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


def position_ids(attention_mask):
    return (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
