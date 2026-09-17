import argparse
import os
from collections.abc import Mapping

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from constants import (
    BOS_SEMANTIC_SESSION,
    BOS_SEMANTIC_TOKEN,
    EOS_SEMANTIC_TOKEN,
)
from data_module import QuantDataCollator, build_amazon_datasets
from trainers import QuanSFTTrainer, QuantConfig


def _flatten_for_logging(
    value: Mapping[str, object],
    prefix: str = "",
) -> dict[str, str | int | float | bool]:
    flattened = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(_flatten_for_logging(item, prefix=full_key))
        elif isinstance(item, list):
            flattened[full_key] = ",".join(str(entry) for entry in item)
        else:
            flattened[full_key] = item
    return flattened


parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, default="configs.yaml")


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    train_dataset, eval_dataset = build_amazon_datasets(config)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Under DeepSpeed/FSDP the Trainer manages device placement, so do NOT pass
    # device_map here. bf16 halves the resident weight footprint vs fp32.
    trainer_cfg = config["trainer"]
    torch_dtype = torch.bfloat16 if trainer_cfg.get("bf16", False) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch_dtype,
        attn_implementation=trainer_cfg.get("attn_implementation", "sdpa"),
    )

    # Add the semantic-ID special tokens and resize the embedding table BEFORE
    # wrapping with LoRA, so the newly added codebook rows are the ones captured
    # by `modules_to_save` and actually get trained. (QuanSFTTrainer re-adds them
    # idempotently, so this stays a no-op there.)
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

    # Apply LoRA if configured. `modules_to_save` keeps the (resized) input
    # embeddings and lm_head fully trainable so the new codebook tokens can be
    # learned even though the rest of the backbone is frozen behind adapters.
    if config.get("use_lora") and "lora" in config:
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["lora_dropout"],
            bias=config["lora"]["bias"],
            task_type=config["lora"]["task_type"],
            modules_to_save=config["lora"].get(
                "modules_to_save", ["embed_tokens", "lm_head"]
            ),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainer_args = QuantConfig(
        output_dir=config["trainer"]["output_dir"],
        per_device_train_batch_size=config["trainer"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["trainer"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["trainer"]["gradient_accumulation_steps"],
        learning_rate=config["trainer"]["learning_rate"],
        lr_scheduler_type=config["trainer"]["lr_scheduler_type"],
        num_train_epochs=config["trainer"]["num_train_epochs"],
        logging_steps=config["trainer"]["logging_steps"],
        save_steps=config["trainer"]["save_steps"],
        eval_steps=config["trainer"]["eval_steps"],
        # warmup_ratio=config["trainer"]["warmup_ratio"],
        warmup_steps=config["trainer"].get("warmup_steps", 0),
        weight_decay=config["trainer"]["weight_decay"],
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=config["trainer"].get("save_total_limit", 1),
        logging_strategy="steps",
        report_to=config["trainer"].get("report_to", []) + ["wandb"],
        seed=config.get("seed", 42),
        codebook_size=config["codebook_size"],
        codebook_range=config["codebook_range"],
        max_source_length=config["max_source_length"],
        max_target_length=config["max_target_length"],
        generation_max_new_tokens=config.get("generation_max_new_tokens", 24),
        generation_temperature=config.get("generation_temperature", 1.0),
        generation_top_p=config.get("generation_top_p", 0.9),
        guessing_weight=config.get("guessing_weight", 1.0),
        reconstruction_weight=config.get("reconstruction_weight", 1.0),
        max_grad_norm=config["trainer"].get("max_grad_norm", 1.0),
        temperature_initial=config["trainer"].get("temperature_initial"),
        temperature_final=config["trainer"].get("temperature_final"),
        gradient_checkpointing=config["trainer"].get("gradient_checkpointing", False),
        # use_reentrant=False is required for our multi-forward compute_loss:
        # the codebook loop builds a graph across several sub-forwards.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=config["trainer"].get("bf16", False),
        # Path to a DeepSpeed ZeRO json (e.g. ds_configs/zero2.json) enables
        # memory sharding across GPUs; None falls back to plain single/DDP.
        deepspeed=config["trainer"].get("deepspeed"),
        ddp_find_unused_parameters=config["trainer"].get(
            "ddp_find_unused_parameters", False
        ),
        dataloader_num_workers=config.get("data_num_workers", 4),
    )

    trainer = QuanSFTTrainer(
        model=model,
        args=trainer_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QuantDataCollator(),
    )

    # Unfreeze codebook token embeddings when use with LoRA
    # codebook_tokens = [f"<|CODE_{id}|>" for id in range(config["codebook_range"])]
    # codebook_token_ids = [tokenizer.vocab[token] for token in codebook_tokens]
    # # Unfreeze only codebook token embeddings
    # trainer.model.get_input_embeddings().weight[
    #     codebook_token_ids
    # ].requires_grad = True  # noqa

    # Under distributed launch every rank runs this script; only the main
    # process should touch the tracker. Training itself runs on all ranks.
    is_main_process = trainer.is_world_process_zero()

    if is_main_process:
        wandb.init(
            entity="quocdat32461997",
            project=config["experiment_name"],
            dir=config["trainer"]["output_dir"],
            config=_flatten_for_logging(config),
        )
    try:
        trainer.train()
    finally:
        if is_main_process:
            wandb.finish()
