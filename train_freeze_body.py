import argparse
import os
from collections.abc import Mapping

import mlflow
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
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

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    # Freeze all model parameters except embeddings
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the embedding layer
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True

    # Also unfreeze output embeddings if they exist
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            for param in output_embeddings.parameters():
                param.requires_grad = True

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

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

    # Initialize wandb
    wandb.init(
        entity="quocdat32461997",
        project=config["experiment_name"],
        dir=config["trainer"]["output_dir"],
    )
    try:
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
        mlflow.set_experiment(config["experiment_name"])

        with mlflow.start_run():
            mlflow.log_params(_flatten_for_logging(config))
            trainer.train()
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        print("Continuing with wandb logging only...")
        trainer.train()
        wandb.finish()
