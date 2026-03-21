import argparse
from collections.abc import Mapping

import mlflow
import wandb
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    trainer_args = QuantConfig(
        output_dir=config["trainer"]["output_dir"],
        per_device_train_batch_size=config["trainer"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["trainer"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["trainer"]["gradient_accumulation_steps"],
        learning_rate=config["trainer"]["learning_rate"],
        num_train_epochs=config["trainer"]["num_train_epochs"],
        logging_steps=config["trainer"]["logging_steps"],
        save_steps=config["trainer"]["save_steps"],
        eval_steps=config["trainer"]["eval_steps"],
        warmup_ratio=config["trainer"]["warmup_ratio"],
        weight_decay=config["trainer"]["weight_decay"],
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=config["trainer"].get("report_to", []),
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
    )

    trainer = QuanSFTTrainer(
        model=model,
        args=trainer_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QuantDataCollator(),
    )

    # Initialize wandb
    wandb.init(entity="quocdat32461997", project=config["experiment_name"])
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
