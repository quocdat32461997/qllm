"""Stage 1: retrain the existing LLM semantic-ID autoencoder."""
import argparse
from pathlib import Path
import shutil

import yaml
from transformers import set_seed
from peft import LoraConfig, get_peft_model

from data_module import QuantDataCollator, build_amazon_datasets
from model_utils import load_checkpoint
from trainers import QuanSFTTrainer, QuantConfig


def run_training(config, resume_from_checkpoint=None):
    set_seed(config.get("seed", 42))
    train_dataset, eval_dataset = build_amazon_datasets(config)
    model, tokenizer = load_checkpoint(config)
    if config.get("use_lora"):
        model = get_peft_model(model, LoraConfig(**config["lora"]))
    trainer_cfg = dict(config["trainer"])
    trainer_cfg.pop("attn_implementation", None)
    trainer_cfg.setdefault("report_to", [])
    trainer_cfg.setdefault("eval_strategy", "no")
    trainer_cfg.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    trainer_cfg.setdefault("ddp_find_unused_parameters", False)
    trainer_cfg["seed"] = config.get("seed", 42)
    trainer_cfg["dataloader_num_workers"] = config.get("data_num_workers", 0)
    for name in ("codebook_size", "codebook_range", "max_source_length", "max_target_length",
                 "generation_max_new_tokens", "generation_temperature", "generation_top_p",
                 "generation_do_sample", "guessing_weight", "reconstruction_weight"):
        if name in config:
            trainer_cfg[name] = config[name]
    trainer = QuanSFTTrainer(model=model, tokenizer=tokenizer, args=QuantConfig(**trainer_cfg),
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=QuantDataCollator())
    tracker = None
    if "wandb" in trainer.args.report_to and trainer.is_world_process_zero():
        import wandb
        tracker = wandb.init(project=config.get("experiment_name", "amazon2014"),
                             entity=config.get("wandb_entity"), config=config,
                             dir=trainer.args.output_dir)
    try:
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        final_dir = str(Path(trainer.args.output_dir) / "final")
        trainer.save_model(final_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final_dir)
            Path(final_dir, "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            if str(config.get("dataset_version")) == "2014":
                shutil.copy2(Path(config["data_dir"]) / "manifest.json", Path(final_dir) / "data_manifest.json")
        return final_dir
    finally:
        if tracker is not None:
            tracker.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="configs.yaml")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config_path).read_text())
    run_training(config, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
