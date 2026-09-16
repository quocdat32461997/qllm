"""Stage 2: fine-tune the stage-1 checkpoint for next-item recommendation."""
import argparse
from pathlib import Path
import shutil

from peft import PeftModel
from transformers import Trainer, TrainingArguments, set_seed
import yaml

from amazon2014 import sha256_file
from model_utils import load_checkpoint
from recommendation import add_item_tokens, load_index, RecommendationDataset, RecommendationCollator


def run_training(config, checkpoint, index_path, resume_from_checkpoint=None):
    set_seed(config.get("seed", 42))
    rec = config["recommendation"]
    trainer_cfg = dict(rec["trainer"])
    trainer_cfg.setdefault("report_to", [])
    trainer_cfg.setdefault("remove_unused_columns", False)
    trainer_cfg.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    trainer_cfg.setdefault("ddp_find_unused_parameters", False)
    trainer_cfg["seed"] = config.get("seed", 42)
    training_args = TrainingArguments(**trainer_cfg)
    index = load_index(index_path, config["data_dir"])
    if (index["codebook_size"], index["codebook_range"]) != (config["codebook_size"], config["codebook_range"]):
        raise ValueError("Config and exported codebook dimensions differ")
    model, tokenizer = load_checkpoint(config, checkpoint, trainable=True)
    # Merge stage-1 adapters before adding the collision suffix vocabulary.
    was_adapter = isinstance(model, PeftModel)
    if was_adapter:
        model = model.merge_and_unload()
    add_item_tokens(model, tokenizer, index)
    if rec.get("use_lora", config.get("use_lora", False)):
        from peft import LoraConfig, get_peft_model
        if was_adapter:
            # A new adapter must reload the merged stage-1 weights, never the
            # original pretrained backbone (which would discard stage 1).
            merged_base = Path(training_args.output_dir).resolve() / "stage1_base"
            if training_args.process_index == 0:
                model.save_pretrained(merged_base)
                tokenizer.save_pretrained(merged_base)
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
            model.name_or_path = str(merged_base)
        model = get_peft_model(model, LoraConfig(**config["lora"]))
    else:
        # Merging a stage-1 adapter preserves PEFT's frozen base parameters.
        # Explicit full fine-tuning must unfreeze them again.
        model.requires_grad_(True)
    datasets = {split: RecommendationDataset(Path(config["data_dir"]) / f"{split}.jsonl", index,
        rec.get("max_history", 20), rec.get(f"max_{split}_samples")) for split in ("train", "validation")}
    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=RecommendationCollator(tokenizer, rec.get("max_length", 1024)))
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    final_dir = Path(trainer.args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        shutil.copy2(index_path, final_dir / "semantic_index.json")
        saved_config = dict(config)
        saved_config["stage1_checkpoint"] = str(checkpoint)
        saved_config["semantic_index_sha256"] = sha256_file(index_path)
        (final_dir / "experiment.yaml").write_text(yaml.safe_dump(saved_config, sort_keys=False))
    return str(final_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="configs.yaml")
    parser.add_argument("--checkpoint", required=True, help="Stage-1 final checkpoint")
    parser.add_argument("--index", required=True, help="Frozen catalog index from stage 1")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    run_training(yaml.safe_load(Path(args.config_path).read_text()), args.checkpoint, args.index, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
