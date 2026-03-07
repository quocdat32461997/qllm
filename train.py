import argparse

import mlflow
import yaml
from datasets import load_dataset
from trainer import QuanSFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str)


if __name__ == "__main__":
    # Get command line arguments
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    meta_data = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{config['category']}",
        split="full",
        trust_remote_code=True,
    )

    # Train
    mlflow.set_experiment(config["experiment_name"])
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    with mlflow.start_run() as run:
        trainer = QuanSFTTrainer()
        trainer.train(meta_data)
