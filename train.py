import argparse

import yaml
from datasets import load_dataset

from trainer import QuanSFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str)
args = parser.parse_args()


def main():
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    meta_data = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{config['category']}",
        split="full",
        trust_remote_code=True,
    )

    trainer = QuanSFTTrainer()
    trainer.train(meta_data)


if __name__ == "__main__":
    main()
