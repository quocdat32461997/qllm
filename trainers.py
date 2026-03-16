from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments

from constants import BOS_SEMANTIC_TOKEN, EOS_SEMANTIC_TOKEN
from data_module import format_semantic_ids, parse_semantic_ids

IGNORE_INDEX = -100


@dataclass
class QuantConfig(TrainingArguments):
    codebook_size: int = field(default=6, metadata={"help": "Number of semantic IDs."})
    codebook_range: int = field(
        default=128, metadata={"help": "Maximum value for each semantic ID."}
    )
    max_source_length: int = field(
        default=256, metadata={"help": "Max length for prompts."}
    )
    max_target_length: int = field(
        default=64, metadata={"help": "Max length for semantic ID and product targets."}
    )
    generation_max_new_tokens: int = field(
        default=24,
        metadata={"help": "Budget used while sampling semantic-ID candidates."},
    )
    generation_temperature: float = field(default=1.0)
    generation_top_p: float = field(default=0.9)
    generation_do_sample: bool = field(default=True)
    guessing_weight: float = field(default=1.0)
    reconstruction_weight: float = field(default=1.0)
    remove_unused_columns: bool = field(default=False)


class QuanSFTTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.processing_class = tokenizer

        # Add special tokens for semantic IDs
        special_tokens = {
            "bos_semantic_token": BOS_SEMANTIC_TOKEN,
            "eos_semantic_token": EOS_SEMANTIC_TOKEN,
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.bos_semantic_token_id = self.tokenizer.vocab[BOS_SEMANTIC_TOKEN]
        self.eos_semantic_token_id = self.tokenizer.vocab[EOS_SEMANTIC_TOKEN]

    def compute_loss(
        self,
        model,
        inputs: dict[str, list[str]],
        return_outputs: bool = False,
        **_: Any,
    ):

        # Encoding: generate semantic IDs
        (
            encoder_generated_texts,
            encoder_attention_mask_size,
            semantic_ids,
            semantic_id_texts,
        ) = self._generate_semantic_ids(
            model=model,
            prompts=inputs["guessing_prompt"],
        )

        guessing_loss = self._compute_autoregressive_loss(
            model=model,
            prompts=inputs["guessing_prompt"],
            targets=semantic_id_texts,
            max_prompt_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
        )

        # Decoding: reconstruct original text from semantic IDs
        reconstruction_prompts = [
            template.format(input=generated_text)
            for template, generated_text in zip(
                inputs["reconstruction_prompt_template"],
                encoder_generated_texts,
            )
        ]
        reconstructed_texts, reconstructed_ids = self._reconstruct_input(
            model=model,
            prompts=reconstruction_prompts,
            encoder_attention_mask_size=encoder_attention_mask_size,
        )

        reconstruction_loss = self._compute_autoregressive_loss(
            model=model,
            prompts=reconstructed_texts,
            targets=inputs["reconstruction_target"],
            max_prompt_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
        )

        diversity_score = self._compute_diversity_score(semantic_ids)
        loss = (
            self.args.guessing_weight * guessing_loss
            + self.args.reconstruction_weight * reconstruction_loss
        )

        self.log(
            {
                "guessing_loss": guessing_loss.detach(),
                "reconstruction_loss": reconstruction_loss.detach(),
                "semantic_id_diversity": diversity_score,
            }
        )

        if return_outputs:
            return loss, {
                "semantic_ids": semantic_ids,
                "semantic_id_texts": semantic_id_texts,
                "guessing_loss": guessing_loss.detach(),
                "reconstruction_loss": reconstruction_loss.detach(),
                "semantic_id_diversity": diversity_score,
            }
        return loss

    def _compute_autoregressive_loss(
        self,
        model,
        prompts: list[str],
        targets: list[str],
        max_prompt_length: int,
        max_target_length: int,
    ) -> torch.Tensor:
        batch = self._build_lm_batch(
            prompts=prompts,
            targets=targets,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            device=next(model.parameters()).device,
        )
        outputs = model(**batch)
        return outputs.loss

    def _build_lm_batch(
        self,
        prompts: list[str],
        targets: list[str],
        max_prompt_length: int,
        max_target_length: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        prompt_encoding = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )
        target_encoding = self.tokenizer(
            targets,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length,
        )

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        input_id_rows = []
        attention_rows = []
        label_rows = []

        for prompt_ids, target_ids in zip(
            prompt_encoding["input_ids"], target_encoding["input_ids"]
        ):
            target_sequence = list(target_ids)
            if eos_token_id is not None:
                target_sequence = target_sequence + [eos_token_id]

            combined_ids = list(prompt_ids) + target_sequence
            labels = [IGNORE_INDEX] * len(prompt_ids) + target_sequence
            attention_mask = [1] * len(combined_ids)

            input_id_rows.append(combined_ids)
            label_rows.append(labels)
            attention_rows.append(attention_mask)

        max_length = max(len(row) for row in input_id_rows)

        padded_inputs = []
        padded_labels = []
        padded_attention = []
        for input_ids, labels, attention_mask in zip(
            input_id_rows, label_rows, attention_rows
        ):
            padding = max_length - len(input_ids)
            padded_inputs.append(input_ids + [pad_token_id] * padding)
            padded_labels.append(labels + [IGNORE_INDEX] * padding)
            padded_attention.append(attention_mask + [0] * padding)

        return {
            "input_ids": torch.tensor(padded_inputs, device=device),
            "attention_mask": torch.tensor(padded_attention, device=device),
            "labels": torch.tensor(padded_labels, device=device),
        }

    def _reconstruct_input(
        self,
        model,
        prompts: list[str],
        encoder_attention_mask_size: torch.Tensor,
    ) -> tuple[list[list[int]], list[str]]:
        device = next(model.parameters()).device
        # This should:
        # 1. Take generated semantic IDs and encoded input
        # 2. Reconstruct the original text
        # 3. Return reconstructed text and decoded IDs

        # 1. Take generated semantic IDs and encoded input
        encoded_prompts = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.args.max_source_length,
            add_special_tokens=False,
            device=device,
        )
        # Masking encoder_prompt
        encoded_prompts["attention_mask"][:, :encoder_attention_mask_size] = 0

        # 2. Reconstruct the original text
        generated = model.generate(
            **encoded_prompts,
            max_new_tokens=self.args.generation_max_new_tokens,
            do_sample=self.args.generation_do_sample,
            temperature=self.args.generation_temperature,
            top_p=self.args.generation_top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 3. Return reconstructed text and reconstructed IDs
        reconstructed_ids = generated[:, encoded_prompts["input_ids"].shape[1] :]
        reconstructed_texts = self.tokenizer.batch_decode(
            reconstructed_ids, skip_special_tokens=True
        )

        return reconstructed_texts, reconstructed_ids

    def _generate_semantic_ids(
        self,
        model,
        prompts: list[str],
    ) -> tuple[list[list[int]], list[str]]:
        device = next(model.parameters()).device
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        encoded_prompts = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.args.max_source_length,
            device=device,
        )
        # encoded = {key: value.to(device) for key, value in encoded.items()}

        # was_training = model.training
        # model.eval()
        # with torch.no_grad():
        generated = model.generate(
            **encoded_prompts,
            max_new_tokens=self.args.codebook_size
            + 1,  # codebook_size semantic IDs + 1 for eos_semantic_token
            do_sample=self.args.generation_do_sample,
            temperature=self.args.generation_temperature,
            top_p=self.args.generation_top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_semantic_token_id,
        )

        # if was_training:
        #     model.train()
        self.tokenizer.padding_side = original_padding_side

        semantic_ids = generated[:, encoded_prompts["input_ids"].shape[1] :]
        semantic_id_texts = self.tokenizer.batch_decode(
            semantic_ids, skip_special_tokens=True
        )
        generated_text = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=False,
        )

        # semantic_ids = []
        # semantic_id_texts = []
        # for generated_text, generated_token_ids in zip(
        #     semantic_id_texts,
        #     semantic_ids.tolist(),
        # ):
        #     fallback_ids = [
        #         (token_id % self.args.codebook_range) + 1
        #         for token_id in generated_token_ids[: self.args.codebook_size]
        #     ]
        #     ids = parse_semantic_ids(
        #         text=generated_text,
        #         codebook_size=self.args.codebook_size,
        #         codebook_range=self.args.codebook_range,
        #         fallback_ids=fallback_ids,
        #     )
        #     semantic_ids.append(ids)
        #     semantic_id_texts.append(format_semantic_ids(ids))

        return (
            generated_text,
            encoded_prompts["attention_mask"].shape[1],  # encoder_attention_mask_size
            semantic_ids,
            semantic_id_texts,
        )

    def _compute_diversity_score(self, semantic_ids: list[list[int]]) -> float:
        if not semantic_ids:
            return 0.0

        per_example_diversity = [
            len(set(example_ids)) / max(1, len(example_ids))
            for example_ids in semantic_ids
        ]
        batch_diversity = len({tuple(example_ids) for example_ids in semantic_ids}) / (
            len(semantic_ids)
        )
        return float(
            (sum(per_example_diversity) / len(per_example_diversity) + batch_diversity)
            / 2.0
        )
