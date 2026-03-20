import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.fx.node import Target
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments
from trl import SFTConfig

from constants import BOS_SEMANTIC_SESSION, BOS_SEMANTIC_TOKEN, EOS_SEMANTIC_TOKEN
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
            "extra_special_tokens": [
                BOS_SEMANTIC_TOKEN,
                EOS_SEMANTIC_TOKEN,
                BOS_SEMANTIC_SESSION,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.bos_semantic_token_id = self.tokenizer.vocab[BOS_SEMANTIC_TOKEN]
        self.eos_semantic_token_id = self.tokenizer.vocab[EOS_SEMANTIC_TOKEN]
        self.bos_semantic_session_id = self.tokenizer.vocab[BOS_SEMANTIC_SESSION]

        # Add special tokens to model
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_digit_num = len(str(self.args.codebook_range))

    def compute_loss(
        self,
        model,
        inputs: dict[str, list[str]],
        return_outputs: bool = False,
        **_: Any,
    ):

        # Encoding: generate semantic IDs
        (
            semantic_id_texts,
            semantic_ids,
            format_loss,
        ) = self._generate_semantic_ids(
            model=model,
            prompts=inputs["guessing_prompt"],
        )

        # Decoding: reconstruct original text from semantic IDs
        reconstruction_prompts = [
            template.format(semantic_ids=semantic_ids)
            for template, semantic_ids in zip(
                inputs["reconstruction_prompt_template"],
                semantic_id_texts,
            )
        ]
        reconstruction_loss = self._reconstruct_input(
            model=model,
            prompts=reconstruction_prompts,
            labels=inputs["reconstruction_target"],
        )

        diversity_score = self._compute_diversity_score(semantic_ids)
        loss = (
            self.args.guessing_weight * format_loss
            + self.args.reconstruction_weight * reconstruction_loss
        )

        self.log(
            {
                "format_loss": format_loss.detach(),
                "reconstruction_loss": reconstruction_loss.detach(),
                "semantic_id_diversity": diversity_score,
            }
        )

        if return_outputs:
            return loss, {
                "semantic_ids": semantic_ids,
                "semantic_id_texts": semantic_id_texts,
                "format_loss": format_loss.detach(),
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
        labels: list[str],
    ) -> tuple[list[list[int]], list[str]]:
        device = next(model.parameters()).device
        # This should:
        # 1. Take generated semantic IDs and encoded input
        # 2. Reconstruct the original text
        # 3. Return reconstructed text and decoded IDs

        # 1. Take generated semantic IDs and encoded input
        tokenized_prompts = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.args.max_source_length,
            add_special_tokens=True,
            device=device,
        )

        # Mask prompt portion (right side with left padding)
        batch_size, seq_len = tokenized_prompts["attention_mask"].shape
        # Mask the prompt portion (right side with left padding) until reaching
        # self.bos_semantic_token_id
        is_bos = tokenized_prompts["input_ids"] == self.bos_semantic_token_id

        # Vectorized masking: mask everything up to and including the
        # FIRST bos_semantic_token_id.
        # cumsum(1) > 0 will be true for all positions at and after the
        # first BOS. We want to mask (set to 0) positions BEFORE and AT
        # the first BOS.

        # Get index of the first occurrence of bos_semantic_token_id.
        # argmax returns the first index where the condition is true.
        first_bos_indices = is_bos.long().argmax(dim=1)
        has_bos = is_bos.any(dim=1)

        range_tensor = torch.arange(
            seq_len,
            device=device,
        ).expand(batch_size, seq_len)
        # Mask where index <= first_bos_index for rows that have a BOS
        mask_to_zero = (
            range_tensor <= first_bos_indices.unsqueeze(1)
        ) & has_bos.unsqueeze(1)
        tokenized_prompts["attention_mask"][mask_to_zero] = 0

        # 2. Reconstruct the original text
        if self.model.training is False:

            generated = model.generate(
                **tokenized_prompts,
                max_new_tokens=self.args.generation_max_new_tokens,
                do_sample=self.args.generation_do_sample,
                temperature=self.args.generation_temperature,
                top_p=self.args.generation_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            return self.tokenizer.batch_decode(
                generated[:, seq_len:], skip_special_tokens=True
            )

        # Calculate reconstruction loss
        tokenized_labels = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=self.args.max_target_length,
            device=device,
        )
        # Expand tokenized_labels to match the length of tokenized_prompts
        tokenized_labels["input_ids"] = torch.cat(
            [
                torch.full((batch_size, seq_len), IGNORE_INDEX, device=device),
                tokenized_labels["input_ids"],
            ],
            dim=-1,
        )

        tokenized_prompts["input_ids"] = torch.cat(
            [tokenized_prompts["input_ids"], tokenized_labels["input_ids"]],
            dim=1,
        )
        tokenized_prompts["attention_mask"] = torch.cat(
            [
                tokenized_prompts["attention_mask"],
                tokenized_labels["attention_mask"],
            ],
            dim=1,
        )
        return model(
            **tokenized_prompts, labels=tokenized_labels["input_ids"]
        ).loss  # noqa

    def _generate_semantic_ids(
        self,
        model,
        prompts: list[str],
    ) -> tuple[list[list[int]], list[str]]:
        device = next(model.parameters()).device

        # Tokenize prompts and input
        tokenized_prompts = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.args.max_source_length,
            device=device,
        )
        prompt_length = tokenized_prompts["input_ids"].shape[1]

        # Generate semantic_ids
        generated = model.generate(
            **tokenized_prompts,
            min_new_tokens=self.args.codebook_size + 1,
            max_new_tokens=self.args.codebook_size * self.max_digit_num
            + 1,  # codebook_size semantic IDs + 1 for eos_semantic_token
            do_sample=self.args.generation_do_sample,
            temperature=self.args.generation_temperature,
            top_p=self.args.generation_top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_semantic_token_id,
            forced_eos_token_id=self.eos_semantic_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        semantic_ids = generated.sequences[:, prompt_length:-1]
        semantic_id_texts = self.tokenizer.batch_decode(
            semantic_ids, skip_special_tokens=True
        )

        if self.model.training is True:
            # Following https://arxiv.org/abs/2305.05065
            # In our fomat:
            # - r_i equals to previous hidden_states
            # - e_i equals to embedding of semantic_ids

            # Check if semantic_id_texts follow format pattern: comma-separated numbers up to codebook_size
            # Create regex pattern for comma-separated numbers within range and limit count to codebook_size
            pattern = rf"^\s*(\d+\s*(,\s*\d+\s*){{0,{self.args.codebook_size-1}}})?\s*$"
            valids = []

            for text in semantic_id_texts:
                # Allow empty string, single number, or up to codebook_size numbers
                if not text.strip():  # Empty string is valid
                    valids.append(1e-2)
                    continue
                elif not re.match(pattern, text):
                    valids.append(1e-2)
                    break

                # Additional check: verify numbers are within codebook_range
                numbers = [int(num.strip()) for num in text.split(",") if num.strip()]
                if any(num < 0 or num >= self.args.codebook_range for num in numbers):
                    valids.append(1e-2)
                    break

                valids.append(1.0)

            # Calculate semantic format loss
            hidden_states = generated.hidden_states[:, prompt_length:]

            # Get semantic_ids only (not other tokens)
            embeddings = model.get_input_embeddings()(semantic_ids)

            # Stop gradients flows of embeddings
            format_loss = (hidden_states.detach() - embeddings).sum(dim=-1) + (
                hidden_states - embeddings.detach()
            ).sum(dim=-1)
            valids = torch.tensor(
                valids,
                device=format_loss.device,
                dtype=format_loss.dtype,
            )
            format_loss = torch.div(format_loss, valids)
        else:
            format_loss = None

        return (
            semantic_ids,
            semantic_id_texts,
            format_loss,
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
