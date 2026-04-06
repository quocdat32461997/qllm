import re
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments

from constants import (  # noqa
    BOS_SEMANTIC_SESSION,
    BOS_SEMANTIC_TOKEN,
    EOS_SEMANTIC_TOKEN,
)

IGNORE_INDEX = -100


@dataclass
class QuantConfig(TrainingArguments):
    codebook_size: int = field(
        default=6,
        metadata={"help": "Number of semantic IDs."},
    )
    codebook_range: int = field(
        default=128, metadata={"help": "Maximum value for each semantic ID."}
    )
    max_source_length: int = field(
        default=256, metadata={"help": "Max length for prompts."}
    )
    max_target_length: int = field(
        default=64,
        metadata={"help": "Max length for semantic ID and product targets."},
    )
    generation_max_new_tokens: int = field(
        default=24,
        metadata={"help": "Budget used while sampling semantic-ID candidates."},  # noqa
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

        self.codebook_token_ids = self.tokenizer(
            [str(i) for i in range(self.args.codebook_range + 1)]
        ).input_ids
        self.codebook_token_ids = set(
            [
                token_id
                for token_ids in self.codebook_token_ids
                for token_id in token_ids
            ]
        )
        self.delimiter_token_id = self.tokenizer(",").input_ids[0]
        self.bad_words_ids = [
            [token_id]
            for token_id in range(self.tokenizer.vocab_size)
            if token_id not in self.codebook_token_ids
        ]
        self.select_token_ids = torch.tensor(
            list(self.codebook_token_ids)
            + [
                self.delimiter_token_id,
                # self.bos_semantic_token_id,
                # self.eos_semantic_token_id,
            ]
        ).to(self.model.device)
        self.special_token_ids = torch.tensor(
            [
                self.bos_semantic_token_id,
                self.eos_semantic_token_id,
            ]
        ).to(self.model.device)

    def compute_loss(
        self,
        model,
        inputs: dict[str, list[str]],
        return_outputs: bool = False,
        **_: Any,
    ):
        """
        model: AutoModelForCausalLM
        """

        # Encoding: generate semantic IDs
        (semantic_ids, semantic_id_texts, format_loss, soft_embeddings) = (
            self._generate_semantic_ids(
                model=model,
                prompts=inputs["guessing_prompt"],
            )
        )

        # Decoding: reconstruct original text from semantic IDs
        for (_, user_prompt, _), (_, target_user_prompt, _), _semantic_ids, idx in zip(
            inputs["reconstruction_prompt_template"],
            inputs["reconstruction_target"],
            semantic_id_texts,
            range(len(semantic_id_texts)),
        ):
            user_prompt["content"] = user_prompt["content"].format(  # noqa
                semantic_id_texts=_semantic_ids
            )
            target_user_prompt["content"] = target_user_prompt[
                "content"
            ].format(  # noqa
                semantic_id_texts=_semantic_ids
            )
            inputs["reconstruction_prompt_template"][idx][1] = user_prompt
            inputs["reconstruction_target"][idx][1] = target_user_prompt

        reconstruction_loss = self._reconstruct_input(
            model=model,
            prompts=inputs["reconstruction_prompt_template"],
            labels=inputs["reconstruction_target"],
            soft_embeddings=soft_embeddings,
        )

        diversity_score = self._compute_diversity_score(semantic_ids)
        format_loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)
        loss = (
            self.args.guessing_weight * format_loss
            + self.args.reconstruction_weight * reconstruction_loss
        )

        self.log(
            {
                "format_loss": format_loss.detach().item(),
                "reconstruction_loss": reconstruction_loss.detach().item(),
                "semantic_id_diversity": torch.tensor(
                    diversity_score, device=next(model.parameters()).device
                )
                .detach()
                .item(),
            }
        )

        if return_outputs:
            return loss, {
                "semantic_ids": semantic_ids,
                "semantic_id_texts": semantic_id_texts,
                "format_loss": format_loss.detach().item(),
                "reconstruction_loss": reconstruction_loss.detach().item(),
                "semantic_id_diversity": torch.tensor(
                    diversity_score,
                    device=next(model.parameters()).device,
                )
                .detach()
                .item(),
            }
        return loss

    def _reconstruct_input(
        self,
        model,
        prompts: list[str],
        labels: list[str],
        soft_embeddings=None,
    ) -> tuple[list[list[int]], list[str]]:
        device = next(model.parameters()).device
        # This should:
        # 1. Take generated semantic IDs and encoded input
        # 2. Reconstruct the original text
        # 3. Return reconstructed text and decoded IDs

        if self.model.training is False:

            # Tokenize prompts and input
            tokenized_prompts = self.tokenizer.apply_chat_template(
                prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                tokenize=True,
                truncation=True,
                max_length=self.args.max_target_length,
                add_special_tokens=True,
            )
            tokenized_prompts = {
                k: v.to(device) for k, v in tokenized_prompts.items()
            }  # noqa
            prompt_length = tokenized_prompts["input_ids"].shape[1]
            generated = model.generate(
                **tokenized_prompts,
                # max_new_tokens=self.args.generation_max_new_tokens,
                max_length=self.args.max_target_length,
                do_sample=self.args.generation_do_sample,
                temperature=self.args.generation_temperature,
                top_p=self.args.generation_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            return self.tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )

        # Calculate reconstruction loss
        tokenized_labels = self.tokenizer.apply_chat_template(
            labels,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            tokenize=True,
            add_special_tokens=True,
            # max_length=self.args.max_target_length,
        )
        tokenized_labels = {
            k: v.to(device) for k, v in tokenized_labels.items()
        }  # noqa

        # Prepare labels by shifting by one token
        tokenized_labels["labels"] = tokenized_labels["input_ids"][:, 1:]
        tokenized_labels["input_ids"] = tokenized_labels["input_ids"][:, :-1]
        tokenized_labels["attention_mask"] = tokenized_labels["attention_mask"][
            :, :-1
        ]  # noqa

        # Find lindices of BOS_TOKEN_ID and EOS_TOKEN_ID
        bos_index = (
            (tokenized_labels["input_ids"] == self.bos_semantic_token_id)
            .nonzero(as_tuple=True)[-1]
            .max()
        )
        eos_index = (
            (tokenized_labels["input_ids"] == self.eos_semantic_token_id)
            .nonzero(as_tuple=True)[-1]
            .max()
        ) + 1

        # Get embeddings of input_ids
        tokenized_labels["inputs_embeds"] = model.get_input_embeddings()(
            tokenized_labels["input_ids"]
        )

        # Replace embeddings from BOS_TOKEN_ID to EOS_TOKEN_ID
        tokenized_labels["inputs_embeds"][
            :, bos_index:eos_index
        ] = soft_embeddings  # noqa

        # Create proper input structure for the model
        tokenized_labels.pop("input_ids")
        return model(**tokenized_labels).loss

    def _generate_semantic_ids(
        self,
        model,
        prompts: list[str],
    ) -> tuple[list[list[int]], list[str]]:
        device = next(model.parameters()).device

        # Tokenize prompts and input
        tokenized_prompts = self.tokenizer.apply_chat_template(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            tokenize=True,
            truncation=True,
            max_length=self.args.max_source_length,
            add_special_tokens=True,
            eos_token_id=None,
        )
        tokenized_prompts = {  # exclude last 2 tokens, <|im_end|>\n
            k: v[:, :-2].to(device) for k, v in tokenized_prompts.items()
        }
        prompt_length = tokenized_prompts["input_ids"].shape[1]

        # Generate semantic_ids
        logits = []
        for idx in range(self.args.codebook_size + 1):
            if idx == 0:
                generated = model.generate(
                    **tokenized_prompts,
                    max_new_tokens=1,
                    do_sample=self.args.generation_do_sample,
                    temperature=self.args.generation_temperature,
                    top_p=self.args.generation_top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    forced_eos_token_id=self.bos_semantic_token_id,
                    bad_words_ids=self.bad_words_ids,  # Restrict to codebook tokens only
                    output_logits=True,
                    return_dict_in_generate=True,
                )
            else:
                generated = model.generate(
                    **tokenized_prompts,
                    min_new_tokens=2,
                    max_new_tokens=self.max_digit_num + 1,
                    do_sample=self.args.generation_do_sample,
                    temperature=self.args.generation_temperature,
                    top_p=self.args.generation_top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    forced_eos_token_id=(
                        self.delimiter_token_id
                        if idx != self.args.codebook_size
                        else self.eos_semantic_token_id
                    ),
                    bad_words_ids=self.bad_words_ids,  # Restrict to codebook tokens only
                    output_logits=True,
                    return_dict_in_generate=True,
                )
            tokenized_prompts["input_ids"] = generated.sequences
            tokenized_prompts["attention_mask"] = torch.cat(
                [
                    tokenized_prompts["attention_mask"],
                    torch.ones(
                        (
                            tokenized_prompts["attention_mask"].shape[0],
                            tokenized_prompts["input_ids"].shape[1]
                            - tokenized_prompts["attention_mask"].shape[1],
                        ),
                        device=tokenized_prompts["attention_mask"].device,
                    ),
                ],
                dim=1,
            )
            logits.extend(generated.logits)
            if idx == self.args.codebook_size:
                generated = model.generate(
                    **tokenized_prompts,
                    max_new_tokens=1,
                    do_sample=self.args.generation_do_sample,
                    temperature=self.args.generation_temperature,
                    top_p=self.args.generation_top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    forced_eos_token_id=self.tokenizer.eos_token_id,
                    output_logits=True,
                    return_dict_in_generate=True,
                )

        # Gumbel-softmax straight-through estimator
        semantic_ids = generated.sequences[
            :, prompt_length:-1
        ]  # batch,  [<|bos_semantic|>, ...tokens..., <|eos_semantic|>]

        semantic_id_texts = self.tokenizer.batch_decode(
            semantic_ids,
            skip_special_tokens=False,
        )  # [<|bos_semantic|>, ...tokens..., <|eos_semantic|>]
        text = self.tokenizer.batch_decode(
            generated.sequences,
            skip_special_tokens=False,
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
                if not text.strip() or not re.match(
                    pattern, text
                ):  # Empty string is valid
                    valids.append(1e-1)
                    continue

                # Additional check: verify numbers are within codebook_range
                numbers = [int(num.strip()) for num in text.split(",") if num.strip()]
                if any(num < 0 or num >= self.args.codebook_range for num in numbers):
                    valids.append(1e-1)
                    continue

                valids.append(1.0)
            # Calculate semantic format loss
            # hidden_states = torch.cat(
            #     [
            #         hidden_states[-1]
            #         for hidden_states in generated.hidden_states[1:]  # noqa
            #     ],
            #     dim=1,
            # )  # [batch_size, num_output_tokens, hidden_size]

            # # Get semantic_ids only (not other tokens)
            # embeddings = model.get_input_embeddings()(semantic_ids)

            # # Stop gradients flows of embeddings
            # format_loss = torch.square(
            #     hidden_states.detach() - embeddings
            # ) + torch.square(hidden_states - embeddings.detach())
            # format_loss = format_loss.sum(-1)

            # valids = torch.tensor(
            #     valids,
            #     device=format_loss.device,
            #     dtype=format_loss.dtype,
            # )
            # format_loss = torch.sqrt(
            #     torch.div(
            #         format_loss.sum(-1),
            #         valids,
            #     ).mean(-1)
            # )
            format_loss = None

            # Gumbel-softmax straight-through estimator for embeddings
            semantic_id_logits = torch.stack(  # [batch, output_tokens, vocab_size]
                logits, dim=1
            ).squeeze()  # Excluding last two tokens (eos_semantic_token and eos_token)
            # semantic_id_logits = semantic_id_logits[
            #     :, :, self.select_token_ids
            # ]  # [batch, output_tokens, codebook_token_size]
            # semantic_onehot_ids = torch.nn.functional.gumbel_softmax(
            #     semantic_id_logits[1:-1], tau=1.0, hard=True
            # )  # [batch, output_tokens, codebook_token_size]
            # special_onehot_ids = torch.nn.functional.gumbel_softmax(
            #     semantic_id_logits[:, [0, -1]], tau=1.0, hard=True
            # )  # [batch, 2, codebook_token_size]
            # semantic_hard_embeddings = model.get_input_embeddings()(
            #     self.select_token_ids
            # )  # [codebook_token_size, hidden_size]
            # special_hard_embeddings = model.get_input_embeddings()(
            #     self.special_token_ids
            # )  # [2, hidden_size]
            # semantic_soft_embeddings = torch.matmul(
            #     semantic_onehot_ids, semantic_hard_embeddings
            # )
            # special_soft_embeddings = torch.matmul(
            #     special_onehot_ids, special_hard_embeddings
            # )
            # soft_embeddings

            # convert semantic_ids to onehot
            semantic_id_onehots = torch.nn.functional.one_hot(
                semantic_ids,
                num_classes=len(self.tokenizer),
            )  # [batch, output_tokens, vocab_size]
            # Differentiable selection of logits for selected tokens
            semantic_id_logits = (
                semantic_id_logits * semantic_id_onehots
            )  # [batch, output_tokens, vocab_size]
            semantic_id_logits = semantic_id_logits.sum(
                dim=-1, keepdim=True
            )  # [batch, output_tokens, 1]
            hard_embeddings = model.get_input_embeddings()(
                semantic_ids
            )  # [batch, output_tokens, hidden_size]
            soft_embeddings = (
                hard_embeddings * semantic_id_logits - semantic_id_logits.detach()
            )

        else:
            format_loss = None
            soft_embeddings = None

        return (
            semantic_ids,
            semantic_id_texts,
            format_loss,
            soft_embeddings,
        )

    def _compute_diversity_score(self, semantic_ids: list[list[int]]) -> float:
        if semantic_ids is None:
            return 0.0

        per_example_diversity = [
            len(set(example_ids)) / max(1, len(example_ids))
            for example_ids in semantic_ids
        ]
        batch_diversity = len(
            {tuple(example_ids) for example_ids in semantic_ids}
        ) / (  # noqa
            len(semantic_ids)
        )
        return float(
            (
                sum(per_example_diversity) / len(per_example_diversity)
                + batch_diversity
            )  # noqa
            / 2.0
        )
