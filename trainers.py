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
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm for clipping."}
    )
    remove_unused_columns: bool = field(default=False)
    temperature_initial: float = field(
        default=1.0,
        metadata={"help": "Initial temperature for Gumbel-softmax annealing."},
    )
    temperature_final: float = field(
        default=1e-10,
        metadata={"help": "Final temperature for Gumbel-softmax annealing."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to save memory."},
    )


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
        self.step_count = 0

        # Debug: Check if max_grad_norm is set
        print(f"max_grad_norm setting: {self.args.max_grad_norm}")

        self.codebook_tokens = [
            f"<|CODE_{id}|>" for id in range(self.args.codebook_range)
        ]  # noqa
        # Add special tokens for semantic IDs
        special_tokens = {
            "extra_special_tokens": [
                BOS_SEMANTIC_TOKEN,
                EOS_SEMANTIC_TOKEN,
                BOS_SEMANTIC_SESSION,
            ]
            + self.codebook_tokens
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.bos_semantic_token_id = self.tokenizer.vocab[BOS_SEMANTIC_TOKEN]
        self.eos_semantic_token_id = self.tokenizer.vocab[EOS_SEMANTIC_TOKEN]
        self.bos_semantic_session_id = self.tokenizer.vocab[BOS_SEMANTIC_SESSION]

        # Add special tokens to model
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.codebook_token_ids = torch.tensor(
            [
                self.tokenizer.vocab[codebook_token]
                for codebook_token in self.codebook_tokens
            ]
        ).to(self.model.device)

        self.bad_words_ids = [
            [token_id]
            for token_id in range(self.tokenizer.vocab_size)
            if token_id not in self.codebook_token_ids
        ]

    def training_step(self, model, inputs, num_items_in_batch=None):
        result = super().training_step(model, inputs, num_items_in_batch)

        # Debug: Check LoRA gradients after backward pass
        # if self.step_count < 3:  # Only check first few steps
        if True:
            lora_grads = []
            for name, param in model.named_parameters():
                if "lora" in name and param.grad is not None:
                    lora_grads.append(param.grad.abs().mean().item())
            if lora_grads:
                print(
                    f"Step {self.step_count} - LoRA gradient magnitude: "
                    f"{sum(lora_grads)/len(lora_grads):.6f}"
                )
            else:
                print(f"Step {self.step_count} - WARNING: No LoRA gradients!")

        self.step_count += 1
        return result

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
        # Debug: Check if model is training
        if not model.training:
            print("WARNING: Model not in training mode!")

        # Debug: Check if LoRA parameters have gradients
        has_lora = hasattr(model, "peft_config") and model.peft_config is not None
        if has_lora:
            lora_params = [p for n, p in model.named_parameters() if "lora" in n]
            if lora_params:
                print(f"LoRA params count: {len(lora_params)}")
                print(
                    f"LoRA params require grad: "
                    f"{all(p.requires_grad for p in lora_params)}"
                )

        # with torch.autograd.set_detect_anomaly(True):
        # Encode input and generate semantic IDs
        encode_outputs = self._generate_semantic_ids(
            model=model,
            prompts=inputs["guessing_prompt"],
        )

        # Look at semantic IDs and reconstruct input
        reconstruction_loss = self._reconstruct_input(
            model=model,
            prompts=inputs["reconstruction_prompt"],
            # semantic_ids_texts=encode_outputs["semantic_id_texts"],
            soft_embeddings=encode_outputs["soft_embeddings"],
        )

        diversity_score = self._compute_diversity_score(
            encode_outputs["semantic_ids"],
        )
        loss = (
            self.args.guessing_weight * encode_outputs["format_loss"]
            + self.args.reconstruction_weight * reconstruction_loss
        )
        # loss.backward()
        print("Semantic id texts", encode_outputs["semantic_id_texts"])
        self.log(
            {
                "format_loss": encode_outputs["format_loss"].detach().item(),
                "reconstruction_loss": reconstruction_loss.detach().item(),
                "semantic_id_diversity": torch.tensor(
                    diversity_score, device=next(model.parameters()).device
                )
                .detach()
                .item(),
            }
        )

        # Debug: Check if loss has gradients
        if loss.requires_grad:
            print(f"Loss requires_grad: {loss.requires_grad}")
        else:
            print("WARNING: Loss does not require grad!")

        if return_outputs:
            return loss, {
                "semantic_ids": encode_outputs["semantic_ids"],
                "semantic_id_texts": encode_outputs["semantic_id_texts"],
                "format_loss": encode_outputs["format_loss"].detach().item(),
                "reconstruction_loss": reconstruction_loss.detach().item(),
                "semantic_id_diversity": torch.tensor(
                    diversity_score,
                    device=next(model.parameters()).device,
                )
                .detach()
                .item(),
            }
        print(f"Allocated: {torch.mps.current_allocated_memory() / 1e6:.2f} MB")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss

    # def _reconstruct_input(
    #     self,
    #     model,
    #     prompts: list[str],
    #     semantic_ids_texts: list[str],
    #     soft_embeddings=None,
    # ) -> tuple[list[list[int]], list[str]]:
    #     device = next(model.parameters()).device

    #     for idx, semantic_ids_text in enumerate(semantic_ids_texts):
    #         prompts[idx][1]["content"] = prompts[idx][1]["content"].format(
    #             SEMANTIC_IDS=semantic_ids_text
    #         )

    #     # This should:
    #     # 1. Take generated semantic IDs and encoded input
    #     # 2. Reconstruct the original text
    #     # 3. Return reconstructed text and decoded IDs

    #     if self.model.training is False:

    #         # Tokenize prompts and input
    #         tokenized_prompts = self.tokenizer.apply_chat_template(
    #             prompts,
    #             return_tensors="pt",
    #             padding=True,
    #             padding_side="left",
    #             tokenize=True,
    #             truncation=True,
    #             max_length=self.args.max_target_length,
    #             add_special_tokens=True,
    #         )
    #         tokenized_prompts = {
    #             k: v.to(device) for k, v in tokenized_prompts.items()
    #         }  # noqa
    #         prompt_length = tokenized_prompts["input_ids"].shape[1]
    #         generated = model.generate(
    #             **tokenized_prompts,
    #             # max_new_tokens=self.args.generation_max_new_tokens,
    #             max_length=self.args.max_target_length,
    #             do_sample=self.args.generation_do_sample,
    #             temperature=self.args.generation_temperature,
    #             top_p=self.args.generation_top_p,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #         )

    #         return self.tokenizer.batch_decode(
    #             generated[:, prompt_length:], skip_special_tokens=True
    #         )

    #     # Calculate reconstruction loss
    #     tokenized_prompts = self.tokenizer.apply_chat_template(
    #         prompts,
    #         return_tensors="pt",
    #         padding=True,
    #         padding_side="left",
    #         truncation=True,
    #         tokenize=True,
    #         add_special_tokens=True,
    #         # max_length=self.args.max_target_length,
    #     )
    #     tokenized_prompts = {
    #         k: v.to(device) for k, v in tokenized_prompts.items()
    #     }  # noqa

    #     # Find indices of SEMANTIC IDS captured by BOS_TOKEN_ID and EOS_TOKEN_ID
    #     semantic_ids_positions = (
    #         (tokenized_prompts["input_ids"] == self.bos_semantic_token_id)
    #         .nonzero(as_tuple=True)[-1]
    #         .reshape(len(prompts), -1)
    #     )[:, -1:]
    #     semantic_ids_positions = (
    #         torch.arange(
    #             start=1,
    #             end=self.args.codebook_size + 1,
    #             device=device,
    #         ).repeat(len(prompts), 1)
    #         + semantic_ids_positions
    #     )
    #     # eos_indices = (
    #     #     (tokenized_prompts["input_ids"] == self.eos_semantic_token_id)
    #     #     .nonzero(as_tuple=True)[-1]
    #     #     .reshape(len(prompts), -1)
    #     # )[:, -1]

    #     # Get embeddings of input_ids
    #     tokenized_prompts["inputs_embeds"] = model.get_input_embeddings()(
    #         tokenized_prompts["input_ids"]
    #     ).scatter_(
    #         1,
    #         semantic_ids_positions.unsqueeze(-1).repeat(
    #             1, 1, soft_embeddings.shape[-1]
    #         ),
    #         soft_embeddings,
    #     )

    #     # Replace embeddings from BOS_TOKEN_ID to EOS_TOKEN_ID
    #     # tokenized_prompts["inputs_embeds"][
    #     #     :, bos_indices + 1 : eos_indices
    #     # ] = soft_embeddings  # noqa

    #     return model(
    #         inputs_embeds=tokenized_prompts["inputs_embeds"],
    #         labels=tokenized_prompts["input_ids"],
    #         attention_mask=tokenized_prompts["attention_mask"],
    #     ).loss

    def _reconstruct_input(
        self,
        model,
        prompts: list[str],
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
        tokenized_prompts = self.tokenizer.apply_chat_template(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            tokenize=True,
            add_special_tokens=True,
            # max_length=self.args.max_target_length,
        )
        self.log(
            {
                "Number of procossed_tokens for reconstructing product-text": (
                    tokenized_prompts["attention_mask"]
                    .sum(dim=1)
                    .float()
                    .mean()
                    .item()  # noqa
                )
            }
        )
        tokenized_prompts = {
            k: v.to(device) for k, v in tokenized_prompts.items()
        }  # noqa

        # Find indices of SEMANTIC IDS captured by BOS_TOKEN_ID and EOS_TOKEN_ID
        semantic_ids_positions = (
            (tokenized_prompts["input_ids"] == self.bos_semantic_token_id)
            .nonzero(as_tuple=True)[-1]
            .reshape(len(prompts), -1)
        )[:, -1:]
        semantic_ids_positions = (
            (
                torch.arange(
                    start=1,  # increase by 1 for 0-index
                    end=self.args.codebook_size + 1,
                    device=device,
                ).repeat(len(prompts), 1)
                + semantic_ids_positions
            )  # currently [batch_size, codebook_size]
            # expand to [batch_size, codebook_size, hidden_size]
            .unsqueeze(-1).repeat(1, 1, soft_embeddings.shape[-1])
        )

        # Get embeddings of input_ids
        tokenized_prompts["inputs_embeds"] = model.get_input_embeddings()(
            tokenized_prompts["input_ids"].clone()
        ).scatter_(
            1,
            semantic_ids_positions,
            soft_embeddings,
        )

        # Consider only AI Message starting with last <|im_start|>
        label_pointer = (tokenized_prompts["input_ids"] == 1).nonzero(  # 1 stands for
            as_tuple=True
        )[-1][
            -1
        ]  # noqa

        tokenized_prompts["input_ids"][:, :label_pointer] = -100

        return model(
            inputs_embeds=tokenized_prompts["inputs_embeds"],
            labels=tokenized_prompts["input_ids"],
            attention_mask=tokenized_prompts["attention_mask"],
        ).loss

    # def _generate_semantic_ids(
    #     self,
    #     model,
    #     prompts: list[str],
    # ) -> tuple[list[list[int]], list[str]]:
    #     device = next(model.parameters()).device

    #     # Tokenize prompts and input
    #     tokenized_prompts = self.tokenizer.apply_chat_template(
    #         prompts,
    #         return_tensors="pt",
    #         padding=True,
    #         padding_side="left",
    #         tokenize=True,
    #         truncation=True,
    #         max_length=self.args.max_source_length,
    #         add_special_tokens=True,
    #         eos_token_id=None,
    #     )
    #     tokenized_prompts = {k: v.to(device) for k, v in tokenized_prompts.items()}
    #     bos_semantic_index = (
    #         tokenized_prompts["input_ids"] == self.bos_semantic_token_id
    #     ).nonzero(as_tuple=True)[-1][-1]
    #     tokenized_prompts = {
    #         k: v[:, : bos_semantic_index + 1]
    #         for k, v in tokenized_prompts.items()  # noqa
    #     }
    #     generated = None
    #     for _ in range(self.args.codebook_size):
    #         generated = model(**tokenized_prompts)
    #         next_best_codebooks = torch.argmax(
    #             torch.nn.functional.softmax(
    #                 generated.logits[:, -1:, self.codebook_token_ids], dim=-1
    #             ),
    #             dim=-1,
    #         )
    #         next_best_codebooks = self.codebook_token_ids[next_best_codebooks]

    #         # Append selected tokens to input sequences
    #         tokenized_prompts["input_ids"] = torch.cat(
    #             [tokenized_prompts["input_ids"], next_best_codebooks],
    #             dim=-1,
    #         )
    #         tokenized_prompts["attention_mask"] = torch.cat(
    #             [
    #                 tokenized_prompts["attention_mask"],
    #                 torch.ones_like(next_best_codebooks),
    #             ],
    #             dim=-1,
    #         )
    #     del next_best_codebooks

    #     # Gumbel-softmax straight-through estimator
    #     semantic_ids = tokenized_prompts["input_ids"][
    #         :, bos_semantic_index + 1 :
    #     ]  # batch,  [<|bos_semantic|>, ...tokens..., <|eos_semantic|>]

    #     semantic_id_texts = self.tokenizer.batch_decode(
    #         semantic_ids,
    #         skip_special_tokens=False,
    #     )  # [<|bos_semantic|>, ...tokens..., <|eos_semantic|>]

    #     if self.model.training is True:
    #         # Gumbel-softmax straight-through estimator for embeddings
    #         soft_embeddings = generated.logits[
    #             :, bos_semantic_index:
    #         ]  # [batch, output_tokens, vocab_size]

    #         # convert semantic_ids to onehot
    #         semantic_id_onehots = torch.nn.functional.one_hot(
    #             semantic_ids,
    #             num_classes=len(self.tokenizer),
    #         )  # [batch, output_tokens, vocab_size]
    #         # Differentiable selection of logits for semantic_ids
    #         soft_embeddings = (
    #             soft_embeddings * semantic_id_onehots
    #         )  # [batch, output_tokens, vocab_size]
    #         soft_embeddings = soft_embeddings.sum(
    #             dim=-1, keepdim=True
    #         )  # [batch, output_tokens, 1]
    #         hard_embeddings = model.get_input_embeddings()(
    #             semantic_ids
    #         )  # [batch, output_tokens, hidden_size]
    #         soft_embeddings = hard_embeddings + (
    #             soft_embeddings - soft_embeddings.detach()
    #         ).repeat(1, 1, hard_embeddings.shape[-1])
    #         # soft_embeddings = hard_embeddings + (
    #         #     soft_embeddings - soft_embeddings.detach()
    #         # ).expand_as(
    #         #     hard_embeddings
    #         # )  # [batch, output_tokens, hidden_size]

    #         # Format loss
    #         prompts = [
    #             (
    #                 system_prompt,
    #                 user_prompt,
    #                 {
    #                     k: v.format(SEMANTIC_IDS=semantic_id_text)
    #                     for k, v in assistant_prompt.items()
    #                 },
    #             )
    #             for semantic_id_text, (
    #                 system_prompt,
    #                 user_prompt,
    #                 assistant_prompt,
    #             ) in zip(semantic_id_texts, prompts)
    #         ]
    #         tokenized_prompts = self.tokenizer.apply_chat_template(
    #             prompts,
    #             return_tensors="pt",
    #             padding=True,
    #             padding_side="left",
    #             tokenize=True,
    #             truncation=True,
    #             max_length=self.args.max_source_length,
    #             add_special_tokens=True,
    #             eos_token_id=None,
    #         )
    #         for k, v in tokenized_prompts.items():
    #             tokenized_prompts[k] = v.to(device)
    #         tokenized_prompts["labels"] = tokenized_prompts["input_ids"]
    #         tokenized_prompts["labels"][
    #             :,
    #             bos_semantic_index
    #             + 1 : bos_semantic_index
    #             + 1
    #             + self.args.codebook_size,
    #         ] = -100  # Ignore semantic-ids in format loss

    #         format_loss = model(**tokenized_prompts).loss
    #     else:
    #         format_loss = None
    #         soft_embeddings = None

    #     return {
    #         "semantic_ids": semantic_ids,
    #         "semantic_id_texts": semantic_id_texts,
    #         "format_loss": format_loss,
    #         "soft_embeddings": soft_embeddings,
    #     }

    def _add_gumbel_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Add Gumbel noise to logits for diverse sampling.
        Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
        """
        return -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)

    def _get_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get the current temperature for sampling with annealing schedule.
        Temperature anneals from temperature_initial to temperature_final over
        the total training steps (max_steps).
        """
        # return torch.full_like(
        #     logits, 1.0, dtype=logits.dtype
        # )  # TODO: Implement temperature scheduling

        progress = min(self.step_count / self.state.max_steps, 1.0)
        current_temp = (
            self.args.temperature_initial * (1 - progress)
            + self.args.temperature_final * progress
        )

        # Log temperature
        self.log({"temperature": current_temp})

        return torch.full_like(logits, current_temp, dtype=logits.dtype)

    def _generate_semantic_ids(
        self,
        model,
        prompts: list[str],
    ) -> tuple[list[list[int]], list[str]]:
        """
        Generate semantic IDs for the given prompts.
        Iteratively generate codebooks and append them to the input sequence.
        """
        device = next(model.parameters()).device

        # Tokenize prompts and input
        tokenized_prompts = self.tokenizer.apply_chat_template(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            tokenize=True,
            truncation=True,
            # max_length=self.args.max_source_length,
            add_special_tokens=True,
            eos_token_id=None,
        )
        self.log(
            {
                "Number of procossed_tokens for generating semantic-ids": (
                    tokenized_prompts["attention_mask"]
                    .sum(dim=1)
                    .float()
                    .mean()
                    .item()  # noqa
                )
            }
        )
        # print("tokenized_prompts", tokenized_prompts, prompts)
        tokenized_prompts = {
            k: v.to(device) for k, v in tokenized_prompts.items()
        }  # noqa

        # Retrieve embeddings for the input sequence
        tokenized_prompts["inputs_embeds"] = model.get_input_embeddings()(
            tokenized_prompts["input_ids"].clone()
        )

        bos_semantic_index = (
            tokenized_prompts["input_ids"] == self.bos_semantic_token_id
        ).nonzero(as_tuple=True)[-1][-1]

        for i in range(1, self.args.codebook_size + 1):
            codebook_idx = bos_semantic_index + i
            # Retrieve logits
            codebook_logits = model(
                inputs_embeds=(
                    tokenized_prompts["inputs_embeds"][:, :codebook_idx]
                    # if len(soft_embeddings) == 0
                    # else torch.cat(
                    #     [
                    #         tokenized_prompts["inputs_embeds"][
                    #             :, : bos_semantic_index + 1
                    #         ],  # noqa
                    #         torch.cat(soft_embeddings, dim=1),
                    #     ],
                    #     dim=1,
                    # )
                ),
                attention_mask=tokenized_prompts["attention_mask"][
                    :, :codebook_idx
                ],  # noqa
            ).logits[
                :, -1:
            ]  # [batch_size, 1, vocab_size]

            # Select best codebook
            # next_best_codebook = torch.argmax(
            #     torch.nn.functional.softmax(
            #         codebook_logits[..., self.codebook_token_ids],
            #         dim=-1,
            #     ),
            #     dim=-1,
            # )  # [batch_size, 1]
            # next_best_codebook = self.codebook_token_ids[
            #     next_best_codebook
            # ]  # [batch_size, 1]

            # # Get full-vocab onehot vector
            # next_best_codebook_onehot = torch.nn.functional.one_hot(
            #     next_best_codebook,
            #     num_classes=len(self.tokenizer),
            # ).to(
            #     dtype=codebook_logits.dtype,
            # )  # [batch_size, 1, vocab_size]

            # Append selected tokens to input sequences
            # tokenized_prompts["input_ids"][
            #     :, codebook_idx : codebook_idx + 1  # noqa
            # ] = next_best_codebook

            # Apply STE trick to allow gradient flow
            # probs = (
            #     codebook_logits  # [batch_size, 1, vocab_size]
            #     * next_best_codebook_onehot  # [batch_size, 1, vocab_size]
            #     / (codebook_logits.detach() + 1e-8)  # [batch_size, 1, vocab_size]
            # )  # [batch_size, 1, vocab_size]

            # # Debug: Check if gradients flow through STE
            # if i == 1 and self.model.training:
            #     print(
            #         f"Codebook logits requires_grad: "
            #         f"{codebook_logits.requires_grad}"
            #     )
            #     print(f"Probs requires_grad: {probs.requires_grad}")

            # codebook_embeds = torch.matmul(
            #     # matmul assigns weight to each vocab and sum up.
            #     # Only vector of the selected vocab gets non-zero weight
            #     probs,  # [batch_size, 1, vocab_size]
            #     # codebook_logits,
            #     model.get_input_embeddings().weight,  # [vocab_size, hidden_size]
            # )  # [batch_size, 1, hidden_size]

            # ---------------

            # Gumbel-softmax
            gumbel_noise = self._add_gumbel_noise(
                codebook_logits[..., self.codebook_token_ids]
            )  # [batch_size, 1, codebook_size]
            codebook_logits = (
                codebook_logits[..., self.codebook_token_ids] + gumbel_noise
            )  # [batch_size, 1, codebook_size]

            # soft update backpropagation
            codebook_logits = torch.nn.functional.softmax(
                codebook_logits / self._get_temperature(codebook_logits),
                dim=-1,
            )  # [batch_size, 1, codebook_size]

            # Hard forward
            next_best_codebook = self.codebook_token_ids[
                torch.argmax(codebook_logits, dim=-1)
            ]  # [batch_size, 1]
            next_best_codebook_onehot = torch.nn.functional.one_hot(
                next_best_codebook,
                num_classes=len(self.codebook_token_ids),
            ).to(
                dtype=codebook_logits.dtype,
            )  # [batch_size, 1, codebook_size]
            codebook_logits = (
                next_best_codebook_onehot
                - codebook_logits.detach()
                + codebook_logits  # noqa
            )  # ensures no mismatch between training and inference

            # Get full-vocab onehot vector
            # next_best_codebook_onehot = torch.nn.functional.one_hot(
            #     next_best_codebook,
            #     num_classes=len(self.tokenizer),
            # ).to(
            #     dtype=codebook_logits.dtype,
            # )  # [batch_size, 1, vocab_size]

            # # Append selected tokens to input sequences
            tokenized_prompts["input_ids"][
                :, codebook_idx : codebook_idx + 1  # noqa
            ] = next_best_codebook

            # Apply Gumbel-softmax trick to allow gradient flow
            codebook_embeds = torch.matmul(
                codebook_logits,  # [batch_size, 1, codebook_size]
                model.get_input_embeddings()(
                    self.codebook_token_ids.clone()
                ),  # [codebook_size, hidden_size]
            )  # [batch_size, 1, hidden_size]
            # probs = (
            #     codebook_logits  # [batch_size, 1, vocab_size]
            #     * next_best_codebook_onehot  # [batch_size, 1, vocab_size]
            #     / (codebook_logits.detach() + 1e-8)  # [batch_size, 1, vocab_size]
            # )  # [batch_size, 1, vocab_size]

            # # Debug: Check if gradients flow through STE
            # if i == 1 and self.model.training:
            #     print(
            #         f"Codebook logits requires_grad: "
            #         f"{codebook_logits.requires_grad}"
            #     )
            #     print(f"Probs requires_grad: {probs.requires_grad}")

            # codebook_embeds = torch.matmul(
            #     # matmul assigns weight to each vocab and sum up.
            #     # Only vector of the selected vocab gets non-zero weight
            #     probs,  # [batch_size, 1, vocab_size]
            #     # codebook_logits,
            #     model.get_input_embeddings().weight,  # [vocab_size, hidden_size]
            # )  # [batch_size, 1, hidden_size]

            # Debug: Check embedding sum
            if i == 1 and self.model.training:
                # Codebook embeds sum: -1.085938 repeated for many times
                print(
                    f"Codebook embeds sum: "
                    f"{model.get_input_embeddings()(self.codebook_token_ids[0].clone()).sum().item():.6f}"
                )

            # Debug: Check if gradients flow through embeddings
            if i == 1 and self.model.training:
                print(
                    f"Codebook embeds requires_grad: "
                    f"{codebook_embeds.requires_grad}"
                )
            # Append embeds of selected codebook tokens
            tokenized_prompts["inputs_embeds"][
                :, codebook_idx : codebook_idx + 1  # noqa
            ] = codebook_embeds
            # soft_embeddings.append(codebook_embeds)

            # After each codebook iteration
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

        del (
            next_best_codebook,
            codebook_logits,
            gumbel_noise,
            next_best_codebook_onehot,
        )  # noqa

        # Gumbel-softmax straight-through estimator
        semantic_ids = tokenized_prompts["input_ids"][
            :, bos_semantic_index + 1 :
        ]  # batch,  [<|bos_semantic|>, ...tokens..., <|eos_semantic|>]

        semantic_id_texts = self.tokenizer.batch_decode(
            semantic_ids,
            skip_special_tokens=False,
        )  # [<|bos_semantic|>, ...tokens..., <|eos_semantic|>]

        if self.model.training is True:
            soft_embeddings = tokenized_prompts["inputs_embeds"][
                :,
                bos_semantic_index
                + 1 : bos_semantic_index  # noqa
                + self.args.codebook_size
                + 1,
            ]  # noqa

            # Ignore semantic-ids in format loss
            # Consider only AI Message starting with last
            label_pointer = (
                tokenized_prompts["input_ids"] == 1
            ).nonzero(  # 1 stands for
                as_tuple=True  # noqa
            )[
                -1
            ][
                -2
            ]

            tokenized_prompts["input_ids"][:, :label_pointer] = -100
            tokenized_prompts["input_ids"][
                :,
                # Ignore prompt tokens that are redundant to recompute
                bos_semantic_index
                + 1 : bos_semantic_index
                + 1
                + self.args.codebook_size,
            ] = -100

            format_loss = model(
                inputs_embeds=torch.cat(
                    [
                        tokenized_prompts["inputs_embeds"][
                            :, : bos_semantic_index + 1
                        ],  # noqa
                        soft_embeddings,
                        tokenized_prompts["inputs_embeds"][
                            :,
                            bos_semantic_index + 1 + self.args.codebook_size :,  # noqa
                        ],  # noqa
                    ],
                    dim=1,
                ),
                attention_mask=tokenized_prompts["attention_mask"],
                labels=tokenized_prompts["input_ids"],
            ).loss
        else:
            format_loss = None
            soft_embeddings = None

        return {
            "semantic_ids": semantic_ids,
            "semantic_id_texts": semantic_id_texts,
            "format_loss": format_loss,
            "soft_embeddings": soft_embeddings,
        }

    def _compute_diversity_score(self, semantic_ids: torch.Tensor) -> float:
        if semantic_ids is None:
            return 0.0

        per_example_diversity = torch.tensor(
            [
                len(set(example_ids)) / max(1, len(example_ids))
                for example_ids in semantic_ids
            ]
        )
        return torch.mean(per_example_diversity)
