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
        self._last_step = -1

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

        self.im_start_token_id = self.tokenizer.vocab["<|im_start|>"]

    def training_step(self, model, inputs, num_items_in_batch=None):
        result = super().training_step(model, inputs, num_items_in_batch)

        # Log current learning rate only once per step
        current_step = (
            self.step_count // self.current_gradient_accumulation_steps
        )  # noqa
        if current_step != self._last_step:
            self.log({"lr": self.optimizer.param_groups[0]["lr"]})
            self._last_step = current_step

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
            soft_embeddings=encode_outputs["soft_embeddings"],
        )

        diversity_score = self._compute_diversity_score(
            encode_outputs["semantic_ids"],
        )
        perplexity_score = self._compute_codebook_perplexity(
            encode_outputs["semantic_ids"],
        )
        per_position_perplexity = self._compute_per_position_perplexity(
            encode_outputs["semantic_ids"],
        )
        loss = (
            self.args.guessing_weight * encode_outputs["format_loss"]
            + self.args.reconstruction_weight * reconstruction_loss
        )
        # loss.backward()
        # print("Semantic id texts", encode_outputs["semantic_id_texts"])
        self.log(
            {
                "format_loss": encode_outputs["format_loss"].detach().item(),
                "reconstruction_loss": reconstruction_loss.detach().item(),
                "semantic_id_diversity": diversity_score,
                "semantic_id_perplexity": perplexity_score,
                **{
                    f"semantic_id_perplexity_pos_{pos}": ppl
                    for pos, ppl in enumerate(per_position_perplexity)
                },
                "guessing_prompt_avg_chars": sum(
                    [len(prompt) for prompt in inputs["guessing_prompt"]]
                )
                / len(inputs["guessing_prompt"]),
                "reconstruction_prompt_avg_chars": sum(
                    [len(prompt) for prompt in inputs["reconstruction_prompt"]]
                )
                / len(inputs["reconstruction_prompt"]),
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
                "format_loss": encode_outputs["format_loss"].detach().item(),
                "reconstruction_loss": reconstruction_loss.detach().item(),
                "semantic_id_perplexity": perplexity_score,
                "semantic_id_diversity": diversity_score,
                **{
                    f"semantic_id_perplexity_pos_{pos}": ppl
                    for pos, ppl in enumerate(per_position_perplexity)
                },
            }

        if torch.backends.mps.is_available():
            print(f"Allocated: {torch.mps.current_allocated_memory() / 1e6:.2f} MB")
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            torch.cuda.empty_cache()
        return loss

    def prediction_step(
        self,
        model,
        inputs: dict[str, list[str]],
    ) -> torch.Tensor:
        """
        Generate semantic IDs at evaluation / prediction time.

        Unlike ``compute_loss`` (which relies on the Gumbel-softmax
        straight-through path to keep gradients flowing), this runs the model
        under ``torch.no_grad`` and only emits the discrete semantic-ID tokens.
        No soft embeddings, format loss, or reconstruction loss are produced.

        Returns the generated semantic-ID token ids of shape
        ``(batch_size, seq_len)``.
        """
        with torch.no_grad():
            encode_outputs = self._generate_semantic_ids(
                model=model,
                prompts=inputs["guessing_prompt"],
            )

        # Log codebook-utilization diagnostics over the eval batch.
        self.log(
            {
                "eval_semantic_id_diversity": self._compute_diversity_score(
                    encode_outputs["semantic_ids"]
                ),
                "eval_semantic_id_perplexity": self._compute_codebook_perplexity(
                    encode_outputs["semantic_ids"]
                ),
                **{
                    f"eval_semantic_id_perplexity_pos_{pos}": ppl
                    for pos, ppl in enumerate(
                        self._compute_per_position_perplexity(
                            encode_outputs["semantic_ids"]
                        )
                    )
                },
            }
        )
        print("Eval semantic id texts", encode_outputs["semantic_id_texts"])

        return encode_outputs["semantic_ids"]

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
            max_length=self.args.max_target_length,
        )
        self.log(
            {
                "Number of procossed_tokens for reconstructing product-text": (
                    tokenized_prompts["attention_mask"].sum(dim=1).float().mean().item()
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
        label_pointer = (
            tokenized_prompts["input_ids"] == self.im_start_token_id
        ).nonzero(as_tuple=True)[-1][
            -1
        ]  # noqa

        tokenized_prompts["input_ids"][:, :label_pointer] = -100

        return model(
            inputs_embeds=tokenized_prompts["inputs_embeds"],
            labels=tokenized_prompts["input_ids"],
            attention_mask=tokenized_prompts["attention_mask"],
        ).loss

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
        current_step = (
            self.step_count // self.current_gradient_accumulation_steps
        )  # noqa
        progress = min(
            current_step / self.state.max_steps,
            1.0,
        )
        current_temp = (
            self.args.temperature_initial * (1 - progress)
            + self.args.temperature_final * progress
        )

        # Log temperature only once per step
        if current_step != self._last_step:
            self.log({"current_temperature": current_temp})
            self._last_step = current_step

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
            max_length=self.args.max_source_length,
            add_special_tokens=True,
            eos_token_id=None,
        )
        self.log(
            {
                "Number of procossed_tokens for generating semantic-ids": (
                    tokenized_prompts["attention_mask"].sum(dim=1).float().mean().item()
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
                ),  # noqa
                attention_mask=tokenized_prompts["attention_mask"][
                    :, :codebook_idx
                ],  # noqa
            ).logits[
                :, -1:
            ]  # [batch_size, 1, vocab_size]

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
            next_best_codebook_idx = torch.argmax(
                codebook_logits, dim=-1
            )  # [batch_size, 1]
            next_best_codebook = self.codebook_token_ids[
                next_best_codebook_idx
            ]  # [batch_size, 1]
            next_best_codebook_onehot = torch.nn.functional.one_hot(
                next_best_codebook_idx,
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

            # Append selected tokens to input sequences
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

        # semantic_id_texts = self.tokenizer.batch_decode(
        #     semantic_ids,
        #     skip_special_tokens=False,
        # )  # [<|bos_semantic|>, ...tokens..., <|eos_semantic|>]

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
                tokenized_prompts["input_ids"] == self.im_start_token_id
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
            # "semantic_id_texts": semantic_id_texts,
            "format_loss": format_loss,
            "soft_embeddings": soft_embeddings,
        }

    def _compute_diversity_score(self, semantic_ids: torch.Tensor) -> float:
        """
        Diversity score is to measure the codebook utilization.

        Within a batch, count the number of unique codebooks are used.

        Input:
            semantic_ids: torch.Tensor of shape (batch_size, seq_len)
        """
        if semantic_ids is None:
            return 0.0

        codebook_ids = self.codebook_token_ids.to(semantic_ids.device)

        # Keep only tokens that belong to the codebook vocabulary, dropping any
        # special/format tokens (e.g. <|eos_semantic|>) in the slice.
        flat_ids = semantic_ids.reshape(-1)
        flat_ids = flat_ids[torch.isin(flat_ids, codebook_ids)]
        if flat_ids.numel() == 0:
            return 0.0

        # Fraction of the codebook that is used at least once.
        num_unique = torch.unique(flat_ids).numel()
        active_codebook_ratio = num_unique / self.args.codebook_range
        return active_codebook_ratio

    def _perplexity_from_ids(
        self,
        ids: torch.Tensor,
        codebook_ids: torch.Tensor,
    ) -> float:
        """
        Perplexity of the code-usage distribution for a set of token ids.

            perplexity = exp(-sum_k p_k * log p_k)

        where ``p_k`` is the empirical probability of codebook entry ``k``
        being selected. Ranges from 1 (all mass on a single code) to
        ``codebook_range`` (uniform usage).
        """
        # Keep only tokens that belong to the codebook vocabulary, dropping any
        # special/format tokens (e.g. <|eos_semantic|>) in the slice.
        ids = ids.reshape(-1)
        ids = ids[torch.isin(ids, codebook_ids)]
        if ids.numel() == 0:
            return 0.0

        # Empirical distribution over the codebook entries.
        counts = torch.bincount(ids, minlength=int(codebook_ids.max().item()) + 1)
        counts = counts[codebook_ids].float()  # [codebook_range]
        probs = counts / counts.sum()

        # Entropy over used entries only (0 * log 0 -> 0).
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        return entropy.exp().item()

    def _compute_codebook_perplexity(self, semantic_ids: torch.Tensor) -> float:
        """
        Perplexity of codebook usage measures how evenly the codebook is
        utilized across a batch, pooling all code positions together.

        Perplexity ranges from 1 (the model collapses onto a single codebook)
        to ``codebook_range`` (all entries used uniformly), so a higher value
        indicates healthier codebook utilization.

        Input:
            semantic_ids: torch.Tensor of shape (batch_size, seq_len) holding
                token ids drawn from ``self.codebook_token_ids``.
        """
        if semantic_ids is None:
            return 0.0

        codebook_ids = self.codebook_token_ids.to(semantic_ids.device)
        return self._perplexity_from_ids(semantic_ids, codebook_ids)

    def _compute_per_position_perplexity(
        self,
        semantic_ids: torch.Tensor,
    ) -> list[float]:
        """
        Per-position codebook perplexity.

        Each item's semantic ID is ``codebook_size`` codes, each drawn from the
        shared ``codebook_range``-entry vocabulary. This computes perplexity
        independently for every position, so a single position collapsing onto
        a few codes is visible even when the pooled perplexity looks healthy.

        Input:
            semantic_ids: torch.Tensor of shape (batch_size, seq_len). Only the
                first ``codebook_size`` columns are generated code positions.

        Returns:
            A list of ``codebook_size`` perplexities, one per position.
        """
        if semantic_ids is None:
            return []

        codebook_ids = self.codebook_token_ids.to(semantic_ids.device)
        positions = semantic_ids[:, : self.args.codebook_size]
        return [
            self._perplexity_from_ids(positions[:, j], codebook_ids)
            for j in range(positions.shape[1])
        ]
