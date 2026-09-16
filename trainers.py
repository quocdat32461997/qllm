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
        default=128, metadata={"help": "Code vocabulary cardinality; values are 0 through range-1."}
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
        default=0.1,
        metadata={"help": "Final temperature for Gumbel-softmax annealing."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to save memory."},
    )


class QuanSFTTrainer(Trainer):
    """Shared causal LM with an autoregressive straight-through code bottleneck."""

    def __init__(self, *args, tokenizer, **kwargs):
        from model_utils import add_semantic_tokens
        model = kwargs.get("model")
        config = kwargs.get("args")
        if model is None or config is None:
            raise ValueError("Pass model and args as keyword arguments")
        if config.codebook_size < 1 or config.codebook_range < 2:
            raise ValueError("Need at least one code position and two code values")
        if min(config.temperature_initial, config.temperature_final) <= 0:
            raise ValueError("Gumbel temperatures must be positive")
        add_semantic_tokens(tokenizer, model, config.codebook_range)
        kwargs["processing_class"] = tokenizer
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        # Our loss is already a mean; Trainer must apply accumulation scaling.
        self.model_accepts_loss_kwargs = False
        self.codebook_tokens = [f"<|CODE_{i}|>" for i in range(config.codebook_range)]
        self.codebook_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(self.codebook_tokens))
        self.bos_semantic_token_id = tokenizer.convert_tokens_to_ids(BOS_SEMANTIC_TOKEN)
        self.eos_semantic_token_id = tokenizer.convert_tokens_to_ids(EOS_SEMANTIC_TOKEN)

    def _tokenize(self, prompts, limit, device):
        from model_utils import tokenize_supervised
        return {k: v.to(device) for k, v in tokenize_supervised(self.tokenizer, prompts, limit).items()}

    def _slots(self, ids):
        # Last semantic BOS belongs to the scaffold, not the system instruction.
        indices = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        bos = indices.masked_fill(ids != self.bos_semantic_token_id, -1).max(-1).values
        slots = bos[:, None] + torch.arange(1, self.args.codebook_size + 1, device=ids.device)
        if (bos < 0).any() or (slots[:, -1] + 1 >= ids.shape[1]).any():
            raise ValueError("Missing or truncated semantic-ID scaffold")
        if not (ids.gather(1, slots) == self.eos_semantic_token_id).all():
            raise ValueError("Semantic-ID scaffold must contain exactly codebook_size placeholders")
        return slots

    def _get_temperature(self, logits):
        progress = min(self.state.global_step / max(self.state.max_steps, 1), 1.0)
        temperature = self.args.temperature_initial * (1 - progress) + self.args.temperature_final * progress
        return max(float(temperature), 1e-5)

    def _generate_semantic_ids(self, model, prompts):
        from model_utils import position_ids
        device = next(model.parameters()).device
        batch = self._tokenize(prompts, self.args.max_source_length, device)
        ids, mask = batch["input_ids"], batch["attention_mask"]
        slots = self._slots(ids)
        # Access the embedding module through the underlying model; all forward
        # passes still go through the Trainer's DDP/DeepSpeed wrapper.
        embed = self.model.get_input_embeddings()
        embeddings = embed(ids)
        code_ids = self.codebook_token_ids.to(device)
        code_embeddings = embed(code_ids)
        codes, soft = [], []
        for i in range(self.args.codebook_size):
            at = slots[:, i]
            end = int(at.max().item())
            prefix_mask = mask[:, :end] * (torch.arange(end, device=device)[None, :] < at[:, None])
            outputs = model(inputs_embeds=embeddings[:, :end], attention_mask=prefix_mask,
                            position_ids=position_ids(prefix_mask), use_cache=False)
            logits = outputs.logits[torch.arange(len(prompts), device=device), at - 1][:, code_ids].float()
            if model.training:
                probabilities = torch.nn.functional.gumbel_softmax(
                    logits, tau=self._get_temperature(logits), hard=True, dim=-1)
                chosen = probabilities.argmax(-1)
                chosen_embeddings = probabilities.to(code_embeddings.dtype) @ code_embeddings
            else:
                chosen = logits.argmax(-1)
                chosen_embeddings = code_embeddings[chosen]
            codes.append(code_ids[chosen])
            soft.append(chosen_embeddings)
            scatter_at = at[:, None, None].expand(-1, 1, embeddings.shape[-1])
            # Out-of-place updates keep earlier forwards valid for backprop.
            embeddings = embeddings.scatter(1, scatter_at, chosen_embeddings[:, None, :])
        semantic_ids = torch.stack(codes, dim=1)
        labels = batch["labels"].clone().scatter(1, slots, IGNORE_INDEX)
        format_loss = model(inputs_embeds=embeddings, attention_mask=mask,
                            position_ids=position_ids(mask), labels=labels, use_cache=False).loss
        return {"semantic_ids": semantic_ids,
                "semantic_id_texts": self.tokenizer.batch_decode(semantic_ids, skip_special_tokens=False),
                "format_loss": format_loss, "soft_embeddings": torch.stack(soft, dim=1)}

    def _reconstruct_input(self, model, prompts, soft_embeddings):
        from model_utils import position_ids
        batch = self._tokenize(prompts, self.args.max_target_length, soft_embeddings.device)
        embeddings = self.model.get_input_embeddings()(batch["input_ids"])
        slots = self._slots(batch["input_ids"]).unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        embeddings = embeddings.scatter(1, slots, soft_embeddings)
        return model(inputs_embeds=embeddings, attention_mask=batch["attention_mask"],
                     position_ids=position_ids(batch["attention_mask"]),
                     labels=batch["labels"], use_cache=False).loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        encoded = self._generate_semantic_ids(model, inputs["guessing_prompt"])
        reconstruction = self._reconstruct_input(model, inputs["reconstruction_prompt"], encoded["soft_embeddings"])
        loss = self.args.guessing_weight * encoded["format_loss"] + self.args.reconstruction_weight * reconstruction
        if model.training and self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({"format_loss": encoded["format_loss"].detach().item(),
                      "reconstruction_loss": reconstruction.detach().item(),
                      "semantic_id_diversity": self._compute_diversity_score(encoded["semantic_ids"]),
                      "semantic_id_perplexity": self._compute_codebook_perplexity(encoded["semantic_ids"]),
                      **{f"semantic_id_perplexity_pos_{i}": p for i, p in enumerate(
                          self._compute_per_position_perplexity(encoded["semantic_ids"]))}})
        return (loss, {"logits": encoded["semantic_ids"]}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        return loss.detach(), None if prediction_loss_only else outputs["logits"].detach(), None

    @torch.no_grad()
    def generate_semantic_ids(self, prompts):
        training = self.model.training
        self.model.eval()
        try:
            outputs = self._generate_semantic_ids(self.model, prompts)
            return outputs["semantic_ids"], outputs["semantic_id_texts"]
        finally:
            self.model.train(training)

    @torch.no_grad()
    def reconstruct_from_semantic_ids(self, prompts, semantic_ids):
        from model_utils import chat_ids
        training = self.model.training
        self.model.eval()
        try:
            rows = []
            for prompt in prompts:
                ids = chat_ids(self.tokenizer, prompt[:-1], generation=True)
                if len(ids) > self.args.max_target_length:
                    raise ValueError("Reconstruction prompt exceeds max_target_length")
                rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            batch = self.tokenizer.pad(rows, padding=True, return_tensors="pt").to(self.model.device)
            slots = self._slots(batch["input_ids"])
            batch["input_ids"] = batch["input_ids"].scatter(1, slots, semantic_ids.to(self.model.device))
            options = {"do_sample": self.args.generation_do_sample}
            if self.args.generation_do_sample:
                options.update(temperature=self.args.generation_temperature, top_p=self.args.generation_top_p)
            generated = self.model.generate(**batch, **options,
                max_new_tokens=self.args.generation_max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True)
            return self.tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            self.model.train(training)

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
