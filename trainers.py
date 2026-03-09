import asyncio
import atexit
import copy
import importlib.resources as pkg_resources
import inspect
import os
import sys
import textwrap
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
import torch.utils.data
import transformers
from accelerate.logging import get_logger
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from huggingface_hub import CommitScheduler, DatasetCard, DatasetCardData, create_repo
from packaging.version import Version
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import (
    is_datasets_available,
    is_peft_available,
    is_rich_available,
)
from trl import BaseConfig, BaseTrainer, GRPOTrainer, SFTTrainer
from trl.callbacks import SyncRefModelCallback
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    parse_response,
)
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.generation.vllm_generation import VLLMGeneration
from trl.import_utils import is_jmespath_available, is_liger_kernel_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection, disable_gradient_checkpointing
from trl.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    shutdown_event_loop_in_daemon,
    split_pixel_values_by_grid,
    split_tensor_dict,
    start_event_loop_in_daemon,
    unsplit_pixel_values_by_grid,
    use_adapter,
)

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

if is_trackio_available():
    import trackio

logger = get_logger(__name__)


@dataclass
class QuantConfig(BaseConfig):
    codebook_range: int = field(default=256, metadata={"help": "Codebook range"})
    codebook_size: int = field(default=6, metadata={"help": "Codebook size"})


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]
RolloutFunc = Callable[[list[str], "GRPOTrainer"], dict[str, Any]]


class QuanTrainer(BaseTrainer):
    """
    Base Quantization RL Trainer to encode and decode text codebooks.
    """

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        args: QuantConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: (
            Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
        ) = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: (
            PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None
        ) = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ):
        # Args
        if args is None:
            model_name = (
                model
                if isinstance(model, str)
                else get_config_model_id(model.config)  # noqa
            )
            model_name = model_name.split("/")[-1]
            args = QuantConfig(f"{model_name}-GRPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `QuantConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config),
                truncation_side="left",
                padding_side="left",
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )

        if is_peft_available() and is_peft_model(model) and args.beta != 0.0:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during GRPO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(
            model, "is_loaded_in_8bit", False
        ):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Rollout function
        if (
            rollout_func is not None
            and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1"
        ):
            warnings.warn(
                "You are using 'rollout_func', which is an experimental feature. This API may change or be removed at "
                "any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func
        if (
            environment_factory is not None
            and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1"
        ):
            warnings.warn(
                "You are using 'environment_factory', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )

        # Tools
        if tools:
            if not Version(transformers.__version__) >= Version("5.0.0"):
                raise ImportError(
                    "Using tools with GRPOTrainer requires transformers version 5.0.0 or higher. Please upgrade "
                    "transformers with `pip install --upgrade transformers` to use this feature."
                )
        if environment_factory:
            if not Version(transformers.__version__) >= Version("5.2.0"):
                raise ImportError(
                    "Using `environment_factory` with GRPOTrainer requires transformers version 5.2.0 or higher. "
                    "Please install transformers from the main branch with `pip install "
                    "git+https://github.com/huggingface/transformers.git@main` to use this feature."
                )
        if tools or environment_factory:
            if not is_jmespath_available():
                raise ImportError(
                    "Using tools with GRPOTrainer requires the jmespath library for response parsing. Please install "
                    "it with `pip install jmespath` to use this feature."
                )

        # Create the environments and extract their methods to be used as tools. We create one environment per rollout
        generation_batch_size = (
            args.per_device_train_batch_size * args.steps_per_generation
        )
        if environment_factory is not None:
            self.environments = [
                environment_factory() for _ in range(generation_batch_size)
            ]
            environment_methods = [[] for _ in range(generation_batch_size)]
            for i, environment in enumerate(self.environments):
                has_reset = False
                for name, member in inspect.getmembers(
                    environment, predicate=inspect.ismethod
                ):
                    if name == "reset":
                        has_reset = True
                    elif not name.startswith("_"):
                        environment_methods[i].append(member)
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define a callable `reset` "
                    )
        else:
            self.environments = None

        tools = tools or []
        self._sync_tool_dicts = [{} for _ in range(generation_batch_size)]
        self._async_tool_dicts = [{} for _ in range(generation_batch_size)]
        for i in range(generation_batch_size):
            for tool in tools + (
                environment_methods[i] if self.environments is not None else []
            ):
                if asyncio.iscoroutinefunction(tool):
                    self._async_tool_dicts[i][tool.__name__] = tool
                else:
                    self._sync_tool_dicts[i][tool.__name__] = tool

        self.tools = tools + (
            environment_methods[0] if self.environments is not None else []
        )

        # Check for async functions to start an event loop on a daemon thread
        self._has_async_funcs = any(
            asyncio.iscoroutinefunction(func) for func in self.reward_funcs + self.tools
        )

        if self._has_async_funcs:
            self.async_loop_thread, self.async_loop, self.async_loop_ready_event = (
                start_event_loop_in_daemon(name="GRPOTrainer-AsyncLoop")
            )
            # wait until the event loop is running in the daemon thread
            self.async_loop_ready_event.wait()
            atexit.register(
                shutdown_event_loop_in_daemon, self.async_loop_thread, self.async_loop
            )

        # At the time of initial implementation, most tokenizers do not have built-in support for response schemas.
        # While waiting for broader adoption, we provide this utility function to manually set the response schema for
        # known chat templates.
        # We need `getattr`` until the base class sets a default None value for response_schema
        if self.tools and not getattr(processing_class, "response_schema", None):
            processing_class = add_response_schema(processing_class)
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.
        if self.tools:
            self.chat_template = get_training_chat_template(processing_class)
        else:
            self.chat_template = None

        # Training arguments
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_tool_calling_iterations = (
            args.max_tool_calling_iterations or sys.maxsize
        )
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = (
            args.vllm_gpu_memory_utilization
        )  # only applies to colocation mode
        self.vllm_tensor_parallel_size = (
            args.vllm_tensor_parallel_size
        )  # only applies to colocation mode
        self.vllm_importance_sampling_correction = (
            args.vllm_importance_sampling_correction
        )
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_kernel = args.use_liger_kernel
        self.loss_type = args.loss_type
        self.multi_objective_aggregation = args.multi_objective_aggregation
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.off_policy_mask_threshold = args.off_policy_mask_threshold
        if self.use_liger_kernel and self.off_policy_mask_threshold is not None:
            raise ValueError(
                "Liger kernel does not support off-policy sequence masking yet."
            )
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_kernel and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_kernel and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict)
                and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        if args.loss_type == "luspo" and args.importance_sampling_level != "sequence":
            logger.warning(
                "When using `'luspo'` loss, `importance_sampling_level` should be set to `'sequence'` to mirror the "
                "paper's setup."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version(
            "5.0.0"
        ):
            args.gradient_checkpointing_kwargs = (
                args.gradient_checkpointing_kwargs or {}
            )
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if self.args.distributed_state.distributed_type in [
                "MULTI_GPU",
                "DEEPSPEED",
            ]:
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(
                get_config_model_id(self.model.config), **model_init_kwargs
            )

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                def cast_inputs_to_fp32(module, inputs):
                    # Preserve other positional args and kwargs untouched
                    if not inputs:
                        return inputs
                    return (inputs[0].to(torch.float32),) + inputs[1:]

                original_dtype_local = target_model.lm_head.weight.dtype
                target_model.lm_head = target_model.lm_head.float()
                target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 (intentional for numerical stability of logits)
                    target_model.model.embed_tokens.register_forward_hook(
                        cast_outputs_to_original_dtype
                    )

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # Initialize vLLM generation backend
            # Wrap rollout_func to capture trainer context if provided
            rollout_func = None
            if self.rollout_func is not None:

                def rollout_func(prompts):
                    return self.rollout_func(prompts, self)

            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                # vLLM configuration
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                # Server mode configuration
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                # Colocate mode configuration
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.steps_per_generation,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                # Generation configuration
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=0,  # we only need the generated token logprobs for the importance sampling correction
                generation_kwargs=args.generation_kwargs,
                # Chat/tool configuration
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
                tools=self.tools,
                rollout_func=rollout_func,
            )
            self._last_loaded_step = (
                -1
            )  # tag to avoid useless loading during grad accumulation
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            if self.beta == 0.0:
                raise ValueError(
                    "You passed `sync_ref_model=True` while `beta=0.0`, which means the reference model is not used "
                    "during training. Consequently, GRPOTrainer does not create a `ref_model` instance, and there is "
                    "nothing to synchronize. Please set `sync_ref_model=False`, or set `beta` to a non-zero value."
                )
            if is_peft_model(model):
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, GRPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(
                        reward_func, self.accelerator
                    )
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(
                os.path.join(self.args.output_dir, "completions"), exist_ok=True
            )
            if self.args.log_completions_hub_repo is not None:
                repo_id = self.args.log_completions_hub_repo
                create_repo(
                    repo_id,
                    private=self.args.hub_private_repo,
                    repo_type="dataset",
                    exist_ok=True,
                )
                template_path = pkg_resources.files("trl").joinpath(
                    "templates/completions_dataset_card.md"
                )
                card_data = DatasetCardData(
                    pretty_name="TRL Completion logs",
                    tags=["trl", "trl-logs", "completions"],
                )
                card = DatasetCard.from_template(
                    card_data=card_data,
                    template_path=str(template_path),
                    repo_id=repo_id,
                    hub_model_id=self.args.hub_model_id,
                )
                card.push_to_hub(repo_id)
                self.commit_scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=f"{self.args.output_dir}/completions",
                    every=2,  # minutes
                    allow_patterns=["*.parquet"],
                )

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:

        model = "train" if self.model.training else "eval"
        if model == "train":
            generation_batch = self._generate_and_score_codebooks(
                generation_batch
            )  # noqa
        else:
            pass

    def _generate_and_score_codebooks(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # Generate codebooks
        (prompt_ids_list, codebook_list) = self._generate(prompts)

        # Convert lists of tokends IDs to padded tensors
        prompt_ids = [
            torch.tensor(ids, device=device) for ids in prompt_ids_list
        ]  # noqa
        prompt_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids
        ]  # noqa
        prompt_ids = pad(
            prompt_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        # Reconstruct completions from codebooks
        pass

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        prompts, completion_ids, logprobs, extra_fields = None

        return None

    def training_step(self, model, inputs):
        """
        Custom training step with two steps:
        1. Forward: Generate 6 tokens following the input.
        2. Reconstruct: Use generated tokens to reconstruct the original input.
        """
        # Step 1: Generate 6 tokens
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Generate 6 tokens following the input
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=6,
            do_sample=True,  # Assuming sampling for RL-like behavior
            pad_token_id=(
                self.tokenizer.pad_token_id if hasattr(self, "tokenizer") else None
            ),
        )

        # Extract the generated tokens (beyond the input)
        generated_tokens = generated_outputs[
            :, input_ids.shape[1] :
        ]  # Assuming batch size

        # Step 2: Reconstruct the input using the generated tokens
        # Placeholder: assuming model predicts original input from generated tokens
        reconstruction_logits = model(
            generated_tokens
        )  # Adjust based on model architecture

        # Compute cross-entropy loss between input and reconstruction
        labels = input_ids  # Reconstruction targets the original input
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            reconstruction_logits.view(-1, reconstruction_logits.size(-1)),
            labels.view(-1),
        )

        return loss
