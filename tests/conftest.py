import gzip
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from amazon2014 import prepare_category


@pytest.fixture
def prepared(tmp_path):
    reviews, metadata = tmp_path / 'reviews.json.gz', tmp_path / 'meta.json.gz'
    with gzip.open(reviews, 'wt') as stream:
        for u in range(5):
            for i in (3, 0, 4, 1, 2):
                stream.write(json.dumps({'reviewerID': f'u{u}', 'asin': f'p{i}', 'unixReviewTime': i,
                                         'reviewText': 'NEVER INCLUDE ME'}) + '\n')
    with gzip.open(metadata, 'wt') as stream:
        for i in range(5):
            stream.write(repr({'asin': f'p{i}', 'title': 'product ' + 'good ' * (i + 1),
                              'description': 'feature useful', 'brand': 'brand',
                              'categories': [['Beauty', 'Care']]}) + '\n')
    directory = tmp_path / 'processed'
    prepare_category('Beauty', reviews, metadata, directory)
    return directory


@pytest.fixture
def tiny_model():
    torch.set_num_threads(1)
    vocabulary = ['[UNK]', '[PAD]', '<|im_start|>', '<|im_end|>', 'system', 'user', 'assistant',
                  'product', 'good', 'feature', 'useful', 'brand', 'Beauty', 'Care', '.', ':', '\n']
    tok = Tokenizer(WordLevel({v: i for i, v in enumerate(vocabulary)}, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]',
        eos_token='<|im_end|>', additional_special_tokens=['<|im_start|>'], padding_side='left')
    tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=32,
        intermediate_size=48, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, attention_dropout=0.0, tie_word_embeddings=False))
    return model, tokenizer


@pytest.fixture
def config(prepared, tmp_path):
    trainer = dict(output_dir=str(tmp_path / 'semantic'), per_device_train_batch_size=2,
                   per_device_eval_batch_size=2, max_steps=2, learning_rate=1e-3,
                   gradient_accumulation_steps=2, report_to=[], use_cpu=True, bf16=False,
                   save_strategy='no', logging_steps=1, disable_tqdm=True,
                   gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    return dict(dataset_version='2014', data_dir=str(prepared), model_name='unused',
                codebook_size=2, codebook_range=4, max_num_chars=150, max_source_length=512,
                max_target_length=512, seed=42, feature_probability=1.0, eval_split_ratio=0.2,
                trainer=trainer, recommendation=dict(max_history=20, max_length=512,
                    trainer={**trainer, 'output_dir': str(tmp_path / 'recommendation')}))
