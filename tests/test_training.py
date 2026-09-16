import json
from pathlib import Path

import pytest
import torch
from transformers import Trainer, TrainingArguments

from data_module import build_amazon_datasets, QuantDataCollator
from export_semantic_ids import build_index, export_index
from model_utils import load_checkpoint, tokenize_supervised
from recommendation import (add_item_tokens, CatalogTrie, load_index, RecommendationCollator,
    RecommendationDataset, evaluate_recommendations, ranking_metrics)
from trainers import QuanSFTTrainer, QuantConfig


def make_trainer(tiny_model, config):
    model, tokenizer = tiny_model
    train, val = build_amazon_datasets(config)
    args = QuantConfig(**config['trainer'], codebook_size=2, codebook_range=4,
                       max_source_length=512, max_target_length=512)
    return QuanSFTTrainer(model=model, tokenizer=tokenizer, args=args, train_dataset=train,
                         eval_dataset=val, data_collator=QuantDataCollator())


def test_semantic_gradient_masking_and_determinism(tiny_model, config):
    trainer = make_trainer(tiny_model, config)
    rows = [trainer.train_dataset[i] for i in range(2)]
    batch = QuantDataCollator()(rows)
    trainer.model.train()
    selector_outputs = []
    def capture_selector(module, inputs, output):
        output.retain_grad()
        selector_outputs.append(output)
    handle = trainer.model.lm_head.register_forward_hook(capture_selector)
    encoded = trainer._generate_semantic_ids(trainer.model, batch['guessing_prompt'])
    handle.remove()
    assert encoded['semantic_ids'].shape == (2, 2)
    assert torch.isin(encoded['semantic_ids'], trainer.codebook_token_ids).all()
    encoded['soft_embeddings'].retain_grad()
    loss = trainer._reconstruct_input(trainer.model, batch['reconstruction_prompt'], encoded['soft_embeddings'])
    loss.backward()
    assert torch.isfinite(loss) and encoded['soft_embeddings'].grad.abs().sum() > 0
    # Reconstruction reaches the selector LM head through Gumbel, not only the decoder embeddings.
    assert selector_outputs[0].grad is not None and selector_outputs[0].grad.abs().sum() > 0
    ids1, _ = trainer.generate_semantic_ids(batch['guessing_prompt'])
    ids2, _ = trainer.generate_semantic_ids(batch['guessing_prompt'])
    assert torch.equal(ids1, ids2) and trainer.model.training
    solo, _ = trainer.generate_semantic_ids([batch['guessing_prompt'][0]])
    assert torch.equal(ids1[0], solo[0])
    tokens = tokenize_supervised(trainer.tokenizer, batch['reconstruction_prompt'], 512)
    assert (tokens['labels'][tokens['attention_mask'] == 0] == -100).all()
    assert (tokens['labels'] != -100).sum() > 0
    with pytest.raises(ValueError, match='exceeding'):
        tokenize_supervised(trainer.tokenizer, batch['guessing_prompt'], 10)
    trainer.model.zero_grad()
    result = trainer.train()
    assert result.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
    assert 'eval_loss' in trainer.evaluate()


def test_collisions_and_ranking(tiny_model):
    model, tokenizer = tiny_model
    from model_utils import add_semantic_tokens
    add_semantic_tokens(tokenizer, model, 4)
    index = build_index(['b', 'a', 'c'], [[0, 1], [0, 1], [2, 3]], 2, 4)
    add_item_tokens(model, tokenizer, index)
    assert index['items']['a']['suffix'] == 0 and index['items']['b']['suffix'] == 1
    trie = CatalogTrie(tokenizer, index)
    assert len(trie.reverse) == 3
    for ids, asin in trie.reverse.items():
        assert trie.decode(ids) == asin
        for i, token in enumerate(ids):
            assert token in trie.allowed(ids[:i])
    metrics = ranking_metrics([['a', 'a', 'b'], ['x']], ['b', 'missing'], (1, 2))
    assert metrics['recall@1'] == 0 and metrics['recall@2'] == 0.5
    assert metrics['ndcg@2'] == pytest.approx(0.5 / 1.584962500721156)


@pytest.mark.parametrize('lora,rec_lora', [(False, False), (False, True), (True, False), (True, True)])
def test_end_to_end_checkpoint_handoff(tiny_model, config, tmp_path, lora, rec_lora):
    from train import run_training as semantic_train
    from train_recommendation import run_training as rec_train
    model, tokenizer = tiny_model
    base = tmp_path / 'base'
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    config['model_name'] = str(base)
    config['use_lora'] = lora
    config['recommendation']['use_lora'] = rec_lora
    config['lora'] = dict(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.0, bias='none', task_type='CAUSAL_LM', modules_to_save=['embed_tokens', 'lm_head'],
        ensure_weight_tying=True)
    semantic_checkpoint = semantic_train(config)
    index_path = tmp_path / 'index.json'
    index = export_index(config, semantic_checkpoint, index_path, batch_size=2)
    assert len(index['items']) == 5
    before, _ = load_checkpoint(config, semantic_checkpoint)
    if hasattr(before, 'merge_and_unload'):
        before = before.merge_and_unload()
    before_query = before.model.layers[0].self_attn.q_proj.weight.detach().clone()
    rec_checkpoint = rec_train(config, semantic_checkpoint, index_path)
    restored, tok = load_checkpoint(config, rec_checkpoint)
    if not rec_lora:
        assert not torch.equal(before_query, restored.model.layers[0].self_attn.q_proj.weight)
    frozen = load_index(Path(rec_checkpoint) / 'semantic_index.json', config['data_dir'])
    dataset = RecommendationDataset(Path(config['data_dir']) / 'test.jsonl', frozen)
    metrics = evaluate_recommendations(restored, tok, dataset, frozen, 5, (1, 5), 2, 512)
    assert metrics['examples'] == 5 and metrics['invalid_sequences'] == 0
    assert metrics['recall@5'] == 1
    assert Path(rec_checkpoint, 'experiment.yaml').exists()
    if lora and rec_lora:
        adapter = json.loads(Path(rec_checkpoint, 'adapter_config.json').read_text())
        assert adapter['base_model_name_or_path'].endswith('stage1_base')
