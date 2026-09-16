import json
import pytest
from amazon2014 import read_records, filter_k_core, sequence_splits, validate_prepared
from data_module import build_amazon_datasets, build_product_text


def test_chronology_no_leakage(prepared):
    manifest = validate_prepared(prepared)
    assert (manifest['users'], manifest['items'], manifest['interactions']) == (5, 5, 25)
    train = list(read_records(prepared / 'train.jsonl'))
    assert len(train) == 10
    assert {r['target'] for r in train} == {'p1', 'p2'}
    assert all('p3' not in r['history'] and 'p4' not in r['history'] for r in train)
    valid = list(read_records(prepared / 'validation.jsonl'))
    test = list(read_records(prepared / 'test.jsonl'))
    assert valid[0]['history'] == ['p0', 'p1', 'p2'] and valid[0]['target'] == 'p3'
    assert test[0]['history'] == ['p0', 'p1', 'p2', 'p3'] and test[0]['target'] == 'p4'
    assert 'NEVER INCLUDE ME' not in (prepared / 'products.jsonl').read_text()


def test_iterative_core_and_history_cap():
    events = [('a','x',0,0), ('a','y',1,1), ('b','y',2,2), ('b','z',3,3)]
    assert filter_k_core(events, 2, 2) == []
    events = [('u', str(i), i, i) for i in range(25)]
    splits = sequence_splits(events, 20)
    assert len(splits['test'][0]['history']) == 20
    assert splits['test'][0]['history'][0] == '4'
    assert max(int(r['target']) for r in splits['train']) == 22


def test_parser_does_not_execute(tmp_path):
    path = tmp_path / 'evil.json'
    path.write_text("__import__('os').system('touch should-never-exist')\n")
    with pytest.raises(ValueError, match='Invalid record'):
        list(read_records(path))


def test_limits_and_manifest(config, prepared):
    config.update(max_train_samples=2, max_eval_samples=1)
    train, val = build_amazon_datasets(config)
    assert len(train) == 2 and len(val) == 1 and train[0]['asin']
    assert len(build_product_text('x'*100, ['y'*100], True, 30)) == 30
    with (prepared / 'train.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='changed'):
        validate_prepared(prepared)
