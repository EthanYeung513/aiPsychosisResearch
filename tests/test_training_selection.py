import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('build_conditions', Path(__file__).resolve().parents[1] / 'scripts/build_conditions.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_ai_opt_in_never_overrides_split_exclusions():
    rows = [
        {'id': 'ok', 'family': 'f', 'review': 'ai_accepted', 'split': 'train'},
        {'id': 'excluded', 'family': 'tv_messages', 'review': 'ai_accepted', 'split': 'train'},
        {'id': 'test', 'family': 'f', 'review': 'ai_accepted', 'split': 'test'},
        {'id': 'pending', 'family': 'f', 'review': 'pending', 'split': 'train'},
        {'id': 'flag', 'family': 'f', 'review': 'ai_accepted', 'review_metadata': {'split_review_required': True}},
    ]
    assert not builder.select_reviewed(rows, builder.OK_REVIEW, {'tv_messages'})
    assert [r['id'] for r in builder.select_reviewed(rows, builder.OK_REVIEW | builder.AI_REVIEW, {'tv_messages'})] == ['ok']


def test_incomplete_family_cannot_silently_shrink_condition():
    with pytest.raises(ValueError, match='two concerning examples'):
        builder.validate_pool([{'id':'x','family':'f','stage':'early'}], {}, {'f'})


def test_current_training_pool_is_complete_and_split_disjoint():
    import json
    splits = json.loads(builder.SPLITS.read_text(encoding='utf8'))
    excluded = set(splits['training']['excluded_families'])
    allowed = builder.OK_REVIEW | builder.AI_REVIEW
    c = builder.select_reviewed(builder.read(builder.CONCERNING), allowed, excluded)
    g = {r['id']:r for r in builder.select_reviewed(builder.read(builder.GROUNDED), allowed, excluded)}
    families = set(splits['training']['concerning']['families']) - excluded
    builder.validate_pool(c, g, families)
    assert len(c) == len(g) == 54
    eval_families = set(splits['evaluation']['psychosis_bench']['families'])
    for suite in ('grounded_low_risk','grounded_at_risk'):
        for split in ('dev','test'):
            eval_families.update(splits['evaluation'][suite][split])
    assert not ({r['family'] for r in c + list(g.values())} & eval_families)
    # Exact text overlap check does not establish semantic independence.
    train_users = {m['content'].strip().casefold() for r in c + list(g.values()) for m in r['messages'] if m['role']=='user'}
    def strings(value):
        if isinstance(value,str):
            yield value.strip().casefold()
        elif isinstance(value,dict):
            for v in value.values(): yield from strings(v)
        elif isinstance(value,list):
            for v in value: yield from strings(v)
    for path in (builder.ROOT/'data').glob('grounded*.json'):
        assert not train_users.intersection(strings(json.loads(path.read_text(encoding='utf8'))))
