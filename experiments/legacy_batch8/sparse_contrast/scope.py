"""Exact dataset-specific registry scope, independent of scientific run configs."""
from __future__ import annotations

import json
import os
from pathlib import Path

ALL_REPRESENTATIONS = ('single_color', 'grayscale', 'color_opponency')
DATASETS = ('mnist', 'cifar10')
ARCHITECTURES = ('vit_small', 'swin_tiny')


def read_scope(root=None):
    root = Path(root) if root is not None else Path(os.environ.get(
        'SPARSE_CONTRAST_ROOT', Path(__file__).resolve().parents[1]))
    path = root / 'experiment_scope.json'
    scope = json.loads(path.read_text()) if path.exists() else {
        'schema_version': 1,
        'active_representations': {dataset: list(ALL_REPRESENTATIONS) for dataset in DATASETS},
    }
    if scope.get('schema_version') != 1:
        raise ValueError('Unsupported experiment scope schema')
    mapping = scope.get('active_representations', {})
    if set(mapping) != set(DATASETS):
        raise ValueError('Scope must explicitly cover MNIST and CIFAR-10')
    for dataset, values in mapping.items():
        if not values or len(set(values)) != len(values) or any(v not in ALL_REPRESENTATIONS for v in values):
            raise ValueError(f'Invalid representation scope for {dataset}')
    return scope


def representations(dataset, scope=None):
    scope = read_scope() if scope is None else scope
    return tuple(scope['active_representations'][dataset])


def configurations(dataset, scope=None):
    reps = representations(dataset, scope)
    return [('raw', 'dense', None)] + [(rep, 'dense', 0) for rep in reps] + [
        (rep, 'compact', p) for rep in reps for p in (0, 20, 40, 60, 80)]


def expected_group_count(dataset, scope=None):
    return 3 * len(configurations(dataset, scope))


def expected_main_count(scope=None):
    scope = read_scope() if scope is None else scope
    return len(ARCHITECTURES) * sum(expected_group_count(d, scope) for d in DATASETS)


def expected_workers_per_block(dataset, scope=None):
    return sum(2 if execution == 'compact' else 1
               for _, execution, _ in configurations(dataset, scope))


def expected_worker_count(scope=None):
    scope = read_scope() if scope is None else scope
    return len(ARCHITECTURES) * 3 * 2 * sum(expected_workers_per_block(d, scope) for d in DATASETS)


def validate_registry(runs, scope=None):
    scope = read_scope() if scope is None else scope
    configs = [item.get('config', item) for item in runs]
    expected = {(d, a, rep, execution, p, seed)
                for d in DATASETS for a in ARCHITECTURES
                for rep, execution, p in configurations(d, scope) for seed in (0, 1, 2)}
    actual = [(c['dataset'], c['architecture'], c['representation'], c['execution'],
               c['sparsity_percent'], c['seed']) for c in configs]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Registry does not exactly match the authorized dataset-specific scope')
    if any('registry_id' in c for c in configs):
        ids = [c['registry_id'] for c in configs]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate registry identities')
    return configs
