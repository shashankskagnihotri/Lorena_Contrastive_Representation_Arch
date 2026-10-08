"""Separate immutable training identities and inference interventions."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from . import PROJECT, BASE_ROOT, BASE_SOURCE_HASH
from sparse_contrast.common import (atomic_json, file_hash, digest, directory_lock,
                                    seed_all, validate_config_artifacts, run_dir)

ROOT = Path(os.environ.get('DENSE_SPARSE_ROOT', PROJECT / 'experiments' / 'training_dense_testing_sparse')).resolve()
PYTHON = sys.executable
STUDY_ID = 'training_dense_testing_sparse'


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def stop_requested():
    return (ROOT / 'STOP').exists() or (BASE_ROOT / 'STOP').exists()


def verify_source():
    executing = Path(__file__).resolve().parents[1]
    path = executing / 'source_manifest.json'
    if not path.is_file():
        raise RuntimeError('Production requires an immutable inference source snapshot')
    manifest = read_json(path)
    for name, expected in manifest.items():
        if file_hash(executing / name) != expected:
            raise ValueError(f'Inference source changed: {name}')
    return digest(manifest)


def load_manifest():
    value = read_json(ROOT / 'study_manifest.json')
    if digest({k:v for k,v in value.items() if k != 'manifest_hash'}) != value['manifest_hash']:
        raise ValueError('Study manifest digest mismatch')
    if value['study_id'] != STUDY_ID or value['base_source_hash'] != BASE_SOURCE_HASH:
        raise ValueError('Wrong study/base computation identity')
    return value


def load_case(value):
    if isinstance(value, dict):
        case = value
    else:
        path = Path(value)
        if not path.is_file():
            path = ROOT / 'configs' / f'{value}.json'
        case = read_json(path)
    if digest({k:v for k,v in case.items() if k != 'test_config_hash'}) != case['test_config_hash']:
        raise ValueError('Inference case digest mismatch')
    return case


def case_dir(case):
    case = load_case(case)
    return ROOT / 'outputs/cases' / case['dataset'] / case['architecture'] / case['representation'] / ('trained_' + case['trained_execution']) / ('infer_' + case['inference_execution']) / ('p' + str(case['inference_sparsity_percent'])) / ('seed' + str(case['seed'])) / case['test_config_hash']


def effective_config(case, original=None):
    """For evaluation/measurement control flow only, never checkpoint validation."""
    original = read_json(case['training_config_path']) if original is None else original
    return dict(original, execution=case['inference_execution'],
                sparsity_percent=case['inference_sparsity_percent'])


def validate_case(case):
    case = load_case(case)
    manifest = load_manifest()
    if case['source_hash'] != verify_source() or case['source_hash'] != manifest['source_hash']:
        raise ValueError('Inference code does not match registered study')
    registered = {x['test_config_hash']:x for x in manifest['cases']}
    if registered.get(case['test_config_hash']) != case:
        raise ValueError('Case not registered in the complete study manifest')
    config = read_json(case['training_config_path'])
    validate_config_artifacts(config)
    for name, original_name in [('training_config_hash','config_hash'), ('training_source_hash','source_hash'),
                                ('normalization_hash','normalization_hash'),('dataset','dataset'),
                                ('architecture','architecture'),('representation','representation'),('seed','seed'),
                                ('trained_execution','execution')]:
        if case[name] != config[original_name]:
            raise ValueError(f'Original training identity changed: {name}')
    if config['role'] != 'main' or config['protocol'] != 'heldout_val':
        raise ValueError('Only main heldout-val checkpoints are eligible')
    if config['representation'] != 'raw' and config['sparsity_percent'] != 0:
        raise ValueError('Only zero-imposed-sparsity training checkpoints are eligible')
    if config['dataset'] == 'mnist' and config['representation'] not in ('raw','grayscale'):
        raise ValueError('MNIST color conditions are excluded')
    checkpoint = run_dir(config) / 'final.pt'
    completed_path = checkpoint.parent / 'completed.json'
    if str(checkpoint) != case['checkpoint_path'] or file_hash(completed_path) != case['completed_receipt_sha256']:
        raise ValueError('Original final-checkpoint receipt changed')
    done = read_json(completed_path)
    if done['checkpoint_sha256'] != case['checkpoint_sha256'] or file_hash(checkpoint) != case['checkpoint_sha256']:
        raise ValueError('Original checkpoint hash mismatch')
    return config


def load_model(case, device):
    import torch
    from sparse_contrast.train import load_stats
    from .adapter import InferenceClassifier
    case = load_case(case)
    original = validate_case(case)
    state = torch.load(case['checkpoint_path'], map_location='cpu', weights_only=False)
    if state['config_hash'] != original['config_hash'] or state['normalization_hash'] != original['normalization_hash']:
        raise ValueError('Checkpoint embedded config/normalization identity mismatch')
    model = InferenceClassifier(original, load_stats(original), case)
    model.load_state_dict(state['model'], strict=True)
    del state
    return model.to(device).eval()
