"""CPU evidence checks for the real workers, without launching an experiment."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dense_sparse import worker


@pytest.fixture
def cell_evidence(tmp_path):
    labels = np.arange(10000, dtype=np.uint16) % 10
    labels = labels.astype(np.uint8)
    predictions = labels.copy()
    predictions[::11] = (predictions[::11] + 1) % 10
    ids = [f'mnist/test/{index:05d}' for index in range(10000)]
    dataset = SimpleNamespace(labels=labels, sample_ids=ids, split_hash=worker.ids_hash(ids))
    identity = {'test_config_hash': 'test', 'training_config_hash': 'original',
                'inference_sparsity_percent': 50, 'inference_execution': 'compact'}
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    correct = int(np.trace(confusion))
    row = dict(state='complete', identity=identity, corruption='clean', severity='test',
               dataset_split_hash=dataset.split_hash, sample_ids_sha256=worker.ids_hash(ids),
               correct=correct, total=10000, accuracy=correct / 10000, error=1 - correct / 10000,
               loss=0.4, confusion=confusion.tolist(),
               per_class_accuracy=(np.diag(confusion) / 1000).tolist())
    path = tmp_path / 'clean__test.json'
    arrays = dict(sample_id=np.asarray(ids), label=labels, prediction=predictions,
                  correct=labels == predictions)
    worker._atomic_npz(path.with_suffix('.npz'), **arrays)
    token_records = [dict(sample_ids=ids[start:start + 5000], metrics={'retained_patches': [10] * 5000})
                     for start in (0, 5000)]
    worker.benchmark._write_jsonl(path.with_suffix('.tokens.jsonl'), token_records)
    row.update(predictions_sha256=worker.file_hash(path.with_suffix('.npz')),
               tokens_sha256=worker.file_hash(path.with_suffix('.tokens.jsonl')))
    worker.atomic_json(path, row)
    return path, row, arrays, dataset, token_records


def test_official_counts_confusion_and_ordered_ids_reconcile(cell_evidence):
    path, row, _arrays, dataset, _tokens = cell_evidence
    assert worker._evaluation_cell(path, dataset, row['identity'], 'clean', 'test') == row
    assert row['total'] == 10000
    assert row['correct'] == 10000 - len(range(0, 10000, 11))


@pytest.mark.parametrize('field', ['sample_id', 'label', 'prediction', 'correct'])
def test_prediction_semantics_fail_even_if_file_hash_is_recommitted(cell_evidence, field):
    path, row, arrays, dataset, _tokens = cell_evidence
    if field == 'sample_id':
        arrays[field][[0, 1]] = arrays[field][[1, 0]]
    elif field == 'correct':
        arrays[field][0] = not arrays[field][0]
    else:
        arrays[field][0] = 255
    worker._atomic_npz(path.with_suffix('.npz'), **arrays)
    row['predictions_sha256'] = worker.file_hash(path.with_suffix('.npz'))
    worker.atomic_json(path, row)
    with pytest.raises(ValueError):
        worker._evaluation_cell(path, dataset, row['identity'], 'clean', 'test')


@pytest.mark.parametrize('field', ['correct', 'accuracy', 'error', 'confusion', 'per_class_accuracy', 'loss'])
def test_summary_metrics_cannot_disagree_with_predictions(cell_evidence, field):
    path, row, _arrays, dataset, _tokens = cell_evidence
    if field == 'confusion':
        row[field][0][0] += 1
    elif field == 'per_class_accuracy':
        row[field][0] += 0.01
    elif field == 'loss':
        row[field] = float('nan')
    else:
        row[field] += 0.1
    with pytest.raises(ValueError):
        worker._verify_predictions(path.with_suffix('.npz'), dataset, row)


@pytest.mark.parametrize('file_suffix', ['.npz', '.tokens.jsonl'])
def test_saved_file_hash_is_required(cell_evidence, file_suffix):
    path, row, _arrays, dataset, _tokens = cell_evidence
    with path.with_suffix(file_suffix).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='files changed'):
        worker._evaluation_cell(path, dataset, row['identity'], 'clean', 'test')


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'reordered'])
def test_token_coverage_requires_each_ordered_official_id(cell_evidence, change):
    path, row, _arrays, dataset, tokens = cell_evidence
    if change == 'missing':
        tokens[-1]['sample_ids'].pop()
    elif change == 'duplicate':
        tokens[-1]['sample_ids'][-1] = tokens[0]['sample_ids'][0]
    else:
        tokens.reverse()
    worker.benchmark._write_jsonl(path.with_suffix('.tokens.jsonl'), tokens)
    row['tokens_sha256'] = worker.file_hash(path.with_suffix('.tokens.jsonl'))
    worker.atomic_json(path, row)
    with pytest.raises(ValueError, match='Token records'):
        worker._evaluation_cell(path, dataset, row['identity'], 'clean', 'test')


def test_inference_intervention_is_part_of_resume_identity(cell_evidence):
    path, row, _arrays, dataset, _tokens = cell_evidence
    different = {**row['identity'], 'inference_sparsity_percent': 60}
    with pytest.raises(ValueError, match='identity'):
        worker._evaluation_cell(path, dataset, different, 'clean', 'test')


def test_identity_separates_training_and_inference_source(monkeypatch):
    case = dict(test_config_hash='testhash', training_config_hash='trainhash',
                checkpoint_path='/original/final.pt', checkpoint_sha256='checkpoint',
                completed_receipt_sha256='completed', training_source_hash='trainingcode',
                normalization_hash='norm', source_hash='newcode', dataset='mnist',
                architecture='vit_small', representation='grayscale', seed=1,
                trained_execution='dense', inference_execution='compact',
                inference_sparsity_percent=50, sparsification_domain='native_contrast',
                training_config_path='/original/config.json')
    original = dict(sparsity_percent=0, data_manifest_sha256='data', environment_hash='env',
                    protocol='heldout_val', split_seed=2026, precision='fp32')
    monkeypatch.setattr(worker, 'verify_source', lambda: 'newcode')
    identity = worker._identity(case, original)
    assert identity['trained_execution'] == 'dense'
    assert identity['trained_sparsity_percent'] == 0
    assert identity['inference_sparsity_percent'] == 50
    assert identity['training_source_hash'] == 'trainingcode'
    assert identity['source_hash'] == 'newcode'
    monkeypatch.setattr(worker, 'verify_source', lambda: 'changed')
    with pytest.raises(ValueError, match='source'):
        worker._identity(case, original)


@pytest.mark.parametrize('domain,percent,expected', [
    ('raw_pixels', 50, 'stable_raw_pixel_abs_sort_and_zero_before_normalization'),
    ('raw_pixels', 0, 'NA_0percent_bypass_no_sort'),
    ('native_contrast', 50, 'stable_native_abs_sort_and_zero'),
    ('none', 0, 'NA_raw_input'),
])
def test_profiler_metadata_names_actual_sparsification_domain(domain, percent, expected):
    original = ('stable_native_abs_sort_and_zero' if domain == 'native_contrast' else 'NA_raw_input')
    row = {'stage_applicability': {'coefficient_ranking': original, 'frontend': 'unchanged'}}
    result = worker._timing_stage_metadata(row, {'sparsification_domain': domain,
                                               'inference_sparsity_percent': percent})
    assert result['sparsification_domain'] == domain
    assert result['stage_applicability']['coefficient_ranking'] == expected
    assert result['stage_applicability']['frontend'] == 'unchanged'


def test_raw_panel_shows_actual_rank_removed_pixels_in_uint8_units():
    from sparse_contrast.visualize import preview_arrays
    raw = torch.tensor([[[[200, 80], [10, 0]]]], dtype=torch.uint8)
    mean, std = torch.tensor([0.13]).reshape(1, 1, 1, 1), torch.tensor([0.3]).reshape(1, 1, 1, 1)
    arrays = preview_arrays(raw, lambda tensor: tensor, mean, std, 50)
    displayed = worker._raw_sparse_display(arrays)
    assert displayed.dtype == np.uint8
    np.testing.assert_array_equal(displayed, [[[[200, 80], [0, 0]]]])
    np.testing.assert_allclose(displayed.astype(np.float32) / 255, arrays['native_sparse'], rtol=0, atol=0)


def test_raw_panel_scales_use_only_fixed_domain_and_frozen_constants():
    model = SimpleNamespace(frontend=None, mean=torch.tensor([0.13]).reshape(1, 1, 1, 1),
                            std=torch.tensor([0.3]).reshape(1, 1, 1, 1))
    (native, normalized), names, provenance = worker._preview_scales(model, {})
    np.testing.assert_array_equal(native, [1])
    np.testing.assert_allclose(normalized, [2.9], rtol=1e-6)
    assert names == ['intensity']
    assert provenance['origin'] == 'fixed_raw_domain_[0,1]_and_frozen_raw_normalization'


def test_contrast_scales_read_existing_shared_metadata_without_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'BASE_ROOT', tmp_path)
    original = dict(dataset='mnist', representation='grayscale', protocol='heldout_val', normalization_hash='norm')
    mean, std = torch.tensor([0.2]), torch.tensor([0.4])
    model = SimpleNamespace(frontend=SimpleNamespace(channel_names=('channel',)),
                            mean=mean.reshape(1, 1, 1, 1), std=std.reshape(1, 1, 1, 1))
    path = tmp_path / 'outputs/previews/heldout_val/mnist/grayscale' / worker.digest(original) / 'complete.json'
    saved = dict(dataset='mnist', representation='grayscale', normalization_hash='norm',
                 mean=mean.tolist(), std=std.tolist(), channel_names=['channel'],
                 native_scales=[0.8], normalized_scales=[2.4], train_panel_split_hash='train-only')
    worker.atomic_json(path, saved)
    before = path.read_bytes()
    scales, names, provenance = worker._preview_scales(model, original)
    assert path.read_bytes() == before
    np.testing.assert_array_equal(scales[0], [0.8])
    assert names == ['channel']
    assert provenance['sha256'] == worker.file_hash(path)
    saved['normalization_hash'] = 'different'
    worker.atomic_json(path, saved)
    with pytest.raises(ValueError, match='preview scales'):
        worker._preview_scales(model, original)


def test_completed_summary_rerun_preserves_bytes_and_rejects_changed_cells(tmp_path):
    path = tmp_path / 'summary.json'
    first = dict(state='complete', cells=[{'path': 'cell', 'sha256': 'exact'}],
                 expected_cells=[('clean', 'test')], completed_at='originaltime')
    worker._save_summary(path, first)
    before = path.read_bytes()
    worker._save_summary(path, {**first, 'completed_at': 'reruntime'})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='summary differs'):
        worker._save_summary(path, {**first, 'cells': []})


def _hardware():
    return dict(hardware_identity='class', gpu='NVIDIA RTX A6000', compute_capability=[8, 6],
                driver_version='driver', torch='torch', cuda='cuda', cudnn='cudnn',
                precision='float32', backend='PyTorch_explicit_attention',
                cpu_model='AMD EPYC 7413 24-Core Processor', torch_num_threads=4,
                torch_num_interop_threads=48, device_uuid='firstGPU', host='node')


def test_hardware_contract_concurrent_first_writers_cannot_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda uuid: worker._freeze_hardware_contract(
            {**_hardware(), 'device_uuid': uuid}, 'source'), ['GPU1', 'GPU2']))
    assert values[0][1] == values[1][1]
    path = values[0][0]
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='contract differs'):
        worker._freeze_hardware_contract({**_hardware(), 'torch_num_threads': 8}, 'source')
    assert path.read_bytes() == before


@pytest.mark.parametrize('field,value', [('gpu', 'Other GPU'), ('cpu_model', 'Other CPU')])
def test_other_hardware_class_rejected(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    with pytest.raises(RuntimeError, match='homogeneous'):
        worker._freeze_hardware_contract({**_hardware(), field: value}, 'source')


def test_stop_checked_at_cell_boundary(monkeypatch):
    monkeypatch.setattr(worker, 'stop_requested', lambda: True)
    with pytest.raises(InterruptedError, match='STOP'):
        worker._check_stop()


def test_timing_panel_is_verified_against_dataset_even_with_updated_hash(tmp_path, monkeypatch):
    dataset = SimpleNamespace(sample_ids=[f'sample{i}' for i in range(10)], split_hash='split')
    monkeypatch.setattr(worker, 'fixed_panel_indices', lambda _dataset, _panel: list(range(10)))
    warm, measured = worker.benchmark.timing_batches(list(range(10)), 1)
    membership = dict(warmup_indices=warm, measured_indices=measured,
                      warmup_ids=[[dataset.sample_ids[i] for i in batch] for batch in warm],
                      measured_ids=[[dataset.sample_ids[i] for i in batch] for batch in measured],
                      seed=worker.benchmark.BENCHMARK_SEED, dataset_split_hash='split',
                      independent_of_predictions_sparsity_latency=True)
    worker.atomic_json(tmp_path / 'batch_membership.json', membership)
    for name in ('timings.jsonl', 'tokens.jsonl', 'profile.json', 'trace.json'):
        (tmp_path / name).write_text('{}\n')
    metadata = {'hardware_identity': 'hardware', 'device_uuid': 'GPU'}
    monkeypatch.setattr(worker.benchmark, 'validate_measurement_session', lambda _measurement: metadata)
    isolation = dict(verified=True, device_uuid='GPU', self_pid=123, observed_compute_pids=[123])
    identity = {'test_config_hash': 'case', 'batch_size': 1}
    row = dict(state='complete', identity=identity, corruption='clean', severity='test',
               warmup_batches_per_scope=50, measured_batches_per_scope=200, batch_size=1,
               scopes={scope: {'count': 200, 'batch_size': 1} for scope in worker.benchmark.SCOPES},
               file_hashes={p.name: worker.file_hash(p) for p in tmp_path.iterdir()}, measurement={},
               isolation_before=isolation, isolation_after=isolation)
    path = tmp_path / 'summary.json'
    worker.atomic_json(path, row)
    worker._verify_timing_cell(path, identity, 'clean', 'test', {'hardware_identity': 'hardware'}, dataset)
    membership['measured_ids'][0][0] = 'other_sample'
    worker.atomic_json(tmp_path / 'batch_membership.json', membership)
    row['file_hashes']['batch_membership.json'] = worker.file_hash(tmp_path / 'batch_membership.json')
    worker.atomic_json(path, row)
    with pytest.raises(ValueError, match='sample panel differs'):
        worker._verify_timing_cell(path, identity, 'clean', 'test', {'hardware_identity': 'hardware'}, dataset)
