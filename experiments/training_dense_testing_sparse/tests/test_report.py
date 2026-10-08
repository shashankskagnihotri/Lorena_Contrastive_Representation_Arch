"""Reporting regressions, including one immutable real parent timing cell."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dense_sparse import report


def case(dataset='mnist', architecture='vit_small', representation='grayscale', trained='dense', seed=0, percent=0, execution='compact'):
    identity = f'{dataset}/{architecture}/{representation}/{trained}/{seed}'
    value = dict(dataset=dataset, architecture=architecture, representation=representation,
        trained_execution=trained, seed=seed, inference_execution=execution,
        inference_sparsity_percent=percent, sparsification_domain='none' if execution == 'dense' else 'raw_pixels' if representation == 'raw' else 'native_contrast',
        training_config_hash=identity, checkpoint_sha256='checkpoint-' + identity, normalization_hash='normalization', source_hash='inference-source')
    value['test_config_hash'] = report.digest(value)
    return value


def full_matrix():
    cases = []
    for dataset in ('mnist', 'cifar10'):
        representations = ('grayscale',) if dataset == 'mnist' else ('single_color', 'grayscale', 'color_opponency')
        families = [('raw', 'dense')] + [(rep, trained) for rep in representations for trained in ('dense', 'compact')]
        for architecture in ('vit_small', 'swin_tiny'):
            for rep, trained in families:
                for seed in (0, 1, 2):
                    cases.extend(case(dataset, architecture, rep, trained, seed, p) for p in range(0, 100, 10))
                    if rep == 'raw': cases.append(case(dataset, architecture, rep, trained, seed, 0, 'dense'))
    return cases


def test_complete_matrix_counts_and_no_duplicate_success():
    cases = full_matrix(); manifest = dict(cases=cases, study_id='training_dense_testing_sparse', source_hash='source', manifest_hash='manifest')
    accuracy = [report.metadata(c) | dict(state='complete', completed_cells=len(report.pairs(c['dataset']))) for c in cases]
    workers = [report.metadata(c) | dict(execution=execution, batch_size=batch, state='complete',
        completed_cells=len(report.pairs(c['dataset'])), hardware_identities='same-hardware') for c in cases
        for execution in (('compact', 'dense_masked') if c['inference_execution'] == 'compact' else ('dense',)) for batch in (1, 8)]
    result = report.coverage_record(manifest, accuracy, workers, [], ['verified_fixture.png'], figures_complete=True)
    assert (result['expected_evaluation_cases'], result['expected_evaluation_cells']) == (612, 35352)
    assert (result['expected_latency_workers'], result['expected_latency_cells']) == (2424, 140304)
    assert result['all_requested_work_complete']
    assert not report.coverage_record(manifest, accuracy, workers, [], [])['all_requested_work_complete']
    assert not report.coverage_record(manifest, [accuracy[0], *accuracy[:-1]], workers, [], [])['accuracy_complete']
    assert not report.coverage_record(manifest, accuracy, [workers[0], *workers[:-1]], [], [])['latency_complete']
    mixed = copy.deepcopy(workers); mixed[0]['hardware_identities'] = 'other-hardware'
    assert not report.coverage_record(manifest, accuracy, mixed, [], [])['all_requested_work_complete']


def test_trained_families_missing_seeds_and_paired_zero_anchor():
    cases = [case(trained=t, seed=s, percent=p) for t in ('dense', 'compact') for s in (0, 1, 2) for p in (0, 10)]
    rows = []
    for c in cases:
        complete = not (c['trained_execution'] == 'compact' and c['seed'] == 2 and c['inference_sparsity_percent'] == 10)
        row = report.metadata(c) | {'state': 'complete' if complete else 'missing'}
        if complete:
            row.update(clean_accuracy=.9 - .01 * c['seed'] - .001 * c['inference_sparsity_percent'],
                clean_loss=.1, mCE_raw=.2, original_clean_accuracy=.95, original_clean_loss=.09, original_mCE_raw=.15,
                delta_clean_accuracy_from_original=-.05 - .01 * c['seed'] - .001 * c['inference_sparsity_percent'])
        rows.append(row)
    report.add_paired_changes(rows, [])
    output = report.aggregate_accuracy(rows, cases)
    assert len(output) == 4
    dense = next(r for r in output if r['trained_execution'] == 'dense' and r['inference_sparsity_percent'] == 10)
    assert dense['clean_accuracy_n'] == 3 and dense['clean_accuracy_sd'] == pytest.approx(.01)
    assert dense['delta_clean_accuracy_from_compact0_mean'] == pytest.approx(-.01)
    assert dense['delta_clean_accuracy_from_compact0_all_seeds_negative']
    partial = next(r for r in output if r['trained_execution'] == 'compact' and r['inference_sparsity_percent'] == 10)
    assert partial['clean_accuracy_n'] == 2 and partial['expected_seeds'] == 3
    assert not partial['delta_clean_accuracy_from_original_all_seeds_negative']


@pytest.fixture
def parent_cell():
    receipt = report.BASE_ROOT / 'outputs/checks/initial_a6000_gray60_latency.json'
    if not receipt.is_file(): pytest.skip('This integration fixture requires the preserved parent timing cell')
    path = Path(report.read_json(receipt)['rows'][0]['path'])
    summary = report.read_json(path)
    assert summary['state'] == 'complete'
    return path, summary


def translated_real_cell(tmp_path, parent_cell):
    original_path, original = parent_cell
    c = case('cifar10', 'vit_small', 'raw', 'dense', 0, 0, 'dense')
    directory = tmp_path / 'latency/dense/batch8/shard0/cells/clean__test'; directory.mkdir(parents=True)
    summary = copy.deepcopy(original); summary['identity'].update({k: c[k] for k in report.IDENTITY})
    timing_path = directory / 'timings.jsonl'
    with (original_path.parent / 'timings.jsonl').open() as source, timing_path.open('w') as target:
        for line in source:
            row = json.loads(line); row.update({k: c[k] for k in report.IDENTITY})
            target.write(json.dumps(row) + '\n')
    for name in summary['file_hashes']:
        if name != 'timings.jsonl': (directory / name).symlink_to(original_path.parent / name)
    summary['file_hashes']['timings.jsonl'] = report.file_hash(timing_path)
    path = directory / 'summary.json'; path.write_text(json.dumps(summary))
    return c, path, summary


def test_real_timing_stream_and_hardware_rejection(tmp_path, parent_cell):
    c, path, summary = translated_real_cell(tmp_path, parent_cell)
    values, panel = report.timing_values(path.parent / 'timings.jsonl', summary['file_hashes']['timings.jsonl'], c, 'dense', 8, summary['measurement'])
    assert len(panel) == 64
    for scope in report.SCOPES:
        assert len(values[scope]) == 200
        assert np.median(values[scope]) == pytest.approx(summary['scopes'][scope]['median_ms'])
        assert np.percentile(values[scope], 95) == pytest.approx(summary['scopes'][scope]['p95_ms'])
    bad = dict(summary['measurement'], measurement_device_uuid='GPU-wrong-device')
    with pytest.raises(ValueError, match='hardware/session'):
        report.timing_values(path.parent / 'timings.jsonl', summary['file_hashes']['timings.jsonl'], c, 'dense', 8, bad)
    with pytest.raises(ValueError, match='UUID/session'):
        report.hardware_label(bad, {})


def test_real_cell_memory_throughput_and_coverage(tmp_path, parent_cell, monkeypatch):
    c, path, cell = translated_real_cell(tmp_path, parent_cell)
    directory = path.parents[2]
    metadata = report.read_json(cell['measurement']['hardware_metadata_file'])
    contract_path = tmp_path / 'hardware_contract.json'
    contract_path.write_text(json.dumps({'contract': {k: metadata[k] for k in ('hardware_identity', 'gpu', 'cpu_model', 'torch_num_threads', 'torch_num_interop_threads')}}))
    zero = directory / 'zero_input_diagnostic.json'; zero.write_text(json.dumps({'state': 'complete'}))
    receipt = dict(state='complete', identity={k: c[k] for k in report.IDENTITY}, execution='dense', batch_size=8,
        shard=0, num_shards=1, expected_cells=[['clean', 'test']], hardware_contract_path=str(contract_path),
        hardware_contract_sha256=report.file_hash(contract_path), zero_input_diagnostic_sha256=report.file_hash(zero),
        cells=[dict(corruption='clean', severity='test', path=str(path), sha256=report.file_hash(path))])
    (directory / 'summary.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(report, 'case_dir', lambda c: tmp_path)
    monkeypatch.setattr(report, 'pairs', lambda d: [('clean', 'test')])
    written = []
    class Sink:
        def write(self, row): written.append(row)
    workers, rows, stages = report.collect_latency(c, {}, {}, Sink())
    assert [w['state'] for w in workers] == ['missing', 'complete']
    assert len(rows) == len(written) == 4
    for row in rows:
        original = cell['scopes'][row['scope']]
        assert row['median_ms'] == pytest.approx(original['median_ms'])
        assert row['max_memory_allocated'] == original['memory']['max_memory_allocated']
        assert row['max_memory_reserved'] == original['memory']['max_memory_reserved']
        assert row['sustained_images_per_second'] == pytest.approx(original['sustained_throughput']['images'] / original['sustained_throughput']['elapsed_seconds'])
        assert row['hardware_identity'] == cell['measurement']['hardware_identity']
    assert stages and all(r['additive'] is False for r in stages)


def test_scope_speedups_never_pair_different_hardware():
    baseline = report.metadata(case(representation='raw', execution='dense')) | dict(batch_size=8, scope='loader_to_cpu_prediction', input_group='clean', hardware_identity='A', median_ms=10, execution='dense')
    method = report.metadata(case(percent=60)) | dict(batch_size=8, scope='loader_to_cpu_prediction', input_group='clean', hardware_identity='A', median_ms=5, execution='compact')
    other = dict(method, hardware_identity='B')
    report.add_paired_changes([], [baseline, method, other])
    assert method['paired_raw_dense_speedup'] == 2
    assert 'paired_raw_dense_speedup' not in other


def test_real_original_reference_hash_and_final_epoch(parent_cell):
    path, summary = parent_cell
    rd = path.parents[6]
    config = report.read_json(rd / 'config.json')
    c = case(config['dataset'], config['architecture'], config['representation'], config['execution'], config['seed'], execution='dense')
    c.update(training_config_hash=config['config_hash'], checkpoint_sha256=summary['identity']['checkpoint_sha256'],
        checkpoint_path=str(rd / 'final.pt'), training_config_path=str(rd / 'config.json'),
        completed_receipt_sha256=report.file_hash(rd / 'completed.json'),
        original_evaluation_path=str(rd / 'evaluation/summary.json'),
        original_evaluation_sha256=report.file_hash(rd / 'evaluation/summary.json'))
    original = report.reference(c, {})
    assert original['original_epochs'] == config['recipe']['epochs'] == 200
    assert original['training_count'] == 45000 and original['validation_count'] == 5000
    assert original['training_batch_size'] == config['recipe']['batch_size'] == 128
    assert 0 <= original['original_validation_accuracy'] <= 1
    with pytest.raises(ValueError, match='evaluation reference changed'):
        report.reference(dict(c, original_evaluation_sha256='0' * 64), {})


def test_full_overview_layout_keeps_missing_labels_and_first_point_inside(tmp_path, parent_cell, monkeypatch):
    _, saved = parent_cell
    cases = [c for c in full_matrix() if c['dataset'] == 'cifar10' and c['architecture'] == 'vit_small']
    conditions = report.aggregate_accuracy([report.metadata(c) | {'state': 'missing'} for c in cases], cases)
    reference = next(c for c in cases if c['inference_execution'] == 'dense' and c['seed'] == 0)
    hardware = report.read_json(saved['measurement']['hardware_metadata_file']); scope = 'gpu_raw_to_logits'
    measured = saved['scopes'][scope]
    row = report.metadata(reference) | dict(execution='dense', batch_size=8, scope=scope, input_group='clean',
        hardware_identity=saved['measurement']['hardware_identity'],
        **{k: hardware[k] for k in ('gpu', 'cpu_model', 'torch_num_threads', 'torch_num_interop_threads')},
        **{k: measured[k] for k in ('mean_ms', 'median_ms', 'p95_ms', 'amortized_mean_ms_per_image', 'serial_images_per_second')},
        **measured['memory'], sustained_images_per_second=measured['sustained_throughput']['images_per_second'])
    aggregate = report.aggregate_timing([row], cases, [])
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    def inspect(fig, stem, rows, writer):
        assert stem.name == 'direct_overview' and len(rows) == 71
        for ax in fig.axes:
            assert ax.get_xlim()[0] == 0
            low, high = sorted(ax.get_ylim())
            assert low < 0 and high > 70
            assert all(ax.get_xlim()[0] <= t.get_position()[0] <= ax.get_xlim()[1] for t in ax.texts)
        report.plt.close(fig)
        return 'layout-fixture.png'
    monkeypatch.setattr(report, 'save_figure', inspect)
    assert report.diagnostic_figures(conditions, aggregate, [row], [], [], []) == ['layout-fixture.png']
