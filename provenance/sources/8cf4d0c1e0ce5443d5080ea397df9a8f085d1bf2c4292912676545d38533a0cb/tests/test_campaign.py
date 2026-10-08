"""Pure controller/ownership checks; scheduler and production artifacts are mocked."""
from copy import deepcopy
import getpass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dense_sparse import common, controller, scheduler, setup


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def parent_artifacts(tmp_path, monkeypatch):
    root = tmp_path / 'parent'
    monkeypatch.setattr(controller, 'BASE_ROOT', root)
    state = dict(phase='completed', main=[dict(state='completed', stage='evaluation') for _ in range(156)],
                 latency_workers=[dict(state='completed') for _ in range(552)])
    coverage = {'all_requested_work_complete': True}
    monitor = {'state': 'completed_verified'}

    def save():
        write_json(root / 'outputs/campaign.json', state)
        write_json(root / 'outputs/tables/coverage.json', coverage)
        write_json(root / 'outputs/agent_monitor/state.json', monitor)

    save()
    return root, state, coverage, monitor, save


def test_parent_gate_requires_complete_training_evaluation_timing_and_agent_verification(parent_artifacts):
    ready, evidence = controller.parent_gate()
    assert ready
    assert evidence['trained_and_evaluated'] == 156
    assert evidence['latency_completed'] == evidence['latency_expected'] == 552
    assert evidence['coverage_complete'] is True
    assert evidence['final_verification'] == 'completed_verified'


@pytest.mark.parametrize('problem', ['training_only', 'partial_timing', 'missing_timing',
                                    'missing_main', 'false_coverage', 'nonboolean_coverage',
                                    'unverified_monitor', 'unfinished_phase'])
def test_parent_gate_rejects_every_incomplete_condition(parent_artifacts, problem):
    root, state, coverage, monitor, save = parent_artifacts
    if problem == 'training_only': state['main'][0]['stage'] = 'train'
    elif problem == 'partial_timing': state['latency_workers'][0]['state'] = 'running'
    elif problem == 'missing_timing': state['latency_workers'].pop()
    elif problem == 'missing_main': state['main'].pop()
    elif problem == 'false_coverage': coverage['all_requested_work_complete'] = False
    elif problem == 'nonboolean_coverage': coverage['all_requested_work_complete'] = 'true'
    elif problem == 'unverified_monitor': monitor['state'] = 'ACTIVE_REPAIR'
    elif problem == 'unfinished_phase': state['phase'] = 'final_report'
    save()
    assert controller.parent_gate()[0] is False


@pytest.mark.parametrize('missing', ['outputs/tables/coverage.json', 'outputs/agent_monitor/state.json'])
def test_parent_gate_missing_receipts_never_launch(parent_artifacts, missing):
    root, *_ = parent_artifacts
    (root / missing).unlink()
    assert controller.parent_gate()[0] is False


def receipt(root, job='123', name='controller', identity='a' * 64):
    value = dict(identity=identity, state='accepted', job_id=job,
                 job_name=f'dss-{name}-{identity[:16]}', comment='dense-sparse:' + identity,
                 request={'name': name})
    write_json(root / 'outputs/submissions' / f'{identity}.json', value)
    return value


def scheduler_outputs(monkeypatch, accounting='', queue=''):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert args[0] in ('sacct', 'squeue'), 'Tests must never invoke a real scheduler'
        if args[0] == 'squeue':
            assert '-u' in args and '-j' not in args
        return SimpleNamespace(stdout=accounting if args[0] == 'sacct' else queue)

    monkeypatch.setattr(scheduler.subprocess, 'run', run)
    return calls


@pytest.mark.parametrize('name', ['controller', 'accuracy-report', 'final-report'])
def test_terminal_empty_comment_requires_unique_local_receipt(tmp_path, monkeypatch, name):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    saved = receipt(tmp_path, name=name)
    scheduler_outputs(monkeypatch, accounting=f"123|COMPLETED|0:0|{getpass.getuser()}|{saved['job_name']}||\n")
    assert scheduler.observed(['123'])['123']['state'] == 'COMPLETED'


@pytest.mark.parametrize('problem', ['wrong_name', 'conflicting_comment', 'missing_receipt', 'duplicate_receipt'])
def test_scheduler_rejects_unowned_or_conflicting_terminal_jobs(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    saved = receipt(tmp_path)
    name, comment = saved['job_name'], ''
    if problem == 'wrong_name': name = 'unrelated'
    elif problem == 'conflicting_comment': comment = 'foreign-campaign'
    elif problem == 'missing_receipt': next((tmp_path / 'outputs/submissions').glob('*.json')).unlink()
    elif problem == 'duplicate_receipt': receipt(tmp_path, identity='b' * 64)
    scheduler_outputs(monkeypatch, accounting=f'123|COMPLETED|0:0|{getpass.getuser()}|{name}|{comment}|\n')
    with pytest.raises(RuntimeError, match='ownership'):
        scheduler.observed(['123'])


def test_scheduler_does_not_adopt_other_user_or_unrequested_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    saved = receipt(tmp_path)
    scheduler_outputs(monkeypatch, accounting=(
        f"123|COMPLETED|0:0|different_user|{saved['job_name']}||\n"
        f"456|COMPLETED|0:0|{getpass.getuser()}|{saved['job_name']}||\n"))
    assert scheduler.observed(['123']) == {}


def test_running_queue_requires_comment_and_overrides_stale_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    saved = receipt(tmp_path)
    scheduler_outputs(monkeypatch,
        accounting=f"123|PENDING||{getpass.getuser()}|{saved['job_name']}||\n",
        queue=f"123|RUNNING|{getpass.getuser()}|{saved['job_name']}|{saved['comment']}\n")
    assert scheduler.observed(['123'])['123']['state'] == 'RUNNING'
    scheduler_outputs(monkeypatch, queue=f"123|RUNNING|{getpass.getuser()}|{saved['job_name']}|\n")
    with pytest.raises(RuntimeError, match='Conflicting ownership'):
        scheduler.observed(['123'])


def test_submission_reuses_accepted_intent_without_second_sbatch(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    monkeypatch.setattr(scheduler, 'stop_requested', lambda: False)
    calls = []

    def run(args, **kwargs):
        assert args[0] == 'sbatch'
        calls.append(args)
        return SimpleNamespace(stdout='123;cluster\n', stderr='')

    monkeypatch.setattr(scheduler.subprocess, 'run', run)
    args = (['python', '-m', 'worker'], 'evaluation', 'case:0', '/immutable/source')
    assert scheduler.submit(*args) == scheduler.submit(*args) == '123'
    assert len(calls) == 1
    with pytest.raises(ValueError, match='stable submission key'):
        scheduler.submit(['different'], *args[1:])


def test_stop_after_intent_is_known_unsubmitted_and_can_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    checks = iter([False, True])
    monkeypatch.setattr(scheduler, 'stop_requested', lambda: next(checks))
    monkeypatch.setattr(scheduler.subprocess, 'run', lambda *a, **k: pytest.fail('STOP must prevent sbatch'))
    args = (['python', '-m', 'worker'], 'evaluation', 'case:0', '/immutable/source')
    with pytest.raises(InterruptedError, match='definitely not submitted'):
        scheduler.submit(*args)
    saved = common.read_json(next((tmp_path / 'outputs/submissions').glob('*.json')))
    assert saved['state'] == 'cancelled_before_submit' and saved['job_id'] is None
    monkeypatch.setattr(scheduler, 'stop_requested', lambda: False)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert args[0] == 'sbatch'
        return SimpleNamespace(stdout='124\n', stderr='')

    monkeypatch.setattr(scheduler.subprocess, 'run', run)
    assert scheduler.submit(*args) == '124' and len(calls) == 1
    assert len(list((tmp_path / 'outputs/submissions').glob('*.cancelled.*'))) == 1


def test_ambiguous_submission_never_creates_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    monkeypatch.setattr(scheduler, 'stop_requested', lambda: False)
    calls = []

    def run(args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(stdout='unparseable' if args[0] == 'sbatch' else '', stderr='')

    monkeypatch.setattr(scheduler.subprocess, 'run', run)
    args = (['python'], 'evaluation', 'case:0', '/immutable/source')
    with pytest.raises(RuntimeError, match='Ambiguous sbatch'):
        scheduler.submit(*args)
    with pytest.raises(RuntimeError, match='Ambiguous submission'):
        scheduler.submit(*args)
    assert calls.count('sbatch') == 1


def test_source_snapshot_verifies_every_frozen_byte(tmp_path, monkeypatch):
    source = tmp_path / 'working'
    (source / 'dense_sparse').mkdir(parents=True)
    (source / 'tests').mkdir()
    (source / 'dense_sparse/setup.py').write_text('# source bytes\n')
    (source / 'dense_sparse/common.py').write_text('# validation bytes\n')
    (source / 'tests/test_one.py').write_text('# evidence tests\n')
    (source / 'PLAN.md').write_text('Scientific protocol\n')
    (source / 'MONITOR.md').write_text('Bounded supervision\n')
    (source / 'README.md').write_text('Commands\n')
    (source / 'audit_zero_checkpoints.py').write_text('# full checkpoint audit\n')
    monkeypatch.setattr(setup, '__file__', str(source / 'dense_sparse/setup.py'))
    monkeypatch.setattr(setup, 'ROOT', tmp_path / 'study')
    identity, frozen = setup.snapshot()
    monkeypatch.setattr(common, '__file__', str(frozen / 'dense_sparse/common.py'))
    assert common.verify_source() == identity
    assert setup.snapshot() == (identity, frozen)
    (frozen / 'dense_sparse/common.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='Inference source changed'):
        common.verify_source()
    with pytest.raises(ValueError, match='snapshot collision'):
        setup.snapshot()


def test_checkpoint_load_requires_embedded_original_identity_and_strict_layout(monkeypatch):
    import torch
    from dense_sparse import adapter
    from sparse_contrast import train

    original = {'config_hash': 'training', 'normalization_hash': 'stats'}
    case = {'checkpoint_path': '/recorded/final.pt'}
    state = dict(config_hash='training', normalization_hash='stats', model={'all_keys': 'weights'})
    calls = []

    class Model:
        def __init__(self, config, stats, inference):
            assert config is original and stats == 'original-frozen-stats' and inference is case
        def load_state_dict(self, values, strict):
            calls.append((values, strict))
        def to(self, device): return self
        def eval(self): return self

    monkeypatch.setattr(common, 'load_case', lambda x: x)
    monkeypatch.setattr(common, 'validate_case', lambda x: original)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: deepcopy(state))
    monkeypatch.setattr(train, 'load_stats', lambda config: 'original-frozen-stats')
    monkeypatch.setattr(adapter, 'InferenceClassifier', Model)
    assert isinstance(common.load_model(case, 'cpu'), Model)
    assert calls == [({'all_keys': 'weights'}, True)]
    for field in ('config_hash', 'normalization_hash'):
        state[field] = 'wrong'
        with pytest.raises(ValueError, match='embedded config/normalization'):
            common.load_model(case, 'cpu')
        state[field] = original[field]
    assert len(calls) == 1


@pytest.fixture
def worker_receipt(tmp_path, monkeypatch):
    from sparse_contrast import data
    monkeypatch.setattr(data, 'corruption_cells', lambda dataset: [('noise', 1)])
    case = dict(test_config_hash='test', training_config_hash='train', checkpoint_sha256='weights',
                completed_receipt_sha256='done', training_source_hash='original-source',
                normalization_hash='stats', source_hash='inference-source', dataset='mnist',
                architecture='vit_small', representation='grayscale', seed=0, trained_execution='dense',
                inference_execution='compact', inference_sparsity_percent=60,
                sparsification_domain='native_contrast')
    record = dict(execution='compact', batch_size=8, shard=0, num_shards=1)
    identity = {**case, **record}
    path = tmp_path / 'worker/summary.json'
    references = []
    for corruption, severity in [('clean', 'test'), ('noise', 1)]:
        cell = path.parent / 'cells' / f'{corruption}__{severity}/summary.json'
        write_json(cell, dict(state='complete', identity=identity, corruption=corruption, severity=severity))
        references.append(dict(path=str(cell), sha256=common.file_hash(cell),
                               corruption=corruption, severity=severity))
    summary = dict(state='complete', identity=identity, cells=references)
    write_json(path, summary)
    return path, case, record, summary


def test_worker_receipt_binds_complete_matrix_original_identity_and_timing_settings(worker_receipt):
    path, case, record, _ = worker_receipt
    assert controller.verify_receipt(path, case, record, 'latency') == common.file_hash(path)


@pytest.mark.parametrize('problem', ['wrong_source', 'wrong_normalization', 'wrong_batch', 'wrong_execution',
                                    'missing_cell', 'duplicate_cell', 'changed_cell', 'cell_other_identity',
                                    'outside_case_directory'])
def test_worker_completion_rejects_relabelled_or_partial_evidence(worker_receipt, problem):
    path, case, record, summary = worker_receipt
    if problem == 'wrong_source': summary['identity']['source_hash'] = 'wrong'
    elif problem == 'wrong_normalization': summary['identity']['normalization_hash'] = 'wrong'
    elif problem == 'wrong_batch': summary['identity']['batch_size'] = 1
    elif problem == 'wrong_execution': summary['identity']['execution'] = 'dense_masked'
    elif problem == 'missing_cell': summary['cells'].pop()
    elif problem == 'duplicate_cell': summary['cells'].append(summary['cells'][0])
    elif problem == 'changed_cell': Path(summary['cells'][0]['path']).write_text('{}')
    elif problem == 'cell_other_identity':
        ref = summary['cells'][0]
        cell = common.read_json(ref['path']); cell['identity']['source_hash'] = 'other'
        write_json(Path(ref['path']), cell); ref['sha256'] = common.file_hash(ref['path'])
    elif problem == 'outside_case_directory':
        ref = summary['cells'][0]; foreign = path.parent.parent / 'foreign.json'
        foreign.write_bytes(Path(ref['path']).read_bytes()); ref['path'] = str(foreign)
    write_json(path, summary)
    with pytest.raises(ValueError):
        controller.verify_receipt(path, case, record, 'latency')


def work_record(identity, job=None):
    return dict(identity=identity, test_config_hash=identity, state='pending' if job else 'unsubmitted',
                job_id=job, attempt=0, transient_retries=0, segments=0)


def test_failed_work_does_not_repeat_history_or_block_independent_launches(tmp_path, monkeypatch):
    failed, ready = work_record('failed', '123'), work_record('ready')
    state = dict(evaluation=[failed, ready], source_path='/immutable', history=[])
    training_config = tmp_path / 'original.json'
    write_json(training_config, {'recipe': {'gpu_partitions': 'gpu-vram-48gb'}})
    manifest = {'cases': [{'test_config_hash': x, 'training_config_path': str(training_config)}
                          for x in ('failed', 'ready')]}
    monkeypatch.setattr(controller, 'publish', lambda state: None)
    monkeypatch.setattr(controller, 'observed', lambda ids: {'123': dict(state='FAILED', exit_code='1:0')})
    monkeypatch.setattr(controller, 'stop_requested', lambda: False)
    monkeypatch.setattr(controller, 'summary_path', lambda record, *args: tmp_path / record['identity'] / 'summary.json')
    monkeypatch.setattr(controller, 'recover_lock', lambda path: None)
    monkeypatch.setattr(controller, 'submit', lambda *args, **kwargs: '124')
    assert controller.advance(state, 'evaluation', manifest) is False
    assert failed['state'] == 'blocked' and ready['state'] == 'pending' and ready['job_id'] == '124'
    assert len(state['history']) == 1
    controller.advance(state, 'evaluation', manifest)
    assert len(state['history']) == 1


@pytest.mark.parametrize('status,retries,expected', [('NODE_FAIL', 0, 'unsubmitted'), ('NODE_FAIL', 3, 'blocked'),
                                                  ('OUT_OF_MEMORY', 0, 'blocked'), ('FAILED', 0, 'blocked')])
def test_worker_retry_budget_is_persisted_and_deterministic_failures_block(monkeypatch, status, retries, expected):
    record = work_record('case', '123'); record['transient_retries'] = retries
    state = dict(evaluation=[record], source_path='/immutable', history=[])
    monkeypatch.setattr(controller, 'publish', lambda state: None)
    monkeypatch.setattr(controller, 'observed', lambda ids: {'123': dict(state=status, exit_code='1:0')})
    monkeypatch.setattr(controller, 'stop_requested', lambda: True)
    controller.advance(state, 'evaluation', {'cases': [{'test_config_hash': 'case'}]})
    assert record['state'] == expected
    assert record['transient_retries'] == retries + (expected == 'unsubmitted')


def test_final_report_exit_zero_requires_complete_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'publish', lambda state: None)
    monkeypatch.setattr(controller, 'observed', lambda ids: {'123': dict(state='COMPLETED', exit_code='0:0')})
    expected = dict(evaluation_cases=612, evaluation_cells=35352)
    monkeypatch.setattr(controller, 'load_manifest', lambda: {'expected': expected})
    write_json(tmp_path / 'outputs/tables/coverage.json', dict(all_requested_work_complete=False,
               accuracy_complete=True, completed_evaluation_cases=612, completed_evaluation_cells=35352))
    state = {'final_report': dict(job_id='123', state='pending', attempt=0, transient_retries=0, segments=0)}
    assert controller.report_step(state, 'final') is False
    assert state['final_report']['state'] == 'blocked'
    assert state['final_report']['failure']['reason'] == 'final_coverage_incomplete'


@pytest.mark.parametrize('field,value', [('accuracy_complete', False), ('completed_evaluation_cases', 611),
                                       ('completed_evaluation_cells', 35351)])
def test_accuracy_report_does_not_launch_latency_with_incomplete_official_results(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'publish', lambda state: None)
    monkeypatch.setattr(controller, 'observed', lambda ids: {'123': dict(state='COMPLETED', exit_code='0:0')})
    monkeypatch.setattr(controller, 'load_manifest', lambda: {'expected': dict(evaluation_cases=612, evaluation_cells=35352)})
    coverage = dict(accuracy_complete=True, completed_evaluation_cases=612, completed_evaluation_cells=35352)
    coverage[field] = value
    write_json(tmp_path / 'outputs/tables/coverage.json', coverage)
    state = {'accuracy_report': dict(job_id='123', state='pending', attempt=0, transient_retries=0, segments=0)}
    assert controller.report_step(state, 'accuracy') is False
    assert state['accuracy_report']['failure']['reason'] == 'accuracy_coverage_incomplete'


@pytest.mark.parametrize('total,per_incident,allowed', [(0, 0, True), (7, 2, True), (8, 0, False), (2, 3, False)])
def test_monitor_repair_budgets_survive_state_reload(total, per_incident, allowed):
    from dense_sparse import monitor
    state = json.loads(json.dumps(dict(agent_calls=total, incidents={'fingerprint': {'calls': per_incident}})))
    assert monitor.eligible({'fingerprint': 'fingerprint'}, state) is allowed


@pytest.mark.parametrize('marker', ['study', 'parent', 'monitor'])
def test_all_authoritative_stop_markers_prevent_monitor_repair(tmp_path, monkeypatch, marker):
    from dense_sparse import monitor
    parent = tmp_path / 'parent'; parent.mkdir()
    study = tmp_path / 'study'; study.mkdir()
    monkeypatch.setattr(common, 'BASE_ROOT', parent)
    monkeypatch.setattr(monitor, 'HALT', False)
    assert monitor.stopped(study) is False
    target = {'study': study / 'STOP', 'parent': parent / 'STOP',
              'monitor': study / 'outputs/agent_monitor/STOP'}[marker]
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text('stop')
    assert monitor.stopped(study) is True


@pytest.mark.parametrize('problem', [None, 'phase', 'evaluation', 'timing', 'coverage', 'active_job'])
def test_monitor_final_gate_requires_complete_current_evidence_and_released_compute(tmp_path, monkeypatch, problem):
    from dense_sparse import monitor
    monkeypatch.setattr(common, 'load_manifest', lambda: {'expected': dict(evaluation_cases=2, latency_workers=3)})
    state = dict(phase='completed', evaluation=[{'state': 'completed'} for _ in range(2)],
                 latency_workers=[{'state': 'completed'} for _ in range(3)])
    coverage = {'all_requested_work_complete': True}
    observed = {'123': {'state': 'COMPLETED'}}
    if problem == 'phase': state['phase'] = 'final_report'
    elif problem == 'evaluation': state['evaluation'][0]['state'] = 'blocked'
    elif problem == 'timing': state['latency_workers'].pop()
    elif problem == 'coverage': coverage['all_requested_work_complete'] = False
    elif problem == 'active_job': observed['123']['state'] = 'RUNNING'
    write_json(tmp_path / 'outputs/tables/coverage.json', coverage)
    assert monitor.complete_gate(tmp_path, state, observed) is (problem is None)
