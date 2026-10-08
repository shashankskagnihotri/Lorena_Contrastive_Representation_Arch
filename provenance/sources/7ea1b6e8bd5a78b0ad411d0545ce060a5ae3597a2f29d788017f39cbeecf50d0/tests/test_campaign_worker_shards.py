"""Mocked scheduler coverage for one homogeneous cohort dispatched per worker."""
import copy
import json
from types import SimpleNamespace

import pytest

from sparse_contrast import benchmark, campaign
from sparse_contrast.common import atomic_json, digest, file_hash
from sparse_contrast.scope import expected_worker_count
from test_campaign_latency import latency

pytestmark = pytest.mark.parametrize('latency', [True], indirect=True)


@pytest.fixture
def shards(latency, monkeypatch):
    state = latency.state
    cohort = dict(id='a6000-complete-v1', gpu_partition='gpu-vram-48gb', gpu_gres_type='nvidia_rtx_a6000')
    state['latency_cohort'] = cohort
    out = latency.root / 'outputs/benchmarks/cohorts' / cohort['id']
    legacy = latency.root / 'outputs/benchmarks'
    plan = latency.plan
    plan['cohort'] = cohort['id']
    plan['blocks_sha256'] = digest({k: v for k, v in plan.items() if k != 'blocks_sha256'})
    contract = dict(gpu_partition=cohort['gpu_partition'], gpu_gres_type=cohort['gpu_gres_type'],
                    contract={'hardware_identity':'a6000-hardware'})
    atomic_json(out / 'hardware_contract.json', contract)
    metadata = dict(hardware_identity='a6000-hardware', measurement_session_id='prepare', device_uuid='GPU-fixture')
    metadata_path = out / 'preparation/hardware.json'; atomic_json(metadata_path, metadata)
    prep = json.loads((legacy / 'prepared.json').read_text())
    prep.update(cohort=cohort['id'], blocks_sha256=plan['blocks_sha256'],
                hardware_contract_sha256=file_hash(out / 'hardware_contract.json'))
    prep['measurement'].update(hardware_metadata_file=str(metadata_path), hardware_metadata_sha256=file_hash(metadata_path),
                               hardware_identity='a6000-hardware')
    atomic_json(out / 'prepared.json', prep)
    def load(root, cohort=None):
        assert root == latency.root and cohort == state['latency_cohort']['id']
        return copy.deepcopy(plan)
    monkeypatch.setattr(benchmark, 'load_frozen_blocks', load)
    # Persistence itself is tested by test_campaign; avoid 552 duplicate full
    # registry disk writes in this scheduler-state fixture.
    monkeypatch.setattr(campaign, '_save_state', lambda value: None)
    checked, finalized, invalid = [], [], set()
    flat = [(b['id'], w) for b in plan['blocks'] for w in b['work']]
    def validate_worker(worker_id, root, cohort):
        assert cohort == state['latency_cohort']['id'] and root == latency.root
        checked.append(worker_id)
        if worker_id in invalid: raise ValueError('fixture: corrupt worker evidence')
        block_id, work = flat[worker_id]
        return dict(state='completed', worker_id=worker_id, block_id=block_id, cohort=cohort, worker=work)
    def validate_block(block_id, root, cohort):
        assert cohort == state['latency_cohort']['id'] and root == latency.root
        return copy.deepcopy(latency.receipts[block_id])
    def finalize(block_id, root, cohort):
        receipt = validate_block(block_id, root, cohort)
        finalized.append(block_id)
        atomic_json(out / 'blocks' / f'block_{block_id:03d}' / 'completed.json', receipt)
        return receipt
    monkeypatch.setattr(benchmark, 'validate_worker_completion', validate_worker, raising=False)
    monkeypatch.setattr(benchmark, 'finalize_block', finalize, raising=False)
    monkeypatch.setattr(benchmark, 'validate_block_completion', validate_block)
    return SimpleNamespace(**vars(latency), out=out, cohort=cohort, checked=checked,
                           finalized=finalized, invalid=invalid, flat=flat)


def dispatch(shards):
    shards.state['latency_prepare'] = dict(campaign._latency_record(), state='completed', job_id='900')
    campaign._advance_postprocessing(shards.state)
    assert len(shards.calls) == expected_worker_count(shards.scope) == 552
    return shards.state['latency_workers']


def test_cohort_prepare_pins_requested_type_without_reusing_legacy_contract(shards):
    legacy = file_hash(shards.root / 'outputs/benchmarks/hardware_contract.json')
    (shards.out / 'hardware_contract.json').unlink()
    campaign._advance_postprocessing(shards.state)
    command, name, options = shards.calls[0]
    assert '--prepare-campaign' in command and command[command.index('--cohort')+1] == shards.cohort['id']
    assert options['gpu_partition'] == 'gpu-vram-48gb' and options['gpu_type'] == 'nvidia_rtx_a6000'
    assert file_hash(shards.root / 'outputs/benchmarks/hardware_contract.json') == legacy
    assert 'latency_workers' not in shards.state


def test_all_552_unique_workers_follow_frozen_order_and_one_typed_gpu_request(shards):
    records = dispatch(shards)
    assert [r['worker_id'] for r in records] == list(range(552))
    assert [(r['block_id'], r['work']) for r in records] == shards.flat
    assert len({(r['work']['config_hash'], r['work']['execution'], r['work']['batch_size']) for r in records}) == 552
    for index, (command, name, options) in enumerate(shards.calls):
        assert command[-2:] == ['--worker-id', str(index)]
        assert options['gpu_partition'] == 'gpu-vram-48gb' and options['gpu_type'] == 'nvidia_rtx_a6000'
        assert options['time_limit'] == '08:00:00'
        assert options['submission_key'].startswith('latency:a6000-complete-v1:')
    assert all(r['job_id'] is None for r in shards.state['latency_blocks'])


def test_restart_retains_unknown_or_running_worker_ownership(shards):
    records = dispatch(shards)
    for record in records[::2]: shards.statuses[record['job_id']] = {'state':'RUNNING'}
    restarted = json.loads(json.dumps(shards.state))
    campaign._advance_postprocessing(restarted)
    assert len(shards.calls) == 552
    assert {r['state'] for r in restarted['latency_workers']} == {'running', 'scheduler_unknown'}
    assert shards.checked == []


def test_worker_timeout_continues_cells_and_transient_retry_is_bounded(shards, monkeypatch):
    records = dispatch(shards); now = [100.]
    monkeypatch.setattr(campaign.time, 'time', lambda: now[0])
    for record in records: shards.statuses[record['job_id']] = {'state':'RUNNING'}
    timeout, transient, exhausted = records[:3]
    shards.statuses[timeout['job_id']] = {'state':'TIMEOUT'}
    shards.statuses[transient['job_id']] = {'state':'NODE_FAIL'}
    shards.statuses[exhausted['job_id']] = {'state':'BOOT_FAIL'}; exhausted['transient_retries'] = 3
    campaign._advance_postprocessing(shards.state)
    assert timeout['segments'] == timeout['attempt'] == 1 and timeout['transient_retries'] == 0
    assert transient['attempt'] == transient['transient_retries'] == 1 and transient['retry_not_before'] == 160
    assert transient['job_id'] is None and exhausted['state'] == 'failed'
    assert len(shards.calls) == 553
    now[0] = 160
    campaign._advance_postprocessing(shards.state)
    assert len(shards.calls) == 554 and transient['job_id'] is not None
    assert shards.calls[-1][2]['submission_key'].endswith(':worker_000001:1')
    assert shards.state['phase'] == 'latency'


def test_newly_complete_workers_validate_once_and_progress_counts_partial_blocks(shards):
    records = dispatch(shards)
    for record in records: shards.statuses[record['job_id']] = {'state':'RUNNING'}
    for record in records[:3]: shards.statuses[record['job_id']] = {'state':'COMPLETED'}
    campaign._advance_postprocessing(shards.state)
    progress = json.loads((shards.out / 'progress.json').read_text())
    assert progress['workers'] == 3 and progress['completed_blocks'] == 0 and progress['cohort'] == shards.cohort['id']
    campaign._advance_postprocessing(shards.state)
    assert shards.checked == [0, 1, 2] and len(shards.calls) == 552


def test_all_workers_produce_24_block_receipts_and_exact_cohort_completion(shards):
    legacy_hash = file_hash(shards.root / 'outputs/benchmarks/hardware_contract.json')
    records = dispatch(shards)
    for record in records: shards.statuses[record['job_id']] = {'state':'COMPLETED'}
    campaign._advance_postprocessing(shards.state)
    progress = json.loads((shards.out / 'progress.json').read_text())
    assert shards.state['phase'] == 'report' and progress['state'] == 'completed'
    assert progress['workers'] == progress['expected_workers'] == 552 and progress['completed_blocks'] == 24
    assert shards.finalized == list(range(24)) and shards.checked == list(range(552))
    assert len(progress['block_receipts']) == 24
    assert all(file_hash(r['path']) == r['sha256'] for r in progress['block_receipts'])
    assert file_hash(shards.root / 'outputs/benchmarks/hardware_contract.json') == legacy_hash
    assert not (shards.root / 'outputs/benchmarks/progress.json').exists()


def test_invalid_worker_evidence_blocks_only_after_other_work_finishes(shards):
    records = dispatch(shards); shards.invalid.add(0)
    for record in records: shards.statuses[record['job_id']] = {'state':'RUNNING'}
    shards.statuses[records[0]['job_id']] = {'state':'COMPLETED'}
    campaign._advance_postprocessing(shards.state)
    assert records[0]['state'] == 'failed' and shards.state['phase'] == 'latency'
    for record in records[1:]: shards.statuses[record['job_id']] = {'state':'COMPLETED'}
    campaign._advance_postprocessing(shards.state)
    assert shards.state['phase'] == 'blocked_latency_workers' and len(shards.finalized) == 23
    assert json.loads((shards.out / 'progress.json').read_text())['state'] == 'blocked'


def test_worker_registry_drift_stops_before_any_duplicate_submission(shards):
    records = dispatch(shards)
    records[0]['work'] = copy.deepcopy(records[1]['work'])
    campaign._advance_postprocessing(shards.state)
    assert shards.state['phase'] == 'blocked_latency_registry' and len(shards.calls) == 552


def test_cohort_selector_cannot_silently_change_hardware(shards):
    shards.state['latency_cohort']['gpu_gres_type'] = 'nvidia_h100_nvl'
    with pytest.raises(ValueError, match='homogeneous GPU class'):
        campaign._latency_selector(shards.state)
    shards.state['latency_cohort']['id'] = '../other'
    with pytest.raises(ValueError, match='safe identity'):
        campaign._latency_selector(shards.state)


def test_stop_and_explicit_resume_include_worker_shards(shards, monkeypatch):
    records = dispatch(shards)
    shards.state.update(phase='stopped', resume_phase='latency', controller_job='800')
    (shards.root / 'STOP').touch()
    shards.statuses['800'] = {'state':'CANCELLED'}
    for record in records: shards.statuses[record['job_id']] = {'state':'CANCELLED'}
    records[0]['transient_retries'] = 2
    monkeypatch.setattr(campaign, 'init_campaign', lambda: shards.state)
    campaign.launch(resume=True)
    assert all(r['job_id'] is None and r['attempt'] == r['segments'] == 1 for r in records)
    assert records[0]['transient_retries'] == 2 and not (shards.root / 'STOP').exists()
    assert shards.state['phase'] == 'latency'


def test_stop_before_dispatch_submits_no_worker(shards):
    shards.state['latency_prepare'] = dict(campaign._latency_record(), state='completed', job_id='900')
    (shards.root / 'STOP').touch()
    campaign._advance_postprocessing(shards.state)
    assert shards.calls == []


def test_cohort_node_exclusion_reaches_preparation_and_worker_requests(shards):
    shards.state['latency_cohort']['exclude_nodes']='dws-10'
    prepare=campaign._latency_record()
    campaign._submit_latency_record(shards.state,prepare)
    shards.state['latency_blocks_sha256']=shards.plan['blocks_sha256']
    worker=campaign._latency_record(block_id=0,worker_id=0)
    campaign._submit_latency_record(shards.state,worker)
    assert [call[2]['exclude_nodes'] for call in shards.calls]==['dws-10','dws-10']
