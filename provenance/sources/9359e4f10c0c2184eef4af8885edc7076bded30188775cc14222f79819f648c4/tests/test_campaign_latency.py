"""Mocked Slurm tests for isolated matched-block scheduling, with no GPU work."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparse_contrast import benchmark, campaign
from sparse_contrast.common import atomic_json, digest, file_hash
from sparse_contrast.scope import read_scope, expected_worker_count, expected_workers_per_block
from test_campaign import primary_runs

_REAL_LOAD_FROZEN_BLOCKS=benchmark.load_frozen_blocks


@pytest.fixture(params=[False,True],ids=['legacy','mnist_grayscale'])
def latency(tmp_path, monkeypatch, request):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'stop_requested',lambda: (tmp_path/'STOP').exists())
    (tmp_path/'outputs').mkdir()
    scope=read_scope(tmp_path)
    if request.param:
        scope['active_representations']['mnist']=['grayscale']
        atomic_json(tmp_path/'experiment_scope.json',scope)
    main=[]
    for run in primary_runs():
        if run['representation']!='raw' and run['representation'] not in scope['active_representations'][run['dataset']]: continue
        config=dict(run,role='main',source_hash='source'); config['config_hash']=digest(config)
        config_path=tmp_path/'configs'/f"{config['config_hash']}.json"
        atomic_json(config_path,config)
        main.append(dict(config=config,config_path=str(config_path),
                         state='completed',stage='evaluation',job_id=None))
    saved=dict(phase='latency',source_hash='source',source_path='/frozen',pilots=[],main=main,
               history=[],controller={},controller_generation=0,latency_job=None)
    blocks=[]
    for dataset in ('mnist','cifar10'):
        for architecture in ('vit_small','swin_tiny'):
            for seed in (0,1,2):
                for batch in (1,8):
                    work=[]
                    for item in main:
                        config=item['config']
                        if (config['dataset'],config['architecture'],config['seed'])!=(dataset,architecture,seed): continue
                        for execution in ([config['execution'],'dense_masked'] if config['execution']=='compact' else [config['execution']]):
                            work.append(dict(config_path=item['config_path'],config_hash=config['config_hash'],
                                             execution=execution,batch_size=batch))
                    blocks.append(dict(id=len(blocks),dataset=dataset,architecture=architecture,seed=seed,batch_size=batch,work=work))
    plan=dict(schema_version=1,source_hash='source',measurement_source_hash='source',scope=scope,scope_sha256=digest(scope),
              frozen_work_sha256=digest([w for b in blocks for w in b['work']]),
              order_sha256='order',blocks=blocks)
    plan['blocks_sha256']=digest(plan)
    contract=dict(gpu_partition='gpu-vram-94gb',gpu_gres_type='h100nvl',contract={'hardware_identity':'hardware'})
    atomic_json(tmp_path/'outputs/benchmarks/hardware_contract.json',contract)
    metadata=dict(hardware_identity='hardware',measurement_session_id='prepare',device_uuid='GPU-fixture')
    metadata_path=tmp_path/'outputs/benchmarks/preparation/hardware.json'; atomic_json(metadata_path,metadata)
    measurement=dict(hardware_metadata_file=str(metadata_path),hardware_metadata_sha256=file_hash(metadata_path),
                     hardware_identity='hardware',measurement_session_id='prepare',measurement_device_uuid='GPU-fixture')
    atomic_json(tmp_path/'outputs/benchmarks/prepared.json',dict(state='completed',source_hash='source',measurement_source_hash='source',
        blocks_sha256=plan['blocks_sha256'],order_sha256=plan['order_sha256'],expected_blocks=24,expected_workers=expected_worker_count(scope),
        hardware_contract_sha256=file_hash(tmp_path/'outputs/benchmarks/hardware_contract.json'),measurement=measurement,
        isolation=dict(verified=True,device_uuid='GPU-fixture',observed_compute_pids=[1],self_pid=1)))
    monkeypatch.setattr(benchmark,'load_frozen_blocks',lambda root: copy.deepcopy(plan),raising=False)
    receipts={}
    for block in blocks:
        receipt=dict(state='completed',block_id=block['id'],source_hash='source',blocks_sha256=plan['blocks_sha256'],
                     order_sha256='order',expected_workers=expected_workers_per_block(block['dataset'],scope),workers=[dict(w,worker_summary_path='fixture',worker_summary_sha256='fixture') for w in block['work']])
        receipts[block['id']]=receipt
        atomic_json(tmp_path/'outputs/benchmarks/blocks'/f"block_{block['id']:03d}"/'completed.json',receipt)
    monkeypatch.setattr(benchmark,'validate_block_completion',lambda block_id,root: copy.deepcopy(receipts[block_id]),raising=False)
    calls=[]; statuses={}
    def submit(command,name,**kwargs):
        calls.append((command,name,kwargs))
        return str(1000+len(calls))
    monkeypatch.setattr(campaign,'submit',submit)
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {job:statuses[job] for job in jobs if job in statuses})
    return SimpleNamespace(root=tmp_path,state=saved,plan=plan,receipts=receipts,calls=calls,statuses=statuses,scope=scope)


def prepared(latency):
    latency.state['latency_prepare']=dict(campaign._latency_record(),state='completed',job_id='900')
    campaign._advance_postprocessing(latency.state)
    assert len(latency.calls)==24
    return latency.state['latency_blocks']


def test_exact_gpu_gres_type_is_in_submission_intent_and_sbatch(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'stop_requested',lambda: False)
    calls=[]
    monkeypatch.setattr(campaign.subprocess,'run',lambda args,**kwargs: calls.append(args) or SimpleNamespace(stdout='51',stderr=''))
    assert campaign.submit(['worker'],'latency',gpu_partition='gpu-vram-94gb',gpu_type='h100nvl',submission_key='typed')=='51'
    assert '--gres=gpu:h100nvl:1' in calls[0] and '--partition=gpu-vram-94gb' in calls[0]
    intent=json.loads(next((tmp_path/'outputs/submissions').glob('*.json')).read_text())
    assert intent['request']['gpu_type']=='h100nvl'
    with pytest.raises(ValueError):
        campaign.submit(['worker'],'latency',gpu_partition='gpu-vram-94gb',gpu_type='a100',submission_key='typed')
    with pytest.raises(ValueError): campaign.submit(['worker'],'latency',gpu_type='a100:2')
    assert len(calls)==1


def test_prepare_is_only_submission_until_terminal_success(latency):
    campaign._advance_postprocessing(latency.state)
    assert len(latency.calls)==1 and '--prepare-campaign' in latency.calls[0][0]
    prepare=latency.state['latency_prepare']; job=prepare['job_id']
    campaign._advance_postprocessing(latency.state)  # unknown ownership is retained
    assert len(latency.calls)==1 and prepare['state']=='scheduler_unknown'
    latency.statuses[job]={'state':'RUNNING'}
    campaign._advance_postprocessing(latency.state)
    assert len(latency.calls)==1 and 'latency_blocks' not in latency.state
    latency.statuses[job]={'state':'COMPLETED'}
    campaign._advance_postprocessing(latency.state)
    assert len(latency.calls)==25 and len(latency.state['latency_blocks'])==24
    assert latency.state['phase']=='latency'
    assert all(call[2]['gpu_partition']=='gpu-vram-94gb' and call[2]['gpu_type']=='h100nvl' for call in latency.calls[1:])
    assert {int(call[0][-1]) for call in latency.calls[1:]}==set(range(24))


def test_restart_preserves_24_block_ownership_without_duplicate_jobs(latency):
    records=prepared(latency)
    for record in records: latency.statuses[record['job_id']]={'state':'RUNNING'}
    restarted=json.loads((latency.root/'outputs/campaign.json').read_text())
    campaign._advance_postprocessing(restarted)
    assert len(latency.calls)==24 and all(record['state']=='running' for record in restarted['latency_blocks'])
    assert restarted['latency_blocks_sha256']==latency.plan['blocks_sha256']


def test_transient_block_retry_has_persistent_backoff_and_timeout_continues(latency, monkeypatch):
    records=prepared(latency); now=[100.]
    monkeypatch.setattr(campaign.time,'time',lambda: now[0])
    for record in records: latency.statuses[record['job_id']]={'state':'RUNNING'}
    failed,timed=records[:2]
    latency.statuses[failed['job_id']]={'state':'NODE_FAIL'}
    latency.statuses[timed['job_id']]={'state':'TIMEOUT'}
    campaign._advance_postprocessing(latency.state)
    assert failed['transient_retries']==1 and failed['retry_not_before']==160 and failed['attempt']==1 and failed['job_id'] is None
    assert timed['segments']==1 and timed['transient_retries']==0 and timed['attempt']==1
    assert len(latency.calls)==25
    now[0]=130
    campaign._advance_postprocessing(latency.state)
    assert len(latency.calls)==25
    now[0]=160
    campaign._advance_postprocessing(latency.state)
    assert len(latency.calls)==26 and failed['job_id'] is not None
    assert latency.calls[-1][2]['submission_key'].endswith(':block_000:1')


def test_failed_block_does_not_stop_other_valid_blocks_and_retry_limit_persists(latency):
    records=prepared(latency)
    records[0]['transient_retries']=3
    latency.statuses[records[0]['job_id']]={'state':'NODE_FAIL'}
    for record in records[1:]: latency.statuses[record['job_id']]={'state':'RUNNING'}
    campaign._advance_postprocessing(latency.state)
    assert records[0]['state']=='failed' and records[0]['transient_retries']==3
    assert latency.state['phase']=='latency' and all(record['state']=='running' for record in records[1:])
    for record in records[1:]: latency.statuses[record['job_id']]={'state':'COMPLETED'}
    campaign._advance_postprocessing(latency.state)
    assert latency.state['phase']=='blocked_latency_blocks'
    assert all(record['state']=='completed' for record in records[1:]) and len(latency.calls)==24


def test_all_24_verified_receipts_enable_report_and_exact_scoped_coverage(latency):
    records=prepared(latency)
    for record in records: latency.statuses[record['job_id']]={'state':'COMPLETED'}
    campaign._advance_postprocessing(latency.state)
    assert latency.state['phase']=='report'
    progress=json.loads((latency.root/'outputs/benchmarks/progress.json').read_text())
    assert progress['state']=='completed' and progress['workers']==expected_worker_count(latency.scope) and progress['completed_blocks']==24
    assert len(progress['block_receipts'])==24
    assert all(file_hash(Path(receipt['path']))==receipt['sha256'] for receipt in progress['block_receipts'])


def test_completed_receipt_union_cannot_overlap_workers(latency):
    records=prepared(latency)
    latency.receipts[1]['workers'][0]=copy.deepcopy(latency.receipts[0]['workers'][0])
    for record in records: latency.statuses[record['job_id']]={'state':'COMPLETED'}
    campaign._advance_postprocessing(latency.state)
    assert latency.state['phase']=='blocked_latency_coverage'
    progress=json.loads((latency.root/'outputs/benchmarks/progress.json').read_text())
    assert progress['state']!='completed'


def test_exit_zero_with_invalid_block_evidence_fails_only_affected_block(latency, monkeypatch):
    records=prepared(latency)
    def validate(block_id,root):
        if block_id==0: raise ValueError('worker summary hash changed')
        return copy.deepcopy(latency.receipts[block_id])
    monkeypatch.setattr(benchmark,'validate_block_completion',validate)
    latency.statuses[records[0]['job_id']]={'state':'COMPLETED'}
    for record in records[1:]: latency.statuses[record['job_id']]={'state':'RUNNING'}
    campaign._advance_postprocessing(latency.state)
    assert records[0]['state']=='failed' and records[0]['failure']['kind']=='completion_evidence'
    assert latency.state['phase']=='latency' and all(record['state']=='running' for record in records[1:])


def test_prepare_cannot_release_mismatched_or_incomplete_plan(latency):
    latency.state['latency_prepare']=dict(campaign._latency_record(),job_id='901',state='submitted')
    latency.plan['blocks'][0]['work'].pop()
    latency.statuses['901']={'state':'COMPLETED'}
    campaign._advance_postprocessing(latency.state)
    assert latency.state['phase']=='blocked_latency_prepare' and latency.calls==[]


def test_hardware_contract_change_cannot_release_further_block_work(latency):
    prepared(latency)
    contract=latency.root/'outputs/benchmarks/hardware_contract.json'
    value=json.loads(contract.read_text()); value['gpu_gres_type']='a100'; atomic_json(contract,value)
    campaign._advance_postprocessing(latency.state)
    assert latency.state['phase']=='blocked_latency_plan' and len(latency.calls)==24


def test_stop_cancels_prepare_and_every_owned_block(latency, monkeypatch):
    records=prepared(latency)
    for record in records: latency.statuses[record['job_id']]={'state':'RUNNING'}
    latency.statuses['900']={'state':'COMPLETED'}
    atomic_json(latency.root/'outputs/campaign.json',latency.state)
    cancelled=[]
    monkeypatch.setattr(campaign.subprocess,'run',lambda args,**kwargs: cancelled.extend(args[1:]) or SimpleNamespace())
    campaign.stop()
    assert set(cancelled)=={record['job_id'] for record in records}
    assert (latency.root/'STOP').exists()
    saved=json.loads((latency.root/'outputs/campaign.json').read_text())
    assert saved['phase']=='stopped' and saved['resume_phase']=='latency'


def test_resume_replans_cancelled_blocks_without_resetting_retry_count(latency, monkeypatch):
    records=prepared(latency)
    records[0]['transient_retries']=2
    records[1]['state']='completed'
    latency.state.update(phase='stopped',resume_phase='latency',controller_job='800')
    (latency.root/'STOP').touch()
    latency.statuses['800']={'state':'CANCELLED'}
    latency.statuses['900']={'state':'COMPLETED'}
    for record in records: latency.statuses[record['job_id']]={'state':'CANCELLED'}
    latency.statuses[records[1]['job_id']]={'state':'COMPLETED'}
    monkeypatch.setattr(campaign,'init_campaign',lambda: latency.state)
    assert campaign.launch(resume=True)=='1025'
    assert records[0]['job_id'] is None and records[0]['attempt']==1 and records[0]['segments']==1
    assert records[0]['transient_retries']==2 and records[1]['state']=='completed'
    assert latency.state['phase']=='latency' and not (latency.root/'STOP').exists()


def test_resume_waits_for_positive_block_termination(latency, monkeypatch):
    records=prepared(latency)
    latency.state.update(phase='stopped',resume_phase='latency',controller_job='800')
    (latency.root/'STOP').touch()
    latency.statuses['800']={'state':'CANCELLED'}
    monkeypatch.setattr(campaign,'init_campaign',lambda: latency.state)
    with pytest.raises(RuntimeError,match='Latency block stop not yet reconciled'):
        campaign.launch(resume=True)
    assert records[0]['job_id'] is not None and (latency.root/'STOP').exists() and len(latency.calls)==24


def test_controller_uses_real_frozen_plan_helper_before_any_block_submission(latency, monkeypatch):
    out=latency.root/'outputs/benchmarks'
    flat=[work for block in latency.plan['blocks'] for work in block['work']]
    order=dict(schema_version=2,source_hash='source',measurement_source_hash='source',scope=latency.scope,scope_sha256=digest(latency.scope),
               work=flat,frozen_work_sha256=digest(flat))
    atomic_json(out/'order.json',order)
    latency.plan['order_sha256']=file_hash(out/'order.json')
    latency.plan['blocks_sha256']=digest({key:value for key,value in latency.plan.items() if key!='blocks_sha256'})
    atomic_json(out/'blocks.json',latency.plan)
    prep=json.loads((out/'prepared.json').read_text())
    prep.update(order_sha256=latency.plan['order_sha256'],blocks_sha256=latency.plan['blocks_sha256'])
    atomic_json(out/'prepared.json',prep)
    monkeypatch.setattr(benchmark,'load_frozen_blocks',_REAL_LOAD_FROZEN_BLOCKS)
    prepared(latency)
    assert len(latency.calls)==24
    plan=json.loads((out/'blocks.json').read_text()); plan['blocks'][0]['work'][0]['execution']='changed'
    atomic_json(out/'blocks.json',plan)
    campaign._advance_postprocessing(latency.state)
    assert latency.state['phase']=='blocked_latency_plan' and len(latency.calls)==24


@pytest.mark.parametrize('mutation',['missing_receipt','foreign_gpu_process'])
def test_prepare_exit_zero_requires_receipt_and_isolated_hardware(latency, mutation):
    path=latency.root/'outputs/benchmarks/prepared.json'
    if mutation=='missing_receipt': path.unlink()
    else:
        prep=json.loads(path.read_text()); prep['isolation']['observed_compute_pids'].append(999)
        atomic_json(path,prep)
    latency.state['latency_prepare']=dict(campaign._latency_record(),job_id='901',state='submitted')
    latency.statuses['901']={'state':'COMPLETED'}
    campaign._advance_postprocessing(latency.state)
    assert latency.state['phase']=='blocked_latency_prepare' and latency.calls==[]


def test_latency_progress_weights_each_completed_dataset_block(latency):
    records=prepared(latency)
    for record in records: record['state']='running'
    records[0]['state']='completed'
    records[12]['state']='completed'
    campaign._latency_progress(latency.state,latency.plan)
    progress=json.loads((latency.root/'outputs/benchmarks/progress.json').read_text())
    assert progress['workers']==expected_workers_per_block('mnist',latency.scope)+expected_workers_per_block('cifar10',latency.scope)
    assert progress['expected_workers']==expected_worker_count(latency.scope)
    assert progress['completed_blocks']==2


def test_latency_and_report_launch_measurement_snapshot_preserving_training_lineage(latency):
    latency.state.update(orchestration_source_hash='measurement_source',orchestration_source_path='/measurement_snapshot')
    record=campaign._latency_record()
    campaign._submit_latency_record(latency.state,record)
    assert latency.calls[-1][2]['source']=='/measurement_snapshot'
    assert latency.calls[-1][2]['submission_key']=='latency:measurement_source:prepare:0'
    latency.state['phase']='report'
    campaign._advance_postprocessing(latency.state)
    assert latency.calls[-1][2]['source']=='/measurement_snapshot'
    assert latency.calls[-1][2]['submission_key']=='report:measurement_source:0'
    assert latency.state['source_hash']=='source' and latency.state['source_path']=='/frozen'


def test_latency_plan_requires_matching_measurement_source(latency):
    latency.state.update(orchestration_source_hash='measurement_source',orchestration_source_path='/measurement_snapshot')
    with pytest.raises(ValueError,match='measurement source'):
        campaign._latency_plan(latency.state)
    latency.plan['measurement_source_hash']='measurement_source'
    prep_path=latency.root/'outputs/benchmarks/prepared.json'
    prep=json.loads(prep_path.read_text()); prep['measurement_source_hash']='measurement_source'
    atomic_json(prep_path,prep)
    assert campaign._latency_plan(latency.state)['source_hash']=='source'
