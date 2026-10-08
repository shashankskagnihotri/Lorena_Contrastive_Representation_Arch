"""Scoped dispatch, failed-worker continuation, and controller handoff ownership."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import types

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('recovery_controller',HERE/'controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    source=tmp_path/'validation_source';source.mkdir()
    for name in ['worker.py','validation.py']:(source/name).write_text(name)
    manifest=dict(scientific_source_hash=c.SCIENTIFIC_SOURCE_HASH,
        base_source_hash='7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0',
        policy={'version':'bound-test-policy'},files={name:c.file_hash(source/name) for name in ['worker.py','validation.py']})
    c.atomic_json(source/'source_manifest.json',manifest)
    proof=tmp_path/'validated.json';c.atomic_json(proof,dict(state='passed',source_path=str(source),source_manifest_sha256=c.file_hash(source/'source_manifest.json')))
    recovery=dict(validation_source_hash=c.digest(manifest),source_path=str(source),validation_receipt_path=str(proof),validation_receipt_sha256=c.file_hash(proof))
    config=tmp_path/'training.json';c.atomic_json(config,dict(recipe={'gpu_partitions':'gpu-vram-48gb'}))
    case=dict(dataset='mnist',architecture='vit_small',representation='raw',seed=1,
        trained_execution='dense',inference_execution='compact',sparsification_domain='raw_pixels',
        source_hash=c.SCIENTIFIC_SOURCE_HASH,inference_sparsity_percent=20,test_config_hash='case',training_config_path=str(config))
    case['test_config_hash']=c.digest({k:v for k,v in case.items() if k!='test_config_hash'})
    record=dict(identity='worker',test_config_hash=case['test_config_hash'],batch_size=8,execution='compact',shard=0,num_shards=1,
        validation_recovery=recovery,state='running',job_id='previous',attempt=2,segments=1,transient_retries=0)
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'load_case',lambda _:case)
    monkeypatch.setattr(c,'stop_requested',lambda:False);monkeypatch.setattr(c,'recover_lock',lambda _:None)
    monkeypatch.setattr(c,'publish',lambda _:None)
    return source,case,record


def test_original_dispatch_is_unchanged(setup):
    _,_,record=setup;record.pop('validation_recovery')
    assert c.command_for(record,'latency')[:4]==[c.PYTHON,'-m','dense_sparse.worker','latency']


def test_explicit_recovery_command_and_source_are_bound(setup):
    source,_,record=setup
    assert c.command_for(record,'latency')[:2]==[c.PYTHON,str(source/'worker.py')]
    assert c.worker_submission_source(record,{'source_path':'original'})==source
    with pytest.raises(ValueError,match='cannot change evaluation'):c.command_for(record,'evaluation')


@pytest.mark.parametrize('field,value',[('seed',0),('inference_sparsity_percent',80),('representation','grayscale')])
def test_dispatch_rejects_undiagnosed_scope(setup,field,value):
    _,case,record=setup;case[field]=value
    with pytest.raises(ValueError,match='sixteen'):c.command_for(record,'latency')


def test_source_byte_tampering_is_rejected(setup):
    source,_,record=setup;(source/'worker.py').write_text('changed')
    with pytest.raises(ValueError,match='source bytes changed'):c.command_for(record,'latency')


def test_walltime_continuation_retains_recovery_identity_and_command(setup,monkeypatch):
    source,case,record=setup;before=copy.deepcopy(record['validation_recovery']);launched=[]
    state=dict(latency_workers=[record],history=[],source_path=str(c.SCIENTIFIC_ROOT))
    monkeypatch.setattr(c,'observed',lambda ids:{'previous':{'state':'TIMEOUT','exit_code':'0:0'}})
    monkeypatch.setattr(c,'submit',lambda *a,**kw:launched.append((a,kw)) or 'continued')
    assert c.advance(state,'latency',{'cases':[case]}) is False
    assert record['validation_recovery']==before and record['segments']==2 and record['attempt']==3
    assert launched[0][0][0][:2]==[c.PYTHON,str(source/'worker.py')]
    assert launched[0][0][3]==source and record['job_id']=='continued'


def test_controller_handoff_executes_same_orchestration_snapshot(setup,monkeypatch):
    source,case,record=setup
    provenance=dict(orchestration_source_hash='orchestration',orchestration_source_path=str(c.ORCHESTRATION_ROOT))
    state=dict(phase='latency',manifest_hash='manifest',source_hash=c.SCIENTIFIC_SOURCE_HASH,
        source_path=str(c.SCIENTIFIC_ROOT),**provenance)
    monkeypatch.setenv('SLURM_JOB_ID','owner')
    monkeypatch.setattr(c,'orchestration_receipt',lambda:provenance)
    monkeypatch.setattr(c,'initialize',lambda:state)
    monkeypatch.setattr(c,'load_manifest',lambda:{'manifest_hash':'manifest'})
    monkeypatch.setattr(c,'verify_source',lambda:c.SCIENTIFIC_SOURCE_HASH)
    times=iter([0.,1.,72001.]);monkeypatch.setattr(c,'time',types.SimpleNamespace(time=lambda:next(times)))
    launched=[];monkeypatch.setattr(c,'submit',lambda *a,**kw:launched.append((a,kw)) or 'next')
    c.supervise()
    args,kwargs=launched[0]
    assert args[0]==[c.PYTHON,str(c.ORCHESTRATION_ROOT/'controller.py'),'supervise']
    assert args[3]==c.ORCHESTRATION_ROOT and kwargs['dependency']=='afterany:owner'
    assert state['next_controller_job']=='next' and state['source_hash']==c.SCIENTIFIC_SOURCE_HASH


def test_controller_diff_is_confined_to_dispatch_provenance_and_handoff():
    def functions(path):return {x.name:ast.dump(x) for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)}
    original=functions(c.SCIENTIFIC_ROOT/'dense_sparse/controller.py');updated=functions(HERE/'controller.py')
    changed={name for name in original if original[name]!=updated[name]}
    assert changed=={'command_for','advance','supervise','launch','main','verify_receipt'}
    assert set(updated)-set(original)=={'orchestration_receipt','require_orchestration','recovery_source','worker_submission_source','verify_recovery_cell'}


@pytest.mark.parametrize('mutation',[None,'archive','source','measurement'])
def test_cell_amendment_binds_archive_source_and_measurement(setup,tmp_path,mutation):
    source,case,record=setup
    manifest_path=source/'source_manifest.json';manifest=c.read_json(manifest_path)
    cell=tmp_path/'cells'/'clean';cell.mkdir(parents=True)
    archive=cell/'reference.npz';archive.write_bytes(b'saved reference evidence')
    row=dict(identity={'test_config_hash':case['test_config_hash']},measurement={'cell':'clean','batch_size':8})
    saved=dict(identity=copy.deepcopy(row['identity']),measurement=copy.deepcopy(row['measurement']),
        source=dict(validation_source_hash=record['validation_recovery']['validation_source_hash'],
            validation_source_path=str(source),validation_manifest_path=str(manifest_path),
            validation_manifest_sha256=c.file_hash(manifest_path),scientific_source_hash=c.SCIENTIFIC_SOURCE_HASH,
            base_source_hash=manifest['base_source_hash'],policy=manifest['policy']),
        state='reference_verified_rounding',fallback_used=True,original_comparison={'all_finite':True},
        original_strict_failure='original FP32 assertion failed',reference_archive=str(archive),
        reference_archive_sha256=c.file_hash(archive),reference_comparisons={'semantic':{'all_finite':True}},
        reference_cleanup=dict(helper_frame_released=True,rng_restored=True,
            synchronize_and_empty_cache_before_primary_warmup=True,allocated_bytes_before=100,allocated_bytes_after=100))
    if mutation=='source':saved['source']['policy']={'version':'unapproved-policy'}
    if mutation=='measurement':saved['measurement']['batch_size']=1
    receipt=cell/'validation.json';c.atomic_json(receipt,saved)
    row['validation_amendment']=dict(path=str(receipt),sha256=c.file_hash(receipt),
        validation_source_hash=record['validation_recovery']['validation_source_hash'],state=saved['state'])
    if mutation=='archive':archive.write_bytes(b'changed after receipt commit')
    if mutation is None:c.verify_recovery_cell(row,record,cell/'summary.json')
    else:
        with pytest.raises(ValueError,match='archive changed|identity/policy/state differs'):
            c.verify_recovery_cell(row,record,cell/'summary.json')


@pytest.mark.parametrize('bad_archive',[False,True])
def test_final_completion_requires_fresh_all_sixteen_amendment_verifications(setup,monkeypatch,bad_archive):
    _,case,record=setup
    records=[dict(copy.deepcopy(record),identity=f'worker-{i}',state='completed') for i in range(16)]
    provenance=dict(orchestration_source_hash='orchestration',orchestration_source_path=str(c.ORCHESTRATION_ROOT))
    state=dict(phase='final_report',manifest_hash='manifest',source_hash=c.SCIENTIFIC_SOURCE_HASH,
        source_path=str(c.SCIENTIFIC_ROOT),latency_workers=records,**provenance)
    monkeypatch.setenv('SLURM_JOB_ID','owner')
    monkeypatch.setattr(c,'orchestration_receipt',lambda:provenance)
    monkeypatch.setattr(c,'initialize',lambda:state)
    monkeypatch.setattr(c,'load_manifest',lambda:{'manifest_hash':'manifest','cases':[case]})
    monkeypatch.setattr(c,'verify_source',lambda:c.SCIENTIFIC_SOURCE_HASH)
    monkeypatch.setattr(c,'report_step',lambda *args:True)
    monkeypatch.setattr(c,'update_scratchpad',lambda *args:None)
    monkeypatch.setattr(c,'time',types.SimpleNamespace(time=lambda:0.,sleep=lambda _:None))
    checked=[]
    def verify(path,selected,worker,kind):
        assert state['phase']=='final_report' and kind=='latency' and selected==case
        checked.append(worker['identity'])
        if bad_archive and len(checked)==9:raise ValueError('Recovery numerical reference archive changed')
        return 'verified-'+worker['identity']
    monkeypatch.setattr(c,'verify_receipt',verify)
    if bad_archive:
        with pytest.raises(ValueError,match='archive changed'):c.supervise()
        assert state['phase']=='final_report' and 'validation_recovery_final_verification' not in state
    else:
        c.supervise()
        assert state['phase']=='completed' and len(checked)==16
        audit=state['validation_recovery_final_verification']
        assert audit['state']=='passed' and audit['worker_count']==16
        assert [r['identity'] for r in audit['workers']]==checked
