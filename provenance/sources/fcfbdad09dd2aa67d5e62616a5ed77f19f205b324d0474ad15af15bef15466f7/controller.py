"""Explicit orchestration extension: durable, scoped validation recovery dispatch."""
from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
import sys
SCIENTIFIC_SOURCE_HASH='8cf4d0c1e0ce5443d5080ea397df9a8f085d1bf2c4292912676545d38533a0cb'
SCIENTIFIC_ROOT=Path('/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/training_dense_testing_sparse/outputs/source')/SCIENTIFIC_SOURCE_HASH
sys.path.insert(0,str(SCIENTIFIC_ROOT))
from dense_sparse.common import *
from dense_sparse.scheduler import ACTIVE, TERMINAL, TRANSIENT, observed, submit, recover_lock



ORCHESTRATION_ROOT=Path(__file__).resolve().parent


def orchestration_receipt():
    path=ORCHESTRATION_ROOT/'orchestration_manifest.json'
    saved=read_json(path)
    if saved.get('scientific_source_hash')!=SCIENTIFIC_SOURCE_HASH or set(saved['files'])!={'controller.py'}:
        raise ValueError('Unexpected recovery orchestration manifest')
    if file_hash(ORCHESTRATION_ROOT/'controller.py')!=saved['files']['controller.py']:
        raise ValueError('Recovery orchestration source changed')
    return dict(orchestration_source_hash=digest(saved),orchestration_source_path=str(ORCHESTRATION_ROOT))


def require_orchestration(state):
    expected=orchestration_receipt()
    if any(state.get(k)!=v for k,v in expected.items()):
        raise ValueError('Controller orchestration identity differs from explicit campaign amendment')
    if state['source_hash']!=SCIENTIFIC_SOURCE_HASH or Path(state['source_path']).resolve()!=SCIENTIFIC_ROOT:
        raise ValueError('Original scientific source identity must remain unchanged')
    return expected


def recovery_source(record):
    specification=record.get('validation_recovery')
    if specification is None:return None
    required={'validation_source_hash','source_path','validation_receipt_path','validation_receipt_sha256'}
    if set(specification)!=required:raise ValueError('Incomplete validation recovery provenance')
    case=load_case(record['test_config_hash'])
    allowed=dict(dataset='mnist',architecture='vit_small',representation='raw',seed=1,
                 trained_execution='dense',inference_execution='compact',sparsification_domain='raw_pixels',
                 source_hash=SCIENTIFIC_SOURCE_HASH)
    if (any(case.get(k)!=v for k,v in allowed.items()) or case['inference_sparsity_percent'] not in range(0,71,10)
            or record['batch_size']!=8 or record['execution'] not in ('compact','dense_masked')
            or record['shard']!=0 or record['num_shards']!=1):
        raise ValueError('Recovery dispatch is limited to the sixteen diagnosed worker identities')
    source=Path(specification['source_path']).resolve();manifest_path=source/'source_manifest.json';manifest=read_json(manifest_path)
    if (digest(manifest)!=specification['validation_source_hash'] or
            manifest['scientific_source_hash']!=SCIENTIFIC_SOURCE_HASH or
            manifest['base_source_hash']!='7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0' or
            set(manifest['files'])!={'worker.py','validation.py'}):
        raise ValueError('Recovery worker source identity changed')
    for name,expected in manifest['files'].items():
        if file_hash(source/name)!=expected:raise ValueError('Recovery worker source bytes changed')
    proof_path=Path(specification['validation_receipt_path'])
    if file_hash(proof_path)!=specification['validation_receipt_sha256']:
        raise ValueError('Allocated recovery validation receipt changed')
    proof=read_json(proof_path)
    if (proof['state']!='passed' or Path(proof['source_path']).resolve()!=source or
            proof['source_manifest_sha256']!=file_hash(manifest_path)):
        raise ValueError('Recovery worker lacks matching allocated validation')
    return source


def worker_submission_source(record,state):
    source=recovery_source(record)
    return source if source is not None else state['source_path']


def verify_recovery_cell(row,record,cell_path):
    """Dereference every extra receipt/archive before recovered work is accepted."""
    source=recovery_source(record);manifest_path=source/'source_manifest.json';manifest=read_json(manifest_path)
    ref=row['validation_amendment'];path=Path(ref['path'])
    path.resolve().relative_to(cell_path.parent.resolve())
    if file_hash(path)!=ref['sha256']:raise ValueError('Recovery validation receipt hash changed')
    saved=read_json(path)
    expected=dict(validation_source_hash=record['validation_recovery']['validation_source_hash'],
        validation_source_path=str(source),validation_manifest_path=str(manifest_path),
        validation_manifest_sha256=file_hash(manifest_path),scientific_source_hash=SCIENTIFIC_SOURCE_HASH,
        base_source_hash=manifest['base_source_hash'],policy=manifest['policy'])
    if (saved['identity']!=row['identity'] or saved['measurement']!=row['measurement'] or
            saved['source']!=expected or ref['validation_source_hash']!=expected['validation_source_hash'] or
            saved['state'] not in ('original_strict_pass','reference_verified_rounding') or
            ref['state']!=saved['state'] or saved['original_comparison'].get('all_finite') is not True or
            saved['fallback_used']!=(saved['state']=='reference_verified_rounding')):
        raise ValueError('Recovery validation identity/policy/state differs')
    if saved['fallback_used']:
        archive=Path(saved['reference_archive']);archive.resolve().relative_to(cell_path.parent.resolve())
        if file_hash(archive)!=saved['reference_archive_sha256']:
            raise ValueError('Recovery numerical reference archive changed')
        cleanup=saved['reference_cleanup']
        if (not saved['original_strict_failure'] or
                any(value.get('all_finite') is not True for value in saved['reference_comparisons'].values()) or
                not all(cleanup.get(k) is True for k in ('helper_frame_released','rng_restored','synchronize_and_empty_cache_before_primary_warmup')) or
                cleanup['allocated_bytes_after']>cleanup['allocated_bytes_before']):
            raise ValueError('Recovery reference/memory validation was not successful')

def parent_gate():
    state=read_json(BASE_ROOT/'outputs/campaign.json')
    coverage_path=BASE_ROOT/'outputs/tables/coverage.json'
    coverage=read_json(coverage_path) if coverage_path.exists() else {}
    monitor_path=BASE_ROOT/'outputs/agent_monitor/state.json'
    monitor=read_json(monitor_path) if monitor_path.exists() else {}
    main=state.get('main',[]);workers=state.get('latency_workers',[])
    ready=(state.get('phase')=='completed' and len(main)==156 and
           all(x.get('state')=='completed' and x.get('stage')=='evaluation' for x in main) and
           len(workers)==552 and all(x.get('state')=='completed' for x in workers) and
           coverage.get('all_requested_work_complete') is True and monitor.get('state')=='completed_verified')
    return ready,{'phase':state.get('phase'),'trained_and_evaluated':sum(x.get('state')=='completed' and x.get('stage')=='evaluation' for x in main),
                  'latency_completed':sum(x.get('state')=='completed' for x in workers),'latency_expected':552,
                  'coverage_complete':coverage.get('all_requested_work_complete',False),
                  'final_verification':monitor.get('state'),'checked':now()}


def initialize():
    path=ROOT/'outputs/campaign.json'
    if path.exists():
        return read_json(path)
    manifest=load_manifest()
    validation=read_json(ROOT/'outputs/checks/validation.json')
    if validation.get('state')!='passed' or validation.get('source_hash')!=manifest['source_hash'] or validation.get('manifest_hash')!=manifest['manifest_hash']:
        raise ValueError('Allocated validation does not match this frozen study')
    audit_path=ROOT/'outputs/checks/zero_checkpoint_audit.json'
    if file_hash(audit_path)!=validation['checkpoint_audit_sha256']:
        raise ValueError('Source-checkpoint audit receipt changed')
    audit=read_json(audit_path)
    wanted={(x['training_config_hash'],x['checkpoint_sha256']) for x in manifest['sources']}
    audited={(x['config_hash'],x['checkpoint_sha256']) for x in audit['checkpoints']}
    if audit['state']!='complete' or len(audit['checkpoints'])!=60 or audited!=wanted:
        raise ValueError('Source checkpoint audit is incomplete')
    def record(identity,**extra):
        return dict(identity=identity,state='unsubmitted',job_id=None,attempt=0,transient_retries=0,segments=0,**extra)
    state=dict(schema_version=1,study_id=STUDY_ID,manifest_hash=manifest['manifest_hash'],source_hash=manifest['source_hash'],
               source_path=manifest['source_path'],created=now(),phase='waiting_parent',controller_generation=0,
               evaluation=[record(c['test_config_hash'],test_config_hash=c['test_config_hash']) for c in manifest['cases']],
               latency_workers=[record(w['worker_id'],**w) for w in manifest['latency_workers']],history=[])
    atomic_json(path,state)
    return state


def publish(state):
    state['updated']=now()
    atomic_json(ROOT/'outputs/campaign.json',state)
    value={k:state[k] for k in ('study_id','phase','updated','source_hash','manifest_hash')}
    value['evaluation']=dict(Counter(x['state'] for x in state['evaluation']))
    value['latency_workers']=dict(Counter(x['state'] for x in state['latency_workers']))
    value['controller']=state.get('controller');value['parent']=state.get('parent');value['blocked']=state.get('blocked')
    atomic_json(ROOT/'STATUS.json',value)


def update_scratchpad(state):
    """Refresh one bounded status block in the user-requested project record."""
    if time.time()-state.get('scratchpad_updated_at',0)<900:
        return
    path=ROOT.parent/'debugging/scratchpad.md'
    begin='<!-- dense-sparse-live:start -->';end='<!-- dense-sparse-live:end -->'
    before=path.read_text()
    block=(begin+'\n'+f"Follow-up live controller update {now()}: phase `{state['phase']}`; "
           +f"evaluation {dict(Counter(r['state'] for r in state['evaluation']))}; "
           +f"latency workers {dict(Counter(r['state'] for r in state['latency_workers']))}. "
           +f"Controller job {state.get('controller_job')}. "
           +'Exact identities/failures: training_dense_testing_sparse/STATUS.json and outputs/campaign.json.\n'+end)
    if begin in before and end in before:
        start=before.index(begin);stop=before.index(end,start)+len(end)
        after=before[:start]+block+before[stop:]
    else:
        after=before+'\n\n'+block+'\n'
    # Refuse to replace if a collaborator changed the file while composing.
    if path.read_text()!=before:
        return
    temporary=path.with_name('.scratchpad.dense-sparse.'+str(os.getpid())+'.tmp')
    temporary.write_text(after);os.replace(temporary,path)
    state['scratchpad_updated_at']=time.time()


def summary_path(record,kind,cases):
    rd=case_dir(cases[record['test_config_hash']])
    if kind=='evaluation':
        return rd/'evaluation/summary.json'
    return rd/'latency'/record['execution']/f"batch{record['batch_size']}"/f"shard{record['shard']}"/'summary.json'


def verify_receipt(path,case,record=None,kind=None):
    from sparse_contrast.data import corruption_cells
    value=read_json(path)
    if value.get('state')!='complete':
        raise ValueError(f'Incomplete worker receipt: {path}')
    identity=value['identity']
    fields=('test_config_hash','training_config_hash','checkpoint_sha256','completed_receipt_sha256',
            'training_source_hash','normalization_hash','source_hash','dataset','architecture','representation',
            'seed','trained_execution','inference_execution','inference_sparsity_percent','sparsification_domain')
    if any(identity.get(key)!=case[key] for key in fields):
        raise ValueError(f'Worker identity mismatch: {path}')
    if record and kind=='latency':
        for key in ('execution','batch_size','shard','num_shards'):
            if identity.get(key)!=record[key]:
                raise ValueError(f'Latency worker setting mismatch: {key}')
    expected={('clean','test')}|{(c,str(s)) for c,s in corruption_cells(case['dataset'])}
    refs=value['cells'];actual=[(r['corruption'],str(r['severity'])) for r in refs]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Worker receipt does not cover the official cell matrix')
    for ref in refs:
        cell_path=Path(ref['path'])
        cell_path.resolve().relative_to(path.parent.resolve())
        if file_hash(cell_path)!=ref['sha256']:
            raise ValueError('Committed cell summary changed')
        row=read_json(cell_path)
        if row['identity']!=identity or row['state']!='complete':
            raise ValueError('Cell/worker identity disagreement')
        if record and record.get('validation_recovery') is not None:
            verify_recovery_cell(row,record,cell_path)
    return file_hash(path)


def command_for(record,kind):
    command=[PYTHON,'-m','dense_sparse.worker','evaluate' if kind=='evaluation' else 'latency',
             '--case',str(ROOT/'configs'/f"{record['test_config_hash']}.json")]
    if record.get('validation_recovery') is not None:
        if kind!='latency':raise ValueError('Validation amendment cannot change evaluation dispatch')
        command=[PYTHON,str(recovery_source(record)/'worker.py'),'--case',str(ROOT/'configs'/f"{record['test_config_hash']}.json")]
    if kind=='latency':
        command+=['--execution',record['execution'],'--batch-size',str(record['batch_size']),
                  '--shard',str(record['shard']),'--num-shards',str(record['num_shards'])]
    return command


def advance(state,kind,manifest):
    records=state['evaluation' if kind=='evaluation' else 'latency_workers']
    cases={x['test_config_hash']:x for x in manifest['cases']}
    unfinished=[r for r in records if r['state']!='completed']
    statuses=observed([r['job_id'] for r in unfinished if r.get('job_id')])
    for record in unfinished:
        if record['state']=='blocked':
            continue
        job=record.get('job_id')
        if not job:
            continue
        seen=statuses.get(str(job))
        if seen is None:
            record['unknown_since']=record.get('unknown_since',time.time())
            if time.time()-record['unknown_since']>900:
                record['state']='blocked';record['failure']={'reason':'scheduler_identity_unknown','job_id':job}
            continue
        record.pop('unknown_since',None)
        status=seen['state'];record['scheduler']=seen
        if status in ACTIVE:
            record['state']='running' if status=='RUNNING' else 'pending'
            continue
        if status=='COMPLETED' and seen.get('exit_code')=='0:0':
            path=summary_path(record,kind,cases)
            try:
                record['summary_sha256']=verify_receipt(path,cases[record['test_config_hash']],record,kind)
                record.update(state='completed',summary_path=str(path),completed=now())
            except (OSError,ValueError,KeyError) as error:
                record.update(state='blocked',failure={'reason':'invalid_completed_receipt','error':repr(error),'job_id':job})
            publish(state)
            continue
        history=dict(time=now(),identity=record['identity'],job_id=job,observed=seen,attempt=record['attempt'])
        if status in TRANSIENT and record['transient_retries']<3:
            record['transient_retries']+=1
            record.update(job_id=None,state='unsubmitted',attempt=record['attempt']+1,
                          not_before=time.time()+min(900,60*2**record['transient_retries']))
            history['action']='transient_retry'
        elif status=='TIMEOUT' and record['segments']<12:
            record['segments']+=1
            record.update(job_id=None,state='unsubmitted',attempt=record['attempt']+1)
            history['action']='walltime_continuation_from_committed_cells'
        else:
            record.update(state='blocked',failure={'reason':'worker_failed','observed':seen,'job_id':job})
            history['action']='diagnosis_required'
        state['history'].append(history);publish(state)
    # Keep independent work moving even if one deterministic failure needs repair.
    active=sum(r['state'] in ('running','pending') for r in records)
    launches=0
    for record in records:
        if active>=128 or launches>=64 or stop_requested():
            break
        if record['state']!='unsubmitted' or time.time()<record.get('not_before',0):
            continue
        case=cases[record['test_config_hash']]
        lock=summary_path(record,kind,cases).parent/'writer.lock'
        recover_lock(lock)
        job=submit(command_for(record,kind),kind, f"{kind}:{record['identity']}:{record['attempt']}",worker_submission_source(record,state),
                   gpu=True,timing=kind=='latency',memory_gb=32,time_limit='1-00:00:00',
                   gpu_partitions=read_json(case['training_config_path'])['recipe']['gpu_partitions'] if kind=='evaluation' else None)
        record.update(job_id=job,state='pending',submitted=now());active+=1;launches+=1;publish(state)
    return all(r['state']=='completed' for r in records)


def report_step(state,kind):
    key=kind+'_report'
    record=state.setdefault(key,dict(job_id=None,state='unsubmitted',attempt=0,transient_retries=0,segments=0))
    if record['state']=='completed':
        return True
    if record['state']=='blocked':
        return False
    if record.get('job_id') is None:
        if time.time()<record.get('not_before',0):
            return False
        job=submit([PYTHON,'-m','dense_sparse.report'],kind+'-report',f"{kind}-report:{record['attempt']}",state['source_path'],
                   memory_gb=96,time_limit='12:00:00')
        record.update(job_id=job,state='pending');publish(state);return False
    seen=observed([record['job_id']]).get(record['job_id'])
    if seen is None:
        record['unknown_since']=record.get('unknown_since',time.time())
        if time.time()-record['unknown_since']>900:
            record.update(state='blocked',failure={'reason':'report_scheduler_identity_unknown'})
        publish(state);return False
    record.pop('unknown_since',None)
    if seen['state'] in ACTIVE:
        record['state']='running' if seen['state']=='RUNNING' else 'pending'
        return False
    if seen['state']!='COMPLETED' or seen.get('exit_code')!='0:0':
        history={'time':now(),'action':'report_failed','kind':kind,'job_id':record['job_id'],'observed':seen}
        if seen['state'] in TRANSIENT and record['transient_retries']<3:
            record['transient_retries']+=1
            record.update(job_id=None,state='unsubmitted',attempt=record['attempt']+1,not_before=time.time()+120)
            history['action']='report_transient_retry'
        elif seen['state']=='TIMEOUT' and record['segments']<3:
            record['segments']+=1
            record.update(job_id=None,state='unsubmitted',attempt=record['attempt']+1)
            history['action']='report_walltime_continuation'
        else:
            record.update(state='blocked',failure=seen)
        state['history'].append(history);publish(state);return False
    coverage=read_json(ROOT/'outputs/tables/coverage.json')
    expected=load_manifest()['expected']
    if (coverage.get('accuracy_complete') is not True or
            coverage.get('completed_evaluation_cases')!=expected['evaluation_cases'] or
            coverage.get('completed_evaluation_cells')!=expected['evaluation_cells']):
        record.update(state='blocked',failure={'reason':'accuracy_coverage_incomplete','coverage':coverage});publish(state);return False
    if kind=='final' and coverage.get('all_requested_work_complete') is not True:
        record['state']='blocked';record['failure']={'reason':'final_coverage_incomplete','coverage':coverage};publish(state);return False
    record.update(state='completed',coverage_sha256=file_hash(ROOT/'outputs/tables/coverage.json'));publish(state);return True


def supervise():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Production supervision requires a Slurm allocation')
    recover_lock(ROOT/'outputs/controller.lock')
    with directory_lock(ROOT/'outputs/controller.lock'):
        state=initialize();manifest=load_manifest();source=verify_source();started=time.time();current=os.environ['SLURM_JOB_ID']
        require_orchestration(state)
        if state['manifest_hash']!=manifest['manifest_hash'] or source!=state['source_hash']:
            raise ValueError('Controller source/manifest identity changed')
        if state.get('next_controller_job')==current:
            state.pop('next_controller_job')
        state['controller_job']=current
        while not stop_requested():
            state['controller']={'job_id':current,'heartbeat':time.time(),'pid':os.getpid(),'host':socket.gethostname(),**orchestration_receipt()}
            publish(state)
            if state['phase']=='completed':
                return
            if time.time()-started>20*3600:
                job=submit([PYTHON,str(ORCHESTRATION_ROOT/'controller.py'),'supervise'],'controller','handoff:'+current,
                           ORCHESTRATION_ROOT,dependency='afterany:'+current,time_limit='1-00:00:00')
                state['next_controller_job']=job;publish(state);return
            try:
                if state['phase']=='waiting_parent':
                    ready,state['parent']=parent_gate()
                    if ready:
                        atomic_json(ROOT/'outputs/parent_completion_gate.json',dict(verified=now(),parent=state['parent'],
                            campaign_sha256=file_hash(BASE_ROOT/'outputs/campaign.json'),
                            coverage_sha256=file_hash(BASE_ROOT/'outputs/tables/coverage.json'),
                            monitor_state_sha256=file_hash(BASE_ROOT/'outputs/agent_monitor/state.json')))
                        state['phase']='evaluation';state['history'].append({'time':now(),'action':'parent_complete_gate_passed'})
                elif state['phase']=='evaluation':
                    if advance(state,'evaluation',manifest):
                        report_step(state,'accuracy')
                        state['phase']='latency'
                elif state['phase']=='accuracy_report':
                    if report_step(state,'accuracy'):
                        state['phase']='latency'
                elif state['phase']=='latency':
                    accuracy_ready=report_step(state,'accuracy')
                    if advance(state,'latency',manifest) and accuracy_ready:
                        state['phase']='final_report'
                elif state['phase']=='final_report':
                    if report_step(state,'final'):
                        recovered=[r for r in state['latency_workers'] if r.get('validation_recovery') is not None]
                        if len(recovered)!=16 or len({r['identity'] for r in recovered})!=16 or any(r['state']!='completed' for r in recovered):
                            raise ValueError('Final validation requires all sixteen diagnosed recovered workers')
                        cases={case['test_config_hash']:case for case in manifest['cases']};verified=[]
                        for record in recovered:
                            path=summary_path(record,'latency',cases)
                            verified.append(dict(identity=record['identity'],summary_path=str(path),
                                summary_sha256=verify_receipt(path,cases[record['test_config_hash']],record,'latency'),
                                validation_source_hash=record['validation_recovery']['validation_source_hash']))
                        state['validation_recovery_final_verification']=dict(state='passed',checked=now(),
                            workers=verified,worker_count=16,**orchestration_receipt())
                        state.update(phase='completed',completed=now())
                elif state['phase']=='stopped':
                    return
                else:
                    raise ValueError('Unknown controller phase')
                state.pop('controller_error',None)
            except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as error:
                state['controller_error']={'time':now(),'error':repr(error),'traceback':traceback.format_exc()}
                publish(state)
                raise  # Failed controller is visible to scheduler and bounded repair monitor.
            update_scratchpad(state)
            publish(state)
            time.sleep(60)
        state['resume_phase']=state['phase'];state['phase']='stopped';publish(state)


def launch(resume=False):
    recover_lock(ROOT/'outputs/launch.lock')
    with directory_lock(ROOT/'outputs/launch.lock'):
        state=initialize()
        require_orchestration(state)
        if state['phase']=='completed':
            return None
        ids=[state.get('controller_job'),state.get('next_controller_job')]
        # Include submission receipts to recover acceptance before state commit.
        for path in (ROOT/'outputs/submissions').glob('*.json'):
            receipt=read_json(path)
            if receipt['request']['name']=='controller' and receipt.get('job_id'):
                ids.append(receipt['job_id'])
        statuses=observed(ids)
        live=[job for job,seen in statuses.items() if seen['state'] in ACTIVE]
        if live:
            return live[-1]
        if any(str(j) not in statuses for j in ids if j):
            raise RuntimeError('Previous controller ownership is unknown')
        if stop_requested():
            if not resume or (BASE_ROOT/'STOP').exists():
                raise RuntimeError('STOP set; explicit authorized resume required')
            (ROOT/'STOP').unlink()
        if state['phase']=='stopped':
            if not resume:
                raise RuntimeError('Stopped study requires explicit resume')
            state['phase']=state['resume_phase']
        if resume:
            for family in ('evaluation','latency_workers'):
                outstanding=[r for r in state[family] if r.get('job_id') and r['state']!='completed']
                worker_states=observed([r['job_id'] for r in outstanding])
                for record in outstanding:
                    terminal=worker_states.get(record['job_id'],{}).get('state')
                    if terminal in ACTIVE or terminal is None:
                        raise RuntimeError('Stopped worker has not relinquished ownership')
                    if terminal=='CANCELLED':
                        state['history'].append({'time':now(),'action':'explicit_resume_cancelled_worker','job_id':record['job_id']})
                        record.update(job_id=None,state='unsubmitted',attempt=record['attempt']+1)
            for family in ('accuracy_report','final_report'):
                record=state.get(family)
                if record and record.get('job_id') and record['state']!='completed':
                    terminal=observed([record['job_id']]).get(record['job_id'],{}).get('state')
                    if terminal in ACTIVE or terminal is None:
                        raise RuntimeError('Stopped report has not relinquished ownership')
                    if terminal=='CANCELLED':
                        state['history'].append({'time':now(),'action':'explicit_resume_cancelled_report','job_id':record['job_id']})
                        record.update(job_id=None,state='unsubmitted',attempt=record['attempt']+1)
            publish(state)
        recover_lock(ROOT/'outputs/controller.lock')
        generation=state.get('controller_generation',0)+1
        job=submit([PYTHON,str(ORCHESTRATION_ROOT/'controller.py'),'supervise'],'controller',f'controller:{generation}',
                   ORCHESTRATION_ROOT,time_limit='1-00:00:00')
        state.update(controller_generation=generation,controller_job=job);publish(state);return job


def launch_monitor():
    recover_lock(ROOT/'outputs/monitor-launch.lock')
    with directory_lock(ROOT/'outputs/monitor-launch.lock'):
        state=initialize();destination=ROOT/'outputs/agent_monitor';destination.mkdir(parents=True,exist_ok=True)
        ledger_path=destination/'state.json'
        if ledger_path.exists():
            ledger=read_json(ledger_path)
            if ledger.get('state')=='completed_verified':
                return None
            if time.time()>=ledger['deadline']:
                raise RuntimeError('Bounded monitor deadline elapsed; do not silently reset its budget')
        receipts=[read_json(p) for p in (ROOT/'outputs/submissions').glob('*.json')]
        previous=[r['job_id'] for r in receipts if r['request']['name']=='monitor' and r.get('job_id')]
        statuses=observed(previous)
        live=[job for job,seen in statuses.items() if seen['state'] in ACTIVE]
        if live:
            return live[-1]
        if any(job not in statuses for job in previous):
            raise RuntimeError('Previous monitor ownership is unknown')
        if (destination/'STOP').exists():
            raise RuntimeError('Monitor STOP requires an explicit authorized removal')
        generation=len(previous)+1
        job=submit([PYTHON,'-m','dense_sparse.monitor','--root',str(ROOT),'--max-hours','96','--poll-seconds','300'],
                   'monitor',f'monitor:{generation}',state['source_path'],memory_gb=8,time_limit='4-01:00:00')
        (destination/'READY').write_text('Authorized bounded repair; source '+state['source_hash']+'\n')
        atomic_json(destination/'launch.json',dict(job_id=job,source_hash=state['source_hash'],source_path=state['source_path'],
            created=now(),max_hours=96,max_agent_calls=8,max_per_incident=3,invocation_limit_seconds=2700,
            runtime_capability_audit_job='356052',generation=generation))
        return job


def stop():
    (ROOT/'STOP').write_text('User-requested study stop '+now()+'\n')
    # All accepted study receipts, including monitor and handoffs, never unrelated work.
    ids=[read_json(p).get('job_id') for p in (ROOT/'outputs/submissions').glob('*.json')]
    live=[job for job,value in observed(ids).items() if value['state'] in ACTIVE]
    if live:
        subprocess.run(['scancel',*live],check=True)
    # Workers/controllers may still be relinquishing ownership; resume reconciles them.


def main():
    orchestration_receipt()
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','launch-monitor','supervise','status','stop','resume']);args=parser.parse_args()
    if args.action=='supervise':supervise()
    elif args.action in ('launch','resume'):
        print(launch(args.action=='resume'))
        if args.action=='resume':print(launch_monitor())
    elif args.action=='launch-monitor':print(launch_monitor())
    elif args.action=='stop':stop()
    else:print(json.dumps(read_json(ROOT/'STATUS.json'),indent=2))

if __name__=='__main__':main()
