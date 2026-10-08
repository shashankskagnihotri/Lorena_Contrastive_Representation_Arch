"""One durable owner for the full inference study, gated on parent completion."""
from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from .common import *
from .scheduler import ACTIVE, TERMINAL, TRANSIENT, observed, submit, recover_lock


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
    return file_hash(path)


def command_for(record,kind):
    command=[PYTHON,'-m','dense_sparse.worker','evaluate' if kind=='evaluation' else 'latency',
             '--case',str(ROOT/'configs'/f"{record['test_config_hash']}.json")]
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
        job=submit(command_for(record,kind),kind, f"{kind}:{record['identity']}:{record['attempt']}",state['source_path'],
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
        if state['manifest_hash']!=manifest['manifest_hash'] or source!=state['source_hash']:
            raise ValueError('Controller source/manifest identity changed')
        if state.get('next_controller_job')==current:
            state.pop('next_controller_job')
        state['controller_job']=current
        while not stop_requested():
            state['controller']={'job_id':current,'heartbeat':time.time(),'pid':os.getpid(),'host':socket.gethostname()}
            publish(state)
            if state['phase']=='completed':
                return
            if time.time()-started>20*3600:
                job=submit([PYTHON,'-m','dense_sparse.controller','supervise'],'controller','handoff:'+current,
                           state['source_path'],dependency='afterany:'+current,time_limit='1-00:00:00')
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
        job=submit([PYTHON,'-m','dense_sparse.controller','supervise'],'controller',f'controller:{generation}',
                   state['source_path'],time_limit='1-00:00:00')
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
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','launch-monitor','supervise','status','stop','resume']);args=parser.parse_args()
    if args.action=='supervise':supervise()
    elif args.action in ('launch','resume'):
        print(launch(args.action=='resume'))
        if args.action=='resume':print(launch_monitor())
    elif args.action=='launch-monitor':print(launch_monitor())
    elif args.action=='stop':stop()
    else:print(json.dumps(read_json(ROOT/'STATUS.json'),indent=2))

if __name__=='__main__':main()
