"""Slurm-native bounded campaign orchestration; no arbitrary autonomous code repair."""
from __future__ import annotations
import argparse, copy, getpass, hashlib, json, os, re, shlex, shutil, socket, subprocess, sys, tempfile, time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from .common import ROOT, atomic_json, digest, file_hash, directory_lock, stop_requested, run_dir
from .scope import (read_scope, validate_registry, expected_main_count,
                    expected_group_count, expected_worker_count, expected_workers_per_block)

PYTHON='/ceph/sagnihot/miniconda3/envs/sparse-contrast-bench/bin/python'
GPU_PARTITIONS='gpu-vram-94gb,gpu-vram-48gb,gpu-vram-32gb,gpu-vram-12gb'
GROUPS={(dataset,architecture) for dataset in ('mnist','cifar10') for architecture in ('vit_small','swin_tiny')}
ACTIVE_STATES={'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','REQUEUED','REQUEUE_FED','RESIZING'}
TERMINAL_STATES={'COMPLETED','CANCELLED','FAILED','TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL','OUT_OF_MEMORY','DEADLINE','REVOKED'}


class SubmissionAmbiguous(RuntimeError):
    """An existing submission may have succeeded; never automatically repeat it."""


def _read(path):
    return json.loads(Path(path).read_text())


def _orchestration_source(state):
    """Keep execution coordination separate from immutable training lineage."""
    has_hash='orchestration_source_hash' in state
    has_path='orchestration_source_path' in state
    if has_hash!=has_path:
        raise ValueError('Orchestration source hash and path must be recorded together')
    return (state['orchestration_source_hash'],state['orchestration_source_path']) if has_hash else (
        state['source_hash'],state['source_path'])


def _save_state(state):
    state['updated']=time.time()
    atomic_json(ROOT/'outputs/campaign.json',state)


def verify_source_snapshot(path):
    path=Path(path); manifest=_read(path/'source_manifest.json')
    for name,expected in manifest.items():
        if file_hash(path/name)!=expected: raise ValueError(f'Frozen source changed: {path/name}')
    return digest(manifest)

def source_snapshot():
    files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.sh'))
    for directory in ('sparse_contrast','scripts','tests'):
        files.update((ROOT/directory).rglob('*.py')); files.update((ROOT/directory).rglob('*.sh'))
    for name in ('requirements.lock.txt','environment.yml','conda-explicit.txt','pytest.ini','experiment_scope.json',
                 'provenance/source_context.json','provenance/original_frontend.py'):
        if (ROOT/name).is_file(): files.add(ROOT/name)
    # Read once: the identity and copied bytes cannot disagree during an edit.
    payload={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(files) if p.is_file()}
    manifest={name:hashlib.sha256(value).hexdigest() for name,value in payload.items()}
    identity=digest(manifest)
    dest=ROOT/'outputs/source'/identity
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if verify_source_snapshot(dest)!=identity: raise ValueError('Source snapshot identity mismatch')
    else:
        temporary=Path(tempfile.mkdtemp(prefix='snapshot-',dir=dest.parent))
        try:
            for name,value in payload.items():
                out=temporary/name; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(value)
            atomic_json(temporary/'source_manifest.json',manifest)
            try: temporary.rename(dest)
            except FileExistsError:
                if verify_source_snapshot(dest)!=identity: raise ValueError('Concurrent source snapshot mismatch')
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
    return identity,dest

def resolved_recipe(manifest,dataset,architecture):
    """One frozen recipe for every representation/seed in a comparison group."""
    recipe=copy.deepcopy(manifest['recipe'])
    for name in ('epochs','warmup_epochs'):
        if isinstance(recipe[name],dict): recipe[name]=recipe[name][dataset]
    overrides=manifest.get('recipe_by_group',{})
    if overrides:
        key=dataset+'/'+architecture
        if key not in overrides: raise ValueError(f'Missing group recipe: {key}')
        recipe.update(copy.deepcopy(overrides[key]))
    return recipe


def resolve_config(run,source_hash,pilot=False):
    from .data import load_manifest
    from .normalization import load_normalization
    from .pipeline import RAW_STATS,RAW_PROVENANCE
    manifest=json.loads((ROOT/'experiment_manifest.json').read_text()); dataset=run['dataset']
    config={k:v for k,v in run.items() if k not in ['status','resolved_config_hash']}
    recipe=resolved_recipe(manifest,dataset,run['architecture'])
    config.update(recipe=recipe,precision='fp32',source_hash=source_hash,environment_hash=file_hash(ROOT/'requirements.lock.txt'),patch_size=2,token_epsilon=0.,normalization_min_std=1e-6,split_seed=2026,data_manifest_sha256=file_hash(ROOT/'datasets_manifest.json'),primary_checkpoint='final_epoch',role='pilot' if pilot else 'main')
    if pilot:
        config['registry_id']='pilots/'+config['registry_id']
        # Pilots remain excluded by role; held-out partition identity is unchanged.
    if run['representation']=='raw':
        config['normalization_hash']=digest(RAW_STATS[dataset]); config['normalization']={'mean':RAW_STATS[dataset][0],'std':RAW_STATS[dataset][1],'input_range':[0,1],'provenance':RAW_PROVENANCE[dataset],'description':'chosen published recipe'}
    else:
        _,stats=load_normalization(ROOT/'normalization'/f"{dataset}_{run['representation']}_{run['protocol']}.json")
        config['normalization_hash']=stats['artifact_sha256']; config['frontend']=stats.get('frontend_config',stats.get('frontend'))
    config['config_hash']=digest(config)
    path=ROOT/'configs/resolved'/f"{config['config_hash']}.json"; atomic_json(path,config)
    return config,path

def _find_intent_jobs(intent):
    """Find only this user's exact persisted job name AND ownership comment."""
    since=datetime.fromtimestamp(intent['prepared_at']-86400).strftime('%Y-%m-%dT%H:%M:%S')
    commands=[
        ['squeue','-h','-u',getpass.getuser(),'--name='+intent['job_name'],'--format=%i|%j|%k'],
        ['sacct','-X','-n','-P','--user='+getpass.getuser(),'--name='+intent['job_name'],'--starttime='+since,
         '--format=JobIDRaw,JobName%100,Comment%180'],
    ]
    found=set()
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        for line in result.stdout.splitlines():
            parts=[part.strip() for part in line.split('|')]
            if len(parts)>=3 and parts[1]==intent['job_name'] and parts[2]==intent['comment'] and parts[0].isdigit():
                found.add(parts[0])
    if len(found)>1: raise RuntimeError(f'Multiple jobs own one submission intent: {sorted(found)}')
    return next(iter(found)) if found else None


def submit(command,name,*,gpu=True,dependency=None,time_limit='4-20:00:00',source=None,array=None,submission_key=None,gpu_partition=None,gpu_type=None):
    if stop_requested(): raise InterruptedError('STOP marker prevents submission')
    command=[str(value) for value in command]
    request=dict(command=command,name=name,gpu=gpu,dependency=dependency,time_limit=time_limit,source=str(source) if source else None,array=array,gpu_partition=gpu_partition)
    if gpu_type is not None:
        if not gpu or re.fullmatch(r'[A-Za-z0-9_.-]+',gpu_type) is None:
            raise ValueError('GPU type must be one exact Slurm GRES type')
        request['gpu_type']=gpu_type
    identity=digest(dict(root=str(ROOT),key=submission_key or request))
    intent_path=ROOT/'outputs/submissions'/f'{identity}.json'
    if intent_path.exists():
        intent=_read(intent_path)
        if intent['request']!=request: raise ValueError('Submission key reused for a different command')
        if intent['state']=='cancelled_before_submit':
            intent_path.rename(intent_path.with_name(intent_path.name+f'.cancelled.{time.time_ns()}'))
            return submit(command,name,gpu=gpu,dependency=dependency,time_limit=time_limit,source=source,array=array,
                          submission_key=submission_key,gpu_partition=gpu_partition,gpu_type=gpu_type)
        if intent.get('job_id'): return intent['job_id']
        adopted=_find_intent_jobs(intent)
        if adopted:
            intent.update(job_id=adopted,state='accepted_reconciled',reconciled_at=time.time())
            atomic_json(intent_path,intent); return adopted
        raise SubmissionAmbiguous(f'No ownership match yet for {intent_path}; do not repeat sbatch. Reconcile Slurm before retrying.')
    intent=dict(schema_version=1,identity=identity,request=request,prepared_at=time.time(),state='prepared',
                job_name=f'scb-{name}-{identity[:16]}',comment='sparse-contrast:'+identity,job_id=None)
    # The intent is durable BEFORE contacting Slurm. Crash recovery looks up its
    # exact ownership key and cannot create another job on an uncertain outcome.
    atomic_json(intent_path,intent)
    logs=ROOT/'outputs/slurm'; logs.mkdir(exist_ok=True,parents=True)
    args=['sbatch','--parsable','--account=ml-staff','--qos=walltime5d',f"--job-name={intent['job_name']}",f"--comment={intent['comment']}",f'--partition={(gpu_partition or GPU_PARTITIONS) if gpu else "cpu"}','--cpus-per-task=4',f'--mem={"16G" if gpu else "8G"}',f'--time={time_limit}',f'--output={logs}/%x-%j.out',f'--error={logs}/%x-%j.err']
    if gpu: args+=[f'--gres=gpu:{gpu_type}:1' if gpu_type else '--gres=gpu:1']
    if dependency: args+=['--dependency='+dependency]
    if array: args+=['--array='+array]
    cwd=Path(source) if source else ROOT
    # shlex.join, never JSON escaping, constructs the shell command safely.
    env={'PATH':str(Path(PYTHON).parent)+':'+os.environ['PATH'],'SPARSE_CONTRAST_ROOT':str(ROOT),'PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'}
    shell='#!/usr/bin/env bash\nset -euo pipefail\n'+''.join(f'export {k}={shlex.quote(v)}\n' for k,v in env.items())+f'cd {shlex.quote(str(cwd))}\nexec '+shlex.join(command)+'\n'
    if stop_requested():
        intent.update(state='cancelled_before_submit',cancelled_at=time.time())
        atomic_json(intent_path,intent)
        raise InterruptedError('STOP marker immediately before Slurm submission')
    try: result=subprocess.run(args,input=shell,text=True,capture_output=True,check=True)
    except (subprocess.CalledProcessError,OSError) as error:
        intent.update(state='ambiguous',error=repr(error),stderr=getattr(error,'stderr',None),observed_at=time.time())
        atomic_json(intent_path,intent)
        raise SubmissionAmbiguous(f'Slurm submission outcome requires reconciliation: {intent_path}') from error
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():
        intent.update(state='ambiguous',stdout=result.stdout,stderr=result.stderr,observed_at=time.time())
        atomic_json(intent_path,intent)
        raise SubmissionAmbiguous(f'Ambiguous Slurm output; ownership intent: {intent_path}')
    intent.update(job_id=job,state='accepted',accepted_at=time.time())
    atomic_json(intent_path,intent)
    return job

def job_states(job_ids):
    if not job_ids:return {}
    result=subprocess.run(['sacct','-X','-n','-P','-j',','.join(job_ids),'--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,check=True)
    states={}
    for line in result.stdout.splitlines():
        parts=line.split('|')
        if len(parts)>=3: states[parts[0]]=dict(state=parts[1].split()[0].rstrip('+'),exit_code=parts[2])
    queue=subprocess.run(['squeue','-h','-u',getpass.getuser(),'--format=%i|%T'],capture_output=True,text=True,check=True)
    for line in queue.stdout.splitlines():
        parts=line.split('|')
        if len(parts)>=2 and parts[0] in job_ids: states[parts[0]]=dict(state=parts[1],exit_code=None)
    return states


def recover_dead_lock(path,expected_job=None):
    """Archive a lease only after positive owner-death evidence."""
    path=Path(path)
    if not path.exists(): return
    owner=_read(path/'owner.json'); job=owner.get('job_id')
    if expected_job is not None and job!=expected_job: raise RuntimeError('Lease belongs to a different job')
    if job:
        status=job_states([str(job)]).get(str(job),{}).get('state')
        if status not in TERMINAL_STATES: raise RuntimeError(f'Lease owner is live or unknown: {job}, {status}')
    elif owner.get('host')==socket.gethostname():
        try: os.kill(int(owner['pid']),0)
        except ProcessLookupError: pass
        else: raise RuntimeError('Lease owner process is still alive')
    else: raise RuntimeError('Cannot establish remote nonscheduler lease owner death')
    path.rename(path.with_name(path.name+f'.retired.{time.time_ns()}'))


@contextmanager
def campaign_lock(name):
    path=ROOT/'outputs'/name
    recover_dead_lock(path)
    with directory_lock(path): yield

def pilot_verdict(config):
    rd=run_dir(config)
    if not (rd/'completed.json').exists():return None
    rows=[json.loads(x) for x in (rd/'epochs.jsonl').read_text().splitlines()]
    if len(rows)!=config['recipe']['epochs']:return {'passed':False,'reason':'Missing pilot epochs'}
    start=rows[0]; last=rows[-1]; tail=rows[-5:]
    # Predeclared clean-validation-only conservative gate; failures require human/agent diagnosis.
    floor=.97 if config['dataset']=='mnist' else .65
    passed=(last['validation']['accuracy']>=floor and last['train_online']['loss']<start['train_online']['loss'] and all(r['validation']['total']==5000 for r in rows))
    return dict(passed=passed,minimum_accuracy=floor,final_validation_accuracy=last['validation']['accuracy'],initial_train_loss=start['train_online']['loss'],final_train_loss=last['train_online']['loss'],last_five_validation_accuracy=[r['validation']['accuracy'] for r in tail],reason='fixed-budget finite completion and prespecified clean validation gate' if passed else 'Pilot convergence requires diagnosis; no main launch')

def init_campaign():
    state_path=ROOT/'outputs/campaign.json'
    if state_path.exists():return json.loads(state_path.read_text())
    gate=json.loads((ROOT/'outputs/checks/completed.json').read_text())
    if not gate['passed']: raise RuntimeError('Required correctness checks failed')
    manifest=json.loads((ROOT/'experiment_manifest.json').read_text())
    if manifest.get('batch_selection_required') and manifest.get('batch_selection',{}).get('state')!='verified':
        raise RuntimeError('Larger-batch campaign requires verified group batch selection before launch')
    source,location=source_snapshot()
    pilots=[]
    for run in manifest['runs']:
        if run['representation']=='raw' and run['seed']==0:
            config,path=resolve_config(run,source,pilot=True)
            pilots.append(dict(config_path=str(path),config=config,stage='train',job_id=None,transient_retries=0,segments=0,state='planned'))
    if len(pilots)!=4 or {(item['config']['dataset'],item['config']['architecture']) for item in pilots}!=GROUPS:
        raise ValueError('Exactly four distinct dataset/architecture raw pilots are required')
    validate_registry(manifest['runs'],read_scope(ROOT))
    state=dict(schema_version=3,phase='pilots',source_hash=source,source_path=str(location),pilots=pilots,main=[],groups={},created=time.time(),history=[],controller={},controller_generation=0)
    atomic_json(state_path,state); return state


def _completion(item):
    config=item['config']; rd=run_dir(config)
    path=rd/('completed.json' if item['stage']=='train' else 'evaluation/summary.json')
    if not path.exists(): return False
    record=_read(path)
    if item['stage']=='train':
        if record['config_hash']!=config['config_hash'] or record['epochs']!=config['recipe']['epochs']:
            raise ValueError('Training completion identity/budget mismatch')
        if record['checkpoint_sha256']!=file_hash(rd/'final.pt'):
            raise ValueError('Completed training checkpoint hash mismatch')
    else:
        expected=75 if config['dataset']=='cifar10' else 15
        if record['config']['config_hash']!=config['config_hash'] or record['state']!='complete':
            raise ValueError('Evaluation completion identity mismatch')
        if record['completed_cells']!=expected or record['total']!=expected*10000 or record['clean']['total']!=10000:
            raise ValueError('Evaluation coverage incomplete')
        if record['checkpoint_sha256']!=_read(rd/'completed.json')['checkpoint_sha256']:
            raise ValueError('Evaluation checkpoint identity mismatch')
    return True


def _block(state,reason,details=None):
    state['resume_phase']=state['phase']
    state['phase']='blocked_'+reason
    state['blocker']=dict(reason=reason,details=details,time=time.time())
    _save_state(state)


def _publish(state):
    _save_state(state)
    from .runtime_estimate import write_runtime_estimate
    scope=read_scope(ROOT)
    estimate_path=write_runtime_estimate(state,output_path=ROOT/'outputs/runtime_estimate.json',scope=scope)
    items=state['pilots']+state['main']
    atomic_json(ROOT/'STATUS.json',dict(state=state['phase'],controller=state.get('controller'),
        completed=sum(i['state']=='completed' for i in items),
        completed_main=sum(i['state']=='completed' for i in state['main']),
        completed_pilots=sum(i['state']=='completed' for i in state['pilots']),
        expected_main_runs=expected_main_count(scope),withdrawn_main_runs=len(state.get('withdrawn_main',[])),
        training_source_hash=state['source_hash'],orchestration_source_hash=_orchestration_source(state)[0],
        groups=state.get('groups',{}),
        running=[{k:i[k] for k in ('job_id','stage','state')} for i in items if i.get('job_id') and i['state'] not in ('failed','completed')],
        failed=[dict(registry_id=i['config']['registry_id'],failure=i.get('failure')) for i in items if i['state'] in ('failed','blocked')],
        latency_prepare=state.get('latency_prepare'),latency_blocks=state.get('latency_blocks',[]),
        blocker=state.get('blocker'),campaign_state=str(ROOT/'outputs/campaign.json'),runtime_estimate=str(estimate_path)))


def _submit_item(state,item):
    config=item['config']; stage=item['stage']
    # Counter changes are committed BEFORE submit; restart reuses the same key.
    key=f"work:{config['config_hash']}:{stage}:{item.get('attempt',0)}"
    item['submission_key']=key; item['state']='submitting'; _save_state(state)
    module='sparse_contrast.train' if stage=='train' else 'sparse_contrast.evaluate'
    time_limit=config['recipe'].get('train_time_limit','4-20:00:00') if stage=='train' else config['recipe'].get('evaluation_time_limit','4-20:00:00')
    job=submit([PYTHON,'-m',module,'--config',item['config_path']],stage,source=state['source_path'],submission_key=key,time_limit=time_limit)
    item.update(job_id=job,state='submitted')
    state['history'].append(dict(time=time.time(),action='submit',job_id=job,config_hash=config['config_hash'],stage=stage,submission_key=key))
    _save_state(state)


def _advance_one(state,item,observed):
    if item['state'] in ('failed','blocked','completed'): return
    if not item.get('job_id') and time.time()<item.get('retry_not_before',0):
        item['state']='waiting_retry'; return
    job=item.get('job_id'); status=observed.get('state')
    if job and status in ACTIVE_STATES:
        item['state']=status.lower(); return
    if job and status is None:
        item['state']='scheduler_unknown'; return
    complete=_completion(item)
    if complete and (not job or status=='COMPLETED'):
        if item['stage']=='train' and item['config']['role']=='main':
            item.update(stage='evaluation',job_id=None,state='planned',attempt=0,transient_retries=0,segments=0)
            item.pop('retry_not_before',None)
            _save_state(state)
        else:
            item['state']='completed'; _save_state(state); return
    elif job:
        rd=run_dir(item['config'])
        if status in ('TIMEOUT','PREEMPTED') and (item['stage']=='evaluation' or (rd/'latest.pt').exists()):
            item['segments']+=1
        elif status in ('NODE_FAIL','BOOT_FAIL') and item['transient_retries']<3:
            item['transient_retries']+=1
            item['retry_not_before']=time.time()+60*2**(item['transient_retries']-1)
        else:
            item.update(state='failed',failure=observed); _save_state(state); return
        lock=rd/('writer.lock' if item['stage']=='train' else 'evaluation/writer.lock')
        recover_dead_lock(lock,expected_job=job)
        item['attempt']=item.get('attempt',0)+1
        item.update(job_id=None,state='planned'); _save_state(state)
        if time.time()<item.get('retry_not_before',0): return
    if not item.get('job_id'): _submit_item(state,item)


def _group_key(config):
    return config['dataset']+'/'+config['architecture']


def _sync_group_manifest(state):
    """Campaign state is authoritative across the two atomic artifact writes."""
    manifest=_read(ROOT/'experiment_manifest.json')
    original=copy.deepcopy(manifest)
    groups=state.get('groups',{})
    frozen=[group for group in groups.values() if group['state']=='frozen']
    manifest['state']='frozen' if len(frozen)==4 else ('partial_frozen' if frozen else 'planned_not_frozen')
    manifest['source_hash']=state['source_hash']
    manifest['groups']=copy.deepcopy(groups)
    reviews=[dict(dataset=group['dataset'],architecture=group['architecture'],**group['verdict'])
             for group in groups.values() if group.get('verdict') is not None]
    manifest['pilot_review']=reviews
    by_registry={item['config']['registry_id']:item['config']['config_hash'] for item in state['main']}
    if len(by_registry)!=len(state['main']): raise ValueError('Duplicate primary registry identity')
    for run in manifest['runs']:
        if run['registry_id'] in by_registry: run['resolved_config_hash']=by_registry[run['registry_id']]
    if manifest!=original: atomic_json(ROOT/'experiment_manifest.json',manifest)
    atomic_json(ROOT/'outputs/pilot_review.json',reviews)


def _freeze_ready_groups(state):
    """Each independent clean pilot releases its exact active-scope group."""
    scope=read_scope(ROOT)
    groups=state.setdefault('groups',{}); new_items=[]
    pilot_keys=[_group_key(item['config']) for item in state['pilots']]
    if len(pilot_keys)!=len(set(pilot_keys)): raise ValueError('Duplicate pilot comparison group')
    for pilot in state['pilots']:
        config=pilot['config']; key=_group_key(config)
        expected_group=expected_group_count(config['dataset'],scope)
        group=groups.setdefault(key,dict(dataset=config['dataset'],architecture=config['architecture'],
            pilot_config_hash=config['config_hash'],state='waiting_pilot',expected_main_runs=expected_group,
            verdict=None,main_config_hashes=[]))
        if group['pilot_config_hash']!=config['config_hash']: raise ValueError('Pilot identity changed under frozen group')
        existing=[item for item in state['main'] if _group_key(item['config'])==key]
        if group['state']=='frozen':
            if (group['expected_main_runs']!=expected_group or len(existing)!=expected_group or
                    {item['config']['config_hash'] for item in existing}!=set(group['main_config_hashes'])):
                raise ValueError('Frozen group lost or duplicated main identities')
            continue
        if group['state'].startswith('blocked_'): continue
        if pilot['state'] in ('failed','blocked'):
            group.update(state='blocked_pilot_execution',failure=pilot.get('failure'))
            _save_state(state); continue
        if pilot['state']!='completed': continue
        verdict=pilot_verdict(config)
        if verdict is None: raise ValueError('Completed pilot has no review evidence')
        group['verdict']=verdict
        if not verdict['passed']:
            group['state']='blocked_pilot_review'; _save_state(state); continue
        manifest=_read(ROOT/'experiment_manifest.json')
        validate_registry(manifest['runs'],scope)
        runs=[run for run in manifest['runs'] if _group_key(run)==key]
        if len(runs)!=expected_group or len({run['registry_id'] for run in runs})!=expected_group:
            raise ValueError('Dataset/architecture group does not match its active scope')
        if existing:
            # Recovery of an already materialized group must be exact; never
            # append another group after a controller restart.
            if len(existing)!=expected_group or {item['config']['registry_id'] for item in existing}!={run['registry_id'] for run in runs}:
                raise ValueError('Partially materialized group registry cannot be silently overwritten')
            if any(item['config']['recipe']!=config['recipe'] for item in existing):
                raise ValueError('Existing main recipe differs from its pilot')
            created=existing
        else:
            created=[]
            for run in runs:
                main,path=resolve_config(run,state['source_hash'])
                if main['recipe']!=config['recipe']: raise ValueError('Main recipe changed from its verified pilot')
                if main['role']!='main' or main['registry_id']==config['registry_id']:
                    raise ValueError('Main initialization/output identity must be separate from the pilot')
                created.append(dict(config_path=str(path),config=main,stage='train',job_id=None,
                                    transient_retries=0,segments=0,attempt=0,state='planned'))
            import random
            random.Random('sparse-contrast:2026:'+key).shuffle(created)
            state['main'].extend(created); new_items.extend(created)
        group.update(state='frozen',frozen_recipe=copy.deepcopy(config['recipe']),frozen_at=time.time(),
                     main_config_hashes=[item['config']['config_hash'] for item in created])
        state['phase']='main'
        # Commit the full group in ONE state write before any main submission.
        _save_state(state)
    if state['pilots']:
        _sync_group_manifest(state)
    return new_items


def _finish_training_phase(state):
    items=state['pilots']+state['main']
    failed=[item for item in items if item['state'] in ('failed','blocked')]
    terminal=all(item['state'] in ('completed','failed','blocked') for item in items)
    groups=state.get('groups',{})
    if failed and terminal:
        _block(state,'work_failed',[dict(config_hash=i['config']['config_hash'],failure=i.get('failure')) for i in failed]); return
    blocked_groups={key:group for key,group in groups.items() if group['state'].startswith('blocked_')}
    if blocked_groups and terminal:
        _block(state,'pilot_review',blocked_groups); return
    if not terminal: return
    scope=read_scope(ROOT); expected_main=expected_main_count(scope)
    registry_error=None
    try: validate_registry(state['main'],scope)
    except (KeyError,ValueError) as error: registry_error=str(error)
    required={dataset+'/'+architecture for dataset,architecture in GROUPS}
    full=(set(groups)==required and all(group['state']=='frozen' for group in groups.values())
          and len(state['pilots'])==4 and all(item['state']=='completed' for item in state['pilots'])
          and registry_error is None and len(state['main'])==expected_main
          and len({item['config']['registry_id'] for item in state['main']})==expected_main
          and all(item['state']=='completed' and item['stage']=='evaluation' and item['config']['role']=='main' for item in state['main']))
    if not full:
        _block(state,'incomplete_registry',dict(groups=list(groups),main_count=len(state['main']),
            expected_main_runs=expected_main,registry_error=registry_error)); return
    state['phase']='latency'; state['latency_attempt']=0; state['latency_job']=None
    state.setdefault('latency_prepare',_latency_record())
    _save_state(state)


def _advance_items(state):
    items=state['pilots']+state['main']
    if not items: raise ValueError('An active training phase cannot have an empty registry')
    active=job_states([item['job_id'] for item in items if item.get('job_id')])
    for item in state['pilots']:
        if stop_requested(): return
        _advance_one(state,item,active.get(item.get('job_id'),{}))
    if stop_requested(): return
    _freeze_ready_groups(state)
    for item in state['main']:
        if stop_requested(): return
        group=state.get('groups',{}).get(_group_key(item['config']),{})
        if item['state'] not in ('completed','failed','blocked') and group.get('state')!='frozen':
            raise ValueError('Main work cannot execute before its own group passes the pilot gate')
        _advance_one(state,item,active.get(item.get('job_id'),{}))
    _finish_training_phase(state)


def _latency_record(block_id=None):
    record=dict(job_id=None,state='planned',attempt=0,transient_retries=0,segments=0)
    if block_id is not None: record['block_id']=block_id
    return record


def _latency_selector(state):
    path=ROOT/'outputs/benchmarks/hardware_contract.json'
    if not path.exists(): return None,None
    contract=_read(path)
    partition=contract.get('gpu_partition'); gpu_type=contract.get('gpu_gres_type')
    if not partition or not gpu_type or ',' in partition:
        raise ValueError('Latency hardware contract requires one frozen partition and exact GPU GRES type')
    if state.get('latency_hardware_contract_sha256') not in (None,file_hash(path)):
        raise ValueError('Frozen latency hardware contract changed')
    return partition,gpu_type


def _latency_plan(state):
    from .benchmark import load_frozen_blocks,validate_measurement_session
    plan=load_frozen_blocks(root=ROOT)
    scope=read_scope(ROOT); expected_workers=expected_worker_count(scope)
    validate_registry(state['main'],scope)
    if plan['source_hash']!=state['source_hash']:
        raise ValueError('Latency plan source differs from frozen campaign source')
    if plan.get('measurement_source_hash',plan['source_hash'])!=_orchestration_source(state)[0]:
        raise ValueError('Latency measurement source differs from frozen orchestration source')
    blocks=plan['blocks']
    if len(blocks)!=24 or {block['id'] for block in blocks}!=set(range(24)):
        raise ValueError('Latency plan requires exactly 24 uniquely identified matched blocks')
    expected={(item['config']['config_hash'],execution,batch)
        for item in state['main'] for batch in (1,8)
        for execution in ([item['config']['execution'],'dense_masked'] if item['config']['execution']=='compact'
                          else [item['config']['execution']])}
    actual=[(work['config_hash'],work['execution'],work['batch_size']) for block in blocks for work in block['work']]
    if (len(state['main'])!=expected_main_count(scope) or len(expected)!=expected_workers or len(actual)!=expected_workers or
            len(set(actual))!=expected_workers or set(actual)!=expected or
            any(len(block['work'])!=expected_workers_per_block(block['dataset'],scope) for block in blocks)):
        raise ValueError('Latency blocks do not cover the exact active-scope primary worker identities once each')
    items={item['config']['config_hash']:item for item in state['main']}
    for block in blocks:
        group=(block['dataset'],block['architecture'],block['seed'],block['batch_size'])
        for work in block['work']:
            item=items[work['config_hash']]; config=item['config']
            if work['config_path']!=item['config_path']:
                raise ValueError('Latency worker config path differs from frozen primary config')
            if (config['dataset'],config['architecture'],config['seed'],work['batch_size'])!=group:
                raise ValueError('Latency matched block contains a worker from another comparison group')
    if state.get('latency_blocks_sha256') not in (None,plan['blocks_sha256']):
        raise ValueError('Latency block assignment changed after submission')
    partition,gpu_type=_latency_selector(state)
    if not partition or not gpu_type: raise ValueError('Latency preparation did not freeze hardware placement')
    out=ROOT/'outputs/benchmarks'; prepared=_read(out/'prepared.json')
    if (prepared['state']!='completed' or prepared['source_hash']!=state['source_hash'] or
            prepared.get('measurement_source_hash',prepared['source_hash'])!=_orchestration_source(state)[0] or
            prepared['blocks_sha256']!=plan['blocks_sha256'] or prepared['order_sha256']!=plan['order_sha256'] or
            prepared['expected_blocks']!=24 or prepared['expected_workers']!=expected_workers or
            prepared['hardware_contract_sha256']!=file_hash(out/'hardware_contract.json')):
        raise ValueError('Latency preparation receipt does not match the frozen plan/hardware')
    metadata=validate_measurement_session(prepared['measurement'])
    contract=_read(out/'hardware_contract.json')['contract']
    if any(metadata.get(key)!=value for key,value in contract.items()):
        raise ValueError('Latency preparation hardware session differs from its frozen contract')
    isolation=prepared['isolation']
    if (isolation.get('verified') is not True or isolation['device_uuid']!=metadata['device_uuid'] or
            set(isolation['observed_compute_pids'])-{isolation['self_pid']}):
        raise ValueError('Latency preparation lacks verified GPU isolation')
    return plan


def _submit_latency_record(state,record):
    is_prepare='block_id' not in record
    kind='prepare' if is_prepare else f"block_{record['block_id']:03d}"
    orchestration_hash,orchestration_path=_orchestration_source(state)
    identity=orchestration_hash if is_prepare else state['latency_blocks_sha256']
    key=f"latency:{identity}:{kind}:{record['attempt']}"
    command=[PYTHON,'benchmark.py','--campaign',str(ROOT/'outputs/campaign.json')]
    command+=['--prepare-campaign'] if is_prepare else ['--block-id',str(record['block_id'])]
    partition,gpu_type=_latency_selector(state)
    record.update(state='submitting',submission_key=key); _save_state(state)
    job=submit(command,'latency-'+kind,source=orchestration_path,submission_key=key,
               gpu_partition=partition,gpu_type=gpu_type,
               time_limit='01:00:00' if is_prepare else '4-20:00:00')
    record.update(state='submitted',job_id=job)
    state['history'].append(dict(time=time.time(),action='submit',stage='latency',kind=kind,
                                job_id=job,submission_key=key))
    _save_state(state)


def _advance_latency_record(state,record,observed,validate):
    if record['state'] in ('completed','failed','blocked'): return
    job=record.get('job_id'); status=observed.get('state')
    if job and status in ACTIVE_STATES:
        record['state']=status.lower(); return
    if job and status is None:
        record['state']='scheduler_unknown'; return
    if job and status=='COMPLETED':
        try:
            validate()
        except (OSError,KeyError,TypeError,ValueError) as error:
            record.update(state='failed',failure=dict(kind='completion_evidence',error=repr(error),job_id=job))
        else:
            record.update(state='completed',completed_at=time.time())
        _save_state(state); return
    if job:
        if status in ('TIMEOUT','PREEMPTED'):
            record['segments']+=1
        elif status in ('NODE_FAIL','BOOT_FAIL') and record['transient_retries']<3:
            record['transient_retries']+=1
            record['retry_not_before']=time.time()+60*2**(record['transient_retries']-1)
        else:
            record.update(state='failed',failure=observed); _save_state(state); return
        # Benchmark workers reclaim their own directory leases only after
        # positive terminal scheduler ownership, and reuse valid cell evidence.
        record['attempt']+=1
        record.update(job_id=None,state='planned'); _save_state(state)
    if time.time()<record.get('retry_not_before',0):
        record['state']='waiting_retry'; return
    if not record.get('job_id') and not stop_requested(): _submit_latency_record(state,record)


def _latency_progress(state,plan):
    blocks=state['latency_blocks']
    complete=sum(record['state']=='completed' for record in blocks)
    failed=[record for record in blocks if record['state'] in ('failed','blocked')]
    terminal=all(record['state'] in ('completed','failed','blocked') for record in blocks)
    progress_state='blocked' if failed and terminal else ('running' if complete<24 else 'verifying')
    by_id={block['id']:block for block in plan['blocks']}
    workers=sum(len(by_id[record['block_id']]['work']) for record in blocks if record['state']=='completed')
    atomic_json(ROOT/'outputs/benchmarks/progress.json',dict(
        state=progress_state,workers=workers,expected_workers=expected_worker_count(read_scope(ROOT)),
        completed_blocks=complete,expected_blocks=24,failed_blocks=failed,
        source_hash=state['source_hash'],measurement_source_hash=_orchestration_source(state)[0],
        withdrawn_main_runs=len(state.get('withdrawn_main',[])),blocks_sha256=plan['blocks_sha256'],
        order_sha256=plan['order_sha256'],time=time.time()))


def _advance_latency(state):
    from .benchmark import validate_block_completion
    prepare=state.setdefault('latency_prepare',_latency_record())
    if prepare['state']!='completed':
        observed=job_states([prepare['job_id']]).get(prepare['job_id'],{}) if prepare.get('job_id') else {}
        _advance_latency_record(state,prepare,observed,lambda: _latency_plan(state))
        if prepare['state'] in ('failed','blocked'):
            _block(state,'latency_prepare',prepare); return
        if prepare['state']!='completed': return
    try:
        plan=_latency_plan(state)
    except (OSError,KeyError,TypeError,ValueError) as error:
        _block(state,'latency_plan',dict(error=repr(error))); return
    if 'latency_blocks' not in state:
        state['latency_blocks_sha256']=plan['blocks_sha256']
        state['latency_hardware_contract_sha256']=file_hash(ROOT/'outputs/benchmarks/hardware_contract.json')
        state['latency_blocks']=[_latency_record(block['id']) for block in plan['blocks']]
        _save_state(state)
    records=state['latency_blocks']
    if len(records)!=24 or {record['block_id'] for record in records}!=set(range(24)):
        _block(state,'latency_registry',dict(error='Persisted latency block registry lost or duplicated identities')); return
    observed=job_states([record['job_id'] for record in records if record.get('job_id')])
    for record in records:
        if stop_requested(): return
        _advance_latency_record(state,record,observed.get(record.get('job_id'),{}),
            lambda record=record: validate_block_completion(record['block_id'],root=ROOT))
    _latency_progress(state,plan)
    if not all(record['state'] in ('completed','failed','blocked') for record in records): return
    failed=[record for record in records if record['state']!='completed']
    if failed:
        _block(state,'latency_blocks',failed); return
    try:
        # Revalidate the complete union, not just individual process exit codes.
        receipts=[validate_block_completion(record['block_id'],root=ROOT) for record in records]
        actual=[(worker['config_hash'],worker['execution'],worker['batch_size'])
                for receipt in receipts for worker in receipt['workers']]
        expected={(work['config_hash'],work['execution'],work['batch_size'])
                  for block in plan['blocks'] for work in block['work']}
        expected_workers=expected_worker_count(read_scope(ROOT))
        if len(actual)!=expected_workers or len(set(actual))!=expected_workers or set(actual)!=expected:
            raise ValueError('Completed latency receipts overlap or omit primary workers')
    except (OSError,KeyError,TypeError,ValueError) as error:
        _block(state,'latency_coverage',dict(error=repr(error))); return
    atomic_json(ROOT/'outputs/benchmarks/progress.json',dict(state='completed',workers=expected_workers,
        completed_blocks=24,expected_blocks=24,expected_workers=expected_workers,
        source_hash=state['source_hash'],measurement_source_hash=_orchestration_source(state)[0],
        withdrawn_main_runs=len(state.get('withdrawn_main',[])),
        blocks_sha256=plan['blocks_sha256'],order_sha256=plan['order_sha256'],
        hardware_contract_sha256=state['latency_hardware_contract_sha256'],
        block_receipts=[dict(block_id=record['block_id'],path=str(ROOT/'outputs/benchmarks/blocks'/
            f"block_{record['block_id']:03d}"/'completed.json'),sha256=file_hash(ROOT/'outputs/benchmarks/blocks'/
            f"block_{record['block_id']:03d}"/'completed.json')) for record in records],time=time.time()))
    state['phase']='report'; _save_state(state)


def _advance_postprocessing(state):
    if state['phase']=='latency':
        # Preserve ownership of any old serial worker until positive termination.
        # Its count-only progress marker can never establish campaign completion.
        legacy=state.get('latency_job')
        if legacy:
            observed=job_states([legacy]).get(legacy,{})
            status=observed.get('state')
            if status in ACTIVE_STATES or status is None: return
            if status!='COMPLETED':
                _block(state,'latency_legacy',dict(job_id=legacy,observed=observed)); return
            state['history'].append(dict(time=time.time(),action='legacy_latency_reconciled',job_id=legacy))
            state['latency_job']=None; _save_state(state)
        return _advance_latency(state)
    if state['phase']!='report': raise ValueError('Unexpected postprocessing phase')
    job=state.get('report_job')
    if job:
        observed=job_states([job]).get(job,{})
        status=observed.get('state')
        if status in ACTIVE_STATES or status is None: return
        if status!='COMPLETED':
            _block(state,'report',observed); return
        path=ROOT/'outputs/tables/coverage.json'
        evidence=_read(path) if path.exists() else None
        if evidence is None or evidence.get('all_requested_work_complete') is not True:
            _block(state,'report_coverage',dict(path=str(path),evidence=evidence,job_id=job)); return
        state['phase']='completed'; _save_state(state); return
    attempt=state.get('report_attempt',0)
    orchestration_hash,orchestration_path=_orchestration_source(state)
    state['report_job']=submit([PYTHON,'report.py'],'report',gpu=False,source=orchestration_path,
        time_limit='04:00:00',submission_key=f"report:{orchestration_hash}:{attempt}")
    _save_state(state)


def supervise():
    state=init_campaign(); start=time.time()
    if state['phase']=='completed' or state['phase'].startswith('blocked_'): return
    if state['phase']=='stopped': raise RuntimeError('Use the explicit resume command to restore the prior phase')
    if verify_source_snapshot(state['source_path'])!=state['source_hash']: raise ValueError('Campaign frozen source mismatch')
    orchestration_hash,orchestration_path=_orchestration_source(state)
    if (orchestration_hash,orchestration_path)!=(state['source_hash'],state['source_path']):
        if verify_source_snapshot(orchestration_path)!=orchestration_hash:
            raise ValueError('Campaign orchestration source mismatch')
    with campaign_lock('controller.lock'):
        current=os.getenv('SLURM_JOB_ID')
        if current: state['controller_job']=current
        if state.get('next_controller_job')==current: state.pop('next_controller_job')
        while not stop_requested():
            state['controller']=dict(job_id=current,pid=os.getpid(),heartbeat=time.time(),host=socket.gethostname())
            _save_state(state)
            try:
                # Applies to pilots, main, latency and reporting alike.
                if time.time()-start>20*3600:
                    if not current: raise RuntimeError('Durable controller must itself run under Slurm')
                    job=submit([PYTHON,'-m','sparse_contrast.campaign','supervise'],'controller',gpu=False,
                        dependency='afterany:'+current,time_limit='1-00:00:00',source=orchestration_path,
                        submission_key=f'controller_handoff:{current}')
                    state['next_controller_job']=job; _save_state(state); return
                if state['phase'] in ('pilots','main'): _advance_items(state)
                elif state['phase'] in ('latency','report'): _advance_postprocessing(state)
                elif state['phase']=='completed' or state['phase'].startswith('blocked_'): break
                else: raise ValueError(f"Unexpected campaign phase: {state['phase']}")
            except SubmissionAmbiguous as error:
                _block(state,'submission',str(error)); break
            _publish(state)
            if state['phase']=='completed' or state['phase'].startswith('blocked_'): break
            time.sleep(60)
        if stop_requested():
            # stop() persists the requested resume phase before cancelling us.
            if state['phase']!='stopped': state['resume_phase']=state['phase']
            state['phase']='stopped'; _save_state(state)
        _publish(state)


def controller_ids(state):
    values=[state.get('controller_job'),state.get('next_controller_job'),state.get('controller',{}).get('job_id')]
    return sorted({str(value) for value in values if value})


def controller_intent_ids():
    result=set()
    for path in (ROOT/'outputs/submissions').glob('*.json'):
        intent=_read(path)
        if intent['request']['name']!='controller': continue
        if intent['state']=='cancelled_before_submit': continue
        job=intent.get('job_id') or _find_intent_jobs(intent)
        if job:
            result.add(job)
            if not intent.get('job_id'):
                intent.update(job_id=job,state='accepted_reconciled',reconciled_at=time.time()); atomic_json(path,intent)
        else: raise SubmissionAmbiguous(f'Controller submission ownership remains unknown: {path}')
    return result


def launch(resume=False):
    with campaign_lock('launch.lock'):
        state=init_campaign()
        if state['phase']=='completed': return None
        controllers=sorted(set(controller_ids(state))|controller_intent_ids()); observed=job_states(controllers)
        active=[job for job in controllers if observed.get(job,{}).get('state') in ACTIVE_STATES]
        unknown=[job for job in controllers if observed.get(job,{}).get('state') is None]
        if active: return active[-1]
        if unknown: raise RuntimeError(f'Previous controller scheduler ownership is unknown: {unknown}')
        if not resume and stop_requested(): raise RuntimeError('STOP exists; use resume to explicitly resume the campaign')
        if resume:
            if state['phase']=='stopped':
                phase=state.get('resume_phase')
                if phase not in ('pilots','main','latency','report'): raise RuntimeError('Stopped campaign has no valid resume phase')
                state['phase']=phase
                for item in state.get('pilots',[])+state.get('main',[]):
                    if item['state'] in ('completed','failed','blocked'): continue
                    job=item.get('job_id')
                    if not job: continue
                    terminal=job_states([job]).get(job,{}).get('state')
                    if terminal in ACTIVE_STATES or terminal is None: raise RuntimeError(f'Worker stop not yet reconciled: {job}')
                    if terminal!='COMPLETED':
                        lock=run_dir(item['config'])/('writer.lock' if item['stage']=='train' else 'evaluation/writer.lock')
                        recover_dead_lock(lock,expected_job=job)
                        item['attempt']=item.get('attempt',0)+1
                        item.update(job_id=None,state='planned')
                if phase=='latency':
                    latency_records=([state['latency_prepare']] if state.get('latency_prepare') else [])+state.get('latency_blocks',[])
                    latency_statuses=job_states([record['job_id'] for record in latency_records if record.get('job_id')])
                    for record in latency_records:
                        if record['state'] in ('completed','failed','blocked') or not record.get('job_id'): continue
                        job=record['job_id']; terminal=latency_statuses.get(job,{}).get('state')
                        if terminal in ACTIVE_STATES or terminal is None:
                            raise RuntimeError(f'Latency block stop not yet reconciled: {job}')
                        # Only an explicit stop cancellation starts a new planned
                        # segment here; other failures retain normal retry accounting.
                        if terminal=='CANCELLED':
                            record['attempt']+=1; record['segments']+=1
                            record.update(job_id=None,state='planned')
                if phase in ('latency','report'):
                    job=state.get(phase+'_job')
                    if job:
                        terminal=job_states([job]).get(job,{}).get('state')
                        if terminal in ACTIVE_STATES or terminal is None: raise RuntimeError(f'Postprocessing stop not yet reconciled: {job}')
                        if terminal!='COMPLETED':
                            state[phase+'_attempt']=state.get(phase+'_attempt',0)+1
                            state[phase+'_job']=None
            elif state['phase']=='blocked_submission':
                state['phase']=state['resume_phase']
            elif state['phase'].startswith('blocked_'):
                raise RuntimeError('Recorded failure requires diagnosis/correction before resume; retry counters cannot be reset by restart')
            if stop_requested(): (ROOT/'STOP').unlink()
            if state.get('blocker'):
                state['history'].append(dict(time=time.time(),action='resume',previous_blocker=state.pop('blocker')))
        elif state['phase']=='stopped' or state['phase'].startswith('blocked_'):
            raise RuntimeError('Use explicit resume after resolving the recorded blocker')
        recover_dead_lock(ROOT/'outputs/controller.lock')
        # A launch crash after Slurm acceptance reuses this generation and the
        # durable submission intent. Increment only once its old job is terminal.
        generation=state.get('controller_generation',0)
        orchestration_hash,orchestration_path=_orchestration_source(state)
        key=f"controller_launch:{orchestration_hash}:{generation}"
        path=ROOT/'outputs/submissions'/f"{digest(dict(root=str(ROOT),key=key))}.json"
        previous=_read(path).get('job_id') if path.exists() else state.get('controller_job')
        if previous and observed.get(previous,{}).get('state') in TERMINAL_STATES:
            generation+=1
        state['controller_generation']=generation
        state['controller_job']=None
        _save_state(state)
        job=submit([PYTHON,'-m','sparse_contrast.campaign','supervise'],'controller',gpu=False,
                   time_limit='1-00:00:00',source=orchestration_path,
                   submission_key=f"controller_launch:{orchestration_hash}:{generation}")
        state['controller_job']=job; _save_state(state)
        return job


def stop():
    # Disable supervision first; then stop its owner before reconciling workers.
    (ROOT/'STOP').touch()
    path=ROOT/'outputs/campaign.json'
    if not path.exists(): return
    state=_read(path)
    if state['phase']!='stopped': state['resume_phase']=state['phase']
    state['phase']='stopped'; _save_state(state)
    controllers=controller_ids(state)
    if controllers:
        statuses=job_states(controllers)
        active=[job for job in controllers if statuses.get(job,{}).get('state') in ACTIVE_STATES]
        if active: subprocess.run(['scancel',*active],check=True)
        for _ in range(30):
            observed=job_states(controllers)
            if all(observed.get(job,{}).get('state') in TERMINAL_STATES for job in controllers): break
            time.sleep(1)
        else: raise RuntimeError('STOP is set, but controller termination is not yet proven; rerun stop to finish reconciliation')
    state=_read(path)
    jobs={str(i['job_id']) for i in state.get('pilots',[])+state.get('main',[]) if i.get('job_id')}
    jobs.update(str(state[key]) for key in ('latency_job','report_job') if state.get(key))
    latency_records=([state['latency_prepare']] if state.get('latency_prepare') else [])+state.get('latency_blocks',[])
    jobs.update(str(record['job_id']) for record in latency_records if record.get('job_id'))
    for intent_path in (ROOT/'outputs/submissions').glob('*.json'):
        intent=_read(intent_path)
        if intent['state']=='cancelled_before_submit': continue
        job=intent.get('job_id') or _find_intent_jobs(intent)
        if job: jobs.add(job)
    states=job_states(sorted(jobs))
    active=[job for job in jobs if states.get(job,{}).get('state') in ACTIVE_STATES]
    if active: subprocess.run(['scancel',*sorted(active)],check=True)
    state['phase']='stopped'; _publish(state)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['launch','supervise','status','stop','resume'])
    args=parser.parse_args()
    if args.action=='supervise': supervise()
    elif args.action in ('launch','resume'): print(launch(resume=args.action=='resume'))
    elif args.action=='stop': stop()
    else:
        path=ROOT/'outputs/campaign.json'
        print(path.read_text() if path.exists() else (ROOT/'STATUS.json').read_text())


if __name__=='__main__': main()
