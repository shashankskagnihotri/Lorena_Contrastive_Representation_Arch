"""Slurm ownership, immutable submissions, and conservative continuation."""
from __future__ import annotations
import getpass
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from .common import ROOT, BASE_ROOT, PYTHON, atomic_json, read_json, digest, stop_requested, directory_lock

ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED','RESIZING'}
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL','OUT_OF_MEMORY','DEADLINE','REVOKED'}
TRANSIENT = {'PREEMPTED','NODE_FAIL','BOOT_FAIL'}


def observed(job_ids):
    """Query current-user queue and require exact accepted local ownership receipts."""
    ids = {str(x) for x in job_ids if x}
    if not ids:
        return {}
    receipts = {}
    for path in (ROOT/'outputs/submissions').glob('*.json'):
        value = read_json(path)
        if str(value.get('job_id')) in ids and value.get('state') in ('accepted','accepted_reconciled'):
            receipts.setdefault(str(value['job_id']), []).append(value)
    rows = []
    sorted_ids = sorted(ids)
    for start in range(0,len(sorted_ids),300):
        args=['sacct','-X','-n','-P','-j',','.join(sorted_ids[start:start+300]),
              '--format=JobIDRaw,State,ExitCode,User%80,JobName%160,Comment%180']
        out=subprocess.run(args,check=True,text=True,capture_output=True,timeout=60)
        rows.extend((line,True) for line in out.stdout.splitlines())
    out=subprocess.run(['squeue','-h','-u',getpass.getuser(),'--format=%i|%T|%u|%j|%k'],
                       check=True,text=True,capture_output=True,timeout=60)
    for line in out.stdout.splitlines():
        parts=line.split('|')
        if len(parts)>=5:
            rows.append(('|'.join(parts[:2]+['']+parts[2:]),False))
    result={}
    for line,accounting in rows:
        parts=[p.strip() for p in line.split('|')]
        if len(parts)<6:
            continue
        job,state,exit_code,user,name,comment=parts[:6]
        if job not in ids or user!=getpass.getuser():
            continue
        candidates=receipts.get(job,[])
        if len(candidates)!=1:
            raise RuntimeError(f'Ambiguous/missing ownership receipt for job {job}')
        receipt=candidates[0]
        if name != receipt['job_name'] or (comment!=receipt['comment'] and not(accounting and comment=='')):
            raise RuntimeError(f'Conflicting ownership for job {job}')
        result[job]={'state':state.split()[0].rstrip('+'),'exit_code':exit_code or None,'name':name}
    return result


def reconcile_intent(intent):
    out=subprocess.run(['squeue','-h','-u',getpass.getuser(),'--name='+intent['job_name'],'--format=%i|%j|%k'],
                       check=True,text=True,capture_output=True,timeout=60)
    matches=set()
    for line in out.stdout.splitlines():
        parts=line.split('|')
        if len(parts)>=3 and parts[1]==intent['job_name'] and parts[2]==intent['comment'] and parts[0].isdigit():
            matches.add(parts[0])
    if len(matches)>1:
        raise RuntimeError('Multiple scheduler jobs claim one intent')
    # An unaccepted terminal receipt cannot safely use site's missing sacct comments.
    # Unknown submission outcome stays blocked for explicit reconciliation.
    return next(iter(matches)) if matches else None


def submit(command, name, key, source, *, gpu=False, timing=False, memory_gb=16,
           time_limit='1-00:00:00', dependency=None, gpu_partitions=None):
    if stop_requested():
        raise InterruptedError('Campaign STOP prevents submission')
    command=[str(x) for x in command]
    request=dict(command=command,name=name,source=str(source),gpu=gpu,timing=timing,
                 memory_gb=memory_gb,time_limit=time_limit,dependency=dependency,gpu_partitions=gpu_partitions)
    identity=digest({'root':str(ROOT),'key':key})
    path=ROOT/'outputs/submissions'/f'{identity}.json'
    if path.exists():
        intent=read_json(path)
        if intent['request']!=request:
            raise ValueError('A stable submission key cannot change its command')
        if intent.get('state')=='cancelled_before_submit':
            path.rename(path.with_name(path.name+'.cancelled.'+str(time.time_ns())))
            return submit(command,name,key,source,gpu=gpu,timing=timing,memory_gb=memory_gb,
                          time_limit=time_limit,dependency=dependency,gpu_partitions=gpu_partitions)
        if intent.get('job_id'):
            return str(intent['job_id'])
        adopted=reconcile_intent(intent)
        if adopted:
            intent.update(job_id=adopted,state='accepted_reconciled',accepted_at=time.time())
            atomic_json(path,intent)
            return adopted
        raise RuntimeError(f'Ambiguous submission, reconcile before retry: {path}')
    intent=dict(identity=identity,request=request,prepared_at=time.time(),state='prepared',job_id=None,
                job_name=f'dss-{name}-{identity[:16]}',comment='dense-sparse:'+identity)
    atomic_json(path,intent)
    logs=ROOT/'outputs/slurm';logs.mkdir(parents=True,exist_ok=True)
    args=['sbatch','--parsable','--account=ml-staff','--qos=walltime5d',
          '--job-name='+intent['job_name'],'--comment='+intent['comment'],
          '--cpus-per-task=4',f'--mem={memory_gb}G','--time='+time_limit,
          f'--output={logs}/%x-%j.out',f'--error={logs}/%x-%j.err']
    if gpu:
        args += ['--partition=gpu-vram-48gb' if timing else '--partition='+(gpu_partitions or 'gpu-vram-94gb,gpu-vram-48gb,gpu-vram-32gb,gpu-vram-12gb'),
                 '--gres=gpu:nvidia_rtx_a6000:1' if timing else '--gres=gpu:1']
        if timing:
            args += ['--exclude=dws-10']
    else:
        args += ['--partition=cpu']
    if dependency:
        args += ['--dependency='+dependency]
    env={'PATH':str(Path(PYTHON).parent)+':'+os.environ['PATH'],'DENSE_SPARSE_ROOT':str(ROOT),
         'SPARSE_CONTRAST_ROOT':str(BASE_ROOT),'PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'}
    script='#!/usr/bin/env bash\nset -euo pipefail\n'+''.join(f'export {k}={shlex.quote(v)}\n' for k,v in env.items())
    script+='cd '+shlex.quote(str(source))+'\nexec '+shlex.join(command)+'\n'
    if stop_requested():
        intent.update(state='cancelled_before_submit',cancelled_at=time.time())
        atomic_json(path,intent)
        raise InterruptedError('STOP immediately before sbatch; definitely not submitted')
    try:
        out=subprocess.run(args,input=script,text=True,capture_output=True,check=True,timeout=90)
    except (OSError,subprocess.SubprocessError) as error:
        intent.update(state='ambiguous',error=repr(error),stderr=getattr(error,'stderr',None))
        atomic_json(path,intent)
        raise
    job=out.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Ambiguous sbatch response retained in intent: {out.stdout!r}')
    intent.update(state='accepted',job_id=job,accepted_at=time.time())
    atomic_json(path,intent)
    return job


def recover_lock(path):
    path=Path(path)
    if not path.exists():
        return
    owner=read_json(path/'owner.json');job=owner.get('job_id')
    if job:
        if observed([job]).get(str(job),{}).get('state') not in TERMINAL:
            raise RuntimeError(f'Lock owner is active/unknown: {job}')
    elif owner.get('host')==socket.gethostname():
        try:
            os.kill(int(owner['pid']),0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Local owner remains alive')
    else:
        raise RuntimeError('Cannot establish remote lock owner death')
    path.rename(path.with_name(path.name+'.retired.'+str(time.time_ns())))
