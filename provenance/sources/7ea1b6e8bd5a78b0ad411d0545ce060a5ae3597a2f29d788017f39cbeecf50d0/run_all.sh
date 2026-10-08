#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/ceph/sagnihot/miniconda3/envs/sparse-contrast-bench/bin:$PATH
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export SPARSE_CONTRAST_ROOT="$PWD"
SCB_DATA_ROOT="${1:-/ceph/sagnihot/datasets}"
if [ ! -f datasets_manifest.json ]; then
    python prepare_data.py --data-root "$SCB_DATA_ROOT" --download
fi
python - "$SCB_DATA_ROOT" <<'PY'
import sys
from sparse_contrast.data import load_manifest
load_manifest(data_root=sys.argv[1])
PY
# Prerequisite work is scheduled; workers never claim SSH-visible idle GPUs.
python - <<'PY'
import json
from sparse_contrast.common import ROOT,atomic_json
from sparse_contrast.campaign import submit,PYTHON,job_states,campaign_lock,TERMINAL_STATES
with campaign_lock('prerequisites.lock'):
    gate=ROOT/'outputs/checks/completed.json'
    if gate.exists():
        if not json.loads(gate.read_text()).get('passed'):
            raise RuntimeError('Prerequisite completion artifact records failed checks')
        import subprocess
        subprocess.run([PYTHON,'-m','sparse_contrast.campaign','launch'],check=True)
        raise SystemExit(0)
    record=ROOT/'outputs/checks/run_all_jobs.json'
    if record.exists():
        saved=json.loads(record.read_text())
        if 'jobs' not in saved: saved={'schema_version':2,'state':'submitted','jobs':saved}
    else:
        saved={'schema_version':2,'state':'preparing','jobs':{}}
        atomic_json(record,saved)
    jobs=saved['jobs']; states=job_states(list(jobs.values()))
    failed={stage:states[job] for stage,job in jobs.items()
            if states.get(job,{}).get('state') in TERMINAL_STATES-{'COMPLETED'}}
    if failed:
        saved.update(state='blocked',failed=failed);atomic_json(record,saved)
        raise RuntimeError('Prerequisite failure is recorded, including any still-pending dependent jobs. '
                           'Inspect and record the correction before creating a distinct repaired attempt: '+json.dumps(failed))
    unknown=[job for job in jobs.values() if job not in states]
    if unknown:
        print(json.dumps({'state':'scheduler_ownership_unknown','job_ids':unknown,'jobs':jobs},indent=2))
        raise SystemExit(0)
    phases=[
        ('core',['/bin/bash','scripts/verify_core.sh'],'verify-core',True,'04:00:00'),
        ('overfit',['/bin/bash','scripts/overfit.sh'],'overfit',True,'03:00:00'),
        ('integration',['/bin/bash','scripts/verify_end_to_end.sh'],'integration',True,'04:00:00'),
        ('launch',[PYTHON,'-m','sparse_contrast.campaign','launch'],'launch',False,'01:00:00'),
    ]
    if all(phase in jobs for phase,_,_,_,_ in phases) and all(states.get(job,{}).get('state')=='COMPLETED' for job in jobs.values()):
        saved.update(state='blocked',failure='all jobs exited but required completion gate is absent')
        atomic_json(record,saved)
        raise RuntimeError('Every prerequisite job exited, but the verified completion artifact is missing')
    previous=None
    for phase,command,name,gpu,limit in phases:
        if phase not in jobs:
            # Persist the intended phase before submit(). Its durable journal
            # reconciles a crash between Slurm acceptance and this ledger update.
            saved.update(state='preparing',pending_stage=phase);atomic_json(record,saved)
            job=submit(command,name,gpu=gpu,dependency='afterok:'+previous if previous else None,time_limit=limit)
            jobs[phase]=job;saved.pop('pending_stage',None);atomic_json(record,saved)
        previous=jobs[phase]
    saved['state']='submitted';atomic_json(record,saved)
    print(json.dumps({'jobs':jobs,'observed_states':states},indent=2))
PY
