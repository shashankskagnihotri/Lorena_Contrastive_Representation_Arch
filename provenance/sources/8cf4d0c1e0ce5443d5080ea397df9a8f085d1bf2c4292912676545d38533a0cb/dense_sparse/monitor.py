#!/usr/bin/env python3
"""Bounded Slurm-hosted diagnosis monitor; the campaign controller owns all jobs."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

CODEX = '/home/sagnihot/.vscode-server/extensions/openai.chatgpt-26.5917.62051-linux-x64/bin/linux-x86_64/codex'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'OUT_OF_MEMORY', 'DEADLINE', 'REVOKED'}
HALT = False


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stopped(root):
    from .common import BASE_ROOT
    return HALT or (root/'STOP').exists() or (BASE_ROOT/'STOP').exists() or (root/'outputs/agent_monitor/STOP').exists()


def records(campaign):
    result=[]
    for family in ('evaluation','latency_workers'):
        for record in campaign.get(family,[]):
            if record.get('state')!='completed':
                result.append((family+':'+record['identity'],record))
    for family in ('accuracy_report','final_report'):
        if campaign.get(family):
            result.append((family,campaign[family]))
    return result


def scheduler(root, ids):
    from .scheduler import observed
    return observed(ids)


def timing_root(root, campaign):
    cohort = campaign.get('latency_cohort', {}).get('id')
    return root/'outputs/benchmarks'/('cohorts/'+cohort if cohort else '')


def progress_signature(root,campaign,label,record):
    from .common import case_dir, read_json
    if record.get('test_config_hash'):
        case=read_json(root/'configs'/f"{record['test_config_hash']}.json")
        rd=case_dir(case)
        if label.startswith('evaluation:'):
            directory=rd/'evaluation'
            paths=list(directory.glob('*.json'))+[directory/'progress.json']
        else:
            directory=rd/'latency'/record['execution']/f"batch{record['batch_size']}"/f"shard{record['shard']}"
            paths=list(directory.glob('cells/*/summary.json'))+[directory/'progress.json',directory/'summary.json']
    else:
        paths=list((root/'outputs/slurm').glob('*-'+str(record.get('job_id'))+'.out'))
        paths+=list((root/'outputs/tables').glob('*.csv'))
        paths+=list((root/'outputs/figures').rglob('*.png'))+[root/'REPORT.md']
    return digest([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(set(paths)) if p.is_file()])


def incidents(root, campaign, observed, ledger, now):
    found = []
    def add(kind, identity, evidence):
        found.append({'kind': kind, 'identity': identity, 'fingerprint': digest([kind,identity]), 'evidence': evidence})
    controller = campaign.get('controller', {})
    controller_ids = {str(value) for value in (controller.get('job_id'),campaign.get('controller_job'),campaign.get('next_controller_job')) if value}
    statuses = [observed.get(job, {}).get('state') for job in controller_ids]
    if campaign.get('phase') != 'completed' and controller_ids:
        if all(status in TERMINAL for status in statuses): add('controller_dead','controller',{'jobs':sorted(controller_ids),'scheduler':statuses})
        elif observed.get(str(controller.get('job_id')), {}).get('state') == 'RUNNING' and now-controller.get('heartbeat',now)>900:
            add('controller_stale','controller',controller)
    if campaign.get('phase','').startswith('blocked'):
        add('campaign_blocked',campaign['phase'],campaign.get('blocker'))
    for label, record in records(campaign):
        status = observed.get(str(record.get('job_id')), {}).get('state')
        if record.get('state','').startswith(('failed','blocked')):
            add('work_failed',label,record.get('failure',record.get('state')))
        elif label.endswith('_report') and status in TERMINAL-{'COMPLETED'}:
            add('report_failed','report',{'job_id':record['job_id'],'scheduler':status})
        if status != 'RUNNING':
            ledger.pop(label, None); continue
        signature = progress_signature(root,campaign,label,record)
        previous = ledger.get(label,{})
        if previous.get('signature') != signature or previous.get('job_id') != record.get('job_id'):
            ledger[label] = {'signature':signature,'changed_at':now,'job_id':record.get('job_id')}
        elif now-previous['changed_at']>2700:
            add('running_without_progress',label,{'job_id':record.get('job_id'),'unchanged_since':previous['changed_at']})
    if campaign.get('controller_error'): add('controller_error','controller',campaign['controller_error'])
    if campaign.get('phase') == 'completed': add('final_verification','campaign',{'phase':'completed'})
    return found


def complete_gate(root,campaign,observed):
    from .common import load_manifest
    manifest=load_manifest();expected=manifest['expected']
    coverage=root/'outputs/tables/coverage.json'
    return (campaign.get('phase')=='completed' and len(campaign['evaluation'])==expected['evaluation_cases'] and
            all(x['state']=='completed' for x in campaign['evaluation']) and
            len(campaign['latency_workers'])==expected['latency_workers'] and
            all(x['state']=='completed' for x in campaign['latency_workers']) and
            coverage.is_file() and read(coverage).get('all_requested_work_complete') is True and
            not any(x['state'] in ACTIVE for x in observed.values()))


def eligible(incident, state):
    record = state['incidents'].get(incident['fingerprint'],{})
    return state['agent_calls']<8 and record.get('calls',0)<3


def audit_failure_incident(state, error, now):
    identity = type(error).__name__+':'+str(error)
    fingerprint = digest(['audit_failed',identity])
    previous = state.get('audit_failures') or {}
    consecutive = previous.get('consecutive',0)+1 if previous.get('fingerprint')==fingerprint else 1
    state['audit_failures']={'fingerprint':fingerprint,'consecutive':consecutive}
    state['last_audit_error']={'time':now,'error':repr(error)}
    if consecutive>=3:
        return {'kind':'audit_failed','identity':identity,'fingerprint':fingerprint,
                'evidence':{'error':repr(error),'consecutive':consecutive}}
    return None


def terminate(process):
    if process.poll() is not None: return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait()


def invoke(root, state, incident, codex, destination, deadline):
    if stopped(root): return {'exit_code':None,'reason':'stopped_before_launch'}
    number = state['agent_calls']+1
    out = destination/'invocations'/f'{number:02d}_{incident["fingerprint"][:12]}'
    out.mkdir(parents=True,exist_ok=False)
    instructions = root/'MONITOR.md'
    prompt = ('ACTIVE_REPAIR. Read '+str(instructions)+'. Handle only the following real campaign incident. '
              'Do not start another agent, Codex process, supervisor or persistent loop. The authoritative controller owns jobs. '
              'Reconcile ownership before interventions; preserve scientific scope and all evidence. Return within this invocation. '
              'If this is final_verification, verify complete coverage, hashes, report/figures and no remaining owned compute. '
              'Final response must be one JSON object with keys incident_resolved (boolean), campaign_complete_verified (boolean), '
              'summary (string), evidence (array of paths). Never claim complete without actual full verification.\n'+json.dumps(incident,indent=2))
    (out/'prompt.txt').write_text(prompt)
    command = [codex,'exec','--skip-git-repo-check','--ephemeral','--json','--color','never','-s','danger-full-access',
               '-c','approval_policy="never"','-C',str(root.parent),'-o',str(out/'final_response.txt'),prompt]
    state['agent_calls']=number
    entry=state['incidents'].setdefault(incident['fingerprint'],{'calls':0})
    entry.update(calls=entry['calls']+1,last_started=time.time(),incident=incident)
    active={'number':number,'directory':str(out),'command':command,'started_at':time.time(),'state':'launching'}
    state['active_invocation']=active; write(destination/'state.json',state); write(out/'invocation.json',active)
    process=None; result={}; reason='exited'
    try:
        with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            if stopped(root):
                result={'exit_code':None,'reason':'stopped_before_launch'};return result
            process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
            active.update(pid=process.pid,state='running');write(destination/'state.json',state)
            until=min(deadline,time.time()+2700)
            while process.poll() is None:
                if stopped(root) or time.time()>=until:
                    reason='stopped' if stopped(root) else 'timeout';terminate(process);break
                time.sleep(2)
            result={'exit_code':process.returncode,'reason':reason}
        final=out/'final_response.txt'
        if final.exists():
            try:
                response=json.loads(final.read_text())
                if not isinstance(response,dict): raise ValueError('Final JSON response must be an object')
                result['response']=response
            except json.JSONDecodeError: result['response_parse_error']='Final response was not one JSON object'
            except ValueError as error: result['response_parse_error']=str(error)
    except OSError as error:
        result={'reason':'runtime_error','error':repr(error),'exit_code':None}
    finally:
        if process is not None: terminate(process)
        active.update(result=result,ended_at=time.time(),state='finished');write(out/'invocation.json',active)
        entry['last_result']=result;state['active_invocation']=None;write(destination/'state.json',state)
    return result


def acquire_lock(root, destination):
    lock=destination/'monitor.lock'
    if lock.exists():
        owner=read(lock/'owner.json');job=str(owner.get('job_id') or '')
        if not job or scheduler(root,{job}).get(job,{}).get('state') not in TERMINAL:
            raise RuntimeError('Monitor lease owner is live or unknown; refusing duplicate monitor')
        lock.rename(lock.with_name('monitor.lock.retired.'+str(time.time_ns())))
    lock.mkdir();write(lock/'owner.json',{'job_id':os.getenv('SLURM_JOB_ID'),'host':socket.gethostname(),'pid':os.getpid()})
    return lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--max-hours',type=float,default=96)
    parser.add_argument('--poll-seconds',type=float,default=300);parser.add_argument('--codex',default=CODEX)
    args=parser.parse_args();root=args.root.resolve()
    if not 0<args.max_hours<=96 or args.poll_seconds<60: parser.error('Require 0 < max-hours <= 96 and poll-seconds >= 60')
    if root.name!='training_dense_testing_sparse' or not (root/'outputs/campaign.json').is_file():
        parser.error('Require the active training_dense_testing_sparse campaign root')
    if not os.getenv('SLURM_JOB_ID'): parser.error('The durable monitor must run inside a scheduler allocation')
    destination=root/'outputs/agent_monitor';destination.mkdir(parents=True,exist_ok=True)
    lock=acquire_lock(root,destination);path=destination/'state.json';now=time.time()
    state=read(path) if path.exists() else {'schema_version':1,'root':str(root),'started_at':now,'deadline':now+args.max_hours*3600,'agent_calls':0,'incidents':{},'progress':{}}
    if state['root']!=str(root): raise ValueError('Monitor state belongs to another root')
    state['deadline']=min(state['deadline'],now+args.max_hours*3600)
    state.update(job_id=os.getenv('SLURM_JOB_ID'),host=socket.gethostname(),pid=os.getpid())
    def halt(*_args):
        global HALT
        HALT=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    try:
        if state.get('state')=='completed_verified': return
        while time.time()<state['deadline'] and not stopped(root):
            now=time.time();state.update(state='monitoring',heartbeat=now,mode='ACTIVE_REPAIR' if (destination/'READY').exists() else 'WAITING_READY')
            try:
                campaign=read(root/'outputs/campaign.json')
                jobs={str(record['job_id']) for _,record in records(campaign) if record.get('job_id')}
                jobs.update(str(value) for value in (campaign.get('controller_job'),campaign.get('next_controller_job'),campaign.get('controller',{}).get('job_id')) if value)
                observed=scheduler(root,jobs)
                found=incidents(root,campaign,observed,state['progress'],now)
                state.update(phase=campaign.get('phase'),scheduler_counts=dict(Counter(x['state'] for x in observed.values())),current_incidents=found,last_audit_error=None,audit_failures=None,
                             blocked_incidents=[x['fingerprint'] for x in found if not eligible(x,state)])
                write(path,state)
                selected=next((x for x in found if eligible(x,state)),None)
                if selected and state['mode']=='ACTIVE_REPAIR' and not stopped(root):
                    result=invoke(root,state,selected,args.codex,destination,state['deadline'])
                    if selected['kind']=='final_verification' and result.get('exit_code')==0 and result.get('response',{}).get('campaign_complete_verified') is True:
                        latest=read(root/'outputs/campaign.json')
                        if complete_gate(root,latest,scheduler(root,jobs)):
                            state.update(state='completed_verified',completed_at=time.time());write(path,state);return
            except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
                failed=audit_failure_incident(state,error,time.time());write(path,state)
                if failed:
                    state['current_incidents']=[failed]
                    state['blocked_incidents']=[] if eligible(failed,state) else [failed['fingerprint']]
                    write(path,state)
                    if eligible(failed,state) and state['mode']=='ACTIVE_REPAIR' and not stopped(root):
                        invoke(root,state,failed,args.codex,destination,state['deadline'])
            until=min(state['deadline'],time.time()+args.poll_seconds)
            while time.time()<until and not stopped(root): time.sleep(min(5,max(0,until-time.time())))
        state.update(state='stopped' if stopped(root) else 'deadline_incomplete',ended_at=time.time());write(path,state)
    finally:
        (lock/'owner.json').unlink();lock.rmdir()


if __name__=='__main__': main()
