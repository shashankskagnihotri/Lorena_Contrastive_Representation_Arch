"""Pure monitor decisions; no scheduler, GPU or actual agent invocation."""
import importlib.util
import json
import os
from pathlib import Path
import pytest

ROOT = Path(os.getenv('SPARSE_CONTRAST_ROOT',Path(__file__).resolve().parents[1]))
MODULE = ROOT.parent/'debugging/monitor_campaign.py'
spec = importlib.util.spec_from_file_location('campaign_agent_monitor',MODULE)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_pending_work_and_unknown_controller_do_not_trigger_stall(tmp_path):
    campaign={'controller':{'job_id':'1','heartbeat':0},'controller_job':'1',
              'latency_workers':[{'worker_id':0,'job_id':'2','state':'pending'}]}
    assert monitor.incidents(tmp_path,campaign,{'2':{'state':'PENDING'}},{},10000)==[]


def test_controller_death_requires_all_known_terminal_and_pending_successor_is_ok(tmp_path):
    campaign={'controller':{'job_id':'1','heartbeat':0},'controller_job':'1','next_controller_job':'2'}
    assert monitor.incidents(tmp_path,campaign,{'1':{'state':'FAILED'},'2':{'state':'PENDING'}},{},10000)==[]
    result=monitor.incidents(tmp_path,campaign,{'1':{'state':'FAILED'},'2':{'state':'FAILED'}},{},10000)
    assert [x['kind'] for x in result]==['controller_dead']
    campaign.pop('next_controller_job')
    assert monitor.incidents(tmp_path,campaign,{'1':{'state':'RUNNING'}},{},10000)[0]['kind']=='controller_stale'


def test_running_stall_resets_on_real_progress_and_attempt_change(tmp_path,monkeypatch):
    campaign={'latency_workers':[{'worker_id':9,'job_id':'2','state':'running'}]};ledger={}
    monkeypatch.setattr(monitor,'progress_signature',lambda *a:'unchanged')
    observed={'2':{'state':'RUNNING'}}
    assert monitor.incidents(tmp_path,campaign,observed,ledger,100)==[]
    assert monitor.incidents(tmp_path,campaign,observed,ledger,2799)==[]
    result=monitor.incidents(tmp_path,campaign,observed,ledger,2801)
    assert result[0]['kind']=='running_without_progress'
    monkeypatch.setattr(monitor,'progress_signature',lambda *a:'new-cell')
    assert monitor.incidents(tmp_path,campaign,observed,ledger,3000)==[]
    campaign['latency_workers'][0]['job_id']='3'
    assert monitor.incidents(tmp_path,campaign,{'3':{'state':'RUNNING'}},ledger,10000)==[]


def test_failed_worker_incident_stable_across_scheduler_attempts_and_limits_persist(tmp_path):
    campaign={'latency_workers':[{'worker_id':9,'job_id':'2','state':'failed','failure':'error'}]}
    first=monitor.incidents(tmp_path,campaign,{}, {},100)[0]
    campaign['latency_workers'][0]['job_id']='3'
    assert monitor.incidents(tmp_path,campaign,{}, {},200)[0]['fingerprint']==first['fingerprint']
    state={'agent_calls':2,'incidents':{first['fingerprint']:{'calls':2}}}
    assert monitor.eligible(first,state)
    state['incidents'][first['fingerprint']]['calls']=3
    monitor.write(tmp_path/'state.json',state)
    assert not monitor.eligible(first,monitor.read(tmp_path/'state.json'))
    state={'agent_calls':8,'incidents':{}}
    assert not monitor.eligible(first,state)


def test_completion_needs_full_evidence_gate_and_no_active_owned_jobs(tmp_path):
    campaign={'phase':'completed','main':[{'state':'completed','stage':'evaluation'}]*156,
              'latency_workers':[{'state':'completed'}]*552}
    assert not monitor.complete_gate(tmp_path,campaign,{})
    monitor.write(tmp_path/'outputs/tables/coverage.json',{'all_requested_work_complete':True})
    assert monitor.complete_gate(tmp_path,campaign,{'1':{'state':'COMPLETED'}})
    assert not monitor.complete_gate(tmp_path,campaign,{'1':{'state':'RUNNING'}})
    campaign['latency_workers'].pop()
    assert not monitor.complete_gate(tmp_path,campaign,{})


def test_monitor_stop_and_campaign_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,'HALT',False)
    assert not monitor.stopped(tmp_path)
    (tmp_path/'STOP').touch()
    assert monitor.stopped(tmp_path)
    (tmp_path/'STOP').unlink()
    (tmp_path/'outputs/agent_monitor').mkdir(parents=True)
    (tmp_path/'outputs/agent_monitor/STOP').touch()
    assert monitor.stopped(tmp_path)


def test_unverified_remote_monitor_lock_is_never_stolen(tmp_path,monkeypatch):
    destination=tmp_path/'outputs/agent_monitor';lock=destination/'monitor.lock'
    monitor.write(lock/'owner.json',{'job_id':'5','host':'other','pid':44})
    monkeypatch.setattr(monitor,'scheduler',lambda *a:{})
    with pytest.raises(RuntimeError,match='live or unknown'):
        monitor.acquire_lock(tmp_path,destination)
    assert lock.exists()
    monkeypatch.setattr(monitor,'scheduler',lambda *a:{'5':{'state':'FAILED'}})
    monkeypatch.setenv('SLURM_JOB_ID','6')
    assert monitor.acquire_lock(tmp_path,destination)==lock
    assert monitor.read(lock/'owner.json')['job_id']=='6'
    assert len(list(destination.glob('monitor.lock.retired.*')))==1


def test_invocation_uses_verified_cli_profile_and_records_response_without_real_agent(tmp_path,monkeypatch):
    destination=tmp_path/'outputs/agent_monitor'
    state={'agent_calls':0,'incidents':{}}
    incident={'kind':'work_failed','fingerprint':'a'*64,'identity':'latency_workers:0','evidence':'fixture'}
    seen=[]
    class FakeProcess:
        pid=123;returncode=0
        def __init__(self,command,**kwargs):
            seen.append(command)
            assert kwargs['start_new_session'] is True
            Path(command[command.index('-o')+1]).write_text(json.dumps({'incident_resolved':True,'campaign_complete_verified':False}))
        def poll(self): return 0
    monkeypatch.setattr(monitor.subprocess,'Popen',FakeProcess)
    result=monitor.invoke(tmp_path,state,incident,'/fixture/codex',destination,monitor.time.time()+100)
    assert result['exit_code']==0 and result['response']['incident_resolved'] is True
    assert state['agent_calls']==1 and state['active_invocation'] is None
    assert '--ephemeral' in seen[0] and 'danger-full-access' in seen[0] and 'approval_policy="never"' in seen[0]
    directory=next((destination/'invocations').iterdir())
    assert all((directory/name).exists() for name in ('prompt.txt','stdout.jsonl','stderr.txt','final_response.txt','invocation.json'))
    assert monitor.read(directory/'invocation.json')['result']==result


def test_report_csv_and_figure_changes_are_progress(tmp_path):
    campaign={};record={'job_id':'1'}
    before=monitor.progress_signature(tmp_path,campaign,'report',record)
    directory=tmp_path/'outputs/tables';directory.mkdir(parents=True)
    (directory/'accuracy.csv').write_text('metric,value\naccuracy,1\n')
    middle=monitor.progress_signature(tmp_path,campaign,'report',record)
    assert before!=middle
    figures=tmp_path/'outputs/figures/mnist/vit_small';figures.mkdir(parents=True)
    (figures/'comparison.png').write_bytes(b'fixture-png-progress')
    assert monitor.progress_signature(tmp_path,campaign,'report',record)!=middle


def test_repeated_audit_errors_trigger_bounded_incident_after_three(tmp_path):
    state={'agent_calls':0,'incidents':{}}
    for n in (1,2):
        assert monitor.audit_failure_incident(state,ValueError('bad state'),n) is None
    incident=monitor.audit_failure_incident(state,ValueError('bad state'),3)
    assert incident['kind']=='audit_failed' and monitor.eligible(incident,state)
    monitor.write(tmp_path/'state.json',state)
    state=monitor.read(tmp_path/'state.json')
    assert monitor.audit_failure_incident(state,ValueError('bad state'),4)['fingerprint']==incident['fingerprint']
    state['incidents'][incident['fingerprint']]={'calls':3}
    assert not monitor.eligible(incident,state)
    assert monitor.audit_failure_incident(state,ValueError('new error'),5) is None
    state['audit_failures']=None
    assert monitor.audit_failure_incident(state,ValueError('bad state'),6) is None
