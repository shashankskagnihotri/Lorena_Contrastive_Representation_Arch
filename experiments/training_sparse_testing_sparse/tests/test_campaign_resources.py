"""Explicit report memory survives durable submission and cancellation recovery."""
import json
from types import SimpleNamespace

import pytest

from sparse_contrast import campaign
from test_campaign import root


@pytest.mark.parametrize('gpu,expected', [(False, 8), (True, 16)])
def test_omitted_memory_preserves_defaults_and_old_intent_identity(root, monkeypatch, gpu, expected):
    calls=[]
    monkeypatch.setattr(campaign.subprocess,'run',lambda command,**kwargs: calls.append(command) or SimpleNamespace(stdout='31'))
    campaign.submit(['worker'],'default-memory',gpu=gpu,submission_key='default')
    assert f'--mem={expected}G' in calls[0]
    intent=json.loads(next((root/'outputs/submissions').glob('*.json')).read_text())
    assert 'memory_gb' not in intent['request']


@pytest.mark.parametrize('value',[0,-1,True,False,1.5,'96'])
def test_memory_override_requires_positive_integer_before_scheduler(root,monkeypatch,value):
    monkeypatch.setattr(campaign.subprocess,'run',lambda *args,**kwargs: pytest.fail('Invalid resource reached scheduler'))
    with pytest.raises(ValueError,match='positive integer'):
        campaign.submit(['worker'],'report',gpu=False,memory_gb=value)
    assert not (root/'outputs/submissions').exists()


def test_report_memory_and_node_exclusion_survive_cancelled_intent_recursive_resubmission(root,monkeypatch):
    calls=[]
    monkeypatch.setattr(campaign.subprocess,'run',lambda command,**kwargs: calls.append(command) or SimpleNamespace(stdout='32'))
    checks=iter([False,True])
    monkeypatch.setattr(campaign,'stop_requested',lambda:next(checks))
    with pytest.raises(InterruptedError):
        campaign.submit(['report.py'],'report',gpu=False,memory_gb=96,exclude_nodes='dws-10,dws-09',submission_key='report-96')
    assert calls==[]
    monkeypatch.setattr(campaign,'stop_requested',lambda:False)
    assert campaign.submit(['report.py'],'report',gpu=False,memory_gb=96,exclude_nodes='dws-10,dws-09',submission_key='report-96')=='32'
    assert '--mem=96G' in calls[0] and '--exclude=dws-10,dws-09' in calls[0]
    intent=json.loads(next((root/'outputs/submissions').glob('*.json')).read_text())
    assert intent['request']['memory_gb']==96 and intent['request']['exclude_nodes']=='dws-10,dws-09'
    with pytest.raises(ValueError,match='different command'):
        campaign.submit(['report.py'],'report',gpu=False,memory_gb=64,submission_key='report-96')
    assert len(calls)==1


@pytest.mark.parametrize('nodes',['','dws-10,','dws-10 --all','--help','dws-[10-14]','dws-10,dws-10',7])
def test_exclusion_accepts_only_distinct_literal_nodes(root,monkeypatch,nodes):
    monkeypatch.setattr(campaign.subprocess,'run',lambda *args,**kwargs: pytest.fail('Invalid node list reached scheduler'))
    with pytest.raises(ValueError,match='literal node names'):
        campaign.submit(['worker'],'latency',exclude_nodes=nodes)


def test_final_report_requests_96gb_and_twelve_hours(root,monkeypatch):
    calls=[]
    monkeypatch.setattr(campaign,'submit',lambda command,name,**kwargs:calls.append((command,name,kwargs)) or '33')
    state=dict(phase='report',source_hash='source',source_path='/frozen')
    campaign._advance_postprocessing(state)
    assert state['report_job']=='33'
    assert calls[0][2]['memory_gb']==96 and calls[0][2]['time_limit']=='12:00:00'
