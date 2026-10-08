"""Pure scheduler/persistence tests. No real jobs are submitted or cancelled."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparse_contrast import campaign
from sparse_contrast.common import atomic_json, digest


def test_group_recipe_preserves_defaults_and_resolves_dataset_epoch_budget():
    manifest={'recipe':{'batch_size':8,'epochs':{'mnist':50,'cifar10':200},
                       'warmup_epochs':{'mnist':5,'cifar10':10},'lr':3e-4}}
    saved=copy.deepcopy(manifest)
    recipe=campaign.resolved_recipe(manifest,'cifar10','vit_small')
    assert recipe=={'batch_size':8,'epochs':200,'warmup_epochs':10,'lr':3e-4}
    assert manifest==saved


def test_group_recipe_has_one_override_independent_of_representation_or_seed():
    manifest={'recipe':{'batch_size':8,'epochs':{'mnist':50,'cifar10':200},
                       'warmup_epochs':{'mnist':5,'cifar10':10},'lr':3e-4},
              'recipe_by_group':{'mnist/vit_small':{'batch_size':128,'effective_batch_size':128,
                  'gradient_accumulation':1,'lr':6e-4,'eval_batch_size':128}}}
    recipe=campaign.resolved_recipe(manifest,'mnist','vit_small')
    assert recipe['batch_size']==recipe['effective_batch_size']==128
    assert recipe['epochs']==50 and recipe['warmup_epochs']==5
    assert recipe['lr']==6e-4 and recipe['eval_batch_size']==128
    recipe['batch_size']=256
    assert campaign.resolved_recipe(manifest,'mnist','vit_small')['batch_size']==128
    with pytest.raises(ValueError,match='Missing group recipe'):
        campaign.resolved_recipe(manifest,'mnist','swin_tiny')


def primary_runs():
    runs=[]
    for dataset in ('mnist','cifar10'):
        for architecture in ('vit_small','swin_tiny'):
            for seed in (0,1,2):
                conditions=[('raw','dense',None)]
                for representation in ('single_color','grayscale','color_opponency'):
                    conditions.append((representation,'dense',0))
                    conditions.extend((representation,'compact',percent) for percent in (0,20,40,60,80))
                for representation,execution,percent in conditions:
                    runs.append(dict(dataset=dataset,architecture=architecture,seed=seed,
                        representation=representation,execution=execution,sparsity_percent=percent,
                        registry_id=f'{dataset}/{architecture}/{representation}/{execution}/p{percent}/seed{seed}'))
    return runs


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    monkeypatch.setattr(campaign, 'stop_requested', lambda: (tmp_path/'STOP').exists())
    (tmp_path/'outputs').mkdir()
    atomic_json(tmp_path/'experiment_manifest.json',dict(state='planned_not_frozen',runs=primary_runs()))
    return tmp_path


def result(stdout=''):
    return SimpleNamespace(stdout=stdout, stderr='', returncode=0)


def test_selected_group_hardware_and_stage_walltime_reach_scheduler(root,monkeypatch):
    work=item(job=None,state='planned',stage='evaluation')
    work['config']['recipe'].update(gpu_partitions='gpu-vram-94gb,gpu-vram-48gb',
                                   evaluation_time_limit='04:00:00')
    campaign_state=state([work],phase='main')
    calls=[]
    def submit(command,name,**kwargs):
        calls.append((command,name,kwargs));return '910'
    monkeypatch.setattr(campaign,'submit',submit)
    campaign._submit_item(campaign_state,work)
    assert calls[0][2]['gpu_partition']=='gpu-vram-94gb,gpu-vram-48gb'
    assert calls[0][2]['time_limit']=='04:00:00'
    assert work['job_id']=='910' and work['stage']=='evaluation'


def item(job='12', stage='train', state='submitted', retries=0):
    return dict(config=dict(config_hash='abc', dataset='mnist', architecture='vit_small',
                            registry_id='condition', recipe={'epochs':50}),
                config_path='config.json', job_id=job, stage=stage, state=state,
                attempt=0, segments=0, transient_retries=retries)


def state(work, phase='pilots'):
    for record in work: record['config']['role']='pilot' if phase=='pilots' else 'main'
    return dict(phase=phase,source_hash='source',source_path='/frozen',pilots=work if phase=='pilots' else [],
                main=work if phase=='main' else [],history=[],controller={},controller_generation=0)


def test_submission_intent_precedes_sbatch_and_repeated_call_is_idempotent(root, monkeypatch):
    calls=[]
    def run(args, **kwargs):
        assert args[0]=='sbatch'
        journal=list((root/'outputs/submissions').glob('*.json'))
        assert len(journal)==1
        assert json.loads(journal[0].read_text())['state']=='prepared'
        assert any(arg.startswith('--comment=sparse-contrast:') for arg in args)
        assert not any('USR1' in arg for arg in args)
        calls.append(args)
        return result('101;cluster\n')
    monkeypatch.setattr(campaign.subprocess,'run',run)
    assert campaign.submit(['python','worker.py'],'train',submission_key='work:abc:train:0')=='101'
    assert campaign.submit(['python','worker.py'],'train',submission_key='work:abc:train:0')=='101'
    assert len(calls)==1


def test_ambiguous_submission_reconciles_ownership_without_another_sbatch(root, monkeypatch):
    submissions=[]
    def run(args, **kwargs):
        if args[0]=='sbatch':
            submissions.append(args)
            return result('connection interrupted after possible acceptance')
        intent=json.loads(next((root/'outputs/submissions').glob('*.json')).read_text())
        assert args[0] in ('squeue','sacct')
        return result(f"711|{intent['job_name']}|{intent['comment']}\n")
    monkeypatch.setattr(campaign.subprocess,'run',run)
    with pytest.raises(campaign.SubmissionAmbiguous):
        campaign.submit(['python','worker.py'],'train',submission_key='key')
    assert campaign.submit(['python','worker.py'],'train',submission_key='key')=='711'
    assert len(submissions)==1


def test_unresolved_submission_never_retries_sbatch(root, monkeypatch):
    calls=[]
    def run(args, **kwargs):
        calls.append(args[0])
        return result('uncertain' if args[0]=='sbatch' else '')
    monkeypatch.setattr(campaign.subprocess,'run',run)
    for _ in range(3):
        with pytest.raises(campaign.SubmissionAmbiguous): campaign.submit(['p'],'train',submission_key='same')
    assert calls.count('sbatch')==1


def test_stop_blocks_submission_before_any_scheduler_call(root, monkeypatch):
    (root/'STOP').touch()
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**k: pytest.fail('STOP ignored'))
    with pytest.raises(InterruptedError): campaign.submit(['p'],'train')


def test_source_snapshot_is_complete_atomic_and_independent_of_live_edits(root):
    (root/'sparse_contrast/nested').mkdir(parents=True)
    (root/'scripts').mkdir()
    files={'run_all.sh':'#!/bin/sh\nexit 0\n','train.py':'print(1)\n',
           'sparse_contrast/nested/module.py':'value = 7\n','scripts/worker.sh':'exit 0\n',
           'requirements.lock.txt':'torch==2.9.1\n','pytest.ini':'[pytest]\ntestpaths=tests\n',
           'tests/nested/test_case.py':'def test_example(): assert True\n',
           'provenance/original_frontend.py':'source_oracle = True\n'}
    for name,value in files.items():
        (root/name).parent.mkdir(exist_ok=True,parents=True)
        (root/name).write_text(value)
    identity,directory=campaign.source_snapshot()
    assert campaign.verify_source_snapshot(directory)==identity
    for name,value in files.items(): assert (directory/name).read_text()==value
    (root/'train.py').write_text('print(2)\n')
    assert (directory/'train.py').read_text()=='print(1)\n'
    (directory/'train.py').write_text('tampered\n')
    with pytest.raises(ValueError): campaign.verify_source_snapshot(directory)


def test_lease_reclaimed_only_after_positive_terminal_owner(root, monkeypatch):
    path=root/'outputs/controller.lock'; path.mkdir()
    atomic_json(path/'owner.json',dict(job_id='22',host='worker',pid=123))
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'22':{'state':'RUNNING'}})
    with pytest.raises(RuntimeError): campaign.recover_dead_lock(path,expected_job='22')
    assert path.exists()
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'22':{'state':'NODE_FAIL'}})
    campaign.recover_dead_lock(path,expected_job='22')
    assert not path.exists()
    assert len(list((root/'outputs').glob('controller.lock.retired.*')))==1


def test_running_pilot_cannot_advance_even_if_completion_file_appears(root, monkeypatch):
    work=item(); saved=state([work])
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'12':{'state':'RUNNING'}})
    monkeypatch.setattr(campaign,'_completion',lambda work: pytest.fail('Must await terminal process'))
    monkeypatch.setattr(campaign,'submit',lambda *a,**k: pytest.fail('Premature submission'))
    campaign._advance_items(saved)
    assert saved['phase']=='pilots' and work['state']=='running'


def test_training_completion_launches_evaluation_not_latency(root, monkeypatch):
    work=item(); saved=state([work],phase='main'); calls=[]
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'12':{'state':'COMPLETED','exit_code':'0:0'}})
    monkeypatch.setattr(campaign,'_completion',lambda work: True)
    monkeypatch.setattr(campaign,'submit',lambda command,*a,**k: calls.append(command) or '13')
    campaign._advance_one(saved,work,{'state':'COMPLETED','exit_code':'0:0'})
    assert work['stage']=='evaluation' and work['job_id']=='13'
    assert saved['phase']=='main'
    assert calls[0][2]=='sparse_contrast.evaluate'


def test_mixed_completed_failed_registry_blocks_instead_of_spinning(root, monkeypatch):
    saved=state([item(state='completed'),item(job='13',state='failed')],phase='main')
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {})
    monkeypatch.setattr(campaign,'submit',lambda *a,**k: pytest.fail('Failed work cannot enable timing'))
    campaign._advance_items(saved)
    assert saved['phase']=='blocked_work_failed'
    assert saved['resume_phase']=='main'


def test_three_infrastructure_retries_are_not_reset_by_restart(root, monkeypatch):
    work=item(retries=3); saved=state([work])
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'12':{'state':'NODE_FAIL','exit_code':'1:0'}})
    monkeypatch.setattr(campaign,'_completion',lambda work: False)
    monkeypatch.setattr(campaign,'run_dir',lambda config: root/'run')
    monkeypatch.setattr(campaign,'submit',lambda *a,**k: pytest.fail('Retry limit exceeded'))
    campaign._advance_items(saved)
    assert work['transient_retries']==3 and work['state']=='failed'
    assert saved['phase']=='blocked_work_failed'


def test_resume_restores_pilot_phase_and_retains_retry_count(root, monkeypatch):
    work=item(job='88',retries=2); saved=state([work])
    saved.update(phase='stopped',resume_phase='pilots',controller_job='77')
    (root/'STOP').touch()
    monkeypatch.setattr(campaign,'init_campaign',lambda: saved)
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {job:{'state':'CANCELLED'} for job in jobs})
    monkeypatch.setattr(campaign,'run_dir',lambda config: root/'run')
    monkeypatch.setattr(campaign,'submit',lambda *a,**k: '99')
    assert campaign.launch(resume=True)=='99'
    assert saved['phase']=='pilots' and saved['controller_job']=='99'
    assert work['job_id'] is None and work['attempt']==1 and work['transient_retries']==2
    assert not (root/'STOP').exists()


def test_handed_off_active_controller_is_reused(root, monkeypatch):
    saved=state([item()]); saved.update(controller_job='1',next_controller_job='2')
    monkeypatch.setattr(campaign,'init_campaign',lambda: saved)
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'1':{'state':'COMPLETED'},'2':{'state':'RUNNING'}})
    monkeypatch.setattr(campaign,'submit',lambda *a,**k: pytest.fail('Duplicate controller'))
    assert campaign.launch()=='2'


def test_stop_cancels_reporting_as_well_as_latency(root, monkeypatch):
    saved=state([],phase='main'); saved.update(phase='report',report_job='44',latency_job='33')
    atomic_json(root/'outputs/campaign.json',saved)
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {job:{'state':'RUNNING'} for job in jobs})
    calls=[]
    monkeypatch.setattr(campaign.subprocess,'run',lambda args,**kwargs: calls.append(args) or result())
    campaign.stop()
    assert calls==[['scancel','33','44']]
    current=json.loads((root/'outputs/campaign.json').read_text())
    assert current['phase']=='stopped' and current['resume_phase']=='report'


def test_unknown_scheduler_state_retains_job_ownership(root, monkeypatch):
    work=item(); saved=state([work])
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {})
    monkeypatch.setattr(campaign,'submit',lambda *a,**k: pytest.fail('Unknown status must not duplicate work'))
    campaign._advance_items(saved)
    assert work['job_id']=='12' and work['state']=='scheduler_unknown'


def test_transient_backoff_and_attempt_survive_controller_ticks(root, monkeypatch):
    work=item(); saved=state([work]); calls=[]; now=[100.]
    monkeypatch.setattr(campaign.time,'time',lambda: now[0])
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'12':{'state':'NODE_FAIL'}} if jobs else {})
    monkeypatch.setattr(campaign,'_completion',lambda work: False)
    monkeypatch.setattr(campaign,'run_dir',lambda config: root/'run')
    monkeypatch.setattr(campaign,'submit',lambda *a,**k: calls.append(k['submission_key']) or '13')
    campaign._advance_items(saved)
    assert work['transient_retries']==1 and work['attempt']==1 and work['retry_not_before']==160.
    assert not calls
    now[0]=130.
    campaign._advance_items(saved)
    assert not calls and work['state']=='waiting_retry'
    now[0]=160.
    campaign._advance_items(saved)
    assert calls==['work:abc:train:1'] and work['job_id']=='13'
    assert work['transient_retries']==1


def test_failed_pilot_verdict_cannot_freeze_or_launch_main(root, monkeypatch):
    saved=state([item(state='completed')])
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {})
    monkeypatch.setattr(campaign,'pilot_verdict',lambda config: {'passed':False,'reason':'loss not reducing'})
    monkeypatch.setattr(campaign,'resolve_config',lambda *a,**k: pytest.fail('Main must remain unfrozen'))
    campaign._advance_items(saved)
    assert saved['phase']=='blocked_pilot_review' and saved['main']==[]


def full_pilot_state():
    pilots=[]
    for index,run in enumerate(run for run in primary_runs() if run['representation']=='raw' and run['seed']==0):
        config=dict(run,role='pilot',registry_id='pilots/'+run['registry_id'],
                    recipe={'epochs':50 if run['dataset']=='mnist' else 200})
        config['config_hash']=digest(config)
        pilots.append(dict(config=config,config_path=f'pilot{index}.json',job_id=str(100+index),
                           stage='train',state='running',attempt=0,segments=0,transient_retries=0))
    return dict(phase='pilots',source_hash='source',source_path='/frozen',pilots=pilots,main=[],
                groups={},history=[],controller={},controller_generation=0)


def group_harness(root, monkeypatch, saved, failed_groups=()):
    submissions=[]; resolutions=[]
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {job:{'state':'RUNNING'} for job in jobs})
    monkeypatch.setattr(campaign,'_completion',lambda work: False)
    monkeypatch.setattr(campaign,'pilot_verdict',lambda config: dict(passed=campaign._group_key(config) not in failed_groups,
                                                                    reason='mocked full clean pilot evidence'))
    def resolve(run,source_hash,pilot=False):
        assert not pilot
        config=dict(run,role='main',source_hash=source_hash,
                    recipe={'epochs':50 if run['dataset']=='mnist' else 200})
        config['config_hash']=digest(config)
        resolutions.append(config['config_hash'])
        return config,root/'configs'/f"{config['config_hash']}.json"
    monkeypatch.setattr(campaign,'resolve_config',resolve)
    def submit(command,*args,**kwargs):
        submissions.append((command,kwargs))
        return str(1000+len(submissions))
    monkeypatch.setattr(campaign,'submit',submit)
    return submissions,resolutions


def test_ready_group_launches_57_fresh_main_runs_while_other_pilots_run(root, monkeypatch):
    saved=full_pilot_state(); saved['pilots'][0]['state']='completed'
    submissions,resolutions=group_harness(root,monkeypatch,saved)
    campaign._advance_items(saved)
    assert len(saved['main'])==len(submissions)==len(resolutions)==57
    assert saved['phase']=='main'
    assert all(work['config']['dataset']=='mnist' and work['config']['architecture']=='vit_small' for work in saved['main'])
    assert all(work['config']['role']=='main' and not work['config']['registry_id'].startswith('pilots/') for work in saved['main'])
    assert all(work['stage']=='train' and work['state']=='submitted' for work in saved['main'])
    assert all(pilot['state']=='running' for pilot in saved['pilots'][1:])
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    assert manifest['state']=='partial_frozen' and len(manifest['runs'])==228
    assert sum(bool(run.get('resolved_config_hash')) for run in manifest['runs'])==57


def test_failed_group_stays_blocked_while_another_group_progresses(root, monkeypatch):
    saved=full_pilot_state()
    saved['pilots'][0]['state']=saved['pilots'][1]['state']='completed'
    submissions,_=group_harness(root,monkeypatch,saved,failed_groups={'mnist/vit_small'})
    campaign._advance_items(saved)
    assert saved['phase']=='main' and len(submissions)==57
    assert saved['groups']['mnist/vit_small']['state']=='blocked_pilot_review'
    assert saved['groups']['mnist/swin_tiny']['state']=='frozen'
    assert all(work['config']['architecture']=='swin_tiny' for work in saved['main'])
    # A later tick continues valid workers even though one group's recipe is blocked.
    campaign._advance_items(saved)
    assert saved['phase']=='main' and all(work['state']=='running' for work in saved['main'])


def test_restart_does_not_duplicate_a_frozen_57_run_group(root, monkeypatch):
    saved=full_pilot_state(); saved['pilots'][0]['state']='completed'
    submissions,resolutions=group_harness(root,monkeypatch,saved)
    campaign._advance_items(saved)
    restarted=json.loads((root/'outputs/campaign.json').read_text())
    before=[work['config']['config_hash'] for work in restarted['main']]
    # Simulate process loss after the authoritative state commit but before
    # the separate manifest write; recovery repairs metadata without new work.
    atomic_json(root/'experiment_manifest.json',dict(state='planned_not_frozen',runs=primary_runs()))
    campaign._advance_items(restarted)
    assert len(submissions)==len(resolutions)==57
    assert [work['config']['config_hash'] for work in restarted['main']]==before
    assert len(set(before))==57
    recovered=json.loads((root/'experiment_manifest.json').read_text())
    assert recovered['state']=='partial_frozen'
    assert sum(bool(run.get('resolved_config_hash')) for run in recovered['runs'])==57


def test_completed_one_group_cannot_enable_partial_latency(root, monkeypatch):
    saved=full_pilot_state(); saved['pilots'][0]['state']='completed'
    submissions,_=group_harness(root,monkeypatch,saved)
    campaign._advance_items(saved)
    for work in saved['main']: work.update(state='completed',stage='evaluation')
    campaign._advance_items(saved)
    assert saved['phase']=='main' and len(saved['main'])==57 and len(submissions)==57
    assert not any('benchmark.py' in command for command,_ in submissions)


def test_latency_requires_all_four_groups_and_all_228_evaluations(root, monkeypatch):
    saved=full_pilot_state()
    for pilot in saved['pilots']: pilot['state']='completed'
    submissions,_=group_harness(root,monkeypatch,saved)
    campaign._advance_items(saved)
    assert len(saved['main'])==228 and saved['phase']=='main'
    assert json.loads((root/'experiment_manifest.json').read_text())['state']=='frozen'
    for work in saved['main'][:-1]: work.update(state='completed',stage='evaluation')
    campaign._advance_items(saved)
    assert saved['phase']=='main'
    saved['main'][-1].update(state='completed',stage='evaluation')
    campaign._advance_items(saved)
    assert saved['phase']=='latency' and saved['latency_job'] is None
    assert len(submissions)==228


@pytest.mark.parametrize('evidence',[None,{'all_requested_work_complete':False}])
def test_report_exit_zero_without_complete_coverage_blocks(root, monkeypatch,evidence):
    saved=state([],phase='main'); saved.update(phase='report',report_job='81')
    if evidence is not None: atomic_json(root/'outputs/tables/coverage.json',evidence)
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'81':{'state':'COMPLETED'}})
    campaign._advance_postprocessing(saved)
    assert saved['phase']=='blocked_report_coverage'


def test_report_verified_coverage_completes_campaign(root, monkeypatch):
    saved=state([],phase='main'); saved.update(phase='report',report_job='81')
    atomic_json(root/'outputs/tables/coverage.json',{'all_requested_work_complete':True})
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'81':{'state':'COMPLETED'}})
    campaign._advance_postprocessing(saved)
    assert saved['phase']=='completed'


@pytest.mark.parametrize('status',[None,'RUNNING'])
def test_legacy_latency_ownership_is_retained_until_positive_terminal(root, monkeypatch,status):
    saved=state([],phase='main'); saved.update(phase='latency',latency_job='91')
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'91':{'state':status}})
    monkeypatch.setattr(campaign,'_advance_latency',lambda saved: pytest.fail('Old worker still owns latency output'))
    campaign._advance_postprocessing(saved)
    assert saved['phase']=='latency' and saved['latency_job']=='91'


def test_legacy_count_only_completion_is_routed_through_strict_block_validation(root, monkeypatch):
    saved=state([],phase='main'); saved.update(phase='latency',latency_job='91')
    atomic_json(root/'outputs/benchmarks/progress.json',{'state':'completed','workers':816})
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {'91':{'state':'COMPLETED'}})
    strict=[]
    monkeypatch.setattr(campaign,'_advance_latency',lambda value: strict.append(value))
    campaign._advance_postprocessing(saved)
    assert saved['phase']=='latency' and saved['latency_job'] is None and strict==[saved]


def grayscale_scope(root):
    from sparse_contrast.scope import read_scope
    scope=read_scope(root); scope['active_representations']['mnist']=['grayscale']
    atomic_json(root/'experiment_scope.json',scope)
    runs=[run for run in primary_runs() if run['dataset']!='mnist' or run['representation'] in ('raw','grayscale')]
    atomic_json(root/'experiment_manifest.json',dict(state='planned_not_frozen',runs=runs))
    return scope


def test_scoped_groups_release_156_main_and_gate_exact_evaluation_coverage(root,monkeypatch):
    grayscale_scope(root)
    saved=full_pilot_state()
    saved.update(orchestration_source_hash='new_controller',orchestration_source_path='/scoped_controller')
    for pilot in saved['pilots']: pilot['state']='completed'
    submissions,_=group_harness(root,monkeypatch,saved)
    campaign._advance_items(saved)
    assert len(saved['main'])==len(submissions)==156
    assert {key:group['expected_main_runs'] for key,group in saved['groups'].items()}=={
        'mnist/vit_small':21,'mnist/swin_tiny':21,'cifar10/vit_small':57,'cifar10/swin_tiny':57}
    assert all(work['config']['source_hash']=='source' for work in saved['main'])
    assert all(kwargs['source']=='/frozen' for _,kwargs in submissions)
    assert not any(work['config']['dataset']=='mnist' and work['config']['representation'] in ('single_color','color_opponency') for work in saved['main'])
    for work in saved['main'][:-1]: work.update(state='completed',stage='evaluation')
    campaign._advance_items(saved)
    assert saved['phase']=='main' and len(submissions)==156
    saved['main'][-1].update(state='completed',stage='evaluation')
    campaign._advance_items(saved)
    assert saved['phase']=='latency'
    assert saved['source_hash']=='source' and saved['source_path']=='/frozen'


def test_scoped_controller_rejects_unamended_frozen_group(root,monkeypatch):
    saved=full_pilot_state(); saved['pilots'][0]['state']='completed'
    submissions,_=group_harness(root,monkeypatch,saved)
    campaign._advance_items(saved)
    assert len(submissions)==57
    grayscale_scope(root)
    with pytest.raises(ValueError,match='Frozen group'):
        campaign._advance_items(saved)
    assert len(submissions)==57


@pytest.mark.parametrize('stage',['train','evaluation'])
def test_retained_training_and_evaluation_use_original_source(root,monkeypatch,stage):
    work=item(job=None,stage=stage,state='planned'); saved=state([work],phase='main')
    saved.update(orchestration_source_hash='new_controller',orchestration_source_path='/scoped_controller')
    calls=[]
    monkeypatch.setattr(campaign,'submit',lambda *args,**kwargs: calls.append((args,kwargs)) or '99')
    campaign._submit_item(saved,work)
    assert calls[0][1]['source']=='/frozen'
    assert saved['source_hash']=='source' and saved['source_path']=='/frozen'


@pytest.mark.parametrize('amended',[False,True])
def test_controller_launch_uses_orchestration_source_with_legacy_fallback(root,monkeypatch,amended):
    saved=state([],phase='main')
    if amended: saved.update(orchestration_source_hash='new_controller',orchestration_source_path='/scoped_controller')
    calls=[]
    monkeypatch.setattr(campaign,'init_campaign',lambda: saved)
    monkeypatch.setattr(campaign,'controller_intent_ids',lambda: set())
    monkeypatch.setattr(campaign,'job_states',lambda jobs: {})
    monkeypatch.setattr(campaign,'submit',lambda *args,**kwargs: calls.append((args,kwargs)) or '99')
    assert campaign.launch()=='99'
    assert calls[0][1]['source']==('/scoped_controller' if amended else '/frozen')
    assert calls[0][1]['submission_key']==f"controller_launch:{'new_controller' if amended else 'source'}:0"
    assert saved['source_path']=='/frozen' and saved['source_hash']=='source'


def test_orchestration_hash_and_path_cannot_be_partially_replaced(root):
    saved=state([],phase='main'); saved['orchestration_source_hash']='new_controller'
    with pytest.raises(ValueError,match='recorded together'):
        campaign._orchestration_source(saved)


def test_scope_manifest_is_part_of_source_snapshot(root):
    grayscale_scope(root)
    _,snapshot=campaign.source_snapshot()
    assert (snapshot/'experiment_scope.json').read_bytes()==(root/'experiment_scope.json').read_bytes()


def test_controller_handoff_uses_amended_orchestration_snapshot(root,monkeypatch):
    saved=state([],phase='main')
    saved.update(orchestration_source_hash='new_controller',orchestration_source_path='/scoped_controller')
    monkeypatch.setattr(campaign,'init_campaign',lambda: saved)
    verified=[]
    def verify(path):
        verified.append(path)
        return {'/frozen':'source','/scoped_controller':'new_controller'}[path]
    monkeypatch.setattr(campaign,'verify_source_snapshot',verify)
    clock=[0]
    def now():
        clock[0]+=1
        return 0 if clock[0]==1 else 21*3600
    monkeypatch.setattr(campaign.time,'time',now)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    calls=[]
    monkeypatch.setattr(campaign,'submit',lambda *args,**kwargs: calls.append((args,kwargs)) or '124')
    campaign.supervise()
    assert verified==['/frozen','/scoped_controller']
    assert calls[0][1]['source']=='/scoped_controller'
    assert calls[0][1]['dependency']=='afterany:123'
    assert saved['next_controller_job']=='124'
    assert saved['source_hash']=='source' and saved['source_path']=='/frozen'
