"""Immutable asynchronous selection from complete real-profile evidence schemas."""
import copy
import json
from pathlib import Path

import pytest

from sparse_contrast.batch_selection import SCIENTIFIC_FILES, refresh_selection
from sparse_contrast.common import atomic_json, digest, file_hash
from sparse_contrast.scope import ARCHITECTURES, DATASETS, configurations


def snapshot(source):
    manifest = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*.py')}
    atomic_json(source / 'source_manifest.json', manifest)
    return digest(manifest)


@pytest.fixture
def evidence(tmp_path):
    root, profile, source = tmp_path, tmp_path / 'profile_source', tmp_path / 'final_source'
    scope = dict(schema_version=1, active_representations=dict(mnist=['grayscale'],
        cifar10=['single_color', 'grayscale', 'color_opponency']))
    atomic_json(root / 'experiment_scope.json', scope)
    for directory in (profile, source):
        for name in SCIENTIFIC_FILES:
            path = directory / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic scientific source fixture: ' + name)
        (directory / 'sparse_contrast/campaign.py').write_text('# ' + directory.name)
    profile_hash = snapshot(profile); snapshot(source)
    launch = dict(source_hash=profile_hash, source_path=str(profile), jobs={}, large_memory_jobs={})
    statuses, paths = {}, {}
    for number, (dataset, architecture) in enumerate((d, a) for d in DATASETS for a in ARCHITECTURES):
        group, job = dataset + '/' + architecture, str(100 + number)
        launch['jobs'][group] = job if architecture == 'swin_tiny' else 'old-' + job
        if architecture == 'vit_small': launch['large_memory_jobs'][group] = job
        statuses[job] = dict(state='COMPLETED')
        directory = root / 'outputs' / ('batch_profile_48gb' if architecture == 'vit_small' else 'batch_profile') / dataset / architecture
        paths[group] = directory
        configs = []
        for rep, execution, percent in configurations(dataset, scope):
            config = dict(dataset=dataset, architecture=architecture, seed=0, representation=rep,
                execution=execution, sparsity_percent=percent, precision='fp32', normalization_hash='fixture')
            config['config_hash'] = digest(config); configs.append(config)
        source_files = json.loads((profile / 'source_manifest.json').read_text())
        plan = dict(dataset=dataset, architecture=architecture, seed=0, conditions=configs,
            root=str(root), output=str(directory), source_root=str(profile), scope=scope,
            candidates=[128, 256], headroom_fraction=.15, plateau_fraction=.05, source_files=source_files,
            input_files={'experiment_scope.json': file_hash(root / 'experiment_scope.json')})
        plan['plan_hash'] = digest(plan); atomic_json(directory / 'plan.json', plan)
        results, candidates = [], []
        for batch in plan['candidates']:
            for config in configs:
                p = config['sparsity_percent']
                condition = f"{config['representation']}_{config['execution']}_p{'NA' if p is None else p}"
                row = dict(batch_size=batch, condition=condition, state='passed', capacity_safe=True,
                    finite_loss_gradients_parameters=True, plan_hash=plan['plan_hash'],
                    template_config_hash=config['config_hash'], normalization_hash=config['normalization_hash'],
                    profile_source_files_sha256=digest(source_files), precision='fp32', gradient_accumulation=1,
                    estimated_headroom_fraction=.2, samples_per_second=100.,
                    isolation_before=dict(verified=True, scheduler_job_id=job),
                    isolation_after=dict(verified=True, scheduler_job_id=job))
                results.append(row); atomic_json(directory / 'workers' / condition / f'batch{batch}.json', row)
            candidates.append(dict(batch_size=batch, conditions_recorded=len(configs), all_conditions_safe=True,
                harmonic_mean_samples_per_second=100., minimum_headroom_fraction=.2))
        atomic_json(directory / 'summary.json', dict(state='completed', dataset=dataset, architecture=architecture,
            plan_hash=plan['plan_hash'], conditions_required=len(configs), records=len(results), results=results,
            candidates=candidates, headroom_fraction=.15, proposed_useful_batch=256))
    atomic_json(root / 'outputs/batch_profile/launch.json', launch)
    return root, source, statuses, paths


def test_ready_swin_starts_independently_and_prefers_48gb_vit(evidence):
    root, source, statuses, _ = evidence
    statuses['100']['state'] = 'RUNNING'; statuses['102']['state'] = 'PENDING'
    ledger = refresh_selection(root, source, statuses)
    assert ledger['state'] == 'pending'
    assert set(ledger['groups']) == {'mnist/swin_tiny', 'cifar10/swin_tiny'}
    assert set(ledger['pending_groups']) == {'mnist/vit_small', 'cifar10/vit_small'}
    assert ledger['groups']['mnist/swin_tiny']['conditions'] == 7
    assert ledger['groups']['cifar10/swin_tiny']['conditions'] == 19
    assert refresh_selection(root, source, statuses) == ledger


def test_all_groups_verified_and_recipe_is_uniform_explicit_new_recipe(evidence):
    root, source, statuses, _ = evidence
    ledger = refresh_selection(root, source, statuses)
    assert ledger['state'] == 'verified' and len(ledger['groups']) == 4
    for group, recipe in ledger['recipe_by_group'].items():
        assert recipe['batch_size'] == recipe['effective_batch_size'] == recipe['eval_batch_size'] == 256
        assert recipe['gradient_accumulation'] == 1
        assert recipe['num_workers'] == recipe['eval_num_workers'] == 0
        assert recipe['lr'] == pytest.approx(3e-4 * 2 ** .5)
        assert 'epochs' not in recipe and 'warmup_epochs' not in recipe
        assert recipe['train_time_limit'] == '24:00:00' and recipe['evaluation_time_limit'] == '04:00:00'
        assert ('gpu-vram-12gb' in recipe['gpu_partitions']) == group.endswith('swin_tiny')


@pytest.mark.parametrize('state', ['FAILED', 'OUT_OF_MEMORY', 'CANCELLED', 'TIMEOUT'])
def test_failed_preferred_job_blocks_even_with_completed_summary(evidence, state):
    root, source, statuses, _ = evidence
    statuses['100']['state'] = state
    ledger = refresh_selection(root, source, statuses)
    assert ledger['state'] == 'blocked' and 'mnist/vit_small' not in ledger['groups']
    assert ledger['blocked_groups']['mnist/vit_small']['state'] == state


@pytest.mark.parametrize('change', ['finite', 'headroom', 'duplicate', 'plan', 'partial', 'worker', 'aggregate', 'other_candidate'])
def test_invalid_profile_evidence_never_releases_a_group(evidence, change):
    root, source, statuses, paths = evidence
    directory = paths['mnist/swin_tiny']; path = directory / 'summary.json'
    summary = json.loads(path.read_text())
    row = next(r for r in summary['results'] if r['batch_size'] == 256)
    if change == 'finite': row['finite_loss_gradients_parameters'] = False
    if change == 'headroom': row['estimated_headroom_fraction'] = .14
    if change == 'duplicate': summary['results'][-1] = copy.deepcopy(row)
    if change == 'plan': summary['plan_hash'] = 'bad'
    if change == 'partial': summary['state'] = 'partial'
    if change == 'worker': row['samples_per_second'] = 1000.
    if change == 'aggregate': summary['candidates'][-1]['harmonic_mean_samples_per_second'] = 1000.
    if change == 'other_candidate': summary['candidates'][0]['harmonic_mean_samples_per_second'] = 1000.
    if change in ('finite', 'headroom'):
        atomic_json(directory / 'workers' / row['condition'] / 'batch256.json', row)
    atomic_json(path, summary)
    ledger = refresh_selection(root, source, statuses)
    assert 'mnist/swin_tiny' in ledger['blocked_groups'] and 'mnist/swin_tiny' not in ledger['groups']


def test_scientific_byte_drift_rejected_but_coordination_changes_allowed(evidence):
    root, source, statuses, _ = evidence
    assert refresh_selection(root, source, statuses)['state'] == 'verified'
    (source / SCIENTIFIC_FILES[0]).write_text('# altered training implementation')
    snapshot(source)
    with pytest.raises(ValueError, match='Frozen group evidence changed'):
        refresh_selection(root, source, statuses)


def test_accepted_group_cannot_be_reselected_after_summary_drift(evidence):
    root, source, statuses, paths = evidence
    first = refresh_selection(root, source, statuses)
    path = paths['mnist/swin_tiny'] / 'summary.json'
    summary = json.loads(path.read_text()); summary['additional_note'] = 'changed evidence'
    atomic_json(path, summary)
    with pytest.raises(ValueError, match='Frozen group selection changed'):
        refresh_selection(root, source, statuses)
    assert json.loads((root / 'outputs/batch_profile/selection.json').read_text()) == first


def test_controller_materializes_ready_pilots_once_and_cannot_finish_while_selection_pending(evidence, monkeypatch):
    from sparse_contrast import campaign
    root, source, statuses, _ = evidence
    monkeypatch.setattr(campaign, 'ROOT', root)
    monkeypatch.setattr(campaign, 'job_states', lambda jobs: {j: statuses[j] for j in jobs if j in statuses})
    runs = [dict(dataset=d, architecture=a, representation='raw', execution='dense',
                 sparsity_percent=None, seed=0, registry_id=d+'/'+a+'/raw/seed0')
            for d in DATASETS for a in ARCHITECTURES]
    manifest = dict(runs=runs, recipe=dict(batch_size=8, epochs={'mnist':50, 'cifar10':200},
        warmup_epochs={'mnist':5, 'cifar10':10}), recipe_by_group={})
    atomic_json(root / 'experiment_manifest.json', manifest)
    resolved = []
    def resolve(run, source_hash, pilot=False):
        current = json.loads((root / 'experiment_manifest.json').read_text())
        config = dict(run, role='pilot' if pilot else 'main', source_hash=source_hash,
                      recipe=campaign.resolved_recipe(current, run['dataset'], run['architecture']))
        config['config_hash'] = digest(config); resolved.append(config)
        return config, root / 'configs' / (config['config_hash'] + '.json')
    monkeypatch.setattr(campaign, 'resolve_config', resolve)
    state = dict(phase='pilots', source_hash=snapshot(source), source_path=str(source),
                 pilots=[], main=[], history=[], groups={}, batch_selection_pending=True)
    for job in statuses: statuses[job]['state'] = 'PENDING'
    statuses['101']['state'] = 'COMPLETED'
    campaign._advance_batch_selection(state)
    assert len(state['pilots']) == len(resolved) == 1 and state['batch_selection_pending']
    pilot = state['pilots'][0]
    assert (pilot['config']['dataset'], pilot['config']['architecture']) == ('mnist', 'swin_tiny')
    assert pilot['config']['recipe']['batch_size'] == 256 and pilot['config']['recipe']['epochs'] == 50
    pilot['state'] = 'completed'
    campaign._finish_training_phase(state)
    assert state['phase'] == 'pilots' and not state['main']
    campaign._advance_batch_selection(state)
    assert len(state['pilots']) == len(resolved) == 1
    for job in statuses: statuses[job]['state'] = 'COMPLETED'
    campaign._advance_batch_selection(state)
    assert len(state['pilots']) == len(resolved) == 4 and not state['batch_selection_pending']
    assert state['phase'] == 'pilots' and not state['main']
