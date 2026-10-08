"""Validate real capacity evidence and freeze independent group recipes, without launches."""
from __future__ import annotations
import json
import math
from pathlib import Path

from .common import atomic_json, digest, file_hash
from .scope import ARCHITECTURES, DATASETS, configurations, read_scope

SCIENTIFIC_FILES = tuple(f'sparse_contrast/{name}.py' for name in (
    'train', 'data', 'models', 'frontend', 'normalization', 'pipeline', 'common', 'evaluate', 'metrics'))
FAILED_STATES = {'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL',
                 'OUT_OF_MEMORY', 'DEADLINE', 'REVOKED'}
ALL_PARTITIONS = 'gpu-vram-94gb,gpu-vram-48gb,gpu-vram-32gb,gpu-vram-12gb'


def _read(path):
    return json.loads(Path(path).read_text())


def _condition(config):
    percent = config['sparsity_percent']
    return f"{config['representation']}_{config['execution']}_p{'NA' if percent is None else percent}"


def _selection(root, source, launch, dataset, architecture, job_id):
    base = 'batch_profile_48gb' if architecture == 'vit_small' else 'batch_profile'
    directory = root / 'outputs' / base / dataset / architecture
    plan_path, summary_path = directory / 'plan.json', directory / 'summary.json'
    plan, summary, scope = _read(plan_path), _read(summary_path), read_scope(root)
    if digest({k: v for k, v in plan.items() if k != 'plan_hash'}) != plan['plan_hash']:
        raise ValueError('Profile plan digest mismatch')
    if (summary['state'] != 'completed' or summary['plan_hash'] != plan['plan_hash'] or
        any(value['dataset'] != dataset or value['architecture'] != architecture for value in (plan, summary)) or
        plan['scope'] != scope or Path(plan['root']).resolve() != root or
        Path(plan['output']).resolve() != directory or Path(plan['source_root']).resolve() != Path(launch['source_path']).resolve()):
        raise ValueError('Completed profile identity, scope or path mismatch')
    profile_source = Path(plan['source_root'])
    profile_manifest = _read(profile_source / 'source_manifest.json')
    if digest(profile_manifest) != launch['source_hash']:
        raise ValueError('Profile source snapshot identity mismatch')
    science = {}
    for name in SCIENTIFIC_FILES:
        expected = plan['source_files'][name]
        if profile_manifest[name] != expected or file_hash(profile_source / name) != expected or file_hash(source / name) != expected:
            raise ValueError('Scientific source differs from measured profile: ' + name)
        science[name] = expected
    for name, expected in plan['input_files'].items():
        if file_hash(root / name) != expected:
            raise ValueError('Profile input changed: ' + name)
    expected = set(configurations(dataset, scope))
    configs = plan['conditions']
    actual = [(c['representation'], c['execution'], c['sparsity_percent']) for c in configs]
    if len(actual) != len(expected) or set(actual) != expected or summary['conditions_required'] != len(expected):
        raise ValueError('Profile conditions do not exactly cover the active group')
    for config in configs:
        if (config['dataset'] != dataset or config['architecture'] != architecture or config['seed'] != plan['seed'] or
            config['precision'] != 'fp32' or config.get('subset_ids') is not None or
            digest({k: v for k, v in config.items() if k != 'config_hash'}) != config['config_hash']):
            raise ValueError('Invalid resolved profile condition')
    if plan['headroom_fraction'] < .15 or summary['headroom_fraction'] != plan['headroom_fraction']:
        raise ValueError('At least 15 percent measured memory headroom is required')
    batch = summary['proposed_useful_batch']
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1 or batch not in plan['candidates']:
        raise ValueError('Completed profile has no usable proposed batch')
    candidates = summary['candidates']
    if (len(candidates) != len(plan['candidates']) or {c['batch_size'] for c in candidates} != set(plan['candidates']) or
        summary['records'] != len(summary['results']) or summary['records'] != len(expected) * len(candidates)):
        raise ValueError('Completed profile candidate coverage is incomplete')
    for candidate in candidates:
        measured = [r for r in summary['results'] if r['batch_size'] == candidate['batch_size']]
        if len(measured) != len(expected) or {r['condition'] for r in measured} != {_condition(c) for c in configs}:
            raise ValueError('Candidate condition coverage is incomplete')
        safe = all(r['state'] == 'passed' and r['capacity_safe'] for r in measured)
        if candidate['all_conditions_safe'] != safe or candidate['conditions_recorded'] != len(measured):
            raise ValueError('Candidate safety aggregate differs from worker evidence')
        if safe:
            if any(not math.isfinite(r['samples_per_second']) or r['samples_per_second'] <= 0 for r in measured):
                raise ValueError('Invalid measured throughput')
            score = len(measured) / sum(1 / r['samples_per_second'] for r in measured)
            if not math.isclose(candidate['harmonic_mean_samples_per_second'], score, rel_tol=1e-12):
                raise ValueError('Candidate throughput aggregate differs from worker evidence')
    eligible = [c for c in candidates if c['all_conditions_safe']]
    if any(not math.isfinite(c['harmonic_mean_samples_per_second']) or c['harmonic_mean_samples_per_second'] <= 0 for c in eligible):
        raise ValueError('Invalid measured throughput')
    best = max((c['harmonic_mean_samples_per_second'] for c in eligible), default=0)
    useful = [c['batch_size'] for c in eligible if c['harmonic_mean_samples_per_second'] >= (1 - plan['plateau_fraction']) * best]
    if not useful or batch != max(useful):
        raise ValueError('Proposed batch does not follow the frozen throughput policy')
    rows = [row for row in summary['results'] if row['batch_size'] == batch]
    by_condition = {_condition(c): c for c in configs}
    if len(rows) != len(expected) or {r['condition'] for r in rows} != set(by_condition):
        raise ValueError('Selected batch lacks exact unique condition coverage')
    worker_hashes = {}
    for row in rows:
        config = by_condition[row['condition']]
        path = directory / 'workers' / row['condition'] / f'batch{batch}.json'
        if (_read(path) != row or row['state'] != 'passed' or row['capacity_safe'] is not True or
            row.get('finite_loss_gradients_parameters') is not True or row['plan_hash'] != plan['plan_hash'] or
            row['template_config_hash'] != config['config_hash'] or row['normalization_hash'] != config['normalization_hash'] or
            row['profile_source_files_sha256'] != digest(plan['source_files']) or row['precision'] != 'fp32' or
            row['gradient_accumulation'] != 1 or not math.isfinite(row['estimated_headroom_fraction']) or
            row['estimated_headroom_fraction'] < plan['headroom_fraction'] or
            any(row[key].get('verified') is not True or str(row[key].get('scheduler_job_id')) != job_id
                for key in ('isolation_before', 'isolation_after'))):
            raise ValueError('Selected worker evidence is invalid: ' + row['condition'])
        worker_hashes[str(path.relative_to(root))] = file_hash(path)
    candidate = next(c for c in candidates if c['batch_size'] == batch)
    score = len(rows) / sum(1 / row['samples_per_second'] for row in rows)
    if (candidate['conditions_recorded'] != len(rows) or candidate['all_conditions_safe'] is not True or
        not math.isclose(candidate['harmonic_mean_samples_per_second'], score, rel_tol=1e-12) or
        candidate['minimum_headroom_fraction'] != min(row['estimated_headroom_fraction'] for row in rows)):
        raise ValueError('Selected aggregate does not match measured workers')
    recipe = dict(batch_size=batch, effective_batch_size=batch, gradient_accumulation=1,
        eval_batch_size=batch, num_workers=0, eval_num_workers=0,
        lr=3e-4 * math.sqrt(batch / 128), min_lr=1e-6, betas=[.9, .999], weight_decay=.05,
        gpu_partitions='gpu-vram-94gb,gpu-vram-48gb' if architecture == 'vit_small' else ALL_PARTITIONS,
        train_time_limit='24:00:00', evaluation_time_limit='04:00:00')
    return dict(dataset=dataset, architecture=architecture, state='verified', profile_job_id=job_id,
        plan_path=str(plan_path), plan_hash=plan['plan_hash'], plan_sha256=file_hash(plan_path),
        summary_path=str(summary_path), summary_sha256=file_hash(summary_path), worker_sha256=worker_hashes,
        scientific_files=science, conditions=len(expected), selected_batch_size=batch, recipe=recipe,
        policy='Fresh group recipe; epochs and warmup epochs unchanged. No equivalence to the retired batch-8 recipe is claimed.')


def refresh_selection(root, source_path, job_states):
    """Caller owns the campaign lock. Return ready groups while others remain pending.

    Job states are supplied by the controller; this helper never queries or changes
    the scheduler. Previously accepted evidence/recipes are immutable and rechecked.
    """
    root, source = Path(root).resolve(), Path(source_path).resolve()
    manifest = _read(source / 'source_manifest.json')
    for name, expected in manifest.items():
        if file_hash(source / name) != expected:
            raise ValueError('Final source snapshot changed: ' + name)
    launch = _read(root / 'outputs/batch_profile/launch.json')
    path = root / 'outputs/batch_profile/selection.json'
    previous = _read(path) if path.exists() else {'groups': {}}
    groups, pending, blocked = {}, [], {}
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            key = dataset + '/' + architecture
            mapping = launch.get('large_memory_jobs' if architecture == 'vit_small' else 'jobs', {})
            job = str(mapping[key]) if key in mapping else None
            status = job_states.get(job, {}) if job else {}
            status = status.get('state') if isinstance(status, dict) else status
            if key in previous['groups'] and status in FAILED_STATES:
                raise ValueError('Accepted profile job no longer has successful scheduler evidence: ' + key)
            if key not in previous['groups'] and status != 'COMPLETED':
                if status in FAILED_STATES:
                    blocked[key] = dict(job_id=job, state=status, reason='Preferred profile job did not complete successfully')
                else:
                    pending.append(key)
                continue
            try:
                selected = _selection(root, source, launch, dataset, architecture, job)
            except (ValueError, KeyError, FileNotFoundError) as error:
                if key in previous['groups']:
                    raise ValueError('Frozen group evidence changed: ' + key) from error
                blocked[key] = dict(job_id=job, state=status, reason=str(error))
                continue
            if key in previous['groups'] and previous['groups'][key] != selected:
                raise ValueError('Frozen group selection changed: ' + key)
            groups[key] = selected
    ledger = dict(schema_version=1, state='blocked' if blocked else 'verified' if len(groups) == 4 else 'pending',
        groups=groups, pending_groups=pending, blocked_groups=blocked,
        recipe_by_group={key: value['recipe'] for key, value in groups.items()})
    if previous != ledger:
        atomic_json(path, ledger)
    return ledger
