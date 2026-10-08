#!/usr/bin/env python
"""Real-data FP32 training capacity/throughput characterization, never accuracy evidence.

Each condition/batch gets fresh parameters, optimizer, allocator and CUDA context.
Only this diagnostic's output/STOP is observed: the retired campaign may be stopped.
The parent uses the standard library and never creates a CUDA context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def stopped(output):
    if (Path(output) / 'STOP').exists():
        raise InterruptedError('Batch characterization output/STOP marker')


def source_files(source):
    names = sorted((Path(source) / 'sparse_contrast').glob('*.py'))
    return {str(p.relative_to(source)): sha(p) for p in names}


def verify_plan(plan):
    if source_files(plan['source_root']) != plan['source_files']:
        raise ValueError('Characterization source bytes changed; freeze source before profiling')
    if sha(__file__) != plan['profiler_sha256']:
        raise ValueError('Characterization script changed after plan creation')
    root = Path(plan['root'])
    for name, expected in plan['input_files'].items():
        if sha(root / name) != expected:
            raise ValueError('Characterization input changed: ' + name)


def condition_key(config):
    p = config['sparsity_percent']
    return f"{config['representation']}_{config['execution']}_p{'NA' if p is None else p}"


def import_worker(plan):
    # Set before importing torch/common/data; immutable source may live elsewhere.
    os.environ['SPARSE_CONTRAST_ROOT'] = plan['root']
    sys.path.insert(0, plan['source_root'])
    import torch
    from sparse_contrast.common import seed_all
    from sparse_contrast.benchmark import device_metadata, gpu_isolation
    seed_all(plan['seed'])
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Real training characterization requires an allocated Slurm GPU')
    device = torch.device('cuda')
    isolation = gpu_isolation(device)
    metadata = device_metadata(device)
    return torch, device, metadata, isolation


def dataset_for(config, seed):
    from sparse_contrast.data import load_clean, StatelessAugment
    dataset = load_clean(config['dataset'], 'train', protocol=config['protocol'])
    if config.get('subset_ids') is not None:
        raise ValueError('Resource characterization requires the complete real training partition')
    if config['dataset'] == 'cifar10' and not config.get('disable_augmentation', False):
        dataset = StatelessAugment(dataset, seed)
        dataset.set_epoch(0)
    return dataset


def make_loader(dataset, batch, seed, workers, *, shuffle=True):
    import torch
    from torch.utils.data import DataLoader
    from sparse_contrast.data import EpochSampler
    sampler = EpochSampler(dataset, seed, shuffle=shuffle)
    sampler.set_epoch(0)
    return DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True,
                      num_workers=workers, pin_memory=True,
                      generator=torch.Generator().manual_seed(71235))


def scan_support(plan, destination):
    """Select upper-tail capacity panels from ALL real epoch-zero train images.

    Only identities and patch counts are cached. Timed workers recompute the
    online frontend from actual uint8 images; no frontend activations are cached.
    """
    torch, device, metadata, isolation = import_worker(plan)
    from torch.utils.data import DataLoader
    from sparse_contrast.frontend import Frontend, sparsify, patch_support
    from sparse_contrast.benchmark import gpu_isolation
    dataset = dataset_for(plan['conditions'][0], plan['seed'])
    reps = sorted({c['representation'] for c in plan['conditions'] if c['representation'] != 'raw'})
    frontends = {rep: Frontend(rep).to(device) for rep in reps}
    counts = {condition_key(c): [] for c in plan['conditions'] if c['execution'] == 'compact'}
    identities = []
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=plan['num_workers'],
                        pin_memory=True, generator=torch.Generator().manual_seed(71235))
    started = time.monotonic()
    with torch.inference_mode():
        for raw, _, ids in loader:
            stopped(plan['output'])
            identities.extend(ids)
            raw = raw.to(device, non_blocking=True).float() / 255.
            for rep, frontend in frontends.items():
                coefficients = frontend(raw)
                for c in plan['conditions']:
                    if c['representation'] == rep and c['execution'] == 'compact':
                        values = sparsify(coefficients, c['sparsity_percent'])
                        counts[condition_key(c)].extend(patch_support(values, 2, 0.).sum(1).cpu().tolist())
    if len(identities) != len(dataset) or len(set(identities)) != len(dataset):
        raise ValueError('Full training partition support scan coverage mismatch')
    panels = {}
    largest = max(plan['candidates'])
    for key, values in counts.items():
        order = sorted(range(len(values)), key=lambda i: (-values[i], i))[:largest]
        panels[key] = {'indices': order, 'sample_ids': [identities[i] for i in order],
                       'retained_patch_counts': [values[i] for i in order],
                       'all_counts_sha256': digest(values), 'minimum_retained': min(values),
                       'maximum_retained': max(values), 'mean_retained': statistics.mean(values)}
    verify_plan(plan)
    result = {'state': 'completed', 'plan_hash': plan['plan_hash'], 'metadata': metadata,
              'isolation_before': isolation, 'isolation_after': gpu_isolation(device),
              'dataset_count': len(dataset), 'split_hash': dataset.split_hash,
              'visited_sample_ids_sha256': digest(identities), 'panels': panels,
              'elapsed_seconds': time.monotonic() - started,
              'policy': 'All clean training images with production epoch-zero augmentation; largest retained-patch-count panel per compact condition. This stresses observed support, not every possible future augmentation.'}
    save(destination, result)


def profile_one(plan, condition, batch, destination):
    torch, device, metadata, isolation = import_worker(plan)
    from torch.utils.data import Subset
    from sparse_contrast.common import initialize_matched, flatten_metrics
    from sparse_contrast.pipeline import Classifier
    from sparse_contrast.metrics import ClassificationCounts
    from sparse_contrast import train
    from sparse_contrast.benchmark import gpu_isolation
    config = plan['conditions'][condition]
    support = json.loads((Path(plan['output']) / 'support_panels.json').read_text())
    if support['plan_hash'] != plan['plan_hash']:
        raise ValueError('Support panel identity mismatch')
    for key in ('gpu', 'memory_bytes', 'driver_version', 'torch', 'cuda', 'cudnn', 'compute_capability'):
        if metadata[key] != support['metadata'][key]:
            raise ValueError('Characterization hardware class changed: ' + key)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    non_allocator_bytes = total_memory - free_before - baseline_reserved
    result = {'schema_version': 1, 'diagnostic': 'training_batch_resource_characterization',
              'accuracy_evidence': False, 'latency_benchmark_evidence': False,
              'plan_hash': plan['plan_hash'], 'condition': condition_key(config),
              'template_registry_id': config['registry_id'], 'template_config_hash': config['config_hash'],
              'template_training_source_hash': config['source_hash'],
              'profile_source_files_sha256': digest(plan['source_files']),
              'batch_size': batch, 'gradient_accumulation': 1, 'precision': 'fp32',
              'normalization_hash': config['normalization_hash'], 'metadata': metadata,
              'isolation_before': isolation, 'free_bytes_before_model': free_before,
              'device_total_bytes': total_memory, 'non_allocator_bytes_at_start': non_allocator_bytes,
              'warmup_steps_requested': plan['warmup_steps'], 'measured_steps_requested': plan['measure_steps'],
              'fixed_profile_lr': plan['profile_lr'], 'phase': 'setup', 'optimizer_steps': 0}
    started = time.monotonic()
    try:
        dataset = dataset_for(config, plan['seed'])
        model = Classifier(config, train.load_stats(config))
        initialize_matched(model.backbone, plan['seed'])
        model.to(device).train()
        model.collect_diagnostics = True
        recipe = dict(config['recipe'], lr=plan['profile_lr'])
        optimizer = train.make_optimizer(model, recipe)
        result['parameters'] = sum(p.numel() for p in model.parameters())
        result['split_hash'] = dataset.split_hash
        result['dataset_count'] = len(dataset)
        diagnostics = train.TrainingDiagnostics() if hasattr(train, 'TrainingDiagnostics') else None
        result['telemetry_policy'] = 'production_TrainingDiagnostics' if diagnostics is not None else 'original_per_field_scalar_sync'
        legacy_totals = {}
        counts = ClassificationCounts()
        iterator = iter(make_loader(dataset, batch, plan['seed'], plan['num_workers']))
        records = []

        def step(raw, target, ids, phase):
            if len(target) != batch or len(ids) != batch or len(set(ids)) != batch:
                raise ValueError('Profiler must execute a full batch of distinct real images')
            raw = raw.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(raw)
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training logits/loss')
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['grad_clip'], error_if_nonfinite=True)
            if not float(gradient_norm) > 0:
                raise FloatingPointError('No training gradient flow')
            optimizer.step()
            # Production sample metrics and support telemetry are part of timing.
            if diagnostics is not None:
                yy, pred = torch.stack((target.detach(), logits.detach().argmax(1))).cpu().numpy()
            else:
                yy = target.detach().cpu().numpy()
                pred = logits.detach().argmax(1).cpu().numpy()
            counts.add(yy, pred, float(loss.detach()) * batch)
            mapping = {**model.latest_diagnostics, **model.backbone.latest_telemetry}
            if diagnostics is not None:
                diagnostics.update(mapping)
            else:
                for name, value in flatten_metrics(mapping):
                    if torch.is_tensor(value):
                        total, n = legacy_totals.get(name, (0., 0))
                        legacy_totals[name] = (total + float(value.detach().float().sum()), n + value.numel())
            result['optimizer_steps'] += 1
            records.append({'phase': phase, 'sample_ids': list(ids), 'sample_ids_sha256': digest(list(ids)),
                            'loss': float(loss.detach()), 'gradient_norm_before_clipping': float(gradient_norm)})

        # Include first optimizer-state allocation in the overall capacity peak.
        torch.cuda.reset_peak_memory_stats(device)
        result['phase'] = 'warmup'
        for _ in range(plan['warmup_steps']):
            stopped(plan['output'])
            step(*next(iterator), 'warmup')
        torch.cuda.synchronize(device)
        result['phase'] = 'measured'
        elapsed = []
        for _ in range(plan['measure_steps']):
            stopped(plan['output'])
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            step(*next(iterator), 'measured')
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - before)
        result['measurement_seconds'] = elapsed
        result['measured_samples'] = batch * len(elapsed)
        result['samples_per_second'] = result['measured_samples'] / sum(elapsed)
        result['median_step_seconds'] = statistics.median(elapsed)
        result['step_seconds_cv'] = statistics.stdev(elapsed) / statistics.mean(elapsed) if len(elapsed) > 1 else None
        result['timing_scope'] = 'DataLoader fetch/augmentation, transfer, online frontend/support, forward, loss, backward, gradient clipping, AdamW, production diagnostics and sample metrics; excludes model/data setup, epoch validation, checkpointing and TensorBoard.'
        result['phase'] = 'upper_support_capacity'
        if config['execution'] == 'compact':
            panel = support['panels'][condition_key(config)]
            selected = Subset(dataset, panel['indices'][:batch])
            stress = next(iter(make_loader(selected, batch, plan['seed'], plan['num_workers'], shuffle=False)))
            if list(stress[2]) != panel['sample_ids'][:batch]:
                raise ValueError('Upper-support capacity batch identity mismatch')
            step(*stress, 'upper_support_capacity')
            result['stress_retained_patch_counts'] = panel['retained_patch_counts'][:batch]
        torch.cuda.synchronize(device)
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        result.update(max_memory_allocated_bytes=allocated, max_memory_reserved_bytes=reserved,
                      estimated_peak_device_bytes=reserved + non_allocator_bytes,
                      estimated_headroom_fraction=1 - (reserved + non_allocator_bytes) / total_memory)
        # This happens outside timing and after recording the capacity peak.
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError('Nonfinite updated parameter: ' + name)
        result['finite_loss_gradients_parameters'] = True
        result['capacity_safe'] = result['estimated_headroom_fraction'] >= plan['headroom_fraction']
        result['support_telemetry'] = diagnostics.means() if diagnostics is not None else {k: v / n for k, (v, n) in legacy_totals.items() if n}
        result['state'] = 'passed'
        result['isolation_after'] = gpu_isolation(device)
        verify_plan(plan)
    except torch.cuda.OutOfMemoryError as error:
        result.update(state='oom', capacity_safe=False, error=str(error), traceback=traceback.format_exc(),
                      max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as error:
        result.update(state='failed', capacity_safe=False, error=repr(error), traceback=traceback.format_exc())
        save(destination, result)
        raise
    finally:
        result['total_worker_seconds'] = time.monotonic() - started
        result['sample_batches'] = locals().get('records', [])
    save(destination, result)


def run_child(command, output, timeout):
    with Path(output).open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise TimeoutError('Characterization child exceeded its explicit time budget; see ' + str(output))
    if code:
        raise RuntimeError(f'Characterization worker failed with exit {code}; see {output}')


def summarize(plan, results, complete):
    candidates = []
    for batch in plan['candidates']:
        rows = [r for r in results if r['batch_size'] == batch]
        safe = len(rows) == len(plan['conditions']) and all(r['state'] == 'passed' and r['capacity_safe'] for r in rows)
        score = len(rows) / sum(1 / r['samples_per_second'] for r in rows) if safe else None
        candidates.append({'batch_size': batch, 'conditions_recorded': len(rows), 'all_conditions_safe': safe,
                           'harmonic_mean_samples_per_second': score,
                           'minimum_headroom_fraction': min((r['estimated_headroom_fraction'] for r in rows if 'estimated_headroom_fraction' in r), default=None)})
    eligible = [r for r in candidates if r['all_conditions_safe']]
    best = max((r['harmonic_mean_samples_per_second'] for r in eligible), default=None)
    useful = [r['batch_size'] for r in eligible if r['harmonic_mean_samples_per_second'] >= (1 - plan['plateau_fraction']) * best] if eligible else []
    return {'state': 'completed' if complete else 'partial', 'plan_hash': plan['plan_hash'],
            'dataset': plan['dataset'], 'architecture': plan['architecture'], 'candidates': candidates,
            'conditions_required': len(plan['conditions']), 'records': len(results),
            'largest_capacity_safe_batch': max((r['batch_size'] for r in eligible), default=None),
            'proposed_useful_batch': max(useful) if useful and complete else None,
            'proposal_policy': 'Largest uniform batch within the stated fraction of the best harmonic-mean condition throughput, among candidates passing every condition and memory headroom. A short resource diagnostic, not an accuracy-optimal batch or scientific latency result.',
            'plateau_fraction': plan['plateau_fraction'], 'headroom_fraction': plan['headroom_fraction'],
            'results': results, 'updated_unix': time.time()}


def parent(args):
    root, source, output = args.root.resolve(), args.source_root.resolve(), args.output.resolve()
    campaign = json.loads(args.campaign.read_text())
    scope = json.loads((root / 'experiment_scope.json').read_text())
    reps = scope['active_representations'][args.dataset]
    expected = [('raw', 'dense', None)] + [(r, 'dense', 0) for r in reps] + [(r, 'compact', p) for r in reps for p in (0, 20, 40, 60, 80)]
    configs = []
    for rep, execution, percent in expected:
        matches = [x.get('config', x) for x in campaign['main'] if (x.get('config', x)['dataset'], x.get('config', x)['architecture'], x.get('config', x)['representation'], x.get('config', x)['execution'], x.get('config', x)['sparsity_percent'], x.get('config', x)['seed']) == (args.dataset, args.architecture, rep, execution, percent, args.seed)]
        if len(matches) != 1:
            raise ValueError(f'Missing or duplicate real condition {(rep, execution, percent)}')
        config = matches[0]
        if digest({k: v for k, v in config.items() if k != 'config_hash'}) != config['config_hash']:
            raise ValueError('Template resolved configuration digest mismatch')
        if config['precision'] != 'fp32':
            raise ValueError('This characterization requires the production FP32 policy')
        configs.append(config)
    names = ['datasets_manifest.json', 'requirements.lock.txt', 'experiment_scope.json']
    names += [f"normalization/{args.dataset}_{rep}_{configs[0]['protocol']}.json" for rep in reps]
    plan = {'schema_version': 1, 'root': str(root), 'source_root': str(source), 'output': str(output),
            'dataset': args.dataset, 'architecture': args.architecture, 'seed': args.seed,
            'conditions': configs, 'scope': scope, 'candidates': sorted(set(args.candidates)),
            'source_files': source_files(source), 'profiler_sha256': sha(__file__),
            'input_files': {name: sha(root / name) for name in names},
            'warmup_steps': args.warmup_steps, 'measure_steps': args.measure_steps,
            'headroom_fraction': args.headroom_fraction, 'plateau_fraction': args.plateau_fraction,
            'profile_lr': args.profile_lr, 'num_workers': args.num_workers}
    plan['plan_hash'] = digest(plan)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing characterization plan differs; use a distinct output directory')
    if not path.exists():
        save(path, plan)
    lock = output / 'writer.lock'
    lock.mkdir()
    save(lock / 'owner.json', {'pid': os.getpid(), 'job_id': os.getenv('SLURM_JOB_ID'), 'host': os.uname().nodename})
    results = []
    started = time.monotonic()
    try:
        stopped(output)
        verify_plan(plan)
        if args.plan_only:
            print(json.dumps({'state': 'planned', 'plan': str(path), 'conditions': len(configs), 'candidates': plan['candidates']}))
            return
        if not os.getenv('SLURM_JOB_ID'):
            raise RuntimeError('Profiling must be launched inside a Slurm GPU allocation')
        panels = output / 'support_panels.json'
        prefix = [sys.executable, str(Path(__file__).resolve()), '--worker-plan', str(path)]
        if not panels.exists():
            run_child(prefix + ['--scan-support'], output / 'support_scan.log', args.worker_timeout_seconds)
        elif json.loads(panels.read_text())['plan_hash'] != plan['plan_hash']:
            raise ValueError('Saved support scan belongs to another plan')
        capacity_limit = None
        for batch in plan['candidates']:
            for condition, config in enumerate(configs):
                stopped(output)
                if time.monotonic() - started > args.max_minutes * 60:
                    save(output / 'summary.json', summarize(plan, results, False))
                    raise TimeoutError('Characterization total time budget reached; valid completed workers are resumable')
                destination = output / 'workers' / condition_key(config) / f'batch{batch}.json'
                if destination.exists():
                    row = json.loads(destination.read_text())
                    if row['plan_hash'] != plan['plan_hash'] or row['batch_size'] != batch or row['condition'] != condition_key(config):
                        raise ValueError('Saved worker identity mismatch')
                elif capacity_limit is not None:
                    row = {'state': 'not_tested_after_group_capacity_limit', 'plan_hash': plan['plan_hash'], 'condition': condition_key(config),
                           'batch_size': batch, 'capacity_safe': False, 'group_capacity_limit': capacity_limit}
                    save(destination, row)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    run_child(prefix + ['--condition-index', str(condition), '--worker-batch', str(batch)],
                              destination.with_suffix('.log'), args.worker_timeout_seconds)
                    row = json.loads(destination.read_text())
                if row['state'] == 'failed':
                    raise RuntimeError('Saved deterministic worker failure requires diagnosis: ' + str(destination))
                if row['state'] == 'oom' or row['state'] == 'passed' and not row['capacity_safe']:
                    capacity_limit = {'batch_size': batch, 'condition': condition_key(config), 'state': row['state'],
                                      'reason': 'One failing condition excludes this and larger batches from a uniform safe group choice'}
                results.append(row)
                save(output / 'summary.json', summarize(plan, results, False))
                print(json.dumps({'condition': condition_key(config), 'batch': batch, 'state': row['state'],
                                  'samples_per_second': row.get('samples_per_second'), 'capacity_safe': row['capacity_safe']}), flush=True)
        save(output / 'summary.json', summarize(plan, results, True))
    except BaseException as error:
        save(output / f'failure-{time.time_ns()}.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'job_id': os.getenv('SLURM_JOB_ID')})
        raise
    finally:
        (lock / 'owner.json').unlink()
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'])
    parser.add_argument('--architecture', choices=['vit_small', 'swin_tiny'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--candidates', type=int, nargs='+', default=[64, 128, 256, 512, 1024])
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--measure-steps', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--headroom-fraction', type=float, default=.15)
    parser.add_argument('--plateau-fraction', type=float, default=.05)
    parser.add_argument('--profile-lr', type=float, default=3e-4)
    parser.add_argument('--max-minutes', type=float, default=90.)
    parser.add_argument('--worker-timeout-seconds', type=float, default=900.)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--scan-support', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--condition-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--worker-batch', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTHONHASHSEED', '0')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', os.environ['OMP_NUM_THREADS'])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.worker_plan:
        plan = json.loads(args.worker_plan.read_text())
        verify_plan(plan)
        if args.scan_support:
            scan_support(plan, Path(plan['output']) / 'support_panels.json')
        else:
            config = plan['conditions'][args.condition_index]
            destination = Path(plan['output']) / 'workers' / condition_key(config) / f'batch{args.worker_batch}.json'
            profile_one(plan, args.condition_index, args.worker_batch, destination)
        return
    if any(getattr(args, name) is None for name in ('root', 'source_root', 'campaign', 'dataset', 'architecture', 'output')):
        parser.error('--root, --source-root, --campaign, --dataset, --architecture and --output are required')
    if min(args.candidates) < 1 or args.warmup_steps < 1 or args.measure_steps < 3 or args.num_workers < 0:
        parser.error('Positive batches, at least one warmup and three measured full batches are required')
    if not 0 < args.headroom_fraction < 1 or not 0 <= args.plateau_fraction < 1:
        parser.error('Invalid headroom or plateau fraction')
    parent(args)


if __name__ == '__main__':
    main()
